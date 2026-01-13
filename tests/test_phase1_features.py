"""
Test suite for new Phase 1 features:
- Calendar features
- Prediction intervals
- Visualization
- Progress bars
- Parallel processing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive matplotlib backend for tests
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_daily_data():
    """Generate sample daily time series data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Create data with trend, seasonality, and noise
    t = np.arange(len(dates))
    trend = 0.05 * t
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)
    yearly_seasonality = 20 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 5, len(dates))
    
    y = pd.DataFrame({
        'sales': 100 + trend + weekly_seasonality + yearly_seasonality + noise,
        'revenue': 200 + 1.5 * trend + 0.8 * weekly_seasonality + noise * 2
    }, index=dates)
    
    return y


@pytest.fixture
def sample_hourly_data():
    """Generate sample hourly time series data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=24*30, freq='H')  # 30 days
    
    t = np.arange(len(dates))
    daily_pattern = 10 * np.sin(2 * np.pi * t / 24)
    weekly_pattern = 5 * np.sin(2 * np.pi * t / (24 * 7))
    noise = np.random.normal(0, 2, len(dates))
    
    y = pd.DataFrame({
        'demand': 50 + daily_pattern + weekly_pattern + noise
    }, index=dates)
    
    return y


@pytest.fixture
def sample_covariates(sample_daily_data):
    """Generate sample covariates."""
    dates = sample_daily_data.index
    np.random.seed(42)
    
    X = pd.DataFrame({
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 3, len(dates)),
        'promo': np.random.binomial(1, 0.2, len(dates)),
        'holiday': np.random.binomial(1, 0.05, len(dates))
    }, index=dates)
    
    return X


# =============================================================================
# Test Calendar Features
# =============================================================================

class TestCalendarFeatures:
    """Test CalendarFeatures class."""
    
    def test_basic_calendar_features(self, sample_daily_data):
        """Test basic calendar feature extraction."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures()
        features = cal.fit_transform(sample_daily_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_daily_data)
        assert features.index.equals(sample_daily_data.index)
        
        # Check that some features exist
        feature_cols = features.columns.tolist()
        assert any('dayofweek' in c or 'weekend' in c for c in feature_cols)
        assert any('month' in c for c in feature_cols)
    
    def test_cyclical_encoding(self, sample_daily_data):
        """Test cyclical encoding for periodic features."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures(
            features=['dayofweek', 'month'],
            cyclical_encoding=True
        )
        features = cal.fit_transform(sample_daily_data)
        
        # Should have sin/cos columns
        assert any('sin' in c for c in features.columns)
        assert any('cos' in c for c in features.columns)
    
    def test_no_cyclical_encoding(self, sample_daily_data):
        """Test without cyclical encoding."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures(
            features=['dayofweek', 'month'],
            cyclical_encoding=False
        )
        features = cal.fit_transform(sample_daily_data)
        
        # Should have raw values
        assert 'dayofweek' in features.columns
        assert 'month' in features.columns
        
        # Check value ranges
        assert features['dayofweek'].min() >= 0
        assert features['dayofweek'].max() <= 6
        assert features['month'].min() >= 1
        assert features['month'].max() <= 12
    
    def test_transform_future(self, sample_daily_data):
        """Test generating features for future dates."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures()
        cal.fit(sample_daily_data)
        
        future_features = cal.transform_future(horizon=30)
        
        assert len(future_features) == 30
        assert future_features.index[0] > sample_daily_data.index[-1]
    
    def test_fourier_terms(self, sample_daily_data):
        """Test Fourier terms for seasonality."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures(
            features=['dayofweek'],
            fourier_terms={'yearly': (365.25, 3), 'weekly': (7, 2)}
        )
        features = cal.fit_transform(sample_daily_data)
        
        # Check Fourier columns exist
        assert any('yearly_sin' in c for c in features.columns)
        assert any('yearly_cos' in c for c in features.columns)
        assert any('weekly_sin' in c for c in features.columns)
    
    def test_hourly_features(self, sample_hourly_data):
        """Test features for hourly data."""
        from autotsforecast.features.calendar import CalendarFeatures
        
        cal = CalendarFeatures(features=['hour', 'dayofweek'])
        features = cal.fit_transform(sample_hourly_data)
        
        # Should detect hourly frequency and include hour features
        assert any('hour' in c for c in features.columns)
    
    def test_convenience_function(self, sample_daily_data):
        """Test add_calendar_features convenience function."""
        from autotsforecast.features.calendar import add_calendar_features
        
        result = add_calendar_features(sample_daily_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should have original columns plus calendar features
        assert len(result.columns) > len(sample_daily_data.columns)


# =============================================================================
# Test Feature Engine
# =============================================================================

class TestFeatureEngine:
    """Test FeatureEngine class."""
    
    def test_basic_feature_engine(self, sample_daily_data):
        """Test basic feature engine."""
        from autotsforecast.features.engine import FeatureEngine
        
        engine = FeatureEngine(
            calendar_features=True,
            lag_features=[1, 7],
            rolling_features={'mean': [7]}
        )
        
        features = engine.fit_transform(sample_daily_data)
        
        assert isinstance(features, pd.DataFrame)
        # Should have lag features
        assert any('lag' in c for c in features.columns)
        # Should have rolling features
        assert any('rolling' in c for c in features.columns)
    
    def test_with_covariates(self, sample_daily_data, sample_covariates):
        """Test feature engine with external covariates."""
        from autotsforecast.features.engine import FeatureEngine
        
        engine = FeatureEngine(
            calendar_features=['dayofweek'],
            lag_features=[1, 7]
        )
        
        features = engine.fit_transform(sample_daily_data, sample_covariates)
        
        # Should include covariate columns
        assert 'temperature' in features.columns
        assert 'promo' in features.columns


# =============================================================================
# Test Prediction Intervals
# =============================================================================

class TestPredictionIntervals:
    """Test PredictionIntervals class."""
    
    def test_conformal_intervals(self, sample_daily_data):
        """Test conformal prediction intervals."""
        from autotsforecast.uncertainty.intervals import PredictionIntervals
        from autotsforecast.models.base import MovingAverageForecaster
        
        # Split data
        train = sample_daily_data.iloc[:-30]
        
        # Fit a simple model
        model = MovingAverageForecaster(horizon=14, window=7)
        model.fit(train)
        forecasts = model.predict()
        
        # Create prediction intervals
        pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
        pi.fit(model, train)
        
        intervals = pi.predict(forecasts)
        
        # Check structure
        assert 'point' in intervals
        assert 'lower_80' in intervals
        assert 'upper_80' in intervals
        assert 'lower_95' in intervals
        assert 'upper_95' in intervals
        
        # Check interval ordering
        for col in forecasts.columns:
            assert (intervals['lower_95'][col] <= intervals['lower_80'][col]).all()
            assert (intervals['upper_80'][col] <= intervals['upper_95'][col]).all()
            assert (intervals['lower_80'][col] <= intervals['point'][col]).all()
            assert (intervals['point'][col] <= intervals['upper_80'][col]).all()
    
    def test_residual_intervals(self, sample_daily_data):
        """Test residual-based intervals."""
        from autotsforecast.uncertainty.intervals import PredictionIntervals
        from autotsforecast.models.base import MovingAverageForecaster
        
        train = sample_daily_data.iloc[:-30]
        
        model = MovingAverageForecaster(horizon=14, window=7)
        model.fit(train)
        forecasts = model.predict()
        
        pi = PredictionIntervals(method='residual', coverage=0.95)
        pi.fit(model, train)
        
        intervals = pi.predict(forecasts)
        
        assert 'lower_95' in intervals
        assert 'upper_95' in intervals
    
    def test_conformal_predictor_online(self, sample_daily_data):
        """Test online conformal predictor."""
        from autotsforecast.uncertainty.intervals import ConformalPredictor
        from autotsforecast.models.base import MovingAverageForecaster
        
        train = sample_daily_data.iloc[:-30]
        
        model = MovingAverageForecaster(horizon=1, window=7)
        model.fit(train)
        
        cp = ConformalPredictor(coverage=0.90)
        cp.fit(model, train)
        
        # Make prediction with interval
        forecast = model.predict()
        point, lower, upper = cp.predict(forecast)
        
        assert lower is not None
        assert upper is not None


# =============================================================================
# Test Visualization
# =============================================================================

class TestVisualization:
    """Test visualization functions."""
    
    def test_plot_forecast(self, sample_daily_data):
        """Test basic forecast plot."""
        pytest.importorskip("matplotlib")
        from autotsforecast.visualization.plots import plot_forecast
        from autotsforecast.models.base import MovingAverageForecaster
        
        train = sample_daily_data.iloc[:-30]
        test = sample_daily_data.iloc[-30:]
        
        model = MovingAverageForecaster(horizon=30, window=7)
        model.fit(train)
        forecasts = model.predict()
        forecasts.index = test.index
        
        fig = plot_forecast(train, forecasts, y_test=test)
        
        assert fig is not None
    
    def test_plot_with_intervals(self, sample_daily_data):
        """Test forecast plot with prediction intervals."""
        pytest.importorskip("matplotlib")
        from autotsforecast.visualization.plots import plot_forecast
        from autotsforecast.models.base import MovingAverageForecaster
        from autotsforecast.uncertainty.intervals import PredictionIntervals
        
        train = sample_daily_data.iloc[:-30]
        
        model = MovingAverageForecaster(horizon=14, window=7)
        model.fit(train)
        forecasts = model.predict()
        
        pi = PredictionIntervals(method='residual', coverage=[0.80, 0.95])
        pi.fit(model, train)
        intervals = pi.predict(forecasts)
        
        fig = plot_forecast(train, forecasts, intervals=intervals)
        
        assert fig is not None
    
    def test_plot_model_comparison(self, sample_daily_data):
        """Test model comparison plot."""
        pytest.importorskip("matplotlib")
        from autotsforecast.visualization.plots import plot_model_comparison
        
        cv_results = {
            'ModelA': {'rmse': 10.5, 'mae': 8.2, 'r2': 0.85},
            'ModelB': {'rmse': 12.3, 'mae': 9.5, 'r2': 0.80},
            'ModelC': {'rmse': 11.0, 'mae': 8.8, 'r2': 0.82},
        }
        
        fig = plot_model_comparison(cv_results, metric='rmse')
        
        assert fig is not None
    
    def test_plot_residuals(self, sample_daily_data):
        """Test residual diagnostic plots."""
        pytest.importorskip("matplotlib")
        from autotsforecast.visualization.plots import plot_residuals
        from autotsforecast.models.base import MovingAverageForecaster
        
        train = sample_daily_data.iloc[:-30]
        test = sample_daily_data.iloc[-30:-16]  # Match 14-day horizon
        
        model = MovingAverageForecaster(horizon=14, window=7)
        model.fit(train)
        forecasts = model.predict()
        forecasts.index = test.index
        
        fig = plot_residuals(test, forecasts)
        
        assert fig is not None


# =============================================================================
# Test Progress Tracking
# =============================================================================

class TestProgressTracking:
    """Test progress bar functionality."""
    
    def test_progress_tracker(self):
        """Test basic progress tracker."""
        from autotsforecast.visualization.progress import ProgressTracker
        
        items = list(range(10))
        results = []
        
        with ProgressTracker(total=len(items), description="Testing") as pbar:
            for item in items:
                results.append(item * 2)
                pbar.update(1)
        
        assert len(results) == 10
    
    def test_progress_track_iterator(self):
        """Test progress tracker as iterator wrapper."""
        from autotsforecast.visualization.progress import ProgressTracker
        
        items = list(range(10))
        results = []
        
        for item in ProgressTracker.track(items, description="Iterating"):
            results.append(item * 2)
        
        assert len(results) == 10
    
    def test_progress_bar_function(self):
        """Test progress_bar convenience function."""
        from autotsforecast.visualization.progress import progress_bar
        
        items = list(range(10))
        results = []
        
        for item in progress_bar(items, description="Processing"):
            results.append(item)
        
        assert len(results) == 10


# =============================================================================
# Test Parallel Processing
# =============================================================================

class TestParallelProcessing:
    """Test parallel processing utilities."""
    
    def test_parallel_map(self):
        """Test parallel map function."""
        from autotsforecast.utils.parallel import parallel_map
        
        items = list(range(20))
        results = parallel_map(lambda x: x * 2, items, n_jobs=2)
        
        assert len(results) == 20
        assert results[0] == 0
        assert results[10] == 20
    
    def test_parallel_forecaster(self, sample_daily_data):
        """Test ParallelForecaster."""
        from autotsforecast.utils.parallel import ParallelForecaster
        from autotsforecast.models.base import MovingAverageForecaster
        
        models = [
            MovingAverageForecaster(horizon=7, window=3),
            MovingAverageForecaster(horizon=7, window=5),
            MovingAverageForecaster(horizon=7, window=7),
        ]
        
        pf = ParallelForecaster(n_jobs=2)
        fitted = pf.parallel_fit(models, sample_daily_data.iloc[:100])
        
        # Check all models are fitted
        valid_fitted = [m for m in fitted if not isinstance(m, Exception)]
        assert len(valid_fitted) == 3
    
    def test_optimal_n_jobs(self):
        """Test optimal n_jobs detection."""
        from autotsforecast.utils.parallel import get_optimal_n_jobs
        
        # Auto-detect
        n = get_optimal_n_jobs(100, -1)
        assert n > 0
        
        # Fixed
        n = get_optimal_n_jobs(100, 4)
        assert n == 4
        
        # More tasks than requested jobs
        n = get_optimal_n_jobs(2, 8)
        assert n == 2


# =============================================================================
# Test AutoForecaster with New Features
# =============================================================================

class TestAutoForecasterNewFeatures:
    """Test AutoForecaster with new features."""
    
    def test_forecast_with_intervals(self, sample_daily_data):
        """Test forecast_with_intervals method."""
        from autotsforecast import AutoForecaster
        from autotsforecast.models.base import MovingAverageForecaster
        
        train = sample_daily_data.iloc[:-30]
        
        # Create simple candidates
        candidates = [
            MovingAverageForecaster(horizon=14, window=5),
            MovingAverageForecaster(horizon=14, window=7),
        ]
        
        auto = AutoForecaster(
            candidate_models=candidates,
            n_splits=2,
            test_size=14,
            verbose=False
        )
        auto.fit(train)
        
        # Get forecasts with intervals
        results = auto.forecast_with_intervals(
            train,
            coverage=[0.80, 0.95],
            method='residual'
        )
        
        assert 'point' in results
        assert 'lower_80' in results
        assert 'upper_80' in results
        assert 'lower_95' in results
        assert 'upper_95' in results


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

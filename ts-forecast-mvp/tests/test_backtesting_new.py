"""Tests for backtesting validator"""

import numpy as np
import pandas as pd
import pytest
from autotsforecast.backtesting.validator import BacktestValidator
from autotsforecast.models.base import VARForecaster, MovingAverageForecaster


@pytest.fixture
def sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'sales_A': np.random.randn(100).cumsum() + 100,
        'sales_B': np.random.randn(100).cumsum() + 150
    }, index=dates)
    return data


class TestBacktestValidator:
    """Tests for Backtest Validator"""
    
    def test_initialization(self):
        """Test validator initialization"""
        model = VARForecaster(horizon=1, lags=2)
        validator = BacktestValidator(model, n_splits=3, test_size=5)
        
        assert validator.n_splits == 3
        assert validator.test_size == 5
        assert validator.window_type == 'expanding'
    
    def test_expanding_window(self, sample_data):
        """Test expanding window backtesting"""
        model = VARForecaster(horizon=1, lags=1)
        validator = BacktestValidator(model, n_splits=3, test_size=10, window_type='expanding')
        
        metrics = validator.run(sample_data)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert len(validator.results) == 3
    
    def test_rolling_window(self, sample_data):
        """Test rolling window backtesting"""
        model = MovingAverageForecaster(horizon=1, window=3)
        validator = BacktestValidator(model, n_splits=3, test_size=10, window_type='rolling')
        
        metrics = validator.run(sample_data)
        assert len(validator.results) == 3
    
    def test_fold_results(self, sample_data):
        """Test fold-by-fold results"""
        model = VARForecaster(horizon=1, lags=2)
        validator = BacktestValidator(model, n_splits=3, test_size=10)
        validator.run(sample_data)
        
        fold_df = validator.get_fold_results()
        assert len(fold_df) == 3
        assert 'fold' in fold_df.columns
        assert 'train_size' in fold_df.columns
        assert 'test_size' in fold_df.columns
        assert 'rmse' in fold_df.columns
    
    def test_summary_statistics(self, sample_data):
        """Test summary statistics"""
        model = VARForecaster(horizon=1, lags=1)
        validator = BacktestValidator(model, n_splits=3, test_size=10)
        validator.run(sample_data)
        
        summary = validator.get_summary()
        assert 'mean' in summary.index
        assert 'std' in summary.index
        assert 'min' in summary.index
        assert 'max' in summary.index
    
    def test_get_predictions(self, sample_data):
        """Test getting all predictions"""
        model = VARForecaster(horizon=1, lags=1)
        validator = BacktestValidator(model, n_splits=3, test_size=10)
        validator.run(sample_data)
        
        actuals, predictions = validator.get_predictions()
        assert len(actuals) == 30  # 3 folds * 10 test_size
        assert len(predictions) == 30
        assert actuals.columns.tolist() == sample_data.columns.tolist()
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        model = VARForecaster(horizon=1, lags=1)
        validator = BacktestValidator(model, n_splits=10, test_size=10)
        
        small_data = pd.DataFrame({
            'sales': np.random.randn(20)
        })
        
        with pytest.raises(ValueError):
            validator.run(small_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

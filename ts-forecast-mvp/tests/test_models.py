"""Tests for model classes"""

import numpy as np
import pandas as pd
import pytest
from autotsforecast.models.base import BaseForecaster, VARForecaster, LinearForecaster, MovingAverageForecaster
from autotsforecast.models.selection import ModelSelector


@pytest.fixture
def sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'series_1': np.random.randn(100).cumsum() + 100,
        'series_2': np.random.randn(100).cumsum() + 150
    }, index=dates)
    return data


@pytest.fixture
def sample_covariates():
    """Generate sample covariates"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    covariates = pd.DataFrame({
        'price': 10 + np.random.randn(100),
        'promotion': np.random.choice([0, 1], 100)
    }, index=dates)
    return covariates


class TestVARForecaster:
    """Tests for VAR Forecaster"""
    
    def test_fit_predict(self, sample_data):
        """Test basic fit and predict"""
        model = VARForecaster(horizon=1, lags=2)
        model.fit(sample_data)
        
        assert model.is_fitted
        assert model.feature_names == sample_data.columns.tolist()
        
        predictions = model.predict()
        assert predictions.shape == (1, 2)
        assert all(predictions.columns == sample_data.columns)
    
    def test_multiple_horizon(self, sample_data):
        """Test with multiple horizon"""
        model = VARForecaster(horizon=5, lags=3)
        model.fit(sample_data)
        predictions = model.predict()
        assert predictions.shape == (5, 2)
    
    def test_fit_predict_workflow(self, sample_data):
        """Test fit_predict method"""
        model = VARForecaster(horizon=1, lags=1)
        predictions = model.fit_predict(sample_data)
        assert predictions.shape == (1, 2)


class TestLinearForecaster:
    """Tests for Linear Forecaster"""
    
    def test_requires_covariates(self, sample_data):
        """Test that LinearForecaster requires X"""
        model = LinearForecaster(horizon=1)
        with pytest.raises(ValueError):
            model.fit(sample_data, X=None)
    
    def test_fit_predict_with_covariates(self, sample_data, sample_covariates):
        """Test fit and predict with covariates"""
        model = LinearForecaster(horizon=1)
        model.fit(sample_data, sample_covariates)
        
        assert model.is_fitted
        predictions = model.predict(sample_covariates)
        assert predictions.shape == (1, 2)
    
    def test_multiple_horizon(self, sample_data, sample_covariates):
        """Test with multiple horizons"""
        model = LinearForecaster(horizon=3)
        model.fit(sample_data, sample_covariates)
        predictions = model.predict(sample_covariates)
        assert predictions.shape == (3, 2)


class TestMovingAverageForecaster:
    """Tests for Moving Average Forecaster"""
    
    def test_fit_predict(self, sample_data):
        """Test basic fit and predict"""
        model = MovingAverageForecaster(horizon=1, window=5)
        model.fit(sample_data)
        
        assert model.is_fitted
        predictions = model.predict()
        assert predictions.shape == (1, 2)
    
    def test_different_windows(self, sample_data):
        """Test with different window sizes"""
        for window in [3, 5, 10]:
            model = MovingAverageForecaster(horizon=1, window=window)
            model.fit(sample_data)
            predictions = model.predict()
            assert predictions.shape == (1, 2)


class TestModelSelector:
    """Tests for Model Selector"""
    
    def test_default_models(self):
        """Test default model initialization"""
        selector = ModelSelector()
        assert len(selector.models) > 0
        assert selector.metric == 'rmse'
    
    def test_fit_and_select(self, sample_data):
        """Test model selection"""
        selector = ModelSelector(metric='rmse')
        selector.fit(sample_data, validation_split=0.2, cv_folds=1)
        
        assert selector.best_model is not None
        assert len(selector.results) > 0
    
    def test_cross_validation(self, sample_data):
        """Test with cross-validation"""
        selector = ModelSelector(metric='mae')
        selector.fit(sample_data, cv_folds=3)
        
        results = selector.get_results()
        assert len(results) > 0
        
        # Check that results have mean and std
        for result in results.values():
            if isinstance(result, dict):
                assert 'mean_score' in result
                assert 'std_score' in result
    
    def test_predict_after_fit(self, sample_data):
        """Test prediction after model selection"""
        selector = ModelSelector()
        selector.fit(sample_data, validation_split=0.2, cv_folds=1)
        
        predictions = selector.predict()
        assert predictions.shape[1] == sample_data.shape[1]
    
    def test_get_best_model(self, sample_data):
        """Test getting best model"""
        selector = ModelSelector()
        selector.fit(sample_data, validation_split=0.2, cv_folds=1)
        
        best_name, best_model = selector.get_best_model()
        assert isinstance(best_name, str)
        assert isinstance(best_model, BaseForecaster)
    
    def test_different_metrics(self, sample_data):
        """Test with different metrics"""
        for metric in ['rmse', 'mae', 'mape']:
            selector = ModelSelector(metric=metric)
            selector.fit(sample_data, validation_split=0.2, cv_folds=1)
            assert selector.best_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
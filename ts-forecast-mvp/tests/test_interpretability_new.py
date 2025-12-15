"""Tests for driver analysis and interpretability"""

import numpy as np
import pandas as pd
import pytest
from ts_forecast.interpretability.drivers import DriverAnalyzer
from ts_forecast.models.base import LinearForecaster


@pytest.fixture
def sample_data_with_covariates():
    """Generate sample data with covariates"""
    np.random.seed(42)
    n = 100
    
    X = pd.DataFrame({
        'price': 10 + np.random.randn(n) * 2,
        'promotion': np.random.choice([0, 1], n),
        'temperature': 20 + np.random.randn(n) * 5
    })
    
    # Create target influenced by covariates
    y = pd.DataFrame({
        'sales': (
            -2 * X['price'] +
            30 * X['promotion'] +
            1.5 * X['temperature'] +
            np.random.randn(n) * 5 +
            100
        )
    })
    
    return X, y


class TestDriverAnalyzer:
    """Tests for Driver Analyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        model = LinearForecaster(horizon=1)
        analyzer = DriverAnalyzer(model)
        
        assert analyzer.model == model
        assert analyzer.feature_importance is None
    
    def test_coefficient_importance(self, sample_data_with_covariates):
        """Test coefficient-based feature importance"""
        X, y = sample_data_with_covariates
        
        model = LinearForecaster(horizon=1)
        model.fit(y, X)
        
        analyzer = DriverAnalyzer(model, feature_names=X.columns.tolist())
        importance = analyzer.calculate_feature_importance(X, y, method='coefficients')
        
        assert importance.shape[0] == X.shape[1]  # One row per feature
        assert importance.shape[1] == y.shape[1]  # One column per target
        assert all(importance.index == X.columns)
    
    def test_sensitivity_analysis(self, sample_data_with_covariates):
        """Test sensitivity analysis"""
        X, y = sample_data_with_covariates
        
        model = LinearForecaster(horizon=1)
        model.fit(y, X)
        
        analyzer = DriverAnalyzer(model)
        sensitivity = analyzer.calculate_feature_importance(X, y, method='sensitivity')
        
        assert sensitivity.shape[0] == X.shape[1]
        assert all(sensitivity.values >= 0)  # Sensitivity should be non-negative
    
    def test_comprehensive_analysis(self, sample_data_with_covariates):
        """Test comprehensive driver analysis"""
        X, y = sample_data_with_covariates
        
        model = LinearForecaster(horizon=1)
        model.fit(y, X)
        
        analyzer = DriverAnalyzer(model)
        analysis = analyzer.analyze_drivers(
            X, y,
            numerical_features=['price', 'temperature'],
            categorical_features=['promotion']
        )
        
        assert 'numerical_features' in analysis
        assert 'categorical_features' in analysis
        assert 'sensitivity' in analysis
        assert 'categorical_analysis' in analysis
    
    def test_requires_fitted_model(self, sample_data_with_covariates):
        """Test that analyzer requires fitted model"""
        X, y = sample_data_with_covariates
        
        model = LinearForecaster(horizon=1)
        # Don't fit the model
        
        analyzer = DriverAnalyzer(model)
        
        with pytest.raises(ValueError):
            analyzer.calculate_feature_importance(X, y, method='coefficients')
    
    def test_coefficient_importance_linear_only(self, sample_data_with_covariates):
        """Test that coefficient importance only works with LinearForecaster"""
        from ts_forecast.models.base import VARForecaster
        X, y = sample_data_with_covariates
        
        # VAR model doesn't have coefficients in the same way
        model = VARForecaster(horizon=1, lags=1)
        
        analyzer = DriverAnalyzer(model)
        
        with pytest.raises(ValueError):
            analyzer.calculate_feature_importance(X, y, method='coefficients')
    
    def test_categorical_analysis(self, sample_data_with_covariates):
        """Test categorical feature analysis"""
        X, y = sample_data_with_covariates
        
        model = LinearForecaster(horizon=1)
        model.fit(y, X)
        
        analyzer = DriverAnalyzer(model)
        cat_analysis = analyzer._analyze_categorical(X, y, ['promotion'])
        
        assert 'promotion' in cat_analysis
        assert len(cat_analysis['promotion']) == 2  # Two unique values: 0 and 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

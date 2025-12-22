from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class BaseForecaster(ABC):
    """Base class for all forecasting models"""
    
    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'BaseForecaster':
        """Fit the forecasting model
        
        Args:
            y: Target time series DataFrame
            X: Optional exogenous variables DataFrame
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts
        
        Args:
            X: Optional exogenous variables for prediction
            
        Returns:
            DataFrame with forecasts
        """
        pass
    
    def fit_predict(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fit and predict in one step"""
        self.fit(y, X)
        return self.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, filepath: str):
        """Save model to disk using joblib
        
        Args:
            filepath: Path to save the model (e.g., 'model.pkl' or 'model.joblib')
            
        Example:
            >>> model.fit(train_data)
            >>> model.save('my_model.joblib')
        """
        import joblib
        from pathlib import Path
        
        # Add metadata
        metadata = {
            'model': self,
            'class_name': self.__class__.__name__,
            'horizon': self.horizon,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'save_timestamp': pd.Timestamp.now()
        }
        
        joblib.dump(metadata, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseForecaster':
        """Load model from disk
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded forecaster model
            
        Example:
            >>> model = RandomForestForecaster.load('my_model.joblib')
            >>> forecasts = model.predict()
        """
        import joblib
        
        metadata = joblib.load(filepath)
        model = metadata['model']
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"  Class: {metadata['class_name']}")
        print(f"  Horizon: {metadata['horizon']}")
        print(f"  Fitted: {metadata['is_fitted']}")
        print(f"  Saved: {metadata['save_timestamp']}")
        
        return model


class VARForecaster(BaseForecaster):
    """Vector Autoregression model for multivariate time series"""
    
    def __init__(self, horizon: int = 1, lags: int = 1, trend: str = 'c'):
        super().__init__(horizon)
        self.lags = lags
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.coefficients = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'VARForecaster':
        """Fit VAR model"""
        from statsmodels.tsa.api import VAR
        
        self.feature_names = y.columns.tolist()
        self.model = VAR(y)
        self.fitted_model = self.model.fit(maxlags=self.lags, trend=self.trend)
        self.coefficients = self.fitted_model.params
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate VAR forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(self.fitted_model.endog, steps=self.horizon)
        return pd.DataFrame(forecast, columns=self.feature_names)


class LinearForecaster(BaseForecaster):
    """Linear regression forecaster with exogenous variables"""
    
    def __init__(self, horizon: int = 1, fit_intercept: bool = True):
        super().__init__(horizon)
        self.fit_intercept = fit_intercept
        from sklearn.linear_model import LinearRegression
        self.models = {}
        self.target_names = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'LinearForecaster':
        """Fit linear models for each target and horizon"""
        if X is None:
            raise ValueError("LinearForecaster requires exogenous variables X")
        
        from sklearn.linear_model import LinearRegression
        
        self.target_names = y.columns.tolist()
        self.feature_names = X.columns.tolist()
        
        for target in self.target_names:
            self.models[target] = []
            for h in range(self.horizon):
                model = LinearRegression(fit_intercept=self.fit_intercept)
                # Shift target by h+1 steps
                y_shifted = y[target].shift(-(h+1))
                valid_idx = ~y_shifted.isna()
                model.fit(X[valid_idx], y_shifted[valid_idx])
                self.models[target].append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate linear forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if X is None:
            raise ValueError("LinearForecaster requires exogenous variables X for prediction")
        
        predictions = {target: [] for target in self.target_names}
        
        for target in self.target_names:
            for h, model in enumerate(self.models[target]):
                pred = model.predict(X.iloc[-1:])
                predictions[target].append(pred[0])
        
        return pd.DataFrame(predictions)


class MovingAverageForecaster(BaseForecaster):
    """Simple moving average forecaster"""
    
    def __init__(self, horizon: int = 1, window: int = 3):
        super().__init__(horizon)
        self.window = window
        self.last_values = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'MovingAverageForecaster':
        """Store last values for moving average"""
        self.feature_names = y.columns.tolist()
        self.last_values = y.tail(self.window)
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate moving average forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simple moving average
        forecast = self.last_values.mean().values
        predictions = np.tile(forecast, (self.horizon, 1))
        
        return pd.DataFrame(predictions, columns=self.feature_names)
"""
External model wrappers for advanced forecasting models.

This module provides wrappers for popular forecasting models from external libraries:
- RandomForest (sklearn)
- XGBoost (xgboost)
- Prophet (facebook prophet)
- NHiTS (darts)

All models automatically handle both categorical and numerical covariates.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
from .base import BaseForecaster
from ..utils.preprocessing import CovariatePreprocessor

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from darts.models import NHiTSModel
    from darts import TimeSeries
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest forecaster with lag features and covariate support.
    
    Automatically handles both categorical and numerical covariates:
    - Categorical features: One-hot encoded or label encoded
    - Numerical features: Used as-is or optionally scaled
    - Auto-detection of feature types based on data
    
    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values to use as features
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    horizon : int, default=1
        Forecast horizon
    random_state : int, default=42
        Random seed for reproducibility
    preprocess_covariates : bool, default=True
        Automatically preprocess categorical and numerical covariates
    **rf_params : dict
        Additional parameters for RandomForestRegressor
    """
    
    def __init__(
        self,
        n_lags: int = 7,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        horizon: int = 1,
        random_state: int = 42,
        preprocess_covariates: bool = True,
        **rf_params
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RandomForestForecaster. Install with: pip install scikit-learn")
        
        super().__init__(horizon)
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.rf_params = rf_params
        self.models = []
        self.covariate_preprocessor_ = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'RandomForestForecaster':
        """Fit Random Forest with lagged features and optional covariates.
        
        Automatically handles categorical and numerical covariates.
        """
        self.feature_names = y.columns.tolist()
        
        # Preprocess covariates (handles categorical/numerical automatically)
        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)
        
        # Create lagged features
        X_train = self._create_features(y, X)
        
        # Train a model for each horizon step
        self.models = []
        for h in range(1, self.horizon + 1):
            # Shift target by h steps
            y_shifted = y.shift(-h)
            
            # X_train already has first n_lags rows removed
            # y_shifted needs to align: also skip first n_lags, and exclude last h (NaN) rows
            # Valid row range: n_lags to len(y) - h
            n_valid = len(y) - self.n_lags - h
            
            # Create model
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.rf_params
            )
            
            model = MultiOutputRegressor(rf)
            # Use first n_valid rows from both X_train and y_shifted
            model.fit(X_train.iloc[:n_valid], y_shifted.iloc[self.n_lags:self.n_lags + n_valid])
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using Random Forest.
        
        Automatically preprocesses covariates using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess covariates if needed
        if X is not None and self.covariate_preprocessor_ is not None:
            X = self.covariate_preprocessor_.transform(X)
        
        # Get last known values for lagging
        last_values = self.last_values_.copy()
        predictions = []
        
        for h, model in enumerate(self.models):
            # Create features for this horizon using CURRENT lag buffer
            X_pred = self._create_features_for_prediction(last_values, X, step=h)
            
            pred = model.predict(X_pred)
            predictions.append(pred[0])
            
            # Update lag buffer with new prediction
            new_row = pd.DataFrame([pred[0]], columns=self.feature_names)
            last_values = pd.concat([last_values, new_row], ignore_index=True)
        
        return pd.DataFrame(predictions, columns=self.feature_names)
    
    def _create_features(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create feature matrix with lags and covariates"""
        features = []
        
        # Add lagged values
        for lag in range(1, self.n_lags + 1):
            lagged = y.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
            features.append(lagged)
        
        # Add covariates if provided
        if X is not None:
            features.append(X)
        
        # Store last values for prediction
        self.last_values_ = y.iloc[-self.n_lags:].copy()
        
        return pd.concat(features, axis=1).dropna()
    
    def _create_features_for_prediction(self, last_values: pd.DataFrame, X: Optional[pd.DataFrame] = None, step: int = 0) -> pd.DataFrame:
        """Create features for prediction"""
        features = []
        
        # Add lagged values (use last n_lags rows)
        for lag in range(1, self.n_lags + 1):
            if lag <= len(last_values):
                lagged = last_values.iloc[-lag]
                features.extend(lagged.values)
        
        # Add covariates if provided (must match training)
        if self.covariate_preprocessor_ is not None:
            if X is not None and len(X) > step:
                features.extend(X.iloc[step].values)
            else:
                # If no covariates provided but model was trained with them, raise error
                raise ValueError("Model was trained with covariates but none provided for prediction")
        
        return pd.DataFrame([features])


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecaster with lag features and covariate support.
    
    Automatically handles both categorical and numerical covariates:
    - Categorical features: One-hot encoded or label encoded
    - Numerical features: Used as-is or optionally scaled
    - Auto-detection of feature types based on data
    
    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values to use as features
    n_estimators : int, default=100
        Number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.1
        Learning rate
    horizon : int, default=1
        Forecast horizon
    random_state : int, default=42
        Random seed
    preprocess_covariates : bool, default=True
        Automatically preprocess categorical and numerical covariates
    **xgb_params : dict
        Additional XGBoost parameters
    """
    
    def __init__(
        self,
        n_lags: int = 7,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        horizon: int = 1,
        random_state: int = 42,
        preprocess_covariates: bool = True,
        **xgb_params
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostForecaster. Install with: pip install xgboost")
        
        super().__init__(horizon)
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.xgb_params = xgb_params
        self.models = []
        self.covariate_preprocessor_ = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'XGBoostForecaster':
        """Fit XGBoost with lagged features and optional covariates.
        
        Automatically handles categorical and numerical covariates.
        """
        self.feature_names = y.columns.tolist()
        
        # Preprocess covariates (handles categorical/numerical automatically)
        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)
        
        # Create lagged features
        X_train = self._create_features(y, X)
        
        # Train a model for each target variable and horizon
        self.models = []
        for h in range(1, self.horizon + 1):
            models_h = []
            y_shifted = y.shift(-h)
            
            # X_train already has first n_lags rows removed
            # y_shifted needs to align: also skip first n_lags, and exclude last h (NaN) rows
            # Valid row range: n_lags to len(y) - h
            n_valid = len(y) - self.n_lags - h
            
            for target_col in y.columns:
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    **self.xgb_params
                )
                # Use first n_valid rows from both X_train and y_shifted
                model.fit(X_train.iloc[:n_valid], y_shifted[target_col].iloc[self.n_lags:self.n_lags + n_valid])
                models_h.append(model)
            
            self.models.append(models_h)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using XGBoost.
        
        Automatically preprocesses covariates using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess covariates if needed
        if X is not None and self.covariate_preprocessor_ is not None:
            X = self.covariate_preprocessor_.transform(X)
        
        last_values = self.last_values_.copy()
        predictions = []
        
        for h, models_h in enumerate(self.models):
            # Create features using current lag buffer
            X_pred = self._create_features_for_prediction(last_values, X, step=h)
            
            pred_h = [model.predict(X_pred)[0] for model in models_h]
            predictions.append(pred_h)
            
            # Update lag buffer with new prediction
            new_row = pd.DataFrame([pred_h], columns=self.feature_names)
            last_values = pd.concat([last_values, new_row], ignore_index=True)
        
        return pd.DataFrame(predictions, columns=self.feature_names)
    
    def _create_features(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create feature matrix with lags and covariates"""
        features = []
        
        for lag in range(1, self.n_lags + 1):
            lagged = y.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
            features.append(lagged)
        
        if X is not None:
            features.append(X)
        
        self.last_values_ = y.iloc[-self.n_lags:].copy()
        
        return pd.concat(features, axis=1).dropna()
    
    def _create_features_for_prediction(self, last_values: pd.DataFrame, X: Optional[pd.DataFrame] = None, step: int = 0) -> pd.DataFrame:
        """Create features for prediction"""
        features = []
        
        for lag in range(1, self.n_lags + 1):
            if lag <= len(last_values):
                lagged = last_values.iloc[-lag]
                features.extend(lagged.values)
        
        # Add covariates if provided (must match training)
        if self.covariate_preprocessor_ is not None:
            if X is not None and len(X) > step:
                features.extend(X.iloc[step].values)
            else:
                # If no covariates provided but model was trained with them, raise error
                raise ValueError("Model was trained with covariates but none provided for prediction")
        
        return pd.DataFrame([features])


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet forecaster wrapper for multivariate forecasting.
    
    Fits a separate Prophet model for each time series variable.
    
    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon
    growth : str, default='linear'
        'linear' or 'logistic' growth
    seasonality_mode : str, default='additive'
        'additive' or 'multiplicative'
    yearly_seasonality : bool or int, default='auto'
        Fit yearly seasonality
    weekly_seasonality : bool or int, default='auto'
        Fit weekly seasonality
    daily_seasonality : bool or int, default='auto'
        Fit daily seasonality
    **prophet_params : dict
        Additional Prophet parameters
    """
    
    def __init__(
        self,
        horizon: int = 1,
        growth: str = 'linear',
        seasonality_mode: str = 'additive',
        yearly_seasonality: Union[bool, int] = 'auto',
        weekly_seasonality: Union[bool, int] = 'auto',
        daily_seasonality: Union[bool, int] = 'auto',
        **prophet_params
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for ProphetForecaster. Install with: pip install prophet")
        
        super().__init__(horizon)
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.prophet_params = prophet_params
        self.models = {}
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'ProphetForecaster':
        """Fit separate Prophet model for each series"""
        self.feature_names = y.columns.tolist()
        self.last_date = y.index[-1]
        
        # Fit a Prophet model for each column
        for col in y.columns:
            # Prepare data for Prophet (needs 'ds' and 'y' columns)
            df_prophet = pd.DataFrame({
                'ds': y.index,
                'y': y[col].values
            })
            
            # Add covariates if provided
            if X is not None:
                for x_col in X.columns:
                    df_prophet[x_col] = X[x_col].values
            
            # Create and fit model
            model = Prophet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                **self.prophet_params
            )
            
            # Add covariates as regressors
            if X is not None:
                for x_col in X.columns:
                    model.add_regressor(x_col)
            
            model.fit(df_prophet)
            self.models[col] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using Prophet"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        # Infer frequency from last date
        freq = pd.infer_freq(pd.date_range(self.last_date - pd.Timedelta(days=7), self.last_date, periods=8))
        if freq is None:
            freq = 'D'  # Default to daily
        
        future_dates = pd.date_range(
            start=self.last_date + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )
        
        predictions = {}
        
        for col, model in self.models.items():
            future = pd.DataFrame({'ds': future_dates})
            
            # Add future covariates if provided
            if X is not None:
                for x_col in X.columns:
                    if len(X) >= self.horizon:
                        future[x_col] = X[x_col].values[:self.horizon]
                    else:
                        # If not enough future covariates, repeat last value
                        future[x_col] = X[x_col].iloc[-1]
            
            forecast = model.predict(future)
            predictions[col] = forecast['yhat'].values
        
        return pd.DataFrame(predictions, index=future_dates)


class NHiTSForecaster(BaseForecaster):
    """
    NHiTS (Neural Hierarchical Interpolation for Time Series) forecaster from Darts.
    
    Parameters
    ----------
    input_chunk_length : int, default=24
        Length of input sequence
    output_chunk_length : int, default=1
        Length of output sequence (forecast horizon)
    num_stacks : int, default=3
        Number of stacks in the model
    num_blocks : int, default=1
        Number of blocks per stack
    num_layers : int, default=2
        Number of layers per block
    layer_widths : int, default=512
        Width of layers
    n_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    **nhits_params : dict
        Additional NHiTS parameters
    """
    
    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 1,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 512,
        n_epochs: int = 100,
        batch_size: int = 32,
        **nhits_params
    ):
        if not DARTS_AVAILABLE:
            raise ImportError("darts is required for NHiTSForecaster. Install with: pip install darts")
        
        super().__init__(output_chunk_length)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nhits_params = nhits_params
        self.model = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'NHiTSForecaster':
        """Fit NHiTS model"""
        self.feature_names = y.columns.tolist()
        
        # Convert to Darts TimeSeries
        series = TimeSeries.from_dataframe(y)
        
        # Create model
        self.model = NHiTSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            **self.nhits_params
        )
        
        # Add covariates if provided
        if X is not None:
            covariates = TimeSeries.from_dataframe(X)
            self.model.fit(series, past_covariates=covariates)
        else:
            self.model.fit(series)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using NHiTS"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecast
        if X is not None:
            covariates = TimeSeries.from_dataframe(X)
            forecast = self.model.predict(n=self.horizon, past_covariates=covariates)
        else:
            forecast = self.model.predict(n=self.horizon)
        
        # Convert back to DataFrame
        return forecast.pd_dataframe()

"""External model wrappers for advanced forecasting models.

This module provides wrappers for popular forecasting models from external libraries:
- RandomForest (sklearn)
- XGBoost (xgboost)
- Prophet (prophet)

All models automatically handle both categorical and numerical covariates.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
from functools import lru_cache
from .base import BaseForecaster
from ..utils.preprocessing import CovariatePreprocessor


@lru_cache(maxsize=1)
def _import_sklearn():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    return RandomForestRegressor, MultiOutputRegressor


@lru_cache(maxsize=1)
def _import_xgboost():
    import xgboost as xgb

    return xgb


@lru_cache(maxsize=1)
def _import_prophet():
    from prophet import Prophet

    return Prophet


@lru_cache(maxsize=1)
def _import_torch():
    import torch
    import torch.nn as nn

    return torch, nn


@lru_cache(maxsize=1)
def _import_chronos():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline


@lru_cache(maxsize=1)
def _import_lightgbm():
    import lightgbm as lgb

    return lgb


@lru_cache(maxsize=1)
def _import_catboost():
    from catboost import CatBoostRegressor

    return CatBoostRegressor


@lru_cache(maxsize=1)
def _import_darts_nbeats():
    from darts.models import NBEATSModel
    from darts import TimeSeries

    return NBEATSModel, TimeSeries


@lru_cache(maxsize=1)
def _import_darts_nhits():
    from darts.models import NHiTSModel
    from darts import TimeSeries

    return NHiTSModel, TimeSeries


@lru_cache(maxsize=1)
def _import_darts_tft():
    from darts.models import TFTModel
    from darts import TimeSeries

    return TFTModel, TimeSeries


class RandomForestForecaster(BaseForecaster):
    """Random Forest forecaster with lag features and covariate support.
    
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

    supports_covariates: bool = True
    
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
        try:
            RandomForestRegressor, MultiOutputRegressor = _import_sklearn()
        except ImportError as exc:
            raise ImportError(
                "sklearn is required for RandomForestForecaster. Install with: pip install scikit-learn"
            ) from exc
        
        super().__init__(horizon)
        self._RandomForestRegressor = RandomForestRegressor
        self._MultiOutputRegressor = MultiOutputRegressor
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.rf_params = rf_params
        self.models = []
        self.covariate_preprocessor_ = None
        self._trained_with_covariates = False
        self._covariate_columns_ = None
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'n_lags': self.n_lags,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'horizon': self.horizon,
            'random_state': self.random_state,
            'preprocess_covariates': self.preprocess_covariates,
            **self.rf_params
        }
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'RandomForestForecaster':
        """Fit Random Forest with lagged features and optional covariates.
        
        Automatically handles categorical and numerical covariates.
        """
        self.feature_names = y.columns.tolist()
        
        self._trained_with_covariates = X is not None

        # Preprocess covariates (handles categorical/numerical automatically)
        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)

        self._covariate_columns_ = list(X.columns) if X is not None else None

        # Create lag-only feature frame once (covariates are horizon-shifted per step)
        lag_features = []
        for lag in range(1, self.n_lags + 1):
            lagged = y.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
            lag_features.append(lagged)
        lag_df = pd.concat(lag_features, axis=1)

        # Store last values for recursive prediction
        self.last_values_ = y.iloc[-self.n_lags:].copy()

        # Train a model for each horizon step
        self.models = []
        for h in range(1, self.horizon + 1):
            y_shifted = y.shift(-h)

            # For h-step ahead prediction, use covariates at forecast time (t+h)
            if X is not None:
                X_h = X.shift(-h)
                X_train_h = pd.concat([lag_df, X_h], axis=1).dropna()
            else:
                X_train_h = lag_df.dropna()

            # Ensure targets are available for this horizon (exclude trailing NaNs)
            train_index = X_train_h.index.intersection(y_shifted.dropna().index)
            X_train_h = X_train_h.loc[train_index]
            y_train_h = y_shifted.loc[train_index]

            rf = self._RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.rf_params
            )
            model = self._MultiOutputRegressor(rf)
            model.fit(X_train_h, y_train_h)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using Random Forest.
        
        Automatically preprocesses covariates using fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self._trained_with_covariates:
            if X is None:
                raise ValueError("Model was trained with covariates but none provided for prediction")
            if self.covariate_preprocessor_ is not None:
                X = self.covariate_preprocessor_.transform(X)
            if self._covariate_columns_ is not None:
                missing = set(self._covariate_columns_) - set(X.columns)
                if missing:
                    raise ValueError(f"Missing required covariates for prediction: {sorted(missing)}")
                X = X[self._covariate_columns_]
        
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
        feature_names = []
        
        # Add lagged values (use last n_lags rows) with proper column names
        for lag in range(1, self.n_lags + 1):
            if lag <= len(last_values):
                lagged = last_values.iloc[-lag]
                features.extend(lagged.values)
                feature_names.extend([f"{col}_lag{lag}" for col in last_values.columns])
        
        # Add covariates if trained with them
        if self._trained_with_covariates:
            if X is None or len(X) == 0:
                raise ValueError("Model was trained with covariates but none provided for prediction")
            row_idx = min(step, len(X) - 1)
            features.extend(X.iloc[row_idx].values)
            feature_names.extend(X.columns.tolist())
        
        return pd.DataFrame([features], columns=feature_names)


class XGBoostForecaster(BaseForecaster):
    """XGBoost forecaster with lag features and covariate support.
    
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

    supports_covariates: bool = True
    
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
        try:
            xgb = _import_xgboost()
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostForecaster. Install with: pip install xgboost") from exc
        
        super().__init__(horizon)
        self._xgb = xgb
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.xgb_params = xgb_params
        self.models = []
        self.covariate_preprocessor_ = None
        self._trained_with_covariates = False
        self._covariate_columns_ = None
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'n_lags': self.n_lags,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'horizon': self.horizon,
            'random_state': self.random_state,
            'preprocess_covariates': self.preprocess_covariates,
            **self.xgb_params
        }
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'XGBoostForecaster':
        """Fit XGBoost with lagged features and optional covariates.
        
        Automatically handles categorical and numerical covariates.
        """
        self.feature_names = y.columns.tolist()
        
        self._trained_with_covariates = X is not None

        # Preprocess covariates (handles categorical/numerical automatically)
        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)

        self._covariate_columns_ = list(X.columns) if X is not None else None

        # Create lag-only feature frame once (covariates are horizon-shifted per step)
        lag_features = []
        for lag in range(1, self.n_lags + 1):
            lagged = y.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
            lag_features.append(lagged)
        lag_df = pd.concat(lag_features, axis=1)

        # Store last values for recursive prediction
        self.last_values_ = y.iloc[-self.n_lags:].copy()

        # Train a model for each target variable and horizon
        self.models = []
        for h in range(1, self.horizon + 1):
            y_shifted = y.shift(-h)

            if X is not None:
                X_h = X.shift(-h)
                X_train_h = pd.concat([lag_df, X_h], axis=1).dropna()
            else:
                X_train_h = lag_df.dropna()

            # Ensure targets are available for this horizon (exclude trailing NaNs)
            train_index = X_train_h.index.intersection(y_shifted.dropna().index)
            X_train_h = X_train_h.loc[train_index]
            y_train_h = y_shifted.loc[train_index]

            models_h = []
            for target_col in y.columns:
                model = self._xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    **self.xgb_params
                )
                model.fit(X_train_h, y_train_h[target_col])
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
        
        if self._trained_with_covariates:
            if X is None:
                raise ValueError("Model was trained with covariates but none provided for prediction")
            if self.covariate_preprocessor_ is not None:
                X = self.covariate_preprocessor_.transform(X)
            if self._covariate_columns_ is not None:
                missing = set(self._covariate_columns_) - set(X.columns)
                if missing:
                    raise ValueError(f"Missing required covariates for prediction: {sorted(missing)}")
                X = X[self._covariate_columns_]
        
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
        feature_names = []
        
        # Create lagged features with proper column names
        for lag in range(1, self.n_lags + 1):
            if lag <= len(last_values):
                lagged = last_values.iloc[-lag]
                features.extend(lagged.values)
                feature_names.extend([f"{col}_lag{lag}" for col in last_values.columns])
        
        # Add covariates if trained with them
        if self._trained_with_covariates:
            if X is None or len(X) == 0:
                raise ValueError("Model was trained with covariates but none provided for prediction")
            row_idx = min(step, len(X) - 1)
            features.extend(X.iloc[row_idx].values)
            feature_names.extend(X.columns.tolist())
        
        return pd.DataFrame([features], columns=feature_names)


class ProphetForecaster(BaseForecaster):
    """

    supports_covariates: bool = True
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

    supports_covariates: bool = True
    
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
        try:
            Prophet = _import_prophet()
        except ImportError as exc:
            raise ImportError("prophet is required for ProphetForecaster. Install with: pip install prophet") from exc
        
        super().__init__(horizon)
        self._Prophet = Prophet
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.prophet_params = prophet_params
        self.models = {}
        self.freq_ = None
        self.regressor_cols_ = []
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'horizon': self.horizon,
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            **self.prophet_params
        }
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'ProphetForecaster':
        """Fit separate Prophet model for each series"""
        self.feature_names = y.columns.tolist()
        self.last_date = y.index[-1]
        self.freq_ = pd.infer_freq(y.index)
        if self.freq_ is None and hasattr(y.index, "freqstr"):
            self.freq_ = y.index.freqstr
        if self.freq_ is None:
            self.freq_ = 'D'

        if X is not None:
            # Prophet regressors must align 1:1 with y timestamps
            X = X.copy()
            if not y.index.equals(X.index):
                X = X.reindex(y.index)
            if X.isna().any().any():
                raise ValueError("ProphetForecaster: covariates X must align to y index with no missing values")
            self.regressor_cols_ = list(X.columns)
        else:
            self.regressor_cols_ = []
        
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
            model = self._Prophet(
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
        freq = self.freq_ or 'D'
        
        future_dates = pd.date_range(
            start=self.last_date + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )

        X_future = None
        if self.regressor_cols_:
            if X is None:
                raise ValueError("Model was trained with covariates but none provided for prediction")

            missing = set(self.regressor_cols_) - set(X.columns)
            if missing:
                raise ValueError(f"Missing required covariates for prediction: {sorted(missing)}")

            if isinstance(X.index, pd.DatetimeIndex):
                X_future = X[self.regressor_cols_].reindex(future_dates)
                if X_future.isna().any().any():
                    raise ValueError("Future covariates X must cover the full forecast horizon dates")
            else:
                if len(X) >= self.horizon:
                    X_future = X[self.regressor_cols_].iloc[:self.horizon].copy()
                    X_future.index = future_dates
                else:
                    # Backtesting-style single-row case: repeat last value
                    last_row = X[self.regressor_cols_].iloc[-1]
                    X_future = pd.DataFrame([last_row.values] * self.horizon, columns=self.regressor_cols_, index=future_dates)
        
        predictions = {}
        
        for col, model in self.models.items():
            future = pd.DataFrame({'ds': future_dates})
            
            # Add future covariates if provided
            if X_future is not None:
                for x_col in self.regressor_cols_:
                    future[x_col] = X_future[x_col].values
            
            forecast = model.predict(future)
            predictions[col] = forecast['yhat'].values
        
        return pd.DataFrame(predictions, index=future_dates)


try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ETS_AVAILABLE = True
except ImportError:
    ETS_AVAILABLE = False

TORCH_AVAILABLE = None


class ARIMAForecaster(BaseForecaster):
    """ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    Fits separate ARIMA models for each time series in multivariate data.
    
    Note: This implementation does NOT use exogenous variables (ARIMAX).
    For pure time-series forecasting without external covariates.
    Use LinearForecaster, RandomForest, XGBoost, or Prophet if you need covariate support.
    
    Parameters
    ----------
    order : tuple, default=(1, 1, 1)
        ARIMA order (p, d, q) where:
        - p: autoregressive order
        - d: differencing order
        - q: moving average order
    seasonal_order : tuple, default=(0, 0, 0, 0)
        Seasonal ARIMA order (P, D, Q, s)
    horizon : int, default=1
        Forecast horizon
    **arima_params : dict
        Additional ARIMA parameters
    """
    
    supports_covariates: bool = False
    
    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (0, 0, 0, 0),
        horizon: int = 1,
        **arima_params
    ):
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMAForecaster. Install with: pip install statsmodels")
        
        super().__init__(horizon)
        self.order = order
        self.seasonal_order = seasonal_order
        self.arima_params = arima_params
        self.models = {}
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'horizon': self.horizon,
            **self.arima_params
        }
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'ARIMAForecaster':
        """Fit separate ARIMA model for each series (ignores X - pure time series)"""
        self.feature_names = y.columns.tolist()
        
        # Fit an ARIMA model for each column (without exogenous variables)
        for col in y.columns:
            model = ARIMA(
                y[col],
                order=self.order,
                seasonal_order=self.seasonal_order,
                **self.arima_params
            )
            fitted_model = model.fit()
            self.models[col] = fitted_model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using ARIMA (ignores X - pure time series)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        for col, model in self.models.items():
            # Generate forecast (no exogenous variables)
            forecast = model.forecast(steps=self.horizon)
            predictions[col] = forecast
        
        # Create future dates
        last_date = list(self.models.values())[0].fittedvalues.index[-1]
        freq = pd.infer_freq(list(self.models.values())[0].fittedvalues.index)
        if freq is None:
            freq = 'D'
        
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )
        
        return pd.DataFrame(predictions, index=future_dates)


class ETSForecaster(BaseForecaster):
    """Holt-Winters / Exponential Smoothing (ETS) forecaster.

    Fits a separate Holt-Winters model per target series.

    Notes:
    - ETS does not support exogenous covariates in statsmodels; any provided X is ignored.
    - For backtesting, the validator sets the model horizon to the test window length.

    Parameters
    ----------
    trend : {'add', 'mul', None}, default=None
        Trend component type.
    seasonal : {'add', 'mul', None}, default=None
        Seasonal component type.
    seasonal_periods : int, optional
        Number of periods in a full seasonal cycle (e.g., 7 for daily-with-weekly seasonality).
    damped_trend : bool, default=False
        Whether to damp the trend component.
    initialization_method : str, default='estimated'
        Initialization method for statsmodels.
    horizon : int, default=1
        Forecast horizon.
    **fit_params : dict
        Additional parameters passed to the fitted model's `.fit()`.
    """

    supports_covariates: bool = False

    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        initialization_method: str = 'estimated',
        horizon: int = 1,
        **fit_params,
    ):
        if not ETS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for ETSForecaster. Install with: pip install statsmodels"
            )

        super().__init__(horizon)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.fit_params = fit_params
        self.models = {}
        self.last_date = None
        self.freq_ = None
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'initialization_method': self.initialization_method,
            'horizon': self.horizon,
            **self.fit_params
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'ETSForecaster':
        """Fit separate ETS model for each series (X is ignored)."""
        self.feature_names = y.columns.tolist()

        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index)
            if self.freq_ is None and hasattr(y.index, 'freqstr'):
                self.freq_ = y.index.freqstr
            if self.freq_ is None:
                self.freq_ = 'D'
        else:
            self.last_date = None
            self.freq_ = None

        for col in y.columns:
            series = y[col].astype(float)
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method=self.initialization_method,
            )
            self.models[col] = model.fit(**self.fit_params)

        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using ETS (X is ignored)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = {}
        for col, fitted in self.models.items():
            forecast = fitted.forecast(steps=self.horizon)
            predictions[col] = np.asarray(forecast)

        if self.last_date is not None:
            freq = self.freq_ or 'D'
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon,
                freq=freq,
            )
            return pd.DataFrame(predictions, index=future_dates)

        return pd.DataFrame(predictions)


class LSTMForecaster(BaseForecaster):
    """

    supports_covariates: bool = False
    LSTM (Long Short-Term Memory) forecaster using PyTorch.
    
    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values to use as features
    hidden_size : int, default=50
        Number of LSTM hidden units
    num_layers : int, default=2
        Number of LSTM layers
    dropout : float, default=0.2
        Dropout rate
    horizon : int, default=1
        Forecast horizon
    epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    learning_rate : float, default=0.001
        Learning rate
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        n_lags: int = 7,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        try:
            torch, nn = _import_torch()
        except ImportError as exc:
            raise ImportError(
                'PyTorch is required for LSTMForecaster. '
                'Install with: pip install "autotsforecast[neural]" '
                'or: pip install torch'
            ) from exc
        
        super().__init__(horizon)
        self._torch = torch
        self._nn = nn
        self.n_lags = n_lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = None
    
    def get_params(self):
        """Get parameters for cloning (excludes internal state)"""
        return {
            'n_lags': self.n_lags,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'horizon': self.horizon,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
        
    def _create_sequences(self, data):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.n_lags - self.horizon + 1):
            seq = data[i:i + self.n_lags]
            target = data[i + self.n_lags:i + self.n_lags + self.horizon]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'LSTMForecaster':
        """Fit LSTM model"""
        from sklearn.preprocessing import StandardScaler
        
        torch = self._torch
        nn = self._nn
        
        self.feature_names = y.columns.tolist()
        torch.manual_seed(self.random_state)
        
        # Scale the data
        self.scaler = StandardScaler()
        y_scaled = self.scaler.fit_transform(y.values)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(y_scaled)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)
        
        # Define LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        input_size = y.shape[1]
        output_size = y.shape[1] * self.horizon
        
        self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, 
                              output_size, self.dropout)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor.view(y_tensor.size(0), -1))
            loss.backward()
            optimizer.step()
        
        self.last_values = y.values[-self.n_lags:]
        self.last_index = y.index[-1]
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using LSTM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        torch = self._torch
        
        self.model.eval()
        
        # Scale last values
        last_scaled = self.scaler.transform(self.last_values)
        
        # Create input tensor
        X_input = torch.FloatTensor(last_scaled).unsqueeze(0)
        
        # Generate forecast
        with torch.no_grad():
            forecast_scaled = self.model(X_input).numpy()
        
        # Reshape and inverse transform
        forecast_scaled = forecast_scaled.reshape(self.horizon, len(self.feature_names))
        forecast = self.scaler.inverse_transform(forecast_scaled)
        
        # Create future dates
        freq = pd.infer_freq(pd.date_range(self.last_index - pd.Timedelta(days=7), 
                                          self.last_index, periods=8))
        if freq is None:
            freq = 'D'
        
        future_dates = pd.date_range(
            start=self.last_index + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )
        
        return pd.DataFrame(forecast, index=future_dates, columns=self.feature_names)


class Chronos2Forecaster(BaseForecaster):
    """Chronos-2 foundation model forecaster from Amazon Science.
    
    Chronos-2 is a state-of-the-art pretrained time series foundation model that achieves
    excellent zero-shot forecasting performance. It supports univariate, multivariate, and
    covariate-informed forecasting.
    
    Key features:
    - Zero-shot forecasting (no training required)
    - Multiple model sizes (tiny, mini, small, base, bolt variants)
    - Fast inference (especially bolt variants - up to 250x faster)
    - State-of-the-art performance on fev-bench and GIFT-Eval
    - Supports prediction intervals
    
    Model sizes:
    - amazon/chronos-2: 120M params (default, best accuracy)
    - autogluon/chronos-2-small: 28M params (smaller, still good)
    - amazon/chronos-bolt-tiny: 9M params (ultra fast)
    - amazon/chronos-bolt-mini: 21M params (fast)
    - amazon/chronos-bolt-small: 48M params (balanced)
    - amazon/chronos-bolt-base: 205M params (accurate + fast)
    
    Example:
        >>> from autotsforecast.models.external import Chronos2Forecaster
        >>> model = Chronos2Forecaster(horizon=24, model_name="amazon/chronos-2")
        >>> model.fit(y_train)
        >>> forecasts = model.predict()
    """
    
    supports_covariates: bool = False  # Currently univariate only in our wrapper
    
    def __init__(
        self,
        horizon: int,
        model_name: str = "amazon/chronos-2",
        device_map: str = "auto",
        dtype: str = "auto",
        quantile_levels: Optional[List[float]] = None,
    ):
        """Initialize Chronos-2 forecaster.
        
        Parameters
        ----------
        horizon : int
            Number of time steps to forecast into the future.
        model_name : str, default="amazon/chronos-2"
            Pretrained model name. Options:
            - "amazon/chronos-2" (120M, best accuracy)
            - "autogluon/chronos-2-small" (28M, good accuracy)
            - "amazon/chronos-bolt-tiny" (9M, ultra fast)
            - "amazon/chronos-bolt-mini" (21M, fast)
            - "amazon/chronos-bolt-small" (48M, balanced)
            - "amazon/chronos-bolt-base" (205M, accurate + fast)
        device_map : str, default="auto"
            Device mapping strategy. "auto" will use GPU if available, else CPU.
            Can also specify "cuda" or "cpu" explicitly.
        dtype : str, default="auto"
            PyTorch dtype for model weights. "auto" will choose automatically.
        quantile_levels : list of float, optional
            Quantile levels for prediction intervals (e.g., [0.1, 0.5, 0.9]).
            If None, only point forecasts are returned.
        """
        super().__init__(horizon=horizon)
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.quantile_levels = quantile_levels or [0.5]  # Default to median
        self._pipeline = None
        self.feature_names = None
        self.last_index = None
    
    @property
    def _chronos(self):
        """Lazy import and cache Chronos pipeline"""
        if self._pipeline is None:
            Chronos2Pipeline = _import_chronos()
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                dtype=self.dtype,
            )
        return self._pipeline
    
    def fit(self, y: Union[pd.Series, pd.DataFrame], X: Optional[pd.DataFrame] = None) -> "Chronos2Forecaster":
        """Fit the Chronos-2 model (stores context for zero-shot inference).
        
        Note: Chronos-2 is a pretrained model, so "fitting" just means storing
        the historical data as context for zero-shot forecasting.
        
        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Historical time series data. If DataFrame, will forecast all columns.
        X : pd.DataFrame, optional
            Covariates (currently not used in this wrapper).
        
        Returns
        -------
        self : Chronos2Forecaster
            Fitted forecaster instance.
        """
        if isinstance(y, pd.Series):
            y = y.to_frame()
        
        self.feature_names = y.columns.tolist()
        self.y_train = y
        self.last_index = y.index[-1]
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using Chronos-2.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Future covariates (currently not used in this wrapper).
        
        Returns
        -------
        pd.DataFrame
            Forecasts with datetime index. If quantile_levels specified,
            returns median forecast only. For full quantiles, use predict_quantiles().
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to list of series for Chronos API
        context_list = [self.y_train[col].values for col in self.feature_names]
        
        # Generate forecasts using correct API
        # Returns list of tensors, shape per tensor: (batch=1, num_quantiles=13, prediction_length)
        forecast_list = self._chronos.predict(
            inputs=context_list,
            prediction_length=self.horizon,
        )
        
        # Extract median (quantile index 6 out of 13)
        # Convert to array: (n_series, horizon)
        forecasts = np.array([f[0, 6, :].numpy() for f in forecast_list]).T  # Shape: (horizon, n_series)
        
        # Create future dates
        freq = pd.infer_freq(self.y_train.index[-10:])
        if freq is None:
            freq = 'D'
        
        future_dates = pd.date_range(
            start=self.last_index + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )
        
        return pd.DataFrame(forecasts, index=future_dates, columns=self.feature_names)
    
    def predict_quantiles(self, quantile_levels: Optional[List[float]] = None) -> pd.DataFrame:
        """Generate probabilistic forecasts with multiple quantiles.
        
        Parameters
        ----------
        quantile_levels : list of float, optional
            Quantile levels to return (e.g., [0.1, 0.5, 0.9]).
            If None, uses quantile_levels from __init__.
        
        Returns
        -------
        pd.DataFrame
            Multi-level DataFrame with columns for each series and quantile.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        quantiles_requested = quantile_levels or self.quantile_levels
        
        # Convert to list of series for Chronos API
        context_list = [self.y_train[col].values for col in self.feature_names]
        
        # Generate forecasts - model returns 13 quantiles by default
        # Quantile levels: [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
        default_quantiles = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        forecast_list = self._chronos.predict(
            inputs=context_list,
            prediction_length=self.horizon,
        )
        
        # Extract requested quantiles
        # forecast shape: (batch=1, num_quantiles=13, prediction_length)
        results = {}
        for i, col in enumerate(self.feature_names):
            forecast_tensor = forecast_list[i][0]  # Shape: (13, horizon)
            for q in quantiles_requested:
                # Find closest quantile in default set
                closest_idx = min(range(len(default_quantiles)), 
                                 key=lambda idx: abs(default_quantiles[idx] - q))
                q_forecast = forecast_tensor[closest_idx, :].numpy()
                results[f"{col}_q{int(q*100)}"] = q_forecast
        
        # Create future dates
        freq = pd.infer_freq(self.y_train.index[-10:])
        if freq is None:
            freq = 'D'
        
        future_dates = pd.date_range(
            start=self.last_index + pd.tseries.frequencies.to_offset(freq),
            periods=self.horizon,
            freq=freq
        )
        
        return pd.DataFrame(results, index=future_dates)


# ---------------------------------------------------------------------------
# New models added in v0.6.0
# ---------------------------------------------------------------------------

# ── Shared helper ────────────────────────────────────────────────────────────
def _build_lag_df(y: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Build a lag feature DataFrame from y (reused by several models)."""
    lag_frames = []
    for lag in range(1, n_lags + 1):
        lagged = y.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
        lag_frames.append(lagged)
    return pd.concat(lag_frames, axis=1)


def _prediction_features(last_values: pd.DataFrame, n_lags: int,
                         trained_with_cov: bool,
                         covariate_columns: Optional[List],
                         X: Optional[pd.DataFrame],
                         step: int) -> pd.DataFrame:
    """Create a one-row feature vector for recursive multi-step prediction."""
    feats: list = []
    feat_names: list = []
    for lag in range(1, n_lags + 1):
        if lag <= len(last_values):
            row = last_values.iloc[-lag]
            feats.extend(row.values.tolist())
            feat_names.extend([f"{col}_lag{lag}" for col in last_values.columns])
    if trained_with_cov:
        if X is None or len(X) == 0:
            raise ValueError("Model was trained with covariates but none were provided for prediction.")
        row_idx = min(step, len(X) - 1)
        feats.extend(X.iloc[row_idx].values.tolist())
        feat_names.extend(X.columns.tolist())
    return pd.DataFrame([feats], columns=feat_names)


def _direct_fit_loop(y: pd.DataFrame, lag_df: pd.DataFrame,
                     X: Optional[pd.DataFrame], horizon: int,
                     make_model_fn):
    """
    Shared direct multi-output training loop used by LightGBM, CatBoost,
    and ElasticNet forecasters.

    Returns: list of lists – outer is horizon step, inner is per-column model.
    """
    models_by_step = []
    for h in range(1, horizon + 1):
        y_shifted = y.shift(-h)
        if X is not None:
            X_h = X.shift(-h)
            X_train_h = pd.concat([lag_df, X_h], axis=1).dropna()
        else:
            X_train_h = lag_df.dropna()
        train_idx = X_train_h.index.intersection(y_shifted.dropna().index)
        X_train_h = X_train_h.loc[train_idx]
        y_train_h = y_shifted.loc[train_idx]
        models_h = []
        for col in y.columns:
            m = make_model_fn()
            m.fit(X_train_h, y_train_h[col])
            models_h.append(m)
        models_by_step.append(models_h)
    return models_by_step


def _direct_predict_loop(models_by_step: list, last_values: pd.DataFrame,
                         feature_names: list, n_lags: int,
                         trained_with_cov: bool,
                         covariate_columns: Optional[List],
                         X: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Shared direct multi-output prediction loop."""
    lv = last_values.copy()
    predictions = []
    for h, models_h in enumerate(models_by_step):
        X_pred = _prediction_features(lv, n_lags, trained_with_cov, covariate_columns, X, h)
        pred_h = [m.predict(X_pred)[0] for m in models_h]
        predictions.append(pred_h)
        new_row = pd.DataFrame([pred_h], columns=feature_names)
        lv = pd.concat([lv, new_row], ignore_index=True)
    return pd.DataFrame(predictions, columns=feature_names)


# ── LightGBMForecaster ───────────────────────────────────────────────────────

class LightGBMForecaster(BaseForecaster):
    """LightGBM forecaster with lag features and covariate support.

    Drop-in replacement for :class:`XGBoostForecaster` backed by LightGBM.
    Typically trains 5-10x faster than XGBoost on CPU for the same depth, and
    handles large numbers of features and series more efficiently.

    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values used as features.
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=-1
        Maximum tree depth (−1 = unlimited, LightGBM default).
    learning_rate : float, default=0.1
        Boosting learning rate.
    num_leaves : int, default=31
        Maximum number of leaves in one tree.
    horizon : int, default=1
        Forecast horizon.
    random_state : int, default=42
        Random seed.
    preprocess_covariates : bool, default=True
        Auto-encode categorical covariates.
    verbose : int, default=-1
        LightGBM verbosity (−1 = silent).
    **lgb_params
        Extra keyword arguments forwarded to ``LGBMRegressor``.
    """

    supports_covariates: bool = True

    def __init__(
        self,
        n_lags: int = 7,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        horizon: int = 1,
        random_state: int = 42,
        preprocess_covariates: bool = True,
        verbose: int = -1,
        **lgb_params,
    ):
        try:
            lgb = _import_lightgbm()
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMForecaster. "
                'Install with: pip install "autotsforecast[lightgbm]" '
                "or: pip install lightgbm"
            ) from exc

        super().__init__(horizon)
        self._lgb = lgb
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.verbose = verbose
        self.lgb_params = lgb_params
        self.models: list = []
        self.covariate_preprocessor_: Optional[CovariatePreprocessor] = None
        self._trained_with_covariates: bool = False
        self._covariate_columns_: Optional[List] = None
        self.last_values_: Optional[pd.DataFrame] = None

    def get_params(self) -> dict:
        return {
            "n_lags": self.n_lags,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "horizon": self.horizon,
            "random_state": self.random_state,
            "preprocess_covariates": self.preprocess_covariates,
            "verbose": self.verbose,
            **self.lgb_params,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "LightGBMForecaster":
        self.feature_names = y.columns.tolist()
        self._trained_with_covariates = X is not None

        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)
        self._covariate_columns_ = list(X.columns) if X is not None else None

        lag_df = _build_lag_df(y, self.n_lags)
        self.last_values_ = y.iloc[-self.n_lags :].copy()

        lgb = self._lgb

        def _make():
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbose=self.verbose,
                **self.lgb_params,
            )

        self.models = _direct_fit_loop(y, lag_df, X, self.horizon, _make)
        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        if self._trained_with_covariates:
            if X is None:
                raise ValueError("Model was trained with covariates; provide X for prediction.")
            if self.covariate_preprocessor_ is not None:
                X = self.covariate_preprocessor_.transform(X)
            if self._covariate_columns_:
                missing = set(self._covariate_columns_) - set(X.columns)
                if missing:
                    raise ValueError(f"Missing covariates for prediction: {sorted(missing)}")
                X = X[self._covariate_columns_]
        return _direct_predict_loop(
            self.models, self.last_values_, self.feature_names,
            self.n_lags, self._trained_with_covariates,
            self._covariate_columns_, X,
        )


# ── CatBoostForecaster ───────────────────────────────────────────────────────

class CatBoostForecaster(BaseForecaster):
    """CatBoost forecaster with lag features and covariate support.

    CatBoost often provides state-of-the-art accuracy with minimal hyperparameter
    tuning, native handling of categorical features, and fast CPU inference.
    Follows the same direct multi-step strategy as :class:`LightGBMForecaster`.

    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values used as features.
    n_estimators : int, default=100
        Number of boosting iterations.
    depth : int, default=6
        Depth of trees.
    learning_rate : float, default=0.1
        Boosting learning rate.
    horizon : int, default=1
        Forecast horizon.
    random_state : int, default=42
        Random seed.
    preprocess_covariates : bool, default=True
        Auto-encode categorical covariates.
    verbose : int, default=0
        CatBoost verbosity (0 = silent).
    **cb_params
        Extra keyword arguments forwarded to ``CatBoostRegressor``.
    """

    supports_covariates: bool = True

    def __init__(
        self,
        n_lags: int = 7,
        n_estimators: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        horizon: int = 1,
        random_state: int = 42,
        preprocess_covariates: bool = True,
        verbose: int = 0,
        **cb_params,
    ):
        try:
            _import_catboost()
        except ImportError as exc:
            raise ImportError(
                "catboost is required for CatBoostForecaster. "
                'Install with: pip install "autotsforecast[catboost]" '
                "or: pip install catboost"
            ) from exc

        super().__init__(horizon)
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.preprocess_covariates = preprocess_covariates
        self.verbose = verbose
        self.cb_params = cb_params
        self.models: list = []
        self.covariate_preprocessor_: Optional[CovariatePreprocessor] = None
        self._trained_with_covariates: bool = False
        self._covariate_columns_: Optional[List] = None
        self.last_values_: Optional[pd.DataFrame] = None

    def get_params(self) -> dict:
        return {
            "n_lags": self.n_lags,
            "n_estimators": self.n_estimators,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "horizon": self.horizon,
            "random_state": self.random_state,
            "preprocess_covariates": self.preprocess_covariates,
            "verbose": self.verbose,
            **self.cb_params,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "CatBoostForecaster":
        CatBoostRegressor = _import_catboost()

        self.feature_names = y.columns.tolist()
        self._trained_with_covariates = X is not None

        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)
        self._covariate_columns_ = list(X.columns) if X is not None else None

        lag_df = _build_lag_df(y, self.n_lags)
        self.last_values_ = y.iloc[-self.n_lags :].copy()

        def _make():
            return CatBoostRegressor(
                iterations=self.n_estimators,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_seed=self.random_state,
                verbose=self.verbose,
                **self.cb_params,
            )

        self.models = _direct_fit_loop(y, lag_df, X, self.horizon, _make)
        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        if self._trained_with_covariates:
            if X is None:
                raise ValueError("Model was trained with covariates; provide X for prediction.")
            if self.covariate_preprocessor_ is not None:
                X = self.covariate_preprocessor_.transform(X)
            if self._covariate_columns_:
                missing = set(self._covariate_columns_) - set(X.columns)
                if missing:
                    raise ValueError(f"Missing covariates for prediction: {sorted(missing)}")
                X = X[self._covariate_columns_]
        return _direct_predict_loop(
            self.models, self.last_values_, self.feature_names,
            self.n_lags, self._trained_with_covariates,
            self._covariate_columns_, X,
        )


# ── ElasticNetForecaster ─────────────────────────────────────────────────────

class ElasticNetForecaster(BaseForecaster):
    """ElasticNet regression forecaster with lag features.

    A fast, interpretable linear forecaster with combined L1+L2 regularisation.
    It works with or without covariates and is the recommended model for the
    ``'fast'`` preset due to its near-instant training time.

    Parameters
    ----------
    n_lags : int, default=7
        Number of lagged values used as features.
    alpha : float, default=0.01
        Regularisation strength.
    l1_ratio : float, default=0.5
        Mixing parameter: 0 = Ridge, 1 = Lasso, 0.5 = ElasticNet.
    horizon : int, default=1
        Forecast horizon.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    preprocess_covariates : bool, default=True
        Auto-encode categorical covariates.
    random_state : int, default=42
        Random seed (used for internal solvers).
    """

    supports_covariates: bool = True

    def __init__(
        self,
        n_lags: int = 7,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        horizon: int = 1,
        fit_intercept: bool = True,
        preprocess_covariates: bool = True,
        random_state: int = 42,
    ):
        try:
            from sklearn.linear_model import ElasticNet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for ElasticNetForecaster. "
                "Install with: pip install scikit-learn"
            ) from exc

        super().__init__(horizon)
        self.n_lags = n_lags
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.preprocess_covariates = preprocess_covariates
        self.random_state = random_state
        self.models: list = []
        self.covariate_preprocessor_: Optional[CovariatePreprocessor] = None
        self._trained_with_covariates: bool = False
        self._covariate_columns_: Optional[List] = None
        self.last_values_: Optional[pd.DataFrame] = None

    def get_params(self) -> dict:
        return {
            "n_lags": self.n_lags,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "horizon": self.horizon,
            "fit_intercept": self.fit_intercept,
            "preprocess_covariates": self.preprocess_covariates,
            "random_state": self.random_state,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "ElasticNetForecaster":
        from sklearn.linear_model import ElasticNet

        self.feature_names = y.columns.tolist()
        self._trained_with_covariates = X is not None

        if X is not None and self.preprocess_covariates:
            self.covariate_preprocessor_ = CovariatePreprocessor()
            X = self.covariate_preprocessor_.fit_transform(X)
        self._covariate_columns_ = list(X.columns) if X is not None else None

        lag_df = _build_lag_df(y, self.n_lags)
        self.last_values_ = y.iloc[-self.n_lags :].copy()

        def _make():
            return ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                max_iter=3000,
            )

        self.models = _direct_fit_loop(y, lag_df, X, self.horizon, _make)
        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        if self._trained_with_covariates:
            if X is None:
                raise ValueError("Model was trained with covariates; provide X for prediction.")
            if self.covariate_preprocessor_ is not None:
                X = self.covariate_preprocessor_.transform(X)
            if self._covariate_columns_:
                missing = set(self._covariate_columns_) - set(X.columns)
                if missing:
                    raise ValueError(f"Missing covariates for prediction: {sorted(missing)}")
                X = X[self._covariate_columns_]
        return _direct_predict_loop(
            self.models, self.last_values_, self.feature_names,
            self.n_lags, self._trained_with_covariates,
            self._covariate_columns_, X,
        )


# ── ThetaForecaster ──────────────────────────────────────────────────────────

class ThetaForecaster(BaseForecaster):
    """Theta method forecaster using statsmodels.

    The Theta method placed first in the M3 forecasting competition and remains
    one of the strongest statistical baselines for univariate series. It decomposes
    the series into two *theta lines* and forecasts them individually before
    recombining.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon.
    period : int, optional
        Seasonal period (e.g., 7 for daily data with weekly seasonality).
        If ``None``, auto-detected from the DatetimeIndex frequency.
    deseasonalize : bool, default=True
        Remove seasonality before fitting. Only applied when ``period > 1``.
    """

    supports_covariates: bool = False

    def __init__(
        self,
        horizon: int = 1,
        period: Optional[int] = None,
        deseasonalize: bool = True,
    ):
        super().__init__(horizon)
        self.period = period
        self.deseasonalize = deseasonalize
        self.models: dict = {}
        self.last_date: Optional[pd.Timestamp] = None
        self.freq_: Optional[str] = None

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon,
            "period": self.period,
            "deseasonalize": self.deseasonalize,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "ThetaForecaster":
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
        except ImportError as exc:
            raise ImportError(
                "statsmodels>=0.12 is required for ThetaForecaster. "
                "Install with: pip install statsmodels"
            ) from exc

        self.feature_names = y.columns.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index) or "D"

        # Auto-detect period from frequency
        period = self.period
        if period is None and self.deseasonalize and self.freq_ is not None:
            _freq_period_map = {
                "D": 7, "W": 52, "M": 12, "MS": 12, "ME": 12,
                "Q": 4, "QS": 4, "QE": 4, "H": 24, "T": 60, "min": 60,
            }
            base = "".join(c for c in str(self.freq_) if c.isalpha())
            period = _freq_period_map.get(base, 1)

        self.models = {}
        for col in y.columns:
            series = y[col].astype(float)
            use_seas = self.deseasonalize and period is not None and period > 1
            try:
                m = ThetaModel(
                    series,
                    period=period if use_seas else None,
                    deseasonalize=use_seas,
                )
                self.models[col] = m.fit()
            except Exception:
                # Fallback without seasonality
                self.models[col] = ThetaModel(series, period=None, deseasonalize=False).fit()

        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        predictions = {col: np.asarray(fitted.forecast(steps=self.horizon))
                       for col, fitted in self.models.items()}

        if self.last_date is not None:
            freq = self.freq_ or "D"
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon,
                freq=freq,
            )
            return pd.DataFrame(predictions, index=future_dates)
        return pd.DataFrame(predictions)


# ── CrostonForecaster ────────────────────────────────────────────────────────

class CrostonForecaster(BaseForecaster):
    """Croston's method for intermittent demand forecasting.

    Specifically designed for sparse time series with many zeros.
    Uses separate exponential smoothing for demand size and inter-demand
    intervals. Includes optional SBA (Syntetos-Boylan Approximation) bias
    correction, which is recommended in most practical applications.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon.
    alpha : float, default=0.1
        Smoothing parameter applied to both demand size and intervals
        (0 < alpha < 1).
    method : str, default='sba'
        ``'croston'`` — original Croston (1972).
        ``'sba'`` — Syntetos-Boylan Approximation (typically lower bias).
    """

    supports_covariates: bool = False

    def __init__(
        self,
        horizon: int = 1,
        alpha: float = 0.1,
        method: str = "sba",
    ):
        super().__init__(horizon)
        self.alpha = alpha
        self.method = method
        self._rates: dict = {}
        self.last_date: Optional[pd.Timestamp] = None
        self.freq_: Optional[str] = None

    def get_params(self) -> dict:
        return {"horizon": self.horizon, "alpha": self.alpha, "method": self.method}

    def _croston_rate(self, y_arr: np.ndarray) -> float:
        """Fit Croston/SBA and return the per-period demand rate."""
        nonzero_idx = np.nonzero(y_arr)[0]
        if len(nonzero_idx) == 0:
            return 0.0

        d = float(y_arr[nonzero_idx[0]])   # demand level estimate
        p = float(nonzero_idx[0] + 1)      # inter-demand interval estimate

        for i in range(nonzero_idx[0] + 1, len(y_arr)):
            if y_arr[i] > 0:
                d = self.alpha * float(y_arr[i]) + (1 - self.alpha) * d
                prev_nonzero = nonzero_idx[nonzero_idx < i]
                interval = float(i - prev_nonzero[-1]) if len(prev_nonzero) else float(i + 1)
                p = self.alpha * interval + (1 - self.alpha) * p

        p = max(p, 1e-6)
        rate = d / p
        if self.method == "sba":
            rate *= (1.0 - self.alpha / 2.0)
        return float(rate)

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "CrostonForecaster":
        self.feature_names = y.columns.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index) or "D"

        self._rates = {col: self._croston_rate(y[col].values.astype(float))
                       for col in y.columns}
        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        preds = {col: [v] * self.horizon for col, v in self._rates.items()}
        if self.last_date is not None:
            freq = self.freq_ or "D"
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon,
                freq=freq,
            )
            return pd.DataFrame(preds, index=future_dates)
        return pd.DataFrame(preds)


# ── Darts-based neural models (N-BEATS, N-HiTS, TFT) ────────────────────────

def _darts_series(y_col: pd.DataFrame, freq: Optional[str]) -> "Any":
    """Convert a single-column DataFrame to a Darts TimeSeries."""
    _, TimeSeries = _import_darts_nbeats()
    df = y_col.copy()
    if isinstance(df.index, pd.DatetimeIndex) and freq:
        try:
            df = df.asfreq(freq)
        except Exception:
            pass
    return TimeSeries.from_dataframe(df, fill_missing_dates=False)


class NBEATSForecaster(BaseForecaster):
    """N-BEATS forecaster (Neural Basis Expansion Analysis for Interpretable TS).

    A pure deep-learning model that achieves state-of-the-art accuracy without
    any manual feature engineering. Fits one N-BEATS model per series.

    Requires ``pip install "autotsforecast[neural]"``.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon (= ``output_chunk_length``).
    n_lags : int, default=28
        Lookback window (= ``input_chunk_length``). Should be ≥ 2 × horizon.
    num_stacks : int, default=30
        Number of N-BEATS stacks.
    num_blocks : int, default=1
        Number of blocks per stack.
    num_layers : int, default=4
        Fully-connected layers per block.
    layer_widths : int, default=256
        Width of each fully-connected layer.
    n_epochs : int, default=100
        Training epochs.
    batch_size : int, default=32
        Mini-batch size.
    random_state : int, default=42
        Random seed.
    """

    supports_covariates: bool = False

    def __init__(
        self,
        horizon: int = 1,
        n_lags: int = 28,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        try:
            _import_darts_nbeats()
        except ImportError as exc:
            raise ImportError(
                "darts is required for NBEATSForecaster. "
                'Install with: pip install "autotsforecast[neural]"'
            ) from exc

        super().__init__(horizon)
        self.n_lags = n_lags
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.models: dict = {}
        self.last_date: Optional[pd.Timestamp] = None
        self.freq_: Optional[str] = None

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon, "n_lags": self.n_lags,
            "num_stacks": self.num_stacks, "num_blocks": self.num_blocks,
            "num_layers": self.num_layers, "layer_widths": self.layer_widths,
            "n_epochs": self.n_epochs, "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "NBEATSForecaster":
        NBEATSModel, _ = _import_darts_nbeats()
        self.feature_names = y.columns.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index) or "D"

        self.models = {}
        for col in y.columns:
            ts = _darts_series(y[[col]], self.freq_)
            m = NBEATSModel(
                input_chunk_length=self.n_lags,
                output_chunk_length=self.horizon,
                num_stacks=self.num_stacks,
                num_blocks=self.num_blocks,
                num_layers=self.num_layers,
                layer_widths=self.layer_widths,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
                pl_trainer_kwargs={"enable_progress_bar": False},
            )
            m.fit(ts)
            self.models[col] = m

        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        predictions = {col: m.predict(self.horizon).values().flatten()
                       for col, m in self.models.items()}

        if self.last_date is not None:
            freq = self.freq_ or "D"
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon, freq=freq,
            )
            return pd.DataFrame(predictions, index=future_dates)
        return pd.DataFrame(predictions)


class NHiTSForecaster(BaseForecaster):
    """N-HiTS forecaster (Neural Hierarchical Interpolation for Time Series).

    N-HiTS improves on N-BEATS with hierarchical interpolation and multi-rate
    data sampling, making it especially accurate for long-horizon forecasts at
    a lower computational cost.

    Requires ``pip install "autotsforecast[neural]"``.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon.
    n_lags : int, default=28
        Lookback window. Should be ≥ 2 × horizon.
    num_stacks : int, default=3
        Number of N-HiTS stacks.
    num_blocks : int, default=1
        Blocks per stack.
    num_layers : int, default=2
        Fully-connected layers per block.
    layer_widths : int, default=512
        Width of each fully-connected layer.
    n_epochs : int, default=100
        Training epochs.
    batch_size : int, default=32
        Mini-batch size.
    random_state : int, default=42
        Random seed.
    """

    supports_covariates: bool = False

    def __init__(
        self,
        horizon: int = 1,
        n_lags: int = 28,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 512,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        try:
            _import_darts_nhits()
        except ImportError as exc:
            raise ImportError(
                "darts is required for NHiTSForecaster. "
                'Install with: pip install "autotsforecast[neural]"'
            ) from exc

        super().__init__(horizon)
        self.n_lags = n_lags
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.models: dict = {}
        self.last_date: Optional[pd.Timestamp] = None
        self.freq_: Optional[str] = None

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon, "n_lags": self.n_lags,
            "num_stacks": self.num_stacks, "num_blocks": self.num_blocks,
            "num_layers": self.num_layers, "layer_widths": self.layer_widths,
            "n_epochs": self.n_epochs, "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "NHiTSForecaster":
        NHiTSModel, _ = _import_darts_nhits()
        self.feature_names = y.columns.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index) or "D"

        self.models = {}
        for col in y.columns:
            ts = _darts_series(y[[col]], self.freq_)
            m = NHiTSModel(
                input_chunk_length=self.n_lags,
                output_chunk_length=self.horizon,
                num_stacks=self.num_stacks,
                num_blocks=self.num_blocks,
                num_layers=self.num_layers,
                layer_widths=self.layer_widths,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
                pl_trainer_kwargs={"enable_progress_bar": False},
            )
            m.fit(ts)
            self.models[col] = m

        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        predictions = {col: m.predict(self.horizon).values().flatten()
                       for col, m in self.models.items()}

        if self.last_date is not None:
            freq = self.freq_ or "D"
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon, freq=freq,
            )
            return pd.DataFrame(predictions, index=future_dates)
        return pd.DataFrame(predictions)


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer (TFT) forecaster.

    TFT is a powerful attention-based model designed for multi-horizon
    forecasting with mixed types of inputs (static metadata, known future
    inputs, and past covariates). It achieves strong results on tabular time
    series benchmarks.

    Requires ``pip install "autotsforecast[neural]"``.

    Parameters
    ----------
    horizon : int, default=1
        Forecast horizon (= ``output_chunk_length``).
    n_lags : int, default=28
        Lookback window (= ``input_chunk_length``).
    hidden_size : int, default=16
        Hidden state size.
    lstm_layers : int, default=1
        Number of LSTM encoder layers.
    num_attention_heads : int, default=4
        Multi-head attention heads.
    dropout : float, default=0.1
        Dropout rate.
    n_epochs : int, default=100
        Training epochs.
    batch_size : int, default=32
        Mini-batch size.
    random_state : int, default=42
        Random seed.
    """

    supports_covariates: bool = False  # future covariates require Darts TimeSeries API

    def __init__(
        self,
        horizon: int = 1,
        n_lags: int = 28,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        try:
            _import_darts_tft()
        except ImportError as exc:
            raise ImportError(
                "darts is required for TFTForecaster. "
                'Install with: pip install "autotsforecast[neural]"'
            ) from exc

        super().__init__(horizon)
        self.n_lags = n_lags
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.models: dict = {}
        self.last_date: Optional[pd.Timestamp] = None
        self.freq_: Optional[str] = None

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon, "n_lags": self.n_lags,
            "hidden_size": self.hidden_size, "lstm_layers": self.lstm_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout, "n_epochs": self.n_epochs,
            "batch_size": self.batch_size, "random_state": self.random_state,
        }

    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> "TFTForecaster":
        TFTModel, _ = _import_darts_tft()
        self.feature_names = y.columns.tolist()
        if isinstance(y.index, pd.DatetimeIndex):
            self.last_date = y.index[-1]
            self.freq_ = pd.infer_freq(y.index) or "D"

        self.models = {}
        for col in y.columns:
            ts = _darts_series(y[[col]], self.freq_)
            m = TFTModel(
                input_chunk_length=self.n_lags,
                output_chunk_length=self.horizon,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                num_attention_heads=self.num_attention_heads,
                dropout=self.dropout,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
                pl_trainer_kwargs={"enable_progress_bar": False},
            )
            m.fit(ts)
            self.models[col] = m

        self.is_fitted = True
        return self

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        predictions = {col: m.predict(self.horizon).values().flatten()
                       for col, m in self.models.items()}

        if self.last_date is not None:
            freq = self.freq_ or "D"
            future_dates = pd.date_range(
                start=self.last_date + pd.tseries.frequencies.to_offset(freq),
                periods=self.horizon, freq=freq,
            )
            return pd.DataFrame(predictions, index=future_dates)
        return pd.DataFrame(predictions)

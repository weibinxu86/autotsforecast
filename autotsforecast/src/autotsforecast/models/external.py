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

 


class RandomForestForecaster(BaseForecaster):
    """

    supports_covariates: bool = True
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
        self._trained_with_covariates = False
        self._covariate_columns_ = None
        
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

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.rf_params
            )
            model = MultiOutputRegressor(rf)
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
    """

    supports_covariates: bool = True
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
        self._trained_with_covariates = False
        self._covariate_columns_ = None
        
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
                model = xgb.XGBRegressor(
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
        self.freq_ = None
        self.regressor_cols_ = []
        
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

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ARIMAForecaster(BaseForecaster):
    """

    supports_covariates: bool = True
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    Fits separate ARIMA models for each time series in multivariate data.
    
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
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'ARIMAForecaster':
        """Fit separate ARIMA model for each series"""
        self.feature_names = y.columns.tolist()
        
        # Fit an ARIMA model for each column
        for col in y.columns:
            model = ARIMA(
                y[col],
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=X if X is not None else None,
                **self.arima_params
            )
            fitted_model = model.fit()
            self.models[col] = fitted_model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts using ARIMA"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        # Handle covariates: ARIMA needs exog for ALL forecast steps
        exog_forecast = None
        if X is not None:
            if len(X) < self.horizon:
                # If only one row provided, repeat it for all horizon steps
                # This handles backtesting case where validator provides one row at a time
                exog_forecast = pd.concat([X] * self.horizon, ignore_index=True)
            else:
                exog_forecast = X.iloc[:self.horizon]
        
        for col, model in self.models.items():
            # Generate forecast
            if exog_forecast is not None:
                forecast = model.forecast(steps=self.horizon, exog=exog_forecast)
            else:
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
        if not TORCH_AVAILABLE:
            raise ImportError("pytorch is required for LSTMForecaster. Install with: pip install torch")
        
        super().__init__(horizon)
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

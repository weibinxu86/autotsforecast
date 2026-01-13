"""
Comprehensive feature engineering engine for time series forecasting.

This module combines calendar features, lag features, rolling statistics,
and other transformations into a single pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from .calendar import CalendarFeatures


class FeatureEngine:
    """
    Comprehensive feature engineering for time series forecasting.
    
    Combines multiple feature extraction methods into a single pipeline:
    - Calendar features (day of week, month, holidays, etc.)
    - Lag features (previous values)
    - Rolling statistics (mean, std, min, max, etc.)
    - Fourier terms for seasonality
    - Interaction features
    - Difference features
    
    Parameters
    ----------
    calendar_features : list of str or bool, optional
        Calendar features to extract. If True, auto-detect.
        If False or None, skip calendar features.
        
    lag_features : list of int, optional
        Lag periods to create. E.g., [1, 7, 14, 28]
        
    rolling_features : dict, optional
        Rolling statistics to compute. Keys are statistic names
        ('mean', 'std', 'min', 'max', 'sum', 'median'),
        values are lists of window sizes.
        Example: {'mean': [7, 14, 28], 'std': [7, 14]}
        
    ewm_features : dict, optional
        Exponentially weighted statistics. Keys are statistic names,
        values are lists of spans/halflife.
        Example: {'mean': [7, 14], 'std': [7]}
        
    fourier_terms : dict, optional
        Fourier terms for seasonality.
        Example: {'yearly': (365.25, 5), 'weekly': (7, 3)}
        
    difference_features : list of int, optional
        Difference orders to create. E.g., [1, 7] for first diff and weekly diff.
        
    country : str, optional
        Country code for holiday features.
        
    cyclical_encoding : bool, default=True
        Use sine/cosine encoding for cyclical features.
        
    drop_na : bool, default=True
        Whether to drop rows with NaN values after feature creation.
        
    Examples
    --------
    >>> from autotsforecast.features import FeatureEngine
    >>> 
    >>> # Create feature engine with multiple feature types
    >>> engine = FeatureEngine(
    ...     calendar_features=['dayofweek', 'month', 'is_weekend'],
    ...     lag_features=[1, 7, 14, 28],
    ...     rolling_features={'mean': [7, 14, 28], 'std': [7, 14]},
    ...     fourier_terms={'yearly': (365.25, 5)},
    ...     country='US'
    ... )
    >>> 
    >>> # Fit and transform
    >>> X = engine.fit_transform(y)
    >>> 
    >>> # Generate features for future
    >>> X_future = engine.transform_future(horizon=30)
    """
    
    def __init__(
        self,
        calendar_features: Optional[Union[List[str], bool]] = True,
        lag_features: Optional[List[int]] = None,
        rolling_features: Optional[Dict[str, List[int]]] = None,
        ewm_features: Optional[Dict[str, List[int]]] = None,
        fourier_terms: Optional[Dict[str, tuple]] = None,
        difference_features: Optional[List[int]] = None,
        country: Optional[str] = None,
        cyclical_encoding: bool = True,
        drop_na: bool = True
    ):
        self.calendar_features = calendar_features
        self.lag_features = lag_features
        self.rolling_features = rolling_features
        self.ewm_features = ewm_features
        self.fourier_terms = fourier_terms or {}
        self.difference_features = difference_features
        self.country = country
        self.cyclical_encoding = cyclical_encoding
        self.drop_na = drop_na
        
        # State
        self.is_fitted_ = False
        self.feature_names_out_ = []
        self.calendar_extractor_ = None
        self._last_values_ = None
        self._last_index_ = None
        self._freq_ = None
        self._target_columns_ = None
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'FeatureEngine':
        """
        Fit the feature engine on training data.
        
        Parameters
        ----------
        y : pd.DataFrame
            Target time series
        X : pd.DataFrame, optional
            Additional covariates (will be included in output)
            
        Returns
        -------
        self
        """
        self._target_columns_ = y.columns.tolist()
        self._last_index_ = y.index[-1]
        self._freq_ = pd.infer_freq(y.index) or 'D'
        
        # Store last values needed for lag/rolling features
        max_lookback = 1
        if self.lag_features:
            max_lookback = max(max_lookback, max(self.lag_features))
        if self.rolling_features:
            for windows in self.rolling_features.values():
                max_lookback = max(max_lookback, max(windows))
        if self.ewm_features:
            for spans in self.ewm_features.values():
                max_lookback = max(max_lookback, max(spans) * 3)
        
        self._last_values_ = y.iloc[-max_lookback:].copy()
        
        # Fit calendar extractor
        if self.calendar_features:
            features = None if self.calendar_features is True else self.calendar_features
            self.calendar_extractor_ = CalendarFeatures(
                features=features,
                country=self.country,
                fourier_terms=self.fourier_terms,
                cyclical_encoding=self.cyclical_encoding
            )
            self.calendar_extractor_.fit(y)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform data to extract all features.
        
        Parameters
        ----------
        y : pd.DataFrame
            Target time series
        X : pd.DataFrame, optional
            Additional covariates
            
        Returns
        -------
        pd.DataFrame
            Feature matrix
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngine must be fitted before transform")
        
        features_list = []
        
        # Calendar features
        if self.calendar_extractor_ is not None:
            cal_features = self.calendar_extractor_.transform(y)
            features_list.append(cal_features)
        
        # Lag features
        if self.lag_features:
            for col in y.columns:
                for lag in self.lag_features:
                    features_list.append(
                        y[col].shift(lag).to_frame(f'{col}_lag{lag}')
                    )
        
        # Rolling features
        if self.rolling_features:
            for stat_name, windows in self.rolling_features.items():
                for col in y.columns:
                    for window in windows:
                        if stat_name == 'mean':
                            feat = y[col].rolling(window).mean()
                        elif stat_name == 'std':
                            feat = y[col].rolling(window).std()
                        elif stat_name == 'min':
                            feat = y[col].rolling(window).min()
                        elif stat_name == 'max':
                            feat = y[col].rolling(window).max()
                        elif stat_name == 'sum':
                            feat = y[col].rolling(window).sum()
                        elif stat_name == 'median':
                            feat = y[col].rolling(window).median()
                        else:
                            continue
                        features_list.append(
                            feat.to_frame(f'{col}_rolling_{stat_name}_{window}')
                        )
        
        # EWM features
        if self.ewm_features:
            for stat_name, spans in self.ewm_features.items():
                for col in y.columns:
                    for span in spans:
                        if stat_name == 'mean':
                            feat = y[col].ewm(span=span).mean()
                        elif stat_name == 'std':
                            feat = y[col].ewm(span=span).std()
                        else:
                            continue
                        features_list.append(
                            feat.to_frame(f'{col}_ewm_{stat_name}_{span}')
                        )
        
        # Difference features
        if self.difference_features:
            for col in y.columns:
                for diff_order in self.difference_features:
                    features_list.append(
                        y[col].diff(diff_order).to_frame(f'{col}_diff{diff_order}')
                    )
        
        # Add external covariates
        if X is not None:
            features_list.append(X)
        
        # Combine all features
        if features_list:
            result = pd.concat(features_list, axis=1)
        else:
            result = pd.DataFrame(index=y.index)
        
        self.feature_names_out_ = list(result.columns)
        
        if self.drop_na:
            result = result.dropna()
        
        return result
    
    def fit_transform(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(y, X).transform(y, X)
    
    def transform_future(
        self,
        horizon: int,
        y_recent: Optional[pd.DataFrame] = None,
        X_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate features for future forecasting periods.
        
        Note: Lag and rolling features will use the last known values
        and cannot be updated during the forecast horizon.
        
        Parameters
        ----------
        horizon : int
            Number of future periods
        y_recent : pd.DataFrame, optional
            Recent target values for lag features. If None, uses stored values.
        X_future : pd.DataFrame, optional
            Future covariate values
            
        Returns
        -------
        pd.DataFrame
            Feature matrix for future periods
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngine must be fitted before generating future features")
        
        # Generate future index
        start = self._last_index_ + pd.tseries.frequencies.to_offset(self._freq_)
        future_index = pd.date_range(start=start, periods=horizon, freq=self._freq_)
        
        features_list = []
        
        # Calendar features
        if self.calendar_extractor_ is not None:
            cal_features = self.calendar_extractor_.transform_future(horizon)
            cal_features.index = future_index
            features_list.append(cal_features)
        
        # For lag/rolling features, we use the last known values
        # These won't be updated during forecast horizon
        y_for_lags = y_recent if y_recent is not None else self._last_values_
        
        if self.lag_features or self.rolling_features or self.ewm_features:
            # Create placeholder with NaN for features that need history
            placeholder = {}
            
            if self.lag_features:
                for col in self._target_columns_:
                    for lag in self.lag_features:
                        placeholder[f'{col}_lag{lag}'] = [np.nan] * horizon
            
            if self.rolling_features:
                for stat_name, windows in self.rolling_features.items():
                    for col in self._target_columns_:
                        for window in windows:
                            placeholder[f'{col}_rolling_{stat_name}_{window}'] = [np.nan] * horizon
            
            if self.ewm_features:
                for stat_name, spans in self.ewm_features.items():
                    for col in self._target_columns_:
                        for span in spans:
                            placeholder[f'{col}_ewm_{stat_name}_{span}'] = [np.nan] * horizon
            
            if placeholder:
                features_list.append(pd.DataFrame(placeholder, index=future_index))
        
        # Difference features (NaN for future)
        if self.difference_features:
            placeholder = {}
            for col in self._target_columns_:
                for diff_order in self.difference_features:
                    placeholder[f'{col}_diff{diff_order}'] = [np.nan] * horizon
            features_list.append(pd.DataFrame(placeholder, index=future_index))
        
        # Add external covariates
        if X_future is not None:
            if len(X_future) != horizon:
                raise ValueError(f"X_future must have {horizon} rows, got {len(X_future)}")
            X_future = X_future.copy()
            X_future.index = future_index
            features_list.append(X_future)
        
        if features_list:
            return pd.concat(features_list, axis=1)
        return pd.DataFrame(index=future_index)
    
    def get_feature_names(self) -> List[str]:
        """Get names of output features."""
        return self.feature_names_out_

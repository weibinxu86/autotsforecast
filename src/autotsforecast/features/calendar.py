"""
Calendar and time-based feature generation for time series forecasting.

This module provides automatic extraction of calendar features from datetime indices,
including day of week, month, holidays, Fourier terms for seasonality, and more.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any
from functools import lru_cache


# Try to import holidays library (optional)
try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


class CalendarFeatures:
    """
    Automatic calendar and time-based feature extraction.
    
    Extracts various temporal features from datetime indices to improve
    forecasting accuracy by capturing seasonality and calendar effects.
    
    Parameters
    ----------
    features : list of str, optional
        List of features to extract. Options:
        - 'dayofweek': Day of week (0=Monday, 6=Sunday)
        - 'dayofmonth': Day of month (1-31)
        - 'dayofyear': Day of year (1-366)
        - 'weekofyear': Week number (1-53)
        - 'month': Month (1-12)
        - 'quarter': Quarter (1-4)
        - 'year': Year
        - 'is_weekend': Binary weekend indicator
        - 'is_month_start': Binary month start indicator
        - 'is_month_end': Binary month end indicator
        - 'is_quarter_start': Binary quarter start indicator
        - 'is_quarter_end': Binary quarter end indicator
        - 'is_year_start': Binary year start indicator
        - 'is_year_end': Binary year end indicator
        - 'hour': Hour of day (0-23) for intraday data
        - 'minute': Minute (0-59) for intraday data
        If None, auto-detects appropriate features based on data frequency.
    
    country : str, optional
        Country code for holiday features (e.g., 'US', 'UK', 'DE', 'FR').
        Requires `holidays` library to be installed.
    
    fourier_terms : dict, optional
        Dictionary specifying Fourier terms for capturing seasonality.
        Keys are period names, values are (period, order) tuples.
        Example: {'yearly': (365.25, 5), 'weekly': (7, 3)}
    
    cyclical_encoding : bool, default=True
        Whether to use sine/cosine encoding for cyclical features
        (dayofweek, month, hour) to preserve cyclical nature.
    
    drop_original : bool, default=True
        Whether to drop the original datetime index information.
    
    Examples
    --------
    >>> from autotsforecast.features import CalendarFeatures
    >>> 
    >>> # Basic usage - auto-detect features
    >>> cal = CalendarFeatures()
    >>> X_cal = cal.fit_transform(y)
    >>> 
    >>> # Specify features explicitly
    >>> cal = CalendarFeatures(
    ...     features=['dayofweek', 'month', 'is_weekend'],
    ...     country='US',
    ...     fourier_terms={'yearly': (365.25, 5)}
    ... )
    >>> X_cal = cal.fit_transform(y)
    >>> 
    >>> # Generate features for future dates
    >>> X_future = cal.transform_future(horizon=30, freq='D')
    """
    
    DEFAULT_FEATURES = [
        'dayofweek', 'dayofmonth', 'month', 'quarter',
        'is_weekend', 'is_month_start', 'is_month_end'
    ]
    
    ALL_FEATURES = [
        'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear',
        'month', 'quarter', 'year', 'is_weekend',
        'is_month_start', 'is_month_end', 'is_quarter_start',
        'is_quarter_end', 'is_year_start', 'is_year_end',
        'hour', 'minute'
    ]
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        country: Optional[str] = None,
        fourier_terms: Optional[Dict[str, tuple]] = None,
        cyclical_encoding: bool = True,
        drop_original: bool = True
    ):
        self.features = features
        self.country = country
        self.fourier_terms = fourier_terms or {}
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original
        
        # State
        self.is_fitted_ = False
        self.detected_features_ = []
        self.feature_names_out_ = []
        self._last_index_ = None
        self._freq_ = None
        self._holidays_cache_ = {}
        
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect the frequency of the datetime index."""
        freq = pd.infer_freq(index)
        if freq is None and len(index) > 1:
            # Estimate from median difference
            diff = pd.Series(index).diff().median()
            if diff <= pd.Timedelta(hours=1):
                freq = 'H'
            elif diff <= pd.Timedelta(days=1):
                freq = 'D'
            elif diff <= pd.Timedelta(days=7):
                freq = 'W'
            else:
                freq = 'M'
        return freq or 'D'
    
    def _auto_detect_features(self, index: pd.DatetimeIndex) -> List[str]:
        """Auto-detect appropriate features based on data frequency."""
        freq = self._detect_frequency(index)
        
        features = []
        
        # Always include basic features
        features.extend(['dayofweek', 'month', 'is_weekend'])
        
        # Add features based on frequency
        if freq in ['H', 'T', 'S', 'min']:  # Hourly or finer
            features.extend(['hour', 'dayofmonth'])
            if freq in ['T', 'S', 'min']:
                features.append('minute')
        elif freq in ['D', 'B']:  # Daily
            features.extend(['dayofmonth', 'dayofyear', 'quarter'])
        elif freq in ['W', 'W-SUN', 'W-MON']:  # Weekly
            features.extend(['weekofyear', 'quarter'])
        else:  # Monthly or longer
            features.extend(['quarter', 'year'])
        
        return list(set(features))
    
    def _get_holidays(self, years: List[int]) -> Dict[pd.Timestamp, str]:
        """Get holidays for specified years."""
        if not HOLIDAYS_AVAILABLE or self.country is None:
            return {}
        
        cache_key = (self.country, tuple(sorted(years)))
        if cache_key in self._holidays_cache_:
            return self._holidays_cache_[cache_key]
        
        try:
            country_holidays = holidays_lib.country_holidays(self.country, years=years)
            result = {pd.Timestamp(date): name for date, name in country_holidays.items()}
            self._holidays_cache_[cache_key] = result
            return result
        except Exception:
            return {}
    
    def _cyclical_encode(self, values: np.ndarray, period: int) -> tuple:
        """Encode values using sine/cosine for cyclical features."""
        sin_values = np.sin(2 * np.pi * values / period)
        cos_values = np.cos(2 * np.pi * values / period)
        return sin_values, cos_values
    
    def _extract_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Extract all configured features from datetime index."""
        features_dict = {}
        
        for feature in self.detected_features_:
            if feature == 'dayofweek':
                values = index.dayofweek.values
                if self.cyclical_encoding:
                    sin_val, cos_val = self._cyclical_encode(values, 7)
                    features_dict['dayofweek_sin'] = sin_val
                    features_dict['dayofweek_cos'] = cos_val
                else:
                    features_dict['dayofweek'] = values
                    
            elif feature == 'dayofmonth':
                features_dict['dayofmonth'] = index.day.values
                
            elif feature == 'dayofyear':
                values = index.dayofyear.values
                if self.cyclical_encoding:
                    sin_val, cos_val = self._cyclical_encode(values, 366)
                    features_dict['dayofyear_sin'] = sin_val
                    features_dict['dayofyear_cos'] = cos_val
                else:
                    features_dict['dayofyear'] = values
                    
            elif feature == 'weekofyear':
                features_dict['weekofyear'] = index.isocalendar().week.values
                
            elif feature == 'month':
                values = index.month.values
                if self.cyclical_encoding:
                    sin_val, cos_val = self._cyclical_encode(values, 12)
                    features_dict['month_sin'] = sin_val
                    features_dict['month_cos'] = cos_val
                else:
                    features_dict['month'] = values
                    
            elif feature == 'quarter':
                features_dict['quarter'] = index.quarter.values
                
            elif feature == 'year':
                features_dict['year'] = index.year.values
                
            elif feature == 'is_weekend':
                features_dict['is_weekend'] = np.asarray((index.dayofweek >= 5).astype(int))
                
            elif feature == 'is_month_start':
                features_dict['is_month_start'] = np.asarray(index.is_month_start.astype(int))
                
            elif feature == 'is_month_end':
                features_dict['is_month_end'] = np.asarray(index.is_month_end.astype(int))
                
            elif feature == 'is_quarter_start':
                features_dict['is_quarter_start'] = np.asarray(index.is_quarter_start.astype(int))
                
            elif feature == 'is_quarter_end':
                features_dict['is_quarter_end'] = np.asarray(index.is_quarter_end.astype(int))
                
            elif feature == 'is_year_start':
                features_dict['is_year_start'] = np.asarray(index.is_year_start.astype(int))
                
            elif feature == 'is_year_end':
                features_dict['is_year_end'] = np.asarray(index.is_year_end.astype(int))
                
            elif feature == 'hour':
                values = index.hour.values
                if self.cyclical_encoding:
                    sin_val, cos_val = self._cyclical_encode(values, 24)
                    features_dict['hour_sin'] = sin_val
                    features_dict['hour_cos'] = cos_val
                else:
                    features_dict['hour'] = values
                    
            elif feature == 'minute':
                values = index.minute.values
                if self.cyclical_encoding:
                    sin_val, cos_val = self._cyclical_encode(values, 60)
                    features_dict['minute_sin'] = sin_val
                    features_dict['minute_cos'] = cos_val
                else:
                    features_dict['minute'] = values
        
        # Add holiday features
        if self.country is not None and HOLIDAYS_AVAILABLE:
            years = list(set(index.year))
            holidays_dict = self._get_holidays(years)
            
            features_dict['is_holiday'] = [
                1 if pd.Timestamp(dt.date()) in holidays_dict else 0
                for dt in index
            ]
            
            # Days until next holiday (useful for retail)
            holiday_dates = sorted(holidays_dict.keys())
            if holiday_dates:
                days_to_holiday = []
                for dt in index:
                    dt_ts = pd.Timestamp(dt.date())
                    future_holidays = [h for h in holiday_dates if h >= dt_ts]
                    if future_holidays:
                        days_to_holiday.append((future_holidays[0] - dt_ts).days)
                    else:
                        days_to_holiday.append(365)  # Default if no future holidays
                features_dict['days_to_holiday'] = days_to_holiday
        
        # Add Fourier terms
        for name, (period, order) in self.fourier_terms.items():
            t = np.arange(len(index))
            for k in range(1, order + 1):
                features_dict[f'{name}_sin_{k}'] = np.sin(2 * np.pi * k * t / period)
                features_dict[f'{name}_cos_{k}'] = np.cos(2 * np.pi * k * t / period)
        
        df = pd.DataFrame(features_dict, index=index)
        self.feature_names_out_ = list(df.columns)
        return df
    
    def fit(self, y: Union[pd.DataFrame, pd.Series]) -> 'CalendarFeatures':
        """
        Fit the calendar feature extractor.
        
        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Time series data with DatetimeIndex
            
        Returns
        -------
        self
        """
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Input must have a DatetimeIndex for calendar features")
        
        self._last_index_ = y.index[-1]
        self._freq_ = self._detect_frequency(y.index)
        
        # Determine features to extract
        if self.features is None:
            self.detected_features_ = self._auto_detect_features(y.index)
        else:
            self.detected_features_ = self.features
        
        self.is_fitted_ = True
        return self
    
    def transform(self, y: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Transform datetime index to calendar features.
        
        Parameters
        ----------
        y : pd.DataFrame or pd.Series
            Time series data with DatetimeIndex
            
        Returns
        -------
        pd.DataFrame
            Calendar features
        """
        if not self.is_fitted_:
            raise ValueError("CalendarFeatures must be fitted before transform")
        
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Input must have a DatetimeIndex")
        
        return self._extract_features(y.index)
    
    def fit_transform(self, y: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
    
    def transform_future(
        self,
        horizon: int,
        start: Optional[pd.Timestamp] = None,
        freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate calendar features for future dates.
        
        Parameters
        ----------
        horizon : int
            Number of future periods
        start : pd.Timestamp, optional
            Start date for future index. If None, uses last fitted index + 1 period.
        freq : str, optional
            Frequency for future index. If None, uses detected frequency.
            
        Returns
        -------
        pd.DataFrame
            Calendar features for future dates
        """
        if not self.is_fitted_:
            raise ValueError("CalendarFeatures must be fitted before generating future features")
        
        freq = freq or self._freq_
        if start is None:
            start = self._last_index_ + pd.tseries.frequencies.to_offset(freq)
        
        future_index = pd.date_range(start=start, periods=horizon, freq=freq)
        return self._extract_features(future_index)
    
    def get_feature_names(self) -> List[str]:
        """Get names of output features."""
        return self.feature_names_out_


def add_calendar_features(
    y: Union[pd.DataFrame, pd.Series],
    features: Optional[List[str]] = None,
    country: Optional[str] = None,
    fourier_terms: Optional[Dict[str, tuple]] = None
) -> pd.DataFrame:
    """
    Convenience function to add calendar features to time series data.
    
    Parameters
    ----------
    y : pd.DataFrame or pd.Series
        Time series data with DatetimeIndex
    features : list of str, optional
        Features to extract (see CalendarFeatures for options)
    country : str, optional
        Country code for holiday features
    fourier_terms : dict, optional
        Fourier terms for seasonality
        
    Returns
    -------
    pd.DataFrame
        Original data with calendar features appended
        
    Examples
    --------
    >>> from autotsforecast.features.calendar import add_calendar_features
    >>> y_with_features = add_calendar_features(y, country='US')
    """
    cal = CalendarFeatures(
        features=features,
        country=country,
        fourier_terms=fourier_terms
    )
    cal_features = cal.fit_transform(y)
    
    if isinstance(y, pd.Series):
        y = y.to_frame()
    
    return pd.concat([y, cal_features], axis=1)

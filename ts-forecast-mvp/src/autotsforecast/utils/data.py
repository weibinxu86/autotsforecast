import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def preprocess_data(data: pd.DataFrame, 
                   handle_missing: str = 'forward_fill',
                   handle_outliers: bool = False,
                   outlier_std: float = 3.0) -> pd.DataFrame:
    """Preprocess time series data
    
    Args:
        data: Input DataFrame
        handle_missing: Method to handle missing values ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        handle_outliers: Whether to cap outliers
        outlier_std: Number of standard deviations for outlier detection
        
    Returns:
        Preprocessed DataFrame
    """
    data = data.copy()
    
    # Handle missing values
    if handle_missing == 'forward_fill':
        data = data.fillna(method='ffill')
    elif handle_missing == 'backward_fill':
        data = data.fillna(method='bfill')
    elif handle_missing == 'interpolate':
        data = data.interpolate(method='linear')
    elif handle_missing == 'drop':
        data = data.dropna()
    
    # Handle outliers
    if handle_outliers:
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - outlier_std * std
            upper_bound = mean + outlier_std * std
            data[col] = data[col].clip(lower_bound, upper_bound)
    
    return data


def split_data(data: pd.DataFrame, 
              train_size: float = 0.8, 
              validation_size: float = 0.0) -> Union[Tuple[pd.DataFrame, pd.DataFrame], 
                                                     Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Split time series data into train/validation/test sets
    
    Args:
        data: Input DataFrame
        train_size: Fraction of data for training
        validation_size: Fraction of data for validation (0 means no validation set)
        
    Returns:
        Tuple of (train, test) or (train, validation, test) DataFrames
    """
    n = len(data)
    train_end = int(n * train_size)
    
    if validation_size > 0:
        val_end = int(n * (train_size + validation_size))
        train = data.iloc[:train_end]
        validation = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]
        return train, validation, test
    else:
        train = data.iloc[:train_end]
        test = data.iloc[train_end:]
        return train, test


def create_time_series_features(data: pd.DataFrame, 
                                time_column: Optional[str] = None,
                                lag_features: Optional[List[int]] = None,
                                rolling_windows: Optional[List[int]] = None,
                                add_date_features: bool = True) -> pd.DataFrame:
    """Create additional features for time series forecasting
    
    Args:
        data: Input DataFrame
        time_column: Name of the time column (if None, uses index)
        lag_features: List of lag periods to create
        rolling_windows: List of window sizes for rolling statistics
        add_date_features: Whether to add date-based features
        
    Returns:
        DataFrame with additional features
    """
    df = data.copy()
    
    # Ensure datetime index
    if time_column is not None:
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Add date features
    if add_date_features:
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Add lag features
    if lag_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            for lag in lag_features:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Add rolling statistics
    if rolling_windows:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            for window in rolling_windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    
    return df


def handle_categorical_covariates(data: pd.DataFrame, 
                                  categorical_columns: List[str],
                                  method: str = 'onehot') -> pd.DataFrame:
    """Convert categorical covariates to numerical format
    
    Args:
        data: Input DataFrame
        categorical_columns: List of categorical column names
        method: Encoding method ('onehot', 'label', 'target')
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df = data.copy()
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
        
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
        elif method == 'target':
            # Target encoding would require target variable
            raise NotImplementedError("Target encoding requires target variable")
    
    return df


def handle_numerical_covariates(data: pd.DataFrame, 
                                numerical_columns: List[str],
                                method: str = 'standard',
                                feature_range: Tuple[float, float] = (0, 1)) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """Scale or normalize numerical covariates
    
    Args:
        data: Input DataFrame
        numerical_columns: List of numerical column names
        method: Scaling method ('standard', 'minmax', 'robust', 'none')
        feature_range: Range for MinMax scaling
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df = data.copy()
    
    if method == 'none':
        return df, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Scale only specified numerical columns
    valid_cols = [col for col in numerical_columns if col in df.columns]
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    
    return df, scaler


def create_sequences(data: pd.DataFrame, 
                    sequence_length: int,
                    target_columns: List[str],
                    feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for deep learning models
    
    Args:
        data: Input DataFrame
        sequence_length: Length of input sequences
        target_columns: Columns to use as targets
        feature_columns: Columns to use as features (if None, uses all)
        
    Returns:
        Tuple of (X sequences, y sequences)
    """
    if feature_columns is None:
        feature_columns = data.columns.tolist()
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[feature_columns].iloc[i:i+sequence_length].values)
        y.append(data[target_columns].iloc[i+sequence_length].values)
    
    return np.array(X), np.array(y)


def detect_seasonality(data: pd.Series, max_lag: int = 50) -> List[int]:
    """Detect seasonal periods using autocorrelation
    
    Args:
        data: Time series data
        max_lag: Maximum lag to check
        
    Returns:
        List of detected seasonal periods
    """
    from statsmodels.tsa.stattools import acf
    
    autocorr = acf(data, nlags=max_lag, fft=True)
    
    # Find peaks in autocorrelation
    seasonal_periods = []
    for i in range(2, len(autocorr)):
        if autocorr[i] > 0.6:  # Threshold for significant autocorrelation
            if i-1 not in seasonal_periods and i+1 not in seasonal_periods:
                seasonal_periods.append(i)
    
    return seasonal_periods
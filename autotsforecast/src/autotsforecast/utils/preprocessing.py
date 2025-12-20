"""
Preprocessing utilities for handling categorical and numerical covariates.

This module provides automatic handling of categorical and numerical features
for time series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class CovariatePreprocessor:
    """
    Automatic preprocessing for categorical and numerical covariates.
    
    Features:
    - Detects categorical vs numerical columns automatically
    - One-hot encoding for categorical variables
    - Optional scaling for numerical variables
    - Handles missing values
    - Preserves column names for interpretability
    
    Parameters
    ----------
    categorical_features : list, optional
        List of categorical feature names. If None, will auto-detect.
    numerical_features : list, optional
        List of numerical feature names. If None, will auto-detect.
    encoding : str, default='onehot'
        Encoding method for categorical variables: 'onehot' or 'label'
    scale_numerical : bool, default=False
        Whether to standardize numerical features
    handle_missing : str, default='forward_fill'
        How to handle missing values: 'forward_fill', 'backward_fill', 'mean', 'drop'
    max_categories : int, default=50
        Maximum number of categories for one-hot encoding (use label encoding beyond this)
    """
    
    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        encoding: str = 'onehot',
        scale_numerical: bool = False,
        handle_missing: str = 'forward_fill',
        max_categories: int = 50
    ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoding = encoding
        self.scale_numerical = scale_numerical
        self.handle_missing = handle_missing
        self.max_categories = max_categories
        
        # State variables
        self.categorical_encoders_ = {}
        self.numerical_scaler_ = None
        self.feature_names_out_ = []
        self.is_fitted_ = False
        self._auto_detected_categorical = []
        self._auto_detected_numerical = []
        
    def fit(self, X: pd.DataFrame) -> 'CovariatePreprocessor':
        """
        Fit the preprocessor on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Covariate dataframe
            
        Returns
        -------
        self
        """
        if X is None or X.empty:
            self.is_fitted_ = True
            return self
        
        X = X.copy()
        
        # Auto-detect feature types if not specified
        if self.categorical_features is None or self.numerical_features is None:
            self._auto_detect_feature_types(X)
        
        # Fit categorical encoders
        for col in self._auto_detected_categorical:
            n_unique = X[col].nunique()
            
            if self.encoding == 'onehot' and n_unique <= self.max_categories:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(X[[col]])
                self.categorical_encoders_[col] = {
                    'type': 'onehot',
                    'encoder': encoder,
                    'feature_names': [f"{col}_{cat}" for cat in encoder.categories_[0]]
                }
            else:
                # Label encoding for high-cardinality features
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
                self.categorical_encoders_[col] = {
                    'type': 'label',
                    'encoder': encoder,
                    'feature_names': [col]
                }
        
        # Fit numerical scaler if requested
        if self.scale_numerical and len(self._auto_detected_numerical) > 0:
            self.numerical_scaler_ = StandardScaler()
            self.numerical_scaler_.fit(X[self._auto_detected_numerical])
        
        # Build feature names
        self._build_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform covariates using fitted preprocessor.
        
        Parameters
        ----------
        X : pd.DataFrame
            Covariate dataframe to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed covariates
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if X is None or X.empty:
            return X
        
        X = X.copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        transformed_parts = []
        
        # Transform categorical features
        for col in self._auto_detected_categorical:
            if col not in X.columns:
                continue
                
            encoder_info = self.categorical_encoders_[col]
            
            if encoder_info['type'] == 'onehot':
                encoded = encoder_info['encoder'].transform(X[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder_info['feature_names'],
                    index=X.index
                )
                transformed_parts.append(encoded_df)
            else:  # label encoding
                # Handle unseen categories
                X_col = X[col].astype(str)
                encoded = np.zeros(len(X_col))
                for i, val in enumerate(X_col):
                    if val in encoder_info['encoder'].classes_:
                        encoded[i] = encoder_info['encoder'].transform([val])[0]
                    else:
                        encoded[i] = -1  # Unknown category
                
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[col],
                    index=X.index
                )
                transformed_parts.append(encoded_df)
        
        # Transform numerical features
        if len(self._auto_detected_numerical) > 0:
            numerical_cols = [col for col in self._auto_detected_numerical if col in X.columns]
            if numerical_cols:
                if self.scale_numerical and self.numerical_scaler_ is not None:
                    scaled = self.numerical_scaler_.transform(X[numerical_cols])
                    numerical_df = pd.DataFrame(
                        scaled,
                        columns=numerical_cols,
                        index=X.index
                    )
                else:
                    numerical_df = X[numerical_cols]
                transformed_parts.append(numerical_df)
        
        # Combine all parts
        if transformed_parts:
            result = pd.concat(transformed_parts, axis=1)
            return result
        else:
            return X
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def _auto_detect_feature_types(self, X: pd.DataFrame):
        """Auto-detect categorical and numerical features."""
        categorical = []
        numerical = []
        
        for col in X.columns:
            # Check if explicitly specified
            if self.categorical_features and col in self.categorical_features:
                categorical.append(col)
            elif self.numerical_features and col in self.numerical_features:
                numerical.append(col)
            else:
                # Auto-detect based on dtype and unique values
                dtype = X[col].dtype
                n_unique = X[col].nunique()
                n_total = len(X[col])
                
                # Heuristics for categorical detection
                is_categorical = (
                    dtype == 'object' or
                    dtype == 'category' or
                    dtype == 'bool' or
                    (dtype in ['int64', 'int32'] and n_unique < min(20, n_total * 0.05))
                )
                
                if is_categorical:
                    categorical.append(col)
                else:
                    numerical.append(col)
        
        self._auto_detected_categorical = categorical
        self._auto_detected_numerical = numerical
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to strategy."""
        if self.handle_missing == 'forward_fill':
            return X.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'backward_fill':
            return X.fillna(method='bfill').fillna(method='ffill')
        elif self.handle_missing == 'mean':
            return X.fillna(X.mean())
        elif self.handle_missing == 'drop':
            return X.dropna()
        else:
            return X
    
    def _build_feature_names(self):
        """Build output feature names."""
        names = []
        
        # Categorical feature names
        for col in self._auto_detected_categorical:
            names.extend(self.categorical_encoders_[col]['feature_names'])
        
        # Numerical feature names
        names.extend(self._auto_detected_numerical)
        
        self.feature_names_out_ = names
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names_out_


def auto_preprocess_covariates(
    X_train: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame] = None,
    **preprocessor_kwargs
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], CovariatePreprocessor]:
    """
    Convenience function to automatically preprocess train and test covariates.
    
    Parameters
    ----------
    X_train : pd.DataFrame, optional
        Training covariates
    X_test : pd.DataFrame, optional
        Test covariates
    **preprocessor_kwargs
        Keyword arguments for CovariatePreprocessor
        
    Returns
    -------
    X_train_processed : pd.DataFrame or None
    X_test_processed : pd.DataFrame or None
    preprocessor : CovariatePreprocessor
    """
    if X_train is None:
        return None, X_test, None
    
    preprocessor = CovariatePreprocessor(**preprocessor_kwargs)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test) if X_test is not None else None
    
    return X_train_processed, X_test_processed, preprocessor

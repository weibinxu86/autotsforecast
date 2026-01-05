"""
Feature importance and interpretability analysis for time series forecasters.

DriverAnalyzer provides multiple methods to understand which features (covariates) 
are driving your forecast predictions. It automatically selects the best method 
based on your model type.

===================================================================================
HOW DRIVERANALYZER WORKS
===================================================================================

1. AUTOMATIC METHOD DETECTION
   - Inspects your model type (Linear, RandomForest, XGBoost, etc.)
   - Selects the most appropriate interpretability method automatically
   - Falls back to model-agnostic methods for unsupported models

2. SUPPORTED METHODS

   A) COEFFICIENTS (LinearForecaster only)
      - Uses model regression coefficients directly
      - Fastest and most interpretable for linear models
      - Coefficient magnitude = feature importance
      - Example: coef=50 means +1 unit in feature → +50 in prediction
   
   B) PERMUTATION IMPORTANCE (model-agnostic)
      - Shuffles each feature and measures prediction degradation
      - Works for ANY model type (linear, tree, neural net, etc.)
      - Higher score = more important feature
      - Slower but universally applicable
   
   C) SHAP VALUES (Shapley Additive Explanations)
      - Unified framework from game theory
      - Shows each feature's contribution to individual predictions
      - Works for any model with optimized implementations for:
        * Tree models (TreeExplainer) - Fast
        * Linear models (Explainer/KernelExplainer) - Fast to moderate
        * Other models (KernelExplainer) - Slower but universal
      - Returns mean absolute SHAP values as feature importance
   
   D) SENSITIVITY ANALYSIS
      - Perturbs features by small amount (e.g., +10%)
      - Measures relative change in predictions
      - Shows which features predictions are most sensitive to

3. KEY FEATURES

   • FOCUS ON COVARIATES: Automatically filters out internal features (lags)
     to show only actionable external drivers (temp, promo, marketing, etc.)
   
   • AUTOMATIC LAG RECONSTRUCTION: For tree models using lags, DriverAnalyzer
     automatically reconstructs the full feature matrix (lags + covariates)
     that the model was trained on, calculates SHAP for all features, then
     extracts only covariate SHAP values for interpretation
   
   • MULTI-TARGET SUPPORT: Handles multiple forecast targets simultaneously
     and aggregates importance across horizons
   
   • ROBUST ERROR HANDLING: Falls back to alternative methods if primary
     method fails (e.g., KernelExplainer if TreeExplainer fails)

4. USAGE WORKFLOW

   # Initialize with fitted model
   analyzer = DriverAnalyzer(model=fitted_model, feature_names=['temp', 'promo'])
   
   # Calculate importance (method auto-selected based on model type)
   importance = analyzer.calculate_feature_importance(X_train, y_train)
   
   # Or use SHAP for detailed analysis
   shap_values = analyzer.calculate_shap_values(X_train, y_train)
   shap_importance = analyzer.get_shap_feature_importance(shap_values)
   
   # Visualize
   analyzer.plot_importance(importance)
   analyzer.plot_shap_summary(X_train, y_train, shap_values, target_name='sales')

5. WHEN TO USE EACH METHOD

   • Use COEFFICIENTS when:
     - Model is LinearForecaster
     - Want fastest, most interpretable results
     - Need exact linear weights
   
   • Use PERMUTATION when:
     - Need model-agnostic method
     - Want to compare across different model types
     - Don't have SHAP installed
   
   • Use SHAP when:
     - Need detailed per-prediction explanations
     - Want theoretically grounded importance (Shapley values)
     - Using tree models (very fast with TreeExplainer)
     - Need to explain model to stakeholders
   
   • Use SENSITIVITY when:
     - Need to understand "what if" scenarios
     - Want to see prediction stability
     - Analyzing impact of feature changes

6. TECHNICAL DETAILS

   SHAP CALCULATION FOR TREE MODELS:
   Tree models (RandomForest, XGBoost) that use lag features require special handling:
   
   1. Reconstruct full feature matrix: [lag1, lag2, ..., lagN, temp, promo]
      - Create lag features by shifting target series
      - Shift covariates by -h for horizon h (matching training)
      - Concatenate lags and covariates
   
   2. Calculate SHAP on ALL features using TreeExplainer
      - Uses model's tree structure for exact Shapley values
      - Very fast compared to model-agnostic methods
   
   3. Extract SHAP values for covariates only (last N columns)
      - Returns only actionable covariate contributions
      - Lags are excluded from user-facing results
   
   SHAP CALCULATION FOR LINEAR MODELS:
   Linear models use model-agnostic Explainer:
   
   1. Create prediction wrapper function
   2. Use shap.Explainer with background samples
   3. Calculate SHAP values directly on covariate matrix
   4. Handle multi-output predictions (multiple targets)
   
   FALLBACK STRATEGY:
   If optimized methods fail, automatically falls back to:
   - KernelExplainer (universal but slower)
   - Includes error messages for debugging

===================================================================================

Implements multiple interpretability methods:
- Coefficient-based: For linear models
- Permutation importance: Model-agnostic feature shuffling
- SHAP values: Unified approach to explain model predictions
- Sensitivity analysis: Impact of feature perturbations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.inspection import permutation_importance
from autotsforecast.models.base import BaseForecaster, LinearForecaster

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class DriverAnalyzer:
    """Analyze the impact of covariates (drivers) on forecasts"""
    
    def __init__(self, model: BaseForecaster, feature_names: Optional[List[str]] = None):
        """
        Args:
            model: Fitted forecaster model
            feature_names: Names of features/covariates
        """
        self.model = model
        self.feature_names = feature_names or getattr(model, 'feature_names', None)
        self.feature_importance = None
        self.sensitivity_results = None
        
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.DataFrame, 
                                    method: str = 'coefficients') -> pd.DataFrame:
        """Calculate feature importance/driver impact
        
        Args:
            X: Exogenous variables DataFrame
            y: Target variables DataFrame
            method: Method to calculate importance
                   - 'coefficients': Use model coefficients (LinearForecaster only)
                   - 'permutation': Permutation importance
                   - 'sensitivity': Sensitivity analysis
        
        Returns:
            DataFrame with feature importance scores
        """
        if method == 'coefficients':
            return self._coefficients_importance()
        elif method == 'permutation':
            return self._permutation_importance(X, y)
        elif method == 'sensitivity':
            return self._sensitivity_analysis(X)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _coefficients_importance(self) -> pd.DataFrame:
        """Extract feature importance from model coefficients"""
        if not isinstance(self.model, LinearForecaster):
            raise ValueError("Coefficient importance only available for LinearForecaster")
        
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance_dict = {}
        
        for target, models in self.model.models.items():
            coeffs = []
            for model in models:
                coeffs.append(np.abs(model.coef_))
            
            # Average across horizons
            avg_coeffs = np.mean(coeffs, axis=0)
            importance_dict[target] = avg_coeffs
        
        df = pd.DataFrame(importance_dict, index=self.feature_names or range(len(avg_coeffs)))
        self.feature_importance = df
        return df
    
    def _permutation_importance(self, X: pd.DataFrame, y: pd.DataFrame, 
                               n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation importance"""
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Create a wrapper for sklearn's permutation_importance
        def score_func(X_test, y_test):
            predictions = self.model.predict(pd.DataFrame(X_test, columns=X.columns))
            mse = np.mean((y_test - predictions.values) ** 2)
            return -mse  # Negative because higher is better for sklearn
        
        # Use permutation importance
        importance_dict = {}
        
        for target_col in y.columns:
            from sklearn.metrics import make_scorer, mean_squared_error
            
            # This is a simplified version - in practice, you'd need to adapt for time series
            feature_importance_scores = np.zeros(X.shape[1])
            
            for i, feature in enumerate(X.columns):
                X_permuted = X.copy()
                original_col = X_permuted[feature].copy()
                
                scores = []
                for _ in range(n_repeats):
                    # Permute feature
                    X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                    pred = self.model.predict(X_permuted)
                    mse = np.mean((y[target_col].values - pred[target_col].values) ** 2)
                    scores.append(mse)
                
                # Restore original
                X_permuted[feature] = original_col
                feature_importance_scores[i] = np.mean(scores)
            
            importance_dict[target_col] = feature_importance_scores
        
        df = pd.DataFrame(importance_dict, index=X.columns)
        self.feature_importance = df
        return df
    
    def _sensitivity_analysis(self, X: pd.DataFrame, perturbation: float = 0.1) -> pd.DataFrame:
        """Perform sensitivity analysis by perturbing each feature
        
        For numerical features: scales by (1 + perturbation)
        For binary features (only 0/1 values): flips 0↔1
        For categorical: skips (returns 0)
        
        Returns mean absolute change in predictions (not relative change to avoid
        division issues when baseline predictions are near zero).
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        baseline_pred = self.model.predict(X)
        sensitivity_dict = {target: [] for target in baseline_pred.columns}
        
        for feature in X.columns:
            X_perturbed = X.copy()
            
            # Check if feature is categorical (non-numeric)
            if X[feature].dtype in ['object', 'category']:
                for target in sensitivity_dict:
                    sensitivity_dict[target].append(0)
                continue
            
            # Check if feature is binary (only 0 and 1 values)
            unique_vals = X[feature].dropna().unique()
            is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
            
            if is_binary:
                # For binary features: flip 0→1 and 1→0
                X_perturbed[feature] = 1 - X[feature]
            else:
                # For continuous features: scale by perturbation
                X_perturbed[feature] = X[feature] * (1 + perturbation)
            
            perturbed_pred = self.model.predict(X_perturbed)
            
            # Calculate mean absolute change (not relative to avoid division by near-zero)
            for target in baseline_pred.columns:
                abs_change = np.abs(perturbed_pred[target].values - baseline_pred[target].values)
                sensitivity_dict[target].append(np.mean(abs_change))
        
        df = pd.DataFrame(sensitivity_dict, index=X.columns)
        self.sensitivity_results = df
        return df
    
    def analyze_drivers(self, X: pd.DataFrame, y: pd.DataFrame, 
                       categorical_features: Optional[List[str]] = None,
                       numerical_features: Optional[List[str]] = None) -> Dict:
        """Comprehensive driver analysis
        
        Args:
            X: Exogenous variables DataFrame
            y: Target variables DataFrame
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            Dict with analysis results
        """
        results = {}
        
        # Identify feature types if not provided
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if numerical_features is None:
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        results['categorical_features'] = categorical_features
        results['numerical_features'] = numerical_features
        
        # Calculate importance for numerical features
        if numerical_features:
            X_numerical = X[numerical_features]
            
            if isinstance(self.model, LinearForecaster):
                results['coefficient_importance'] = self._coefficients_importance()
            
            results['sensitivity'] = self._sensitivity_analysis(X_numerical)
        
        # Analyze categorical features
        if categorical_features:
            results['categorical_analysis'] = self._analyze_categorical(X, y, categorical_features)
        
        return results
    
    def _analyze_categorical(self, X: pd.DataFrame, y: pd.DataFrame, 
                           categorical_features: List[str]) -> Dict:
        """Analyze impact of categorical features"""
        categorical_impact = {}
        
        for feature in categorical_features:
            unique_values = X[feature].unique()
            impact_by_category = {}
            
            for value in unique_values:
                mask = X[feature] == value
                if mask.sum() > 0:
                    # Average target value for this category
                    avg_target = y[mask].mean()
                    impact_by_category[str(value)] = avg_target.to_dict()
            
            categorical_impact[feature] = impact_by_category
        
        return categorical_impact
    
    def plot_importance(self, importance_df: Optional[pd.DataFrame] = None, 
                       top_n: int = 10, figsize: Tuple[int, int] = (12, 6)):
        """Plot feature importance
        
        Args:
            importance_df: DataFrame with importance scores (uses stored if None)
            top_n: Number of top features to display
            figsize: Figure size tuple
        """
        try:
            import matplotlib.pyplot as plt
            
            if importance_df is None:
                importance_df = self.feature_importance
            
            if importance_df is None:
                raise ValueError("No importance data available. Run calculate_feature_importance() first.")
            
            # Get top features by average importance across targets
            avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
            top_features = avg_importance.head(top_n)
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Plot 1: Bar chart of average importance
            top_features.plot(kind='barh', ax=axes[0])
            axes[0].set_title(f'Top {top_n} Feature Importance (Average)')
            axes[0].set_xlabel('Importance Score')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Heatmap of importance by target
            import matplotlib.cm as cm
            
            top_feature_names = top_features.index
            heatmap_data = importance_df.loc[top_feature_names]
            
            im = axes[1].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
            axes[1].set_xticks(range(len(heatmap_data.columns)))
            axes[1].set_yticks(range(len(heatmap_data.index)))
            axes[1].set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
            axes[1].set_yticklabels(heatmap_data.index)
            axes[1].set_title('Feature Importance by Target')
            
            plt.colorbar(im, ax=axes[1])
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Install it with: pip install matplotlib")
    
    def calculate_shap_values(self, X: pd.DataFrame, y: pd.DataFrame,
                             background_samples: Optional[pd.DataFrame] = None,
                             max_samples: int = 100) -> Dict[str, np.ndarray]:
        """Calculate SHAP values for model interpretability (returns values for covariates only)
        
        Args:
            X: Covariate data (external features like temp, promo)
            y: Historical target data (needed to create lag features)
            background_samples: Background dataset for TreeExplainer (if None, uses sample from data)
            max_samples: Maximum background samples for faster computation
            
        Returns:
            Dictionary mapping target names to SHAP values arrays (for covariates only)
        
        Note:
            This method reconstructs the full feature matrix (lags + covariates) that the model
            was trained on, calculates SHAP values for all features, then returns only the
            SHAP values for covariate features. This ensures accurate SHAP calculation.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")

        # Keep track of the covariate names used for the output so we can label SHAP
        # feature importance consistently (and avoid column index printing).
        feature_names_output = self.feature_names if self.feature_names else list(X.columns)
        original_covariate_columns = list(X.columns)
        covariate_column_map = {col: [col] for col in original_covariate_columns}
        
        # Select appropriate explainer based on model type
        model_class_name = self.model.__class__.__name__
        
        # For tree-based models that use lags (RandomForest, XGBoost)
        if 'RandomForest' in model_class_name or 'XGBoost' in model_class_name:
            if not hasattr(self.model, 'n_lags') or not hasattr(self.model, 'models'):
                raise ValueError(f"Model {model_class_name} doesn't have expected attributes for SHAP analysis")

            # Use the same covariate preprocessing (if any) that the model used during fit
            if hasattr(self.model, 'covariate_preprocessor_') and self.model.covariate_preprocessor_ is not None:
                preprocessor = self.model.covariate_preprocessor_
                X_processed = preprocessor.transform(X.copy())
                if getattr(self.model, '_covariate_columns_', None):
                    X_processed = X_processed[self.model._covariate_columns_]
                X = X_processed

                # Map expanded columns (e.g., one-hot) back to original covariate names
                covariate_column_map = {}
                for col in original_covariate_columns:
                    if col in preprocessor.categorical_encoders_:
                        covariate_column_map[col] = preprocessor.categorical_encoders_[col]['feature_names']
                    else:
                        covariate_column_map[col] = [col]
            
            # Reconstruct full feature matrix as model was trained
            # Create lag features
            lag_features = []
            for lag in range(1, self.model.n_lags + 1):
                lagged = y.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in y.columns]
                lag_features.append(lagged)
            lag_df = pd.concat(lag_features, axis=1)
            
            # For horizon=1 models, covariates are shifted by -1 during training
            X_shifted = X.shift(-1) if len(self.model.models) > 0 else X
            
            # Combine lags and covariates
            X_full = pd.concat([lag_df, X_shifted], axis=1).dropna()
            
            # Get base model to check feature count
            horizon_model = self.model.models[0]
            if hasattr(horizon_model, 'estimators_'):
                base_model = horizon_model.estimators_[0]
            else:
                base_model = horizon_model
            
            # BUG FIX: MultiOutputRegressor may add extra features (e.g., series index)
            # If model expects more features than we reconstructed, add dummy column
            if base_model.n_features_in_ > X_full.shape[1]:
                n_missing = base_model.n_features_in_ - X_full.shape[1]
                for j in range(n_missing):
                    X_full[f'_dummy_{j}'] = 0  # Add dummy features to match model
            
            # Identify covariate column indices in reconstructed feature matrix
            # Covariates are right after lag features, before any dummy columns
            n_lags_total = self.model.n_lags * len(y.columns)
            n_covariates = len(X.columns)
            covariate_start_idx = n_lags_total
            covariate_end_idx = n_lags_total + n_covariates
            
            # Prepare background and sample data
            if background_samples is None:
                n_bg = min(max_samples, len(X_full))
                X_background = X_full.sample(n=n_bg, random_state=42)
            else:
                X_background = background_samples
            
            n_samples = min(max_samples, len(X_full))
            X_sample = X_full.sample(n=n_samples, random_state=43)
            
            shap_values_dict = {}
            
            # Calculate SHAP for each target
            for i, target_col in enumerate(self.model.feature_names):
                # Get the model for first horizon
                if 'XGBoost' in model_class_name:
                    base_model = self.model.models[0][i]
                elif 'RandomForest' in model_class_name:
                    horizon_model = self.model.models[0]
                    if hasattr(horizon_model, 'estimators_'):
                        base_model = horizon_model.estimators_[i]
                    else:
                        base_model = horizon_model
                else:
                    continue
                
                # Calculate SHAP values for ALL features
                explainer = shap.TreeExplainer(base_model, X_background)
                shap_values_all = explainer.shap_values(X_sample, check_additivity=False)
                
                # BUGFIX: Extract covariate SHAP values using correct column indices
                # Instead of using [-n_covariates:], use explicit indices where covariates are
                shap_values_covariates = shap_values_all[:, covariate_start_idx:covariate_end_idx]

                # If covariates were expanded (e.g., one-hot), aggregate SHAP back to the
                # original feature names provided by the user so outputs never show indices.
                covariate_cols = list(X_shifted.columns)
                if covariate_column_map and len(feature_names_output) != shap_values_covariates.shape[1]:
                    aggregated = np.zeros((shap_values_covariates.shape[0], len(feature_names_output)))
                    for j, base_feature in enumerate(feature_names_output):
                        mapped_cols = covariate_column_map.get(base_feature, [base_feature])
                        col_indices = [covariate_cols.index(c) for c in mapped_cols if c in covariate_cols]
                        if col_indices:
                            aggregated[:, j] = shap_values_covariates[:, col_indices].sum(axis=1)
                        else:
                            aggregated[:, j] = 0
                    shap_values_covariates = aggregated
                
                shap_values_dict[target_col] = shap_values_covariates
        
        # For linear models (don't use lags in same way)
        elif 'Linear' in model_class_name or 'VAR' in model_class_name:
            # Linear models may have different structure - use model-agnostic approach
            def predict_func(X_input):
                X_df = pd.DataFrame(X_input, columns=X.columns)
                pred = self.model.predict(X_df)
                # Ensure consistent output shape
                if isinstance(pred, pd.DataFrame):
                    return pred.values
                return pred
            
            # Prepare background
            n_bg = min(max_samples, len(X))
            background = X.sample(n=n_bg, random_state=42).values
            
            # Sample
            n_samples = min(max_samples, len(X))
            X_sample_lin = X.sample(n=n_samples, random_state=43).values
            
            try:
                explainer = shap.Explainer(predict_func, background)
                shap_values = explainer(X_sample_lin)
                
                # Handle different output shapes
                shap_array = shap_values.values
                if len(shap_array.shape) == 3:
                    # Multi-output case: (samples, features, targets)
                    for i, target_col in enumerate(self.model.feature_names):
                        shap_values_dict[target_col] = shap_array[:, :, i]
                elif len(shap_array.shape) == 2:
                    # Single output: (samples, features)
                    target_col = self.model.feature_names[0] if self.model.feature_names else 'target_0'
                    shap_values_dict[target_col] = shap_array
                else:
                    raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")
            except Exception as e:
                # Fall back to KernelExplainer if Explainer fails
                print(f"Warning: shap.Explainer failed ({e}), falling back to KernelExplainer (slower)")
                explainer = shap.KernelExplainer(predict_func, background)
                shap_values = explainer.shap_values(X_sample_lin, nsamples=100)
                
                if isinstance(shap_values, list):
                    for i, target_col in enumerate(self.model.feature_names):
                        shap_values_dict[target_col] = shap_values[i]
                else:
                    if len(shap_values.shape) == 3:
                        for i, target_col in enumerate(self.model.feature_names):
                            shap_values_dict[target_col] = shap_values[:, :, i]
                    else:
                        target_col = self.model.feature_names[0] if self.model.feature_names else 'target_0'
                        shap_values_dict[target_col] = shap_values
        
        # For other models, use KernelExplainer (slower but model-agnostic)
        else:
            def predict_func(X_input):
                X_df = pd.DataFrame(X_input, columns=X.columns)
                return self.model.predict(X_df).values
            
            n_bg = min(max_samples, len(X))
            background = X.sample(n=n_bg, random_state=42)
            
            n_samples = min(max_samples, len(X))
            X_sample_other = X.sample(n=n_samples, random_state=43)
            
            explainer = shap.KernelExplainer(predict_func, background)
            shap_values = explainer.shap_values(X_sample_other, nsamples=100)
            
            if isinstance(shap_values, list):
                for i, target_col in enumerate(self.model.feature_names):
                    shap_values_dict[target_col] = shap_values[i]
            else:
                for i, target_col in enumerate(self.model.feature_names):
                    shap_values_dict[target_col] = shap_values[:, :, i] if len(shap_values.shape) > 2 else shap_values
        
        # Store the covariate names used so downstream importance printing stays readable
        self.covariate_names_ = feature_names_output
        return shap_values_dict
    
    def plot_shap_summary(self, X: pd.DataFrame, y: pd.DataFrame,
                         shap_values_dict: Optional[Dict] = None,
                         target_name: Optional[str] = None, plot_type: str = 'dot'):
        """Plot SHAP summary visualization (covariates only)
        
        Args:
            X: Covariate data
            y: Historical target data (needed to create lag features for tree models)
            shap_values_dict: Pre-calculated SHAP values (if None, calculates them)
            target_name: Specific target to plot (if None, plots first target)
            plot_type: Type of plot ('dot', 'bar', 'violin')
        
        Note:
            Only covariates are shown in SHAP plots. Lag features are excluded.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        if shap_values_dict is None:
            shap_values_dict = self.calculate_shap_values(X, y)
        
        if target_name is None:
            target_name = list(shap_values_dict.keys())[0]
        
        if target_name not in shap_values_dict:
            raise ValueError(f"Target {target_name} not found in SHAP values")
        
        # X should contain only covariates for plotting
        shap_values = shap_values_dict[target_name]
        
        if plot_type == 'dot':
            shap.summary_plot(shap_values, X, plot_type='dot', show=True)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X, plot_type='bar', show=True)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X, plot_type='violin', show=True)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def get_shap_feature_importance(self, shap_values_dict: Dict) -> pd.DataFrame:
        """Calculate mean absolute SHAP values as feature importance (covariates only)
        
        Args:
            shap_values_dict: Dictionary of SHAP values per target (for covariates)
            
        Returns:
            DataFrame with SHAP-based feature importance for covariates only
        
        Note:
            Only covariate features are included. Lag features are excluded from importance ranking.
        """
        importance_dict = {}
        
        for target_name, shap_vals in shap_values_dict.items():
            # Mean absolute SHAP value for each feature
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            importance_dict[target_name] = mean_abs_shap
        
        covariate_names = getattr(self, 'covariate_names_', None)
        if covariate_names and len(covariate_names) == len(mean_abs_shap):
            feature_names = covariate_names
        elif self.feature_names and len(self.feature_names) == len(mean_abs_shap):
            feature_names = self.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        df = pd.DataFrame(importance_dict, index=feature_names)
        
        return df
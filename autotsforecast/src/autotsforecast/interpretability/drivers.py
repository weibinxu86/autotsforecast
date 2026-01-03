"""
Feature importance and interpretability analysis for time series forecasters.

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
        """Perform sensitivity analysis by perturbing each feature"""
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        baseline_pred = self.model.predict(X)
        sensitivity_dict = {target: [] for target in baseline_pred.columns}
        
        for feature in X.columns:
            X_perturbed = X.copy()
            
            # Perturb feature by perturbation %
            if X[feature].dtype in ['object', 'category']:
                # For categorical, we skip or handle differently
                for target in sensitivity_dict:
                    sensitivity_dict[target].append(0)
                continue
            
            X_perturbed[feature] = X[feature] * (1 + perturbation)
            perturbed_pred = self.model.predict(X_perturbed)
            
            # Calculate relative change in predictions
            for target in baseline_pred.columns:
                rel_change = np.abs((perturbed_pred[target].values - baseline_pred[target].values) / 
                                   (baseline_pred[target].values + 1e-10))
                sensitivity_dict[target].append(np.mean(rel_change))
        
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
    
    def calculate_shap_values(self, X: pd.DataFrame, background_samples: Optional[pd.DataFrame] = None,
                             max_samples: int = 100) -> Dict[str, np.ndarray]:
        """Calculate SHAP values for model interpretability (covariates only, excludes lag features)
        
        Args:
            X: Covariate data to explain (should not include lag features)
            background_samples: Background dataset for TreeExplainer (if None, uses X sample)
            max_samples: Maximum background samples for faster computation
            
        Returns:
            Dictionary mapping target names to SHAP values arrays for covariates only
        
        Note:
            This method only calculates SHAP values for external covariates.
            Lag features are excluded from the interpretability analysis.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Select appropriate explainer based on model type
        model_class_name = self.model.__class__.__name__
        
        # Filter out lag features - only keep covariates for SHAP analysis
        # Lag features typically have patterns like: 'target_lag1', 'target_lag2', etc.
        covariate_columns = [col for col in X.columns if '_lag' not in col.lower()]
        
        if len(covariate_columns) == 0:
            raise ValueError("No covariate features found. SHAP analysis is only performed on covariates, not lag features.")
        
        X_covariates = X[covariate_columns]
        
        # Prepare background data
        if background_samples is None:
            if len(X_covariates) > max_samples:
                background_samples = X_covariates.sample(n=max_samples, random_state=42)
            else:
                background_samples = X_covariates
        else:
            # Filter background samples to only include covariates
            background_samples = background_samples[[col for col in background_samples.columns if '_lag' not in col.lower()]]
        
        shap_values_dict = {}
        
        # For tree-based models (RandomForest, XGBoost)
        if 'RandomForest' in model_class_name or 'XGBoost' in model_class_name:
            # Get the underlying sklearn models
            if hasattr(self.model, 'models'):
                for i, target_col in enumerate(self.model.feature_names):
                    # Use first horizon model for interpretation
                    if len(self.model.models) > 0:
                        horizon_model = self.model.models[0]
                        
                        # For MultiOutputRegressor, get the underlying estimator
                        if hasattr(horizon_model, 'estimators_'):
                            base_model = horizon_model.estimators_[i]
                        else:
                            base_model = horizon_model
                        
                        # Create wrapper function that only uses covariates
                        def predict_func(X_cov):
                            # Reconstruct full feature matrix with lags
                            # This is a simplified approach - assumes model needs full features
                            # but we only explain covariates
                            return base_model.predict(X_cov)
                        
                        # Create TreeExplainer with covariates only
                        explainer = shap.TreeExplainer(base_model, background_samples)
                        shap_values = explainer.shap_values(X_covariates)
                        shap_values_dict[target_col] = shap_values
        
        # For linear models
        elif 'Linear' in model_class_name or 'VAR' in model_class_name:
            # Use LinearExplainer for linear models - covariates only
            def predict_func(X_input):
                X_df = pd.DataFrame(X_input, columns=covariate_columns)
                return self.model.predict(X_df).values
            
            explainer = shap.Explainer(predict_func, background_samples)
            shap_values = explainer(X_covariates)
            
            for i, target_col in enumerate(self.model.feature_names):
                shap_values_dict[target_col] = shap_values.values[:, :, i] if len(shap_values.values.shape) > 2 else shap_values.values
        
        # For other models, use KernelExplainer (slower but model-agnostic)
        else:
            def predict_func(X_input):
                X_df = pd.DataFrame(X_input, columns=covariate_columns)
                return self.model.predict(X_df).values
            
            explainer = shap.KernelExplainer(predict_func, background_samples)
            shap_values = explainer.shap_values(X_covariates, nsamples=100)
            
            if isinstance(shap_values, list):
                for i, target_col in enumerate(self.model.feature_names):
                    shap_values_dict[target_col] = shap_values[i]
            else:
                for i, target_col in enumerate(self.model.feature_names):
                    shap_values_dict[target_col] = shap_values[:, :, i] if len(shap_values.shape) > 2 else shap_values
        
        return shap_values_dict
    
    def plot_shap_summary(self, X: pd.DataFrame, shap_values_dict: Optional[Dict] = None,
                         target_name: Optional[str] = None, plot_type: str = 'dot'):
        """Plot SHAP summary visualization (covariates only)
        
        Args:
            X: Covariate data (lag features will be automatically filtered out)
            shap_values_dict: Pre-calculated SHAP values (if None, calculates them)
            target_name: Specific target to plot (if None, plots first target)
            plot_type: Type of plot ('dot', 'bar', 'violin')
        
        Note:
            Only covariates are shown in SHAP plots. Lag features are excluded.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        if shap_values_dict is None:
            shap_values_dict = self.calculate_shap_values(X)
        
        if target_name is None:
            target_name = list(shap_values_dict.keys())[0]
        
        if target_name not in shap_values_dict:
            raise ValueError(f"Target {target_name} not found in SHAP values")
        
        # Filter X to only include covariates (exclude lag features)
        covariate_columns = [col for col in X.columns if '_lag' not in col.lower()]
        X_covariates = X[covariate_columns]
        
        shap_values = shap_values_dict[target_name]
        
        if plot_type == 'dot':
            shap.summary_plot(shap_values, X_covariates, plot_type='dot', show=True)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_covariates, plot_type='bar', show=True)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X_covariates, plot_type='violin', show=True)
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
        
        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        df = pd.DataFrame(importance_dict, index=feature_names)
        
        return df
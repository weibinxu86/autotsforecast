import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.inspection import permutation_importance
from ts_forecast.models.base import BaseForecaster, LinearForecaster


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
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from ts_forecast.models.base import BaseForecaster, VARForecaster, LinearForecaster, MovingAverageForecaster


class ModelSelector:
    """Automatic model selection based on performance metrics"""
    
    def __init__(self, models: Optional[List[BaseForecaster]] = None, metric: str = 'rmse'):
        """
        Args:
            models: List of forecaster instances to compare
            metric: Metric to use for selection ('rmse', 'mae', 'mape', 'r2')
        """
        self.models = models or self._get_default_models()
        self.metric = metric
        self.best_model = None
        self.results = {}
        self.cv_results = []
        
    def _get_default_models(self) -> List[BaseForecaster]:
        """Get default set of models to compare"""
        return [
            VARForecaster(horizon=1, lags=1),
            VARForecaster(horizon=1, lags=2),
            VARForecaster(horizon=1, lags=3),
            MovingAverageForecaster(horizon=1, window=3),
            MovingAverageForecaster(horizon=1, window=5),
        ]
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None, 
            validation_split: float = 0.2, cv_folds: int = 3) -> 'ModelSelector':
        """Fit all models and select the best one
        
        Args:
            y: Target time series DataFrame
            X: Optional exogenous variables
            validation_split: Fraction of data to use for validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            self
        """
        
        if cv_folds > 1:
            # Use time series cross-validation
            best_score = float('inf') if self.metric != 'r2' else float('-inf')
            
            for i, model in enumerate(self.models):
                try:
                    scores = self._cross_validate(model, y, X, cv_folds)
                    avg_score = np.mean(scores)
                    
                    model_name = f"{model.__class__.__name__}_lag{getattr(model, 'lags', 'NA')}"
                    self.results[model_name] = {
                        'mean_score': avg_score,
                        'std_score': np.std(scores),
                        'scores': scores
                    }
                    
                    is_better = (avg_score < best_score if self.metric != 'r2' 
                               else avg_score > best_score)
                    
                    if is_better:
                        best_score = avg_score
                        self.best_model = model
                        
                except Exception as e:
                    print(f"Model {model.__class__.__name__} failed: {str(e)}")
                    continue
        else:
            # Simple train/validation split
            split_idx = int(len(y) * (1 - validation_split))
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            X_train = X.iloc[:split_idx] if X is not None else None
            X_val = X.iloc[split_idx:] if X is not None else None
            
            best_score = float('inf') if self.metric != 'r2' else float('-inf')
            
            for i, model in enumerate(self.models):
                try:
                    # Check if model requires exogenous variables
                    if isinstance(model, LinearForecaster) and X is None:
                        continue
                    
                    model.fit(y_train, X_train)
                    predictions = self._generate_validation_predictions(model, y_train, y_val, X_train, X_val)
                    
                    score = self._calculate_metric(y_val, predictions)
                    model_name = f"{model.__class__.__name__}_lag{getattr(model, 'lags', 'NA')}"
                    self.results[model_name] = score
                    
                    is_better = (score < best_score if self.metric != 'r2' 
                               else score > best_score)
                    
                    if is_better:
                        best_score = score
                        self.best_model = model
                        
                except Exception as e:
                    print(f"Model {model.__class__.__name__} failed: {str(e)}")
                    continue
        
        # Refit best model on full data
        if self.best_model is not None:
            self.best_model.fit(y, X)
        else:
            raise ValueError("No model could be fitted successfully")
        
        return self
    
    def _cross_validate(self, model: BaseForecaster, y: pd.DataFrame, 
                       X: Optional[pd.DataFrame], n_folds: int) -> List[float]:
        """Perform time series cross-validation"""
        scores = []
        fold_size = len(y) // (n_folds + 1)
        
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, len(y))
            
            y_train = y.iloc[:train_end]
            y_test = y.iloc[train_end:test_end]
            X_train = X.iloc[:train_end] if X is not None else None
            X_test = X.iloc[train_end:test_end] if X is not None else None
            
            # Check if model requires exogenous variables
            if isinstance(model, LinearForecaster) and X is None:
                return [float('inf')]
            
            model.fit(y_train, X_train)
            predictions = self._generate_validation_predictions(model, y_train, y_test, X_train, X_test)
            
            score = self._calculate_metric(y_test, predictions)
            scores.append(score)
        
        return scores
    
    def _generate_validation_predictions(self, model: BaseForecaster, 
                                        y_train: pd.DataFrame, y_val: pd.DataFrame,
                                        X_train: Optional[pd.DataFrame], 
                                        X_val: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Generate predictions for validation period"""
        predictions = []
        
        for j in range(len(y_val)):
            if isinstance(model, LinearForecaster):
                if X_val is None:
                    raise ValueError("LinearForecaster requires X")
                X_curr = pd.concat([X_train, X_val.iloc[:j+1]]) if j > 0 else X_train
                pred = model.predict(X_curr)
            else:
                pred = model.predict()
            predictions.append(pred.values[0])
        
        return pd.DataFrame(predictions, columns=y_val.columns, index=y_val.index)
    
    def predict(self, X: Optional[pd.DataFrame] = None, steps: int = None) -> pd.DataFrame:
        """Generate predictions using the best model
        
        Args:
            X: Optional exogenous variables
            steps: Number of steps to forecast (overrides model horizon)
            
        Returns:
            DataFrame with forecasts
        """
        if self.best_model is None:
            raise ValueError("No model selected. Run fit() first.")
        
        if steps is not None and steps != self.best_model.horizon:
            # Update model horizon
            self.best_model.horizon = steps
        
        return self.best_model.predict(X)
    
    def _calculate_metric(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Calculate the selected metric"""
        if self.metric == 'mape':
            epsilon = 1e-10
            return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
        elif self.metric == 'rmse':
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        elif self.metric == 'mae':
            return float(np.mean(np.abs(y_true - y_pred)))
        elif self.metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            return float(1 - (ss_res / ss_tot))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get performance results for all models"""
        return self.results
    
    def get_best_model(self) -> Tuple[str, BaseForecaster]:
        """Get the best model name and instance"""
        if self.best_model is None:
            raise ValueError("No model selected. Run fit() first.")
        
        best_name = None
        for name, result in self.results.items():
            score = result if isinstance(result, (int, float)) else result.get('mean_score')
            if self.best_model in self.models:
                best_name = name
                break
        
        return best_name or "Unknown", self.best_model
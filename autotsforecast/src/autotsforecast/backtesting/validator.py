import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from autotsforecast.models.base import BaseForecaster


class BacktestValidator:
    """Time series cross-validation and backtesting with expanding/rolling windows
    
    This validator performs time-respecting cross-validation that maximizes training data usage.
    For n_splits CV folds, the first fold uses maximum available training data, working backwards.
    
    Example with 226 points, n_splits=3, test_size=14:
    - Fold 1: Train [0:212] → Validate [212:226]  (most recent data)
    - Fold 2: Train [0:198] → Validate [198:212]
    - Fold 3: Train [0:184] → Validate [184:198]
    
    Optionally, you can reserve a holdout period at the end for final evaluation.
    """
    
    def __init__(self, model: Optional[BaseForecaster] = None, n_splits: int = 5, 
                 test_size: int = 10, window_type: str = 'expanding',
                 holdout_period: Optional[int] = None):
        """
        Args:
            model: Forecaster instance to validate. Required for run()/run_with_holdout().
            n_splits: Number of train/test splits for CV
            test_size: Size of each validation fold
            window_type: 'expanding' (growing train set) or 'rolling' (fixed train size)
            holdout_period: If provided, reserves last N points as holdout test set.
                           CV will only use data before the holdout period.
                           Use run_with_holdout() to get both CV and holdout metrics.
        """
        self.model = model
        self.n_splits = n_splits
        self.test_size = test_size
        self.window_type = window_type
        self.holdout_period = holdout_period
        self.results = []
        self.predictions_list = []
        self.actuals_list = []
        self.holdout_metrics_ = None
        self.holdout_predictions_ = None

    def validate_results(self, results: Dict) -> bool:
        """Validate a simple metrics dict.

        This is a lightweight helper for quick sanity checks and backwards compatibility.
        It does not depend on an initialized model.
        """
        try:
            mse = float(results.get('mse'))
            mae = float(results.get('mae'))
        except (TypeError, ValueError):
            return False

        return mse >= 0 and mae >= 0
        
    def run(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Run cross-validation on the provided data (excluding holdout if specified)
        
        Args:
            y: Target time series DataFrame
            X: Optional exogenous variables
            
        Returns:
            Dict with aggregated CV performance metrics
            
        Note:
            If holdout_period was specified in __init__, this only uses data up to
            len(y) - holdout_period for CV. Use run_with_holdout() to also evaluate
            on the holdout period.
        """
        
        if self.model is None:
            raise ValueError("BacktestValidator requires a model for run().")

        # Determine the effective data length for CV
        effective_len = len(y) - (self.holdout_period or 0)
        
        if effective_len < self.n_splits * self.test_size + 10:
            raise ValueError(
                f"Not enough data for backtesting. Need at least {self.n_splits * self.test_size + 10} points "
                f"for CV (got {effective_len} after excluding holdout period)"
            )
        
        # Use only the CV portion of data
        y_cv = y.iloc[:effective_len]
        X_cv = X.iloc[:effective_len] if X is not None else None
        
        all_predictions = []
        all_actuals = []
        
        # Work backwards from end of CV data to maximize training data on first fold
        for i in range(self.n_splits):
            # Fold 0 (first): validates on last test_size points of CV data [n-k:n]
            # Fold 1: validates on second-to-last test_size points [n-2k:n-k], etc.
            test_end = len(y_cv) - (i * self.test_size)
            test_start = test_end - self.test_size
            
            if self.window_type == 'expanding':
                train_start = 0
                train_end = test_start
            else:  # rolling window
                # For rolling, maintain consistent training size
                min_train_size = len(y_cv) - (self.n_splits * self.test_size)
                train_end = test_start
                train_start = max(0, train_end - min_train_size)
            
            y_train = y_cv.iloc[train_start:train_end]
            y_test = y_cv.iloc[test_start:test_end]

            # Only use covariates for models that support them.
            use_covariates = bool(getattr(self.model, 'supports_covariates', False))
            X_train = X_cv.iloc[train_start:train_end] if (X_cv is not None and use_covariates) else None
            X_test = X_cv.iloc[test_start:test_end] if (X_cv is not None and use_covariates) else None

            # Fit model and generate predictions for the full test window.
            # This is required for multi-step models and models that need full-horizon
            # future covariates (e.g., Prophet with regressors).
            original_horizon = getattr(self.model, 'horizon', None)
            try:
                if original_horizon is not None:
                    self.model.horizon = len(y_test)

                self.model.fit(y_train, X_train)

                if X_test is not None:
                    predictions = self.model.predict(X_test)
                else:
                    predictions = self.model.predict()
            finally:
                if original_horizon is not None:
                    self.model.horizon = original_horizon

            # Normalize output shape/index
            if len(predictions) != len(y_test):
                raise ValueError(
                    f"Model produced {len(predictions)} predictions for a test window of {len(y_test)}"
                )
            predictions = predictions.copy()
            predictions.index = y_test.index

            # Ensure column order matches y_cv
            if list(predictions.columns) != list(y_cv.columns):
                predictions = predictions.reindex(columns=y_cv.columns)
            
            all_predictions.append(predictions)
            all_actuals.append(y_test)
            
            # Store fold results
            fold_metrics = self._calculate_metrics(y_test, predictions)
            self.results.append({
                'fold': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(y_train),
                'test_size': len(y_test),
                **fold_metrics
            })
        
        # Store for plotting
        self.predictions_list = all_predictions
        self.actuals_list = all_actuals
        
        # Calculate overall metrics
        all_predictions_concat = pd.concat(all_predictions)
        all_actuals_concat = pd.concat(all_actuals)
        overall_metrics = self._calculate_metrics(all_actuals_concat, all_predictions_concat)
        
        return overall_metrics
    
    def run_with_holdout(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Run CV on training portion, then evaluate on holdout period
        
        This method requires holdout_period to be specified in __init__.
        
        Args:
            y: Full time series DataFrame (including holdout period)
            X: Optional exogenous variables (including holdout period)
            
        Returns:
            Tuple of (cv_metrics, holdout_metrics):
            - cv_metrics: Aggregated CV performance on training portion
            - holdout_metrics: Performance on holdout test period
            
        Example:
            >>> validator = BacktestValidator(model, n_splits=3, test_size=14, holdout_period=14)
            >>> cv_metrics, holdout_metrics = validator.run_with_holdout(y_full, X_full)
            >>> print(f"CV RMSE: {cv_metrics['rmse']:.2f}")
            >>> print(f"Holdout RMSE: {holdout_metrics['rmse']:.2f}")
        """
        if self.holdout_period is None:
            raise ValueError(
                "holdout_period must be specified in __init__ to use run_with_holdout(). "
                "Either set holdout_period when creating BacktestValidator, or use run() for CV only."
            )

        if self.model is None:
            raise ValueError("BacktestValidator requires a model for run_with_holdout().")
        
        if len(y) < self.holdout_period:
            raise ValueError(f"Data length {len(y)} is less than holdout_period {self.holdout_period}")
        
        # Step 1: Run CV on training portion (excludes holdout)
        cv_metrics = self.run(y, X)
        
        # Step 2: Evaluate on holdout period
        holdout_start = len(y) - self.holdout_period
        y_train_full = y.iloc[:holdout_start]
        y_holdout = y.iloc[holdout_start:]
        X_train_full = X.iloc[:holdout_start] if X is not None else None
        X_holdout = X.iloc[holdout_start:] if X is not None else None
        
        # Train final model on all training data (before holdout)
        use_covariates = bool(getattr(self.model, 'supports_covariates', False))
        X_fit = X_train_full if (X_train_full is not None and use_covariates) else None
        X_pred = X_holdout if (X_holdout is not None and use_covariates) else None
        
        # Set horizon to match holdout period
        original_horizon = getattr(self.model, 'horizon', None)
        try:
            if original_horizon is not None:
                self.model.horizon = len(y_holdout)
            
            self.model.fit(y_train_full, X_fit)
            
            if X_pred is not None:
                holdout_pred = self.model.predict(X_pred)
            else:
                holdout_pred = self.model.predict()
        finally:
            if original_horizon is not None:
                self.model.horizon = original_horizon
        
        # Normalize predictions
        if len(holdout_pred) != len(y_holdout):
            raise ValueError(
                f"Model produced {len(holdout_pred)} predictions for holdout period of {len(y_holdout)}"
            )
        holdout_pred = holdout_pred.copy()
        holdout_pred.index = y_holdout.index
        if list(holdout_pred.columns) != list(y.columns):
            holdout_pred = holdout_pred.reindex(columns=y.columns)
        
        # Calculate holdout metrics
        holdout_metrics = self._calculate_metrics(y_holdout, holdout_pred)
        
        # Store for later access
        self.holdout_metrics_ = holdout_metrics
        self.holdout_predictions_ = holdout_pred
        
        return cv_metrics, holdout_metrics
    
    def _calculate_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        """Calculate multiple evaluation metrics"""
        metrics = {}
        
        # MAPE (Mean Absolute Percentage Error)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        metrics['mape'] = float(mape.mean() if isinstance(mape, pd.Series) else mape)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = float(rmse.mean() if isinstance(rmse, pd.Series) else rmse)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = float(mae.mean() if isinstance(mae, pd.Series) else mae)
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
        metrics['smape'] = float(smape.mean() if isinstance(smape, pd.Series) else smape)

        # R2 (Coefficient of determination), averaged across series
        y_true_vals = y_true.values
        y_pred_vals = y_pred.values
        ss_res = np.sum((y_true_vals - y_pred_vals) ** 2, axis=0)
        ss_tot = np.sum((y_true_vals - np.mean(y_true_vals, axis=0)) ** 2, axis=0)
        r2_per_series = 1.0 - (ss_res / (ss_tot + epsilon))
        metrics['r2'] = float(np.mean(r2_per_series))
        
        return metrics
    
    def get_fold_results(self) -> pd.DataFrame:
        """Get results for each fold as DataFrame"""
        return pd.DataFrame(self.results)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics across all folds"""
        df = self.get_fold_results()
        metric_cols = ['mape', 'rmse', 'mae', 'smape']
        
        summary = df[metric_cols].agg(['mean', 'std', 'min', 'max'])
        return summary
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot backtesting results
        
        Args:
            figsize: Figure size tuple
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                print("No results to plot. Run backtesting first.")
                return
            
            df = self.get_fold_results()
            n_series = len(self.actuals_list[0].columns)
            
            # Create subplots
            fig = plt.figure(figsize=figsize)
            
            # Plot 1: Metrics by fold
            ax1 = plt.subplot(2, 2, 1)
            df.plot(x='fold', y=['mape', 'rmse', 'mae'], ax=ax1, marker='o')
            ax1.set_title('Error Metrics by Fold')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('Error')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: SMAPE by fold
            ax2 = plt.subplot(2, 2, 2)
            df.plot(x='fold', y='smape', ax=ax2, marker='o', color='purple')
            ax2.set_title('SMAPE by Fold')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('SMAPE (%)')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Actual vs Predicted for last fold
            ax3 = plt.subplot(2, 1, 2)
            last_actual = self.actuals_list[-1]
            last_pred = self.predictions_list[-1]
            
            for col in last_actual.columns:
                ax3.plot(last_actual.index, last_actual[col], 
                        label=f'{col} (Actual)', marker='o', linestyle='-')
                ax3.plot(last_pred.index, last_pred[col], 
                        label=f'{col} (Predicted)', marker='x', linestyle='--')
            
            ax3.set_title('Actual vs Predicted (Last Fold)')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Install it with: pip install matplotlib")
    
    def get_predictions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get concatenated actuals and predictions from all folds
        
        Returns:
            Tuple of (actuals_df, predictions_df)
        """
        if not self.actuals_list:
            raise ValueError("No results available. Run backtesting first.")
        
        actuals = pd.concat(self.actuals_list)
        predictions = pd.concat(self.predictions_list)
        
        return actuals, predictions

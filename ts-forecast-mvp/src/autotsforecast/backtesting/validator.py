import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from autotsforecast.models.base import BaseForecaster


class BacktestValidator:
    """Time series cross-validation and backtesting with expanding/rolling windows"""
    
    def __init__(self, model: BaseForecaster, n_splits: int = 5, 
                 test_size: int = 10, window_type: str = 'expanding'):
        """
        Args:
            model: Forecaster instance to validate
            n_splits: Number of train/test splits
            test_size: Size of each test set
            window_type: 'expanding' (growing train set) or 'rolling' (fixed train size)
        """
        self.model = model
        self.n_splits = n_splits
        self.test_size = test_size
        self.window_type = window_type
        self.results = []
        self.predictions_list = []
        self.actuals_list = []
        
    def run(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Run backtesting with specified window type
        
        Args:
            y: Target time series DataFrame
            X: Optional exogenous variables
            
        Returns:
            Dict with aggregated performance metrics
        """
        
        min_train_size = len(y) - (self.n_splits * self.test_size)
        
        if min_train_size < 10:
            raise ValueError("Not enough data for backtesting with given parameters")
        
        all_predictions = []
        all_actuals = []
        
        for i in range(self.n_splits):
            # Define train and test sets
            test_end = len(y) - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            
            if self.window_type == 'expanding':
                train_start = 0
                train_end = test_start
            else:  # rolling window
                train_end = test_start
                train_start = max(0, train_end - min_train_size)
            
            y_train = y.iloc[train_start:train_end]
            y_test = y.iloc[test_start:test_end]
            
            X_train = X.iloc[train_start:train_end] if X is not None else None
            X_test = X.iloc[test_start:test_end] if X is not None else None
            
            # Fit model
            self.model.fit(y_train, X_train)
            
            # Generate predictions for test period
            predictions = []
            for j in range(len(y_test)):
                if X_test is not None:
                    pred = self.model.predict(X_test.iloc[[j]])
                else:
                    pred = self.model.predict()
                predictions.append(pred.values[0])
            
            predictions = pd.DataFrame(predictions, columns=y.columns, index=y_test.index)
            
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
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        metrics['r2'] = float(r2.mean() if isinstance(r2, pd.Series) else r2)
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
        metrics['smape'] = float(smape.mean() if isinstance(smape, pd.Series) else smape)
        
        return metrics
    
    def get_fold_results(self) -> pd.DataFrame:
        """Get results for each fold as DataFrame"""
        return pd.DataFrame(self.results)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics across all folds"""
        df = self.get_fold_results()
        metric_cols = ['mape', 'rmse', 'mae', 'r2', 'smape']
        
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
            
            # Plot 2: R-squared by fold
            ax2 = plt.subplot(2, 2, 2)
            df.plot(x='fold', y='r2', ax=ax2, marker='o', color='green')
            ax2.set_title('R² by Fold')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('R²')
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
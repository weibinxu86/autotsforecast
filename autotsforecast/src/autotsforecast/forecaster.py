"""
High-level forecasting interface that combines model selection, backtesting, and forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
import copy
from .models.base import BaseForecaster
from .models.selection import ModelSelector
from .backtesting.validator import BacktestValidator


def get_default_candidate_models(horizon: int) -> List[BaseForecaster]:
    """Return a default pool of candidate models for `AutoForecaster`.

    Includes a mix of fast baselines, classical models (ETS), and ML models.
    Optional models (e.g., LSTM) are included only when their dependencies are installed.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    from .models.base import VARForecaster, MovingAverageForecaster, LinearForecaster
    from .models.external import RandomForestForecaster, XGBoostForecaster, ETSForecaster, LSTMForecaster

    candidates: List[BaseForecaster] = [
        MovingAverageForecaster(horizon=horizon, window=5),
        MovingAverageForecaster(horizon=horizon, window=7),
        VARForecaster(horizon=horizon, lags=1),
        VARForecaster(horizon=horizon, lags=2),
        RandomForestForecaster(horizon=horizon, n_lags=7, n_estimators=100, random_state=42),
        XGBoostForecaster(horizon=horizon, n_lags=7, n_estimators=100, random_state=42),
        LinearForecaster(horizon=horizon),
        ETSForecaster(horizon=horizon, trend='add', seasonal=None),
    ]

    # Optional deep learning model
    try:
        candidates.append(LSTMForecaster(horizon=horizon))
    except ImportError:
        pass

    return candidates


class AutoForecaster:
    """
    High-level interface for automatic model selection and forecasting.
    
    This class:
    1. Evaluates candidate models using backtesting
    2. Selects the best model based on specified metric
    3. Retrains best model on full dataset
    4. Generates forecasts for specified horizon
    
    Parameters
    ----------
    candidate_models : List[BaseForecaster]
        List of candidate forecasting models to evaluate
    metric : str, default='rmse'
        Metric to use for model selection ('rmse', 'mae', 'mape', 'r2')
    n_splits : int, default=5
        Number of backtesting splits
    test_size : int, default=20
        Size of test set in each backtest split
    window_type : str, default='expanding'
        Type of cross-validation window ('expanding' or 'rolling')
    verbose : bool, default=True
        Whether to print progress information
        
    Attributes
    ----------
    best_model_ : BaseForecaster
        The best performing model after selection
    cv_results_ : Dict
        Cross-validation results for all models
    best_model_name_ : str
        Name of the best model
    forecasts_ : pd.DataFrame
        Generated forecasts (after calling forecast())
    
    Examples
    --------
    >>> from autotsforecast import AutoForecaster, VARForecaster, MovingAverageForecaster
    >>> 
    >>> # Define candidate models
    >>> candidates = [
    ...     VARForecaster(lags=1, horizon=30),
    ...     VARForecaster(lags=3, horizon=30),
    ...     VARForecaster(lags=7, horizon=30),
    ...     MovingAverageForecaster(window=7, horizon=30)
    ... ]
    >>> 
    >>> # Create auto forecaster
    >>> auto = AutoForecaster(
    ...     candidate_models=candidates,
    ...     metric='rmse',
    ...     n_splits=5,
    ...     test_size=20
    ... )
    >>> 
    >>> # Fit and select best model
    >>> auto.fit(train_data)
    >>> 
    >>> # Generate forecasts
    >>> forecasts = auto.forecast()
    >>> 
    >>> # Get performance summary
    >>> summary = auto.get_summary()
    """
    
    def __init__(
        self,
        candidate_models: List[BaseForecaster],
        metric: str = 'rmse',
        n_splits: int = 5,
        test_size: int = 20,
        window_type: str = 'expanding',
        verbose: bool = True,
        per_series_models: bool = False,
        n_jobs: int = -1,
    ):
        self.candidate_models = candidate_models
        self.metric = metric
        self.n_splits = n_splits
        self.test_size = test_size
        self.window_type = window_type
        self.verbose = verbose
        self.per_series_models = per_series_models
        self.n_jobs = n_jobs
        
        # Initialize attributes
        self.best_model_ = None
        self.best_models_ = None
        self.cv_results_ = {}
        self.cv_results_by_series_ = None
        self.best_model_name_ = None
        self.best_model_names_ = None
        self.forecasts_ = None
        self.is_fitted = False
        self.feature_names_ = None
        self._last_index_ = None
        self._freq_ = None

    def _infer_freq(self, y_index) -> Optional[str]:
        if not isinstance(y_index, pd.DatetimeIndex):
            return None
        freq = pd.infer_freq(y_index)
        if freq is None and hasattr(y_index, 'freqstr'):
            freq = y_index.freqstr
        return freq

    def _future_index(self, horizon: int):
        if isinstance(self._last_index_, pd.Timestamp):
            freq = self._freq_ or 'D'
            return pd.date_range(
                start=self._last_index_ + pd.tseries.frequencies.to_offset(freq),
                periods=horizon,
                freq=freq,
            )
        return pd.RangeIndex(start=0, stop=horizon, step=1)

    def _clone_model(self, model: BaseForecaster) -> BaseForecaster:
        """Best-effort cloning of a candidate model to avoid state bleed between fits."""
        try:
            return copy.deepcopy(model)
        except Exception as e:
            raise TypeError(
                f"Failed to clone model {model.__class__.__name__}. "
                "Please pass fresh model instances or models that can be deep-copied. "
                f"Original error: {e}"
            )
        
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None) -> 'AutoForecaster':
        """
        Evaluate candidate models and select the best one.
        
        Parameters
        ----------
        y : pd.DataFrame
            Historical time series data
        X : pd.DataFrame, optional
            Exogenous variables (if needed by models)
            
        Returns
        -------
        self : AutoForecaster
            Fitted forecaster
        """
        if self.verbose:
            print("="*80)
            print("AUTO FORECASTER: Model Selection with Backtesting")
            print("="*80)
            print(f"\n📊 Data: {len(y)} observations, {y.shape[1]} variables")
            print(f"🔍 Evaluating {len(self.candidate_models)} candidate models")
            print(f"📈 Backtesting: {self.n_splits} splits, {self.test_size} test size")
            print(f"🎯 Selection metric: {self.metric.upper()}")
            print(f"🔄 Window type: {self.window_type}")
            if self.per_series_models:
                print(f"🧩 Per-series selection: enabled ({y.shape[1]} models will be selected)")
                print(f"⚡ Parallelism: n_jobs={self.n_jobs}")
            print()
        
        self.feature_names_ = y.columns.tolist()
        self._last_index_ = y.index[-1] if len(y.index) else None
        self._freq_ = self._infer_freq(y.index)

        # Per-series model selection mode: select a best model for each time series.
        if self.per_series_models:
            from joblib import Parallel, delayed

            def fit_one_series(series_name: str):
                y_single = y[[series_name]]

                best_score = float('inf') if self.metric != 'r2' else float('-inf')
                best_model = None
                best_name = None
                series_cv_results: Dict[str, Any] = {}

                for candidate in self.candidate_models:
                    # Skip VAR in univariate mode (requires 2+ series)
                    if candidate.__class__.__name__ == 'VARForecaster':
                        continue

                    model = self._clone_model(candidate)
                    model_name = model.__class__.__name__
                    if hasattr(model, 'lags'):
                        model_name += f"(lags={getattr(model, 'lags')})"
                    elif hasattr(model, 'window'):
                        model_name += f"(window={getattr(model, 'window')})"

                    try:
                        validator = BacktestValidator(
                            model=model,
                            n_splits=self.n_splits,
                            test_size=self.test_size,
                            window_type=self.window_type,
                        )
                        cv = validator.run(y_single, X)
                        series_cv_results[model_name] = cv
                        score = cv[self.metric]
                        is_better = (score < best_score) if self.metric != 'r2' else (score > best_score)
                        if is_better:
                            best_score = score
                            best_model = model
                            best_name = model_name
                    except Exception:
                        continue

                if best_model is None:
                    raise ValueError(f"No valid models found for series '{series_name}'.")

                # Retrain best model on full series
                best_model.fit(y_single, X)
                return series_name, best_model, best_name, best_score, series_cv_results

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_one_series)(col) for col in self.feature_names_
            )

            self.best_models_ = {name: model for name, model, _, _, _ in results}
            self.best_model_names_ = {name: model_name for name, _, model_name, _, _ in results}
            self.cv_results_by_series_ = {name: cv for name, _, _, _, cv in results}
            self.best_model_ = None
            self.best_model_name_ = 'per-series'
            self.is_fitted = True

            if self.verbose:
                print("="*80)
                print("🏆 PER-SERIES MODELS SELECTED")
                print("="*80)
                # Show a short summary (first few)
                for j, col in enumerate(self.feature_names_[:10], 1):
                    print(f"  {j}. {col}: {self.best_model_names_[col]}")
                if len(self.feature_names_) > 10:
                    print(f"  ... ({len(self.feature_names_)} total)")
                print()

            return self
        best_score = float('inf') if self.metric != 'r2' else float('-inf')
        
        # Evaluate each candidate model
        for i, model in enumerate(self.candidate_models, 1):
            model_name = f"{model.__class__.__name__}"
            if hasattr(model, 'lags'):
                model_name += f"(lags={model.lags})"
            elif hasattr(model, 'window'):
                model_name += f"(window={model.window})"
            
            if self.verbose:
                print(f"[{i}/{len(self.candidate_models)}] Testing {model_name}...")
            
            try:
                # Create backtesting validator
                validator = BacktestValidator(
                    model=model,
                    n_splits=self.n_splits,
                    test_size=self.test_size,
                    window_type=self.window_type
                )
                
                # Run backtesting
                cv_results = validator.run(y, X)
                
                # Store results
                self.cv_results_[model_name] = cv_results
                
                # Get mean metric
                score = cv_results[self.metric]
                
                if self.verbose:
                    print(f"   {self.metric.upper()}: {score:.4f}")
                
                # Update best model
                is_better = (score < best_score) if self.metric != 'r2' else (score > best_score)
                if is_better:
                    best_score = score
                    self.best_model_ = model
                    self.best_model_name_ = model_name
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Error: {str(e)}")
                continue
        
        if self.best_model_ is None:
            raise ValueError("No valid models found. All candidates failed.")
        
        if self.verbose:
            print()
            print("="*80)
            print(f"🏆 BEST MODEL SELECTED: {self.best_model_name_}")
            print(f"   Performance: {self.metric.upper()} = {best_score:.4f}")
            print("="*80)
            print()
            print("🔄 Retraining best model on full dataset...")
        
        # Retrain best model on full dataset
        self.best_model_.fit(y, X)
        self.is_fitted = True
        
        if self.verbose:
            print("✅ Training complete!")
        
        return self
    
    def forecast(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts using the best model.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Future exogenous variables (if required by model)
            
        Returns
        -------
        forecasts : pd.DataFrame
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before forecasting. Call fit() first.")
        
        if self.verbose:
            print("\n🔮 Generating forecasts...")
        
        # Generate forecasts
        if self.per_series_models:
            if not self.best_models_:
                raise ValueError("Per-series forecaster is not fitted. Call fit() first.")

            horizon = None
            # Determine horizon from any selected model
            for m in self.best_models_.values():
                horizon = getattr(m, 'horizon', None)
                if horizon is not None:
                    break
            if horizon is None:
                raise ValueError("Unable to determine forecast horizon from selected models")

            future_index = self._future_index(horizon)
            preds: Dict[str, np.ndarray] = {}
            for col, model in self.best_models_.items():
                p = model.predict(X)
                # Accept either single-column df or multi; normalize to 1d array
                if isinstance(p, pd.DataFrame):
                    if col in p.columns:
                        vals = p[col].to_numpy()
                    else:
                        vals = p.iloc[:, 0].to_numpy()
                else:
                    vals = np.asarray(p)
                preds[col] = vals

            df = pd.DataFrame(preds)
            if len(df) != len(future_index):
                # Best-effort alignment
                df = df.iloc[: len(future_index)].copy()
                if len(df) < len(future_index):
                    df = df.reindex(range(len(future_index)))
            df.index = future_index
            self.forecasts_ = df
        else:
            self.forecasts_ = self.best_model_.predict(X)
        
        if self.verbose:
            horizon = len(self.forecasts_)
            print(f"✅ Generated {horizon} step forecast")
            print(f"   Variables: {', '.join(self.feature_names_)}")
        
        return self.forecasts_
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of model selection and forecasting.
        
        Returns
        -------
        summary : dict
            Dictionary containing:
            - best_model: Name of best model
            - best_score: Performance score
            - all_results: Results for all models
            - forecast_summary: Summary statistics of forecasts
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before getting summary.")
        
        summary = {
            'best_model': self.best_model_name_,
            'best_score': self.cv_results_[self.best_model_name_][self.metric],
            'selection_metric': self.metric,
            'backtesting_config': {
                'n_splits': self.n_splits,
                'test_size': self.test_size,
                'window_type': self.window_type
            },
            'all_results': {}
        }
        
        # Add results for all models
        for model_name, results in self.cv_results_.items():
            summary['all_results'][model_name] = {
                'rmse': results['rmse'],
                'mae': results['mae'],
                'r2': results['r2'],
                'mape': results.get('mape', None),
                'smape': results.get('smape', None)
            }
        
        # Add forecast summary if available
        if self.forecasts_ is not None:
            forecast_values = self.forecasts_.values
            summary['forecast_summary'] = {
                'horizon': len(self.forecasts_),
                'variables': self.feature_names_,
                'mean': forecast_values.mean(axis=0).tolist(),
                'std': forecast_values.std(axis=0).tolist(),
                'min': forecast_values.min(axis=0).tolist(),
                'max': forecast_values.max(axis=0).tolist()
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of results."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("AUTO FORECASTER SUMMARY")
        print("="*80)
        
        print(f"\n🏆 Best Model: {summary['best_model']}")
        print(f"   {summary['selection_metric'].upper()}: {summary['best_score']:.4f}")
        
        print(f"\n📊 Backtesting Configuration:")
        print(f"   Splits: {summary['backtesting_config']['n_splits']}")
        print(f"   Test size: {summary['backtesting_config']['test_size']}")
        print(f"   Window: {summary['backtesting_config']['window_type']}")
        
        print(f"\n📈 All Models Performance:")
        print(f"{'Model':<40} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 80)
        
        # Sort by metric
        sorted_models = sorted(
            summary['all_results'].items(),
            key=lambda x: x[1][summary['selection_metric']],
            reverse=(summary['selection_metric'] == 'r2')
        )
        
        for model_name, metrics in sorted_models:
            marker = "🏆" if model_name == summary['best_model'] else "  "
            print(f"{marker} {model_name:<38} {metrics['rmse']:>10.4f}  {metrics['mae']:>10.4f}  {metrics['r2']:>10.4f}")
        
        if 'forecast_summary' in summary:
            print(f"\n🔮 Forecast Summary:")
            print(f"   Horizon: {summary['forecast_summary']['horizon']} steps")
            print(f"   Variables: {', '.join(summary['forecast_summary']['variables'])}")
            print(f"\n   {'Variable':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("   " + "-" * 68)
            for i, var in enumerate(summary['forecast_summary']['variables']):
                print(f"   {var:<20} "
                      f"{summary['forecast_summary']['mean'][i]:>10.4f}  "
                      f"{summary['forecast_summary']['std'][i]:>10.4f}  "
                      f"{summary['forecast_summary']['min'][i]:>10.4f}  "
                      f"{summary['forecast_summary']['max'][i]:>10.4f}")
        
        print("\n" + "="*80)
    
    def save(self, filepath: str):
        """Save AutoForecaster to disk
        
        Args:
            filepath: Path to save (e.g., 'autoforecaster.pkl')
            
        Example:
            >>> auto.fit(train_data)
            >>> auto.save('best_autoforecaster.joblib')
        """
        import joblib
        
        metadata = {
            'autoforecaster': self,
            'best_model_name': self.best_model_name_,
            'cv_results': self.cv_results_,
            'metric': self.metric,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names_,
            'save_timestamp': pd.Timestamp.now()
        }
        
        joblib.dump(metadata, filepath)
        print(f"✓ AutoForecaster saved to: {filepath}")
        print(f"  Best Model: {self.best_model_name_}")
        print(f"  Metric: {self.metric}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AutoForecaster':
        """Load AutoForecaster from disk
        
        Args:
            filepath: Path to saved AutoForecaster
            
        Returns:
            Loaded AutoForecaster instance
            
        Example:
            >>> auto = AutoForecaster.load('best_autoforecaster.joblib')
            >>> forecasts = auto.forecast()
        """
        import joblib
        
        metadata = joblib.load(filepath)
        auto = metadata['autoforecaster']
        
        print(f"✓ AutoForecaster loaded from: {filepath}")
        print(f"  Best Model: {metadata['best_model_name']}")
        print(f"  Metric: {metadata['metric']}")
        print(f"  Saved: {metadata['save_timestamp']}")
        
        return auto


"""
Parallel processing utilities for time series forecasting.

Provides efficient concurrent processing for:
- Model evaluation across multiple series
- Backtesting with parallel folds
- Batch forecasting
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Callable, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import warnings


# Try to import joblib for robust parallelization
try:
    from joblib import Parallel, delayed, cpu_count
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def get_optimal_n_jobs(n_tasks: int, n_jobs: int = -1) -> int:
    """
    Determine optimal number of parallel jobs.
    
    Parameters
    ----------
    n_tasks : int
        Number of tasks to parallelize
    n_jobs : int
        Requested number of jobs. -1 for auto-detect.
        
    Returns
    -------
    int
        Optimal number of jobs
    """
    max_workers = multiprocessing.cpu_count()
    
    if n_jobs == -1:
        # Auto-detect: use min of available CPUs and tasks
        return min(max_workers, n_tasks)
    elif n_jobs < -1:
        # Negative values: max_workers + 1 + n_jobs
        return max(1, max_workers + 1 + n_jobs)
    else:
        return min(n_jobs, max_workers, n_tasks)


def parallel_map(
    func: Callable,
    items: List[Any],
    n_jobs: int = -1,
    backend: str = 'auto',
    verbose: bool = False,
    prefer: str = 'threads',
    desc: str = 'Processing'
) -> List[Any]:
    """
    Apply a function to items in parallel.
    
    Parameters
    ----------
    func : callable
        Function to apply to each item
    items : list
        Items to process
    n_jobs : int
        Number of parallel jobs (-1 for auto)
    backend : str
        'auto', 'joblib', 'threads', 'processes'
    verbose : bool
        Show progress
    prefer : str
        Preferred backend for joblib: 'threads' or 'processes'
    desc : str
        Description for progress bar
        
    Returns
    -------
    list
        Results in same order as input items
    """
    n_items = len(items)
    if n_items == 0:
        return []
    
    n_workers = get_optimal_n_jobs(n_items, n_jobs)
    
    # Single item or single worker - no parallelization needed
    if n_workers == 1 or n_items == 1:
        results = []
        for i, item in enumerate(items):
            if verbose:
                print(f"\r{desc}: {i+1}/{n_items}", end='', flush=True)
            results.append(func(item))
        if verbose:
            print()
        return results
    
    # Select backend
    if backend == 'auto':
        backend = 'joblib' if JOBLIB_AVAILABLE else 'threads'
    
    if backend == 'joblib' and JOBLIB_AVAILABLE:
        verbosity = 10 if verbose else 0
        results = Parallel(n_jobs=n_workers, prefer=prefer, verbose=verbosity)(
            delayed(func)(item) for item in items
        )
        return results
    
    elif backend == 'threads':
        results = [None] * n_items
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                completed += 1
                if verbose:
                    print(f"\r{desc}: {completed}/{n_items}", end='', flush=True)
        if verbose:
            print()
        return results
    
    elif backend == 'processes':
        results = [None] * n_items
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                completed += 1
                if verbose:
                    print(f"\r{desc}: {completed}/{n_items}", end='', flush=True)
        if verbose:
            print()
        return results
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


class ParallelForecaster:
    """
    Wrapper for parallel forecasting operations.
    
    Efficiently parallelizes operations across multiple time series
    or across multiple models.
    
    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for auto-detect)
    backend : str, default='auto'
        Parallelization backend ('auto', 'joblib', 'threads', 'processes')
    verbose : bool, default=False
        Show progress information
        
    Examples
    --------
    >>> from autotsforecast.utils.parallel import ParallelForecaster
    >>> 
    >>> pf = ParallelForecaster(n_jobs=-1)
    >>> 
    >>> # Fit multiple models in parallel
    >>> fitted_models = pf.parallel_fit(models, y_train)
    >>> 
    >>> # Evaluate models in parallel
    >>> scores = pf.parallel_evaluate(models, y_train, y_test)
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = 'auto',
        verbose: bool = False
    ):
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
    
    def parallel_fit(
        self,
        models: List[Any],
        y: pd.DataFrame,
        X: Optional[pd.DataFrame] = None
    ) -> List[Any]:
        """
        Fit multiple models in parallel.
        
        Parameters
        ----------
        models : list
            List of forecaster instances
        y : pd.DataFrame
            Training data
        X : pd.DataFrame, optional
            Covariates
            
        Returns
        -------
        list
            Fitted model instances
        """
        import copy
        
        def fit_single(model):
            model_copy = copy.deepcopy(model)
            try:
                model_copy.fit(y, X)
                return model_copy
            except Exception as e:
                return e
        
        results = parallel_map(
            fit_single,
            models,
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose,
            desc='Fitting models'
        )
        
        return results
    
    def parallel_predict(
        self,
        models: List[Any],
        X: Optional[pd.DataFrame] = None
    ) -> List[pd.DataFrame]:
        """
        Generate predictions from multiple models in parallel.
        
        Parameters
        ----------
        models : list
            List of fitted forecasters
        X : pd.DataFrame, optional
            Future covariates
            
        Returns
        -------
        list
            List of prediction DataFrames
        """
        def predict_single(model):
            try:
                return model.predict(X)
            except Exception as e:
                return e
        
        results = parallel_map(
            predict_single,
            models,
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose,
            desc='Generating predictions'
        )
        
        return results
    
    def parallel_backtest(
        self,
        model: Any,
        y: pd.DataFrame,
        X: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
        test_size: int = 20,
        window_type: str = 'expanding'
    ) -> Dict[str, float]:
        """
        Run backtesting with parallel fold evaluation.
        
        Parameters
        ----------
        model : forecaster
            Forecaster to evaluate
        y : pd.DataFrame
            Full dataset
        X : pd.DataFrame, optional
            Covariates
        n_splits : int
            Number of CV splits
        test_size : int
            Size of each test fold
        window_type : str
            'expanding' or 'rolling'
            
        Returns
        -------
        dict
            Aggregated performance metrics
        """
        import copy
        
        def evaluate_fold(fold_idx):
            test_end = len(y) - (fold_idx * test_size)
            test_start = test_end - test_size
            
            if window_type == 'expanding':
                train_start = 0
            else:
                min_train = len(y) - (n_splits * test_size)
                train_start = max(0, test_start - min_train)
            
            y_train = y.iloc[train_start:test_start]
            y_test = y.iloc[test_start:test_end]
            X_train = X.iloc[train_start:test_start] if X is not None else None
            X_test = X.iloc[test_start:test_end] if X is not None else None
            
            try:
                model_copy = copy.deepcopy(model)
                original_horizon = getattr(model_copy, 'horizon', None)
                if original_horizon is not None:
                    model_copy.horizon = len(y_test)
                
                model_copy.fit(y_train, X_train)
                predictions = model_copy.predict(X_test)
                
                if original_horizon is not None:
                    model_copy.horizon = original_horizon
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                predictions.index = y_test.index
                rmse = np.sqrt(mean_squared_error(y_test.values, predictions.values))
                mae = mean_absolute_error(y_test.values, predictions.values)
                r2 = r2_score(y_test.values, predictions.values)
                
                return {'rmse': rmse, 'mae': mae, 'r2': r2}
            except Exception as e:
                return {'error': str(e)}
        
        fold_results = parallel_map(
            evaluate_fold,
            list(range(n_splits)),
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose,
            desc='Evaluating folds'
        )
        
        # Aggregate results
        valid_results = [r for r in fold_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'All folds failed'}
        
        return {
            'rmse': np.mean([r['rmse'] for r in valid_results]),
            'mae': np.mean([r['mae'] for r in valid_results]),
            'r2': np.mean([r['r2'] for r in valid_results]),
            'n_valid_folds': len(valid_results)
        }
    
    def parallel_series_fit(
        self,
        model_factory: Callable,
        y: pd.DataFrame,
        X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None
    ) -> Dict[str, Any]:
        """
        Fit models for each series in parallel.
        
        Parameters
        ----------
        model_factory : callable
            Function that returns a new model instance
        y : pd.DataFrame
            Multi-series data
        X : pd.DataFrame or dict, optional
            Covariates (shared or per-series)
            
        Returns
        -------
        dict
            Mapping from series name to fitted model
        """
        import copy
        
        def fit_series(series_name):
            y_single = y[[series_name]]
            
            if isinstance(X, dict):
                X_series = X.get(series_name)
            else:
                X_series = X
            
            try:
                model = model_factory()
                model.fit(y_single, X_series)
                return (series_name, model)
            except Exception as e:
                return (series_name, e)
        
        results = parallel_map(
            fit_series,
            y.columns.tolist(),
            n_jobs=self.n_jobs,
            backend=self.backend,
            verbose=self.verbose,
            desc='Fitting per-series models'
        )
        
        return dict(results)


def batch_forecast(
    models: Dict[str, Any],
    X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Generate forecasts from multiple per-series models in parallel.
    
    Parameters
    ----------
    models : dict
        Mapping from series name to fitted model
    X : pd.DataFrame or dict, optional
        Future covariates
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    pd.DataFrame
        Combined forecasts for all series
    """
    def predict_series(item):
        series_name, model = item
        if isinstance(X, dict):
            X_series = X.get(series_name)
        else:
            X_series = X
        
        try:
            pred = model.predict(X_series)
            if isinstance(pred, pd.DataFrame):
                if series_name in pred.columns:
                    return (series_name, pred[series_name].values)
                return (series_name, pred.iloc[:, 0].values)
            return (series_name, np.array(pred).flatten())
        except Exception as e:
            return (series_name, e)
    
    results = parallel_map(
        predict_series,
        list(models.items()),
        n_jobs=n_jobs,
        desc='Generating forecasts'
    )
    
    # Combine into DataFrame
    forecasts = {}
    for series_name, values in results:
        if not isinstance(values, Exception):
            forecasts[series_name] = values
    
    if not forecasts:
        raise ValueError("All forecasts failed")
    
    return pd.DataFrame(forecasts)

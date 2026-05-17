"""
High-level forecasting interface that combines model selection, backtesting, and forecasting.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
import copy
from .models.base import BaseForecaster
from .models.selection import ModelSelector
from .backtesting.validator import BacktestValidator

# Import progress tracking utilities
try:
    from .visualization.progress import ProgressTracker, progress_bar
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False


def get_default_candidate_models(horizon: int) -> List[BaseForecaster]:
    """Return a default pool of candidate models for `AutoForecaster`.

    Includes a mix of fast baselines, classical models (ETS, Theta), linear
    models (ElasticNet), and ensemble models (RandomForest).  Optional models
    (XGBoost, LSTM) are included only when their dependencies are installed.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    from .models.base import VARForecaster, MovingAverageForecaster
    from .models.external import (
        RandomForestForecaster, XGBoostForecaster, ETSForecaster,
        LSTMForecaster, ElasticNetForecaster, ThetaForecaster,
    )

    candidates: List[BaseForecaster] = [
        MovingAverageForecaster(horizon=horizon, window=5),
        MovingAverageForecaster(horizon=horizon, window=7),
        VARForecaster(horizon=horizon, lags=1),
        VARForecaster(horizon=horizon, lags=2),
        ElasticNetForecaster(horizon=horizon, n_lags=7),
        RandomForestForecaster(horizon=horizon, n_lags=7, n_estimators=100, random_state=42),
        ETSForecaster(horizon=horizon, trend='add', seasonal=None),
        ThetaForecaster(horizon=horizon),
    ]

    # Optional models — silently skip if the dependency is not installed
    try:
        candidates.append(XGBoostForecaster(horizon=horizon, n_lags=7, n_estimators=100, random_state=42))
    except ImportError:
        pass

    try:
        candidates.append(LSTMForecaster(horizon=horizon))
    except ImportError:
        pass

    return candidates


# ---------------------------------------------------------------------------
# Preset model pools (v0.6.0)
# ---------------------------------------------------------------------------

PRESETS: Dict[str, str] = {
    "fast": (
        "Very fast statistical + linear models. Runs in seconds. "
        "Best for quick exploration and large numbers of series."
    ),
    "balanced": (
        "Speed/accuracy balance: statistical + gradient boosting. Recommended default. "
        "Typically completes in under a minute per series."
    ),
    "accuracy": (
        "Maximum accuracy: all ML + neural models. Takes longer but wins competitions. "
        "Enable per_series_models=True and n_jobs=-1 to parallelise."
    ),
    "zero_shot": (
        "Foundation models only (Chronos-2 variants). Zero training required — "
        "just pass data and get state-of-the-art predictions instantly."
    ),
    "intermittent": (
        "Optimised for sparse / intermittent demand series with many zeros. "
        "Uses Croston SBA, ETS, and Theta."
    ),
    "hierarchical": (
        "Models suited for hierarchical reconciliation: statistical baselines + "
        "LightGBM. Produces smooth, coherent forecasts."
    ),
}


def get_preset_models(preset: str, horizon: int) -> List[BaseForecaster]:
    """Return a curated list of candidate models for *preset*.

    Parameters
    ----------
    preset : str
        One of ``'fast'``, ``'balanced'``, ``'accuracy'``, ``'zero_shot'``,
        ``'intermittent'``, ``'hierarchical'``.
    horizon : int
        Forecast horizon shared by all candidate models.

    Returns
    -------
    List[BaseForecaster]
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
        )

    from .models.base import MovingAverageForecaster
    from .models.external import (
        ARIMAForecaster,
        ETSForecaster,
        RandomForestForecaster,
    )

    if preset == "fast":
        candidates: List[BaseForecaster] = [
            MovingAverageForecaster(horizon=horizon, window=7),
            ETSForecaster(horizon=horizon, trend="add", seasonal=None),
            ARIMAForecaster(horizon=horizon),
        ]
        try:
            from .models.external import ElasticNetForecaster
            candidates.append(ElasticNetForecaster(horizon=horizon, n_lags=7))
        except ImportError:
            pass
        try:
            from .models.external import ThetaForecaster
            candidates.append(ThetaForecaster(horizon=horizon))
        except ImportError:
            pass
        return candidates

    if preset == "balanced":
        candidates = [
            MovingAverageForecaster(horizon=horizon, window=7),
            ETSForecaster(horizon=horizon, trend="add", seasonal=None),
            ARIMAForecaster(horizon=horizon),
            RandomForestForecaster(horizon=horizon, n_lags=7, n_estimators=100),
        ]
        try:
            from .models.external import ThetaForecaster
            candidates.append(ThetaForecaster(horizon=horizon))
        except ImportError:
            pass
        try:
            from .models.external import XGBoostForecaster
            candidates.append(XGBoostForecaster(horizon=horizon, n_lags=7, n_estimators=100))
        except ImportError:
            pass
        try:
            from .models.external import LightGBMForecaster
            candidates.append(LightGBMForecaster(horizon=horizon, n_lags=7, n_estimators=100))
        except ImportError:
            pass
        return candidates

    if preset == "accuracy":
        candidates = [
            MovingAverageForecaster(horizon=horizon, window=7),
            ETSForecaster(horizon=horizon, trend="add", seasonal=None),
            ARIMAForecaster(horizon=horizon),
            RandomForestForecaster(horizon=horizon, n_lags=14, n_estimators=200),
        ]
        _optional_accuracy = [
            ("ThetaForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["ThetaForecaster"]
            ).ThetaForecaster(horizon=horizon)),
            ("XGBoostForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["XGBoostForecaster"]
            ).XGBoostForecaster(horizon=horizon, n_lags=14, n_estimators=200)),
            ("LightGBMForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["LightGBMForecaster"]
            ).LightGBMForecaster(horizon=horizon, n_lags=14, n_estimators=200)),
            ("CatBoostForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["CatBoostForecaster"]
            ).CatBoostForecaster(horizon=horizon, n_lags=14, n_estimators=200)),
            ("ProphetForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["ProphetForecaster"]
            ).ProphetForecaster(horizon=horizon)),
            ("NBEATSForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["NBEATSForecaster"]
            ).NBEATSForecaster(horizon=horizon, n_lags=max(28, horizon * 2))),
            ("NHiTSForecaster", lambda: __import__(
                "autotsforecast.models.external", fromlist=["NHiTSForecaster"]
            ).NHiTSForecaster(horizon=horizon, n_lags=max(28, horizon * 2))),
        ]
        for _name, _factory in _optional_accuracy:
            try:
                candidates.append(_factory())
            except (ImportError, Exception):
                pass
        return candidates

    if preset == "zero_shot":
        candidates = []
        for _model_name in [
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
            "autogluon/chronos-2-small",
        ]:
            try:
                from .models.external import Chronos2Forecaster
                candidates.append(Chronos2Forecaster(horizon=horizon, model_name=_model_name))
                break  # Use the first one that loads successfully
            except (ImportError, Exception):
                pass
        if not candidates:
            # Graceful degradation when Chronos is not installed
            candidates = [
                ETSForecaster(horizon=horizon, trend="add", seasonal=None),
                ARIMAForecaster(horizon=horizon),
            ]
        return candidates

    if preset == "intermittent":
        candidates = [
            MovingAverageForecaster(horizon=horizon, window=7),
            ETSForecaster(horizon=horizon, trend=None, seasonal=None),
        ]
        try:
            from .models.external import CrostonForecaster
            candidates.insert(0, CrostonForecaster(horizon=horizon, method="sba"))
            candidates.insert(1, CrostonForecaster(horizon=horizon, method="croston"))
        except ImportError:
            pass
        try:
            from .models.external import ThetaForecaster
            candidates.append(ThetaForecaster(horizon=horizon))
        except ImportError:
            pass
        return candidates

    if preset == "hierarchical":
        candidates = [
            ETSForecaster(horizon=horizon, trend="add", seasonal=None),
            ARIMAForecaster(horizon=horizon),
            MovingAverageForecaster(horizon=horizon, window=7),
        ]
        try:
            from .models.external import LightGBMForecaster, ThetaForecaster
            candidates.append(LightGBMForecaster(horizon=horizon, n_lags=7, n_estimators=100))
            candidates.append(ThetaForecaster(horizon=horizon))
        except ImportError:
            pass
        return candidates

    # Should never reach here due to guard at the top, but be safe
    return get_default_candidate_models(horizon)


# ---------------------------------------------------------------------------
# Module-level helper for candidate parallelism (must be picklable)
# ---------------------------------------------------------------------------

def _evaluate_candidate_worker(args):
    """Evaluate a single candidate model via backtesting. Module-level for joblib."""
    model, model_name, validator_kwargs, y, X = args
    validator = BacktestValidator(**validator_kwargs)
    cv = validator.run(y, X)
    return model_name, cv


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
        candidate_models: Optional[List[BaseForecaster]] = None,
        metric: str = 'rmse',
        n_splits: int = 5,
        test_size: int = 20,
        window_type: str = 'expanding',
        verbose: bool = True,
        per_series_models: bool = False,
        n_jobs: int = 1,
        # ── v0.6.0 new parameters ─────────────────────────────────────────────
        preset: Optional[str] = None,
        horizon: Optional[int] = None,
        time_limit: Optional[float] = None,
        max_models: Optional[int] = None,
        backtest_mode: str = 'full',
    ):
        # Resolve candidate models from preset if needed
        if preset is not None and candidate_models is not None:
            raise ValueError(
                "Specify either preset= or candidate_models=, not both. "
                "Use preset= for automatic candidate selection, or pass an "
                "explicit candidate_models list."
            )
        if preset is not None and candidate_models is None:
            if horizon is None:
                raise ValueError(
                    "horizon is required when using preset. "
                    "Example: AutoForecaster(preset='balanced', horizon=14)"
                )
            candidate_models = get_preset_models(preset, horizon)

        if not candidate_models:
            raise ValueError(
                "candidate_models must not be empty. "
                "Provide at least one BaseForecaster instance, or use preset=."
            )
        if backtest_mode not in ('full', 'fast', 'last_fold'):
            raise ValueError(
                "backtest_mode must be 'full', 'fast', or 'last_fold'."
            )
        self.candidate_models = candidate_models
        self.metric = metric
        self.n_splits = n_splits
        self.test_size = test_size
        self.window_type = window_type
        self.verbose = verbose
        self.per_series_models = per_series_models
        self.n_jobs = n_jobs
        self.preset = preset
        self.horizon = horizon
        self.time_limit = time_limit
        self.max_models = max_models
        self.backtest_mode = backtest_mode
        
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
        self._per_series_covariates_ = None  # Store per-series covariate mapping

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
        """Clone a candidate model to avoid state bleed between fits.

        Tries deepcopy first; if that fails, falls back to reconstructing
        the model from ``get_params()`` (which every built-in model exposes).
        """
        try:
            return copy.deepcopy(model)
        except Exception as deep_err:
            # Fallback: reconstruct from declared parameters
            if hasattr(model, 'get_params'):
                try:
                    return type(model)(**model.get_params())
                except Exception as param_err:
                    raise TypeError(
                        f"Failed to clone model {model.__class__.__name__} via get_params(). "
                        f"deepcopy error: {deep_err}. get_params error: {param_err}"
                    ) from param_err
            raise TypeError(
                f"Failed to clone model {model.__class__.__name__}. "
                "Implement get_params() or ensure the model supports deepcopy. "
                f"deepcopy error: {deep_err}"
            ) from deep_err
        
    def fit(self, y: pd.DataFrame, X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None) -> 'AutoForecaster':
        """
        Evaluate candidate models and select the best one.
        
        Parameters
        ----------
        y : pd.DataFrame
            Historical time series data
        X : pd.DataFrame or Dict[str, pd.DataFrame], optional
            Exogenous variables. Can be:
            - pd.DataFrame: Same covariates used for all series
            - Dict[str, pd.DataFrame]: Different covariates per series
              Keys are series names, values are covariate DataFrames
            
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
            if self.preset:
                print(f"📦 Preset: {self.preset}")
            # Effective n_splits after backtest_mode
            _eff_splits = {'full': self.n_splits, 'fast': min(2, self.n_splits), 'last_fold': 1}[self.backtest_mode]
            print(f"📈 Backtesting: {_eff_splits} splits, {self.test_size} test size"
                  + (f" (mode={self.backtest_mode})" if self.backtest_mode != 'full' else ""))
            print(f"🎯 Selection metric: {self.metric.upper()}")
            print(f"🔄 Window type: {self.window_type}")
            if self.time_limit:
                print(f"⏱️  Time budget: {self.time_limit}s")
            if self.max_models:
                print(f"🔢 Max models: {self.max_models}")
            if self.per_series_models:
                print(f"🧩 Per-series selection: enabled ({y.shape[1]} models will be selected)")
            print(f"⚡ Parallelism: n_jobs={self.n_jobs}")
            print()
        
        self.feature_names_ = y.columns.tolist()
        self._last_index_ = y.index[-1] if len(y.index) else None
        self._freq_ = self._infer_freq(y.index)

        # Effective splits based on backtest_mode
        _effective_n_splits = {
            'full': self.n_splits,
            'fast': min(2, self.n_splits),
            'last_fold': 1,
        }[self.backtest_mode]

        # Candidate pool respecting max_models budget
        _candidates = (
            self.candidate_models[:self.max_models]
            if self.max_models is not None
            else self.candidate_models
        )

        # Handle per-series covariates
        if isinstance(X, dict):
            self._per_series_covariates_ = X
            # Validate that all series have covariates if dict is provided
            missing_series = set(self.feature_names_) - set(X.keys())
            if missing_series:
                raise ValueError(f"Per-series covariates missing for: {missing_series}. "
                               f"Provide covariates for all series or use a single DataFrame.")
        else:
            self._per_series_covariates_ = None

        # Per-series model selection mode: select a best model for each time series.
        if self.per_series_models:
            from joblib import Parallel, delayed

            def fit_one_series(series_name: str):
                y_single = y[[series_name]]
                
                # Get covariates for this series
                if self._per_series_covariates_ is not None:
                    X_series = self._per_series_covariates_.get(series_name)
                else:
                    X_series = X

                best_score = float('inf') if self.metric != 'r2' else float('-inf')
                best_model = None
                best_name = None
                series_cv_results: Dict[str, Any] = {}
                errors = []  # Track errors for debugging

                for candidate in _candidates:
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
                            n_splits=_effective_n_splits,
                            test_size=self.test_size,
                            window_type=self.window_type,
                        )
                        cv = validator.run(y_single, X_series)
                        series_cv_results[model_name] = cv
                        score = cv[self.metric]
                        is_better = (score < best_score) if self.metric != 'r2' else (score > best_score)
                        if is_better:
                            best_score = score
                            best_model = model
                            best_name = model_name
                    except Exception as ex:
                        errors.append(f"{model_name}: {str(ex)[:100]}")
                        continue

                if best_model is None:
                    error_msg = f"No valid models found for series '{series_name}'."
                    if errors:
                        error_msg += f" Errors: {'; '.join(errors)}"
                    raise ValueError(error_msg)

                # Retrain best model on full series with its covariates
                best_model.fit(y_single, X_series)
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

        # ── Global model selection (non-per-series) ───────────────────────────
        best_score = float('inf') if self.metric != 'r2' else float('-inf')

        _validator_kwargs = dict(
            n_splits=_effective_n_splits,
            test_size=self.test_size,
            window_type=self.window_type,
        )

        # ── Parallel path (n_jobs != 1) ───────────────────────────────────────
        if self.n_jobs != 1 and len(_candidates) > 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                pass
            else:
                if self.verbose:
                    print(f"⚡ Running {len(_candidates)} candidates in parallel "
                          f"(n_jobs={self.n_jobs})...")

                def _eval_one(idx_model):
                    idx, model = idx_model
                    nm = model.__class__.__name__
                    if hasattr(model, 'lags'):
                        nm += f"(lags={model.lags})"
                    elif hasattr(model, 'window'):
                        nm += f"(window={model.window})"
                    try:
                        v = BacktestValidator(model=model, **_validator_kwargs)
                        cv = v.run(y, X)
                        return idx, nm, cv, None
                    except Exception as ex:
                        return idx, nm, None, str(ex)[:150]

                par_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_eval_one)((i, m)) for i, m in enumerate(_candidates)
                )
                for idx, model_name, cv_results, err in par_results:
                    if err is not None:
                        if self.verbose:
                            print(f"   ⚠️ {model_name}: {err}")
                        continue
                    self.cv_results_[model_name] = cv_results
                    score = cv_results[self.metric]
                    if self.verbose:
                        print(f"   {model_name}: {self.metric.upper()}={score:.4f}")
                    is_better = (score < best_score) if self.metric != 'r2' else (score > best_score)
                    if is_better:
                        best_score = score
                        self.best_model_ = _candidates[idx]
                        self.best_model_name_ = model_name

                if self.best_model_ is None:
                    raise ValueError("No valid models found. All candidates failed.")

                if self.verbose:
                    print()
                    print("=" * 80)
                    print(f"🏆 BEST MODEL SELECTED: {self.best_model_name_}")
                    print(f"   Performance: {self.metric.upper()} = {best_score:.4f}")
                    print("=" * 80)
                    print()
                    print("🔄 Retraining best model on full dataset...")

                self.best_model_.fit(y, X)
                self.is_fitted = True
                if self.verbose:
                    print("✅ Training complete!")
                return self

        # ── Sequential path (n_jobs == 1 or joblib unavailable) ──────────────
        _start_time = time.monotonic()
        for i, model in enumerate(_candidates, 1):
            # Time-budget check (between models)
            if self.time_limit is not None:
                elapsed = time.monotonic() - _start_time
                if elapsed >= self.time_limit:
                    if self.verbose:
                        print(f"⏱️  Time limit ({self.time_limit}s) reached after "
                              f"{i - 1} model(s). Using best found so far.")
                    break

            model_name = model.__class__.__name__
            if hasattr(model, 'lags'):
                model_name += f"(lags={model.lags})"
            elif hasattr(model, 'window'):
                model_name += f"(window={model.window})"

            if self.verbose:
                print(f"[{i}/{len(_candidates)}] Testing {model_name}...")

            try:
                validator = BacktestValidator(model=model, **_validator_kwargs)
                cv_results = validator.run(y, X)
                self.cv_results_[model_name] = cv_results
                score = cv_results[self.metric]

                if self.verbose:
                    print(f"   {self.metric.upper()}: {score:.4f}")

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
    
    def fit_forecast(
        self,
        y: pd.DataFrame,
        X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        X_future: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    ) -> pd.DataFrame:
        """Fit and generate forecasts in one step.

        Parameters
        ----------
        y : pd.DataFrame
            Historical time series data.
        X : pd.DataFrame or dict, optional
            Historical exogenous variables used during fitting.
        X_future : pd.DataFrame or dict, optional
            Future exogenous variables for the forecast horizon.

        Returns
        -------
        pd.DataFrame
            Forecasted values.

        Example
        -------
        >>> forecasts = AutoForecaster(preset='balanced', horizon=14).fit_forecast(y_train)
        """
        self.fit(y, X)
        return self.forecast(X_future)

    def forecast(self, X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None) -> pd.DataFrame:
        """
        Generate forecasts using the best model.
        
        Parameters
        ----------
        X : pd.DataFrame or Dict[str, pd.DataFrame], optional
            Future exogenous variables. Can be:
            - pd.DataFrame: Same covariates for all series
            - Dict[str, pd.DataFrame]: Different covariates per series
              Keys are series names, values are covariate DataFrames
            
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
            
            # Handle per-series covariates for prediction
            X_dict = {}
            if isinstance(X, dict):
                X_dict = X
            elif self._per_series_covariates_ is not None and X is None:
                # Use the same per-series structure but no future values provided
                pass
            else:
                # Single X for all series
                pass
            
            for col, model in self.best_models_.items():
                # Get covariates for this series
                if isinstance(X, dict):
                    X_series = X.get(col)
                elif self._per_series_covariates_ is not None:
                    # If trained with per-series covariates, must provide them for prediction
                    if X is None:
                        raise ValueError(f"Model for '{col}' was trained with covariates but none provided for prediction. "
                                       f"Provide a dict with covariates for each series.")
                    X_series = X
                else:
                    X_series = X
                
                p = model.predict(X_series)
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
    
    def forecast_with_intervals(
        self,
        y_train: pd.DataFrame,
        X: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        coverage: Union[float, List[float]] = 0.95,
        method: str = 'conformal',
        n_cal: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts with prediction intervals.
        
        This method provides uncertainty quantification for forecasts using
        various interval estimation methods.
        
        Parameters
        ----------
        y_train : pd.DataFrame
            Training data used for calibrating prediction intervals
        X : pd.DataFrame or Dict[str, pd.DataFrame], optional
            Future exogenous variables for forecasting
        coverage : float or list of float, default=0.95
            Target coverage probability (e.g., 0.95 for 95% intervals)
            Can be a list for multiple intervals: [0.80, 0.95]
        method : str, default='conformal'
            Method for generating intervals:
            - 'conformal': Conformal prediction (recommended, distribution-free)
            - 'residual': Normal approximation from residuals
            - 'empirical': Empirical quantiles from residuals
        n_cal : int, optional
            Size of calibration set. If None, uses min(len(y_train)//4, 100)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'point': Point forecasts (pd.DataFrame)
            - 'lower_{coverage}': Lower bounds for each coverage level
            - 'upper_{coverage}': Upper bounds for each coverage level
            
        Examples
        --------
        >>> auto = AutoForecaster(candidate_models=models)
        >>> auto.fit(y_train)
        >>> 
        >>> # Get forecasts with 80% and 95% prediction intervals
        >>> results = auto.forecast_with_intervals(
        ...     y_train,
        ...     coverage=[0.80, 0.95]
        ... )
        >>> 
        >>> print(results['point'])      # Point forecasts
        >>> print(results['lower_95'])   # Lower bound of 95% interval
        >>> print(results['upper_95'])   # Upper bound of 95% interval
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before forecasting. Call fit() first.")
        
        from .uncertainty.intervals import PredictionIntervals
        
        if self.verbose:
            print("\n🔮 Generating forecasts with prediction intervals...")
            print(f"   Method: {method}")
            coverage_list = [coverage] if isinstance(coverage, (int, float)) else coverage
            print(f"   Coverage levels: {[f'{c*100:.0f}%' for c in coverage_list]}")
        
        # Create prediction interval estimator
        pi = PredictionIntervals(
            method=method,
            coverage=coverage
        )
        
        # Fit on training data
        if self.verbose:
            print("   Calibrating intervals...")
        pi.fit(self, y_train, X, n_cal=n_cal)
        
        # Generate point forecasts
        forecasts = self.forecast(X)
        
        # Generate intervals
        if self.verbose:
            print("   Computing intervals...")
        intervals = pi.predict(forecasts)
        
        if self.verbose:
            print("✅ Forecasts with intervals generated!")
        
        return intervals

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of model selection and forecasting.
        
        Returns
        -------
        summary : dict
            Dictionary containing:
            - best_model: Name of best model
            - best_score: Performance score (None in per-series mode)
            - all_results: Results for all models
            - forecast_summary: Summary statistics of forecasts
        """
        if not self.is_fitted:
            raise ValueError("Forecaster must be fitted before getting summary.")

        # Per-series mode: results are stored per-series, not globally
        if self.per_series_models:
            summary = {
                'best_model': 'per-series',
                'best_score': None,
                'selection_metric': self.metric,
                'backtesting_config': {
                    'n_splits': self.n_splits,
                    'test_size': self.test_size,
                    'window_type': self.window_type
                },
                'per_series_models': dict(self.best_model_names_ or {}),
                'all_results': self.cv_results_by_series_ or {}
            }
        else:
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
        if summary['best_score'] is not None:
            print(f"   {summary['selection_metric'].upper()}: {summary['best_score']:.4f}")
        
        print(f"\n📊 Backtesting Configuration:")
        print(f"   Splits: {summary['backtesting_config']['n_splits']}")
        print(f"   Test size: {summary['backtesting_config']['test_size']}")
        print(f"   Window: {summary['backtesting_config']['window_type']}")
        
        if 'per_series_models' in summary:
            print(f"\n🧩 Per-Series Models Selected ({len(summary['per_series_models'])} series):")
            for j, (series, model_name) in enumerate(list(summary['per_series_models'].items())[:10], 1):
                print(f"   {j}. {series}: {model_name}")
            if len(summary['per_series_models']) > 10:
                print(f"   ... ({len(summary['per_series_models'])} total)")
        elif summary['all_results']:
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

    # ── v0.6.0 additions ─────────────────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        """Return a structured model-selection report.

        Works in both global and per-series modes.

        Returns
        -------
        dict
            Keys include ``best_model``, ``metric``, ``n_candidates_evaluated``,
            ``backtest_config``, ``model_ranking``, ``selection_rationale``,
            ``forecast_horizon``, and ``preset``.
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before get_report().")

        is_lower_better = self.metric != 'r2'

        report: Dict[str, Any] = {
            'version': '0.6.0',
            'preset': self.preset,
            'best_model': self.best_model_name_,
            'metric': self.metric,
            'n_candidates_evaluated': len(self.cv_results_) if not self.per_series_models else len(self.candidate_models),
            'backtest_config': {
                'n_splits': self.n_splits,
                'backtest_mode': self.backtest_mode,
                'test_size': self.test_size,
                'window_type': self.window_type,
                'n_jobs': self.n_jobs,
                'time_limit': self.time_limit,
                'max_models': self.max_models,
            },
            'model_ranking': [],
            'selection_rationale': '',
            'forecast_horizon': None,
        }

        # Model ranking (global mode only)
        if not self.per_series_models and self.cv_results_:
            ranked = sorted(
                self.cv_results_.items(),
                key=lambda x: x[1][self.metric],
                reverse=not is_lower_better,
            )
            report['model_ranking'] = [
                {
                    'rank': i + 1,
                    'model': name,
                    self.metric: metrics[self.metric],
                    'rmse': metrics.get('rmse'),
                    'mae': metrics.get('mae'),
                    'mape': metrics.get('mape'),
                    'r2': metrics.get('r2'),
                }
                for i, (name, metrics) in enumerate(ranked)
            ]
            if len(ranked) >= 2:
                best_score = ranked[0][1][self.metric]
                second_score = ranked[1][1][self.metric]
                if is_lower_better:
                    pct = (second_score - best_score) / (abs(second_score) + 1e-10) * 100
                else:
                    pct = (best_score - second_score) / (abs(second_score) + 1e-10) * 100
                report['selection_rationale'] = (
                    f"{ranked[0][0]} outperformed {ranked[1][0]} "
                    f"by {pct:.1f}% on {self.metric.upper()}."
                )

        # Per-series summary
        if self.per_series_models and self.best_model_names_:
            from collections import Counter
            counts = Counter(self.best_model_names_.values())
            report['per_series_model_distribution'] = dict(counts)
            report['selection_rationale'] = (
                f"Per-series selection across {len(self.best_model_names_)} series. "
                f"Most common: {counts.most_common(1)[0]}."
            )

        # Forecast horizon
        if self.best_model_ is not None:
            report['forecast_horizon'] = getattr(self.best_model_, 'horizon', None)
        elif self.best_models_:
            first_m = next(iter(self.best_models_.values()))
            report['forecast_horizon'] = getattr(first_m, 'horizon', None)

        return report

    def print_report(self):
        """Print a formatted model-selection report to stdout."""
        r = self.get_report()
        print("\n" + "=" * 70)
        print("AUTOFORECASTER REPORT  (v{})".format(r.get('version', '?')))
        print("=" * 70)
        if r.get('preset'):
            print(f"  Preset           : {r['preset']}")
        print(f"  Best model       : {r['best_model']}")
        print(f"  Metric           : {r['metric'].upper()}")
        print(f"  Horizon          : {r['forecast_horizon']}")
        print(f"  Candidates eval  : {r['n_candidates_evaluated']}")
        cfg = r['backtest_config']
        print(f"  Backtest         : {cfg['n_splits']} splits, "
              f"mode={cfg['backtest_mode']}, test_size={cfg['test_size']}")
        print(f"  n_jobs           : {cfg['n_jobs']}")
        if r.get('selection_rationale'):
            print(f"\n  Rationale  : {r['selection_rationale']}")
        if r.get('model_ranking'):
            print(f"\n  {'Rank':<5} {'Model':<35} {r['metric'].upper():<12} RMSE")
            print("  " + "-" * 60)
            for entry in r['model_ranking']:
                marker = "🏆" if entry['rank'] == 1 else "  "
                score = entry.get(r['metric'], float('nan'))
                rmse = entry.get('rmse') or float('nan')
                print(f"  {marker} {entry['rank']:<3} {entry['model']:<35} {score:<12.4f} {rmse:.4f}")
        if r.get('per_series_model_distribution'):
            print(f"\n  Per-series model distribution:")
            for model, cnt in sorted(r['per_series_model_distribution'].items(),
                                     key=lambda x: -x[1]):
                print(f"    {model}: {cnt} series")
        print("=" * 70)

    @staticmethod
    def profile_data(y: pd.DataFrame):
        """Profile a dataset and return a recommended forecasting strategy.

        Parameters
        ----------
        y : pd.DataFrame
            Training time series.

        Returns
        -------
        ProfileResult

        Example
        -------
        >>> result = AutoForecaster.profile_data(y_train)
        >>> result.print_summary()
        >>> auto = AutoForecaster(preset=result.recommended_preset, horizon=14)
        """
        from .utils.profiler import DatasetProfiler
        return DatasetProfiler().profile(y)

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

    def to_structured(self, forecasts: Optional[pd.DataFrame] = None):
        """
        Return a structured ``ForecastResult`` Pydantic model.

        Enables machine-readable output for agent frameworks, LangChain,
        OpenAI function calling, and the MCP server.

        Parameters
        ----------
        forecasts : pd.DataFrame, optional
            Forecast DataFrame from ``forecast()``. If ``None``,
            uses ``self.forecasts_`` if available.

        Returns
        -------
        ForecastResult
            Pydantic model with series names, horizon, dates, values,
            best model name, metric, and CV score.

        Example
        -------
        >>> auto.fit(y_train)
        >>> fc = auto.forecast()
        >>> result = auto.to_structured(fc)
        >>> print(result.model_dump_json())
        """
        from autotsforecast.schemas import ForecastResult

        fc = forecasts if forecasts is not None else self.forecasts_
        if fc is None:
            raise ValueError("No forecasts available. Call forecast() first or pass forecasts=.")

        dates = [str(d) for d in fc.index.tolist()]
        values = {col: fc[col].tolist() for col in fc.columns}

        # Best model name
        if self.per_series_models and self.best_model_names_:
            best_model: Any = dict(self.best_model_names_)
        else:
            best_model = self.best_model_name_ or "unknown"

        # CV score
        metric_value = None
        if not self.per_series_models and self.best_model_name_ and self.cv_results_:
            metric_value = self.cv_results_.get(self.best_model_name_, {}).get(self.metric)

        return ForecastResult(
            series_names=list(fc.columns),
            horizon=len(fc),
            dates=dates,
            values=values,
            best_model=best_model,
            metric=self.metric,
            metric_value=metric_value,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def quick_forecast(
    y: pd.DataFrame,
    horizon: int,
    preset: str = 'balanced',
    metric: str = 'rmse',
    n_splits: int = 3,
    test_size: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """One-liner forecasting: auto-select models and generate forecasts.

    Parameters
    ----------
    y : pd.DataFrame
        Historical time series data with a DatetimeIndex (recommended) or
        integer index.
    horizon : int
        Number of future periods to forecast.
    preset : str, default='balanced'
        Model preset — one of ``'fast'``, ``'balanced'``, ``'accuracy'``,
        ``'zero_shot'``, ``'intermittent'``, ``'hierarchical'``.
    metric : str, default='rmse'
        Metric used for automatic model selection.
    n_splits : int, default=3
        Number of backtesting splits.
    test_size : int, optional
        Backtest window size per split.  Defaults to ``horizon``.
    n_jobs : int, default=-1
        Number of parallel workers.  ``-1`` uses all available CPU cores.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Forecasted values.  Index is a future ``DatetimeIndex`` when *y*
        had one, otherwise a ``RangeIndex``.

    Example
    -------
    >>> from autotsforecast import quick_forecast
    >>> forecasts = quick_forecast(y_train, horizon=14)
    >>> forecasts = quick_forecast(y_train, horizon=7, preset='fast', verbose=False)
    """
    auto = AutoForecaster(
        preset=preset,
        horizon=horizon,
        metric=metric,
        n_splits=n_splits,
        test_size=test_size if test_size is not None else horizon,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return auto.fit_forecast(y)


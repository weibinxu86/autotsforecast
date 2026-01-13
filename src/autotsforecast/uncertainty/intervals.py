"""
Prediction intervals and uncertainty quantification for time series forecasting.

This module provides multiple methods for generating prediction intervals:
- Conformal prediction (distribution-free, model-agnostic)
- Quantile regression
- Bootstrap
- Residual-based intervals

Reference:
- Conformal Prediction: Vovk, V., Gammerman, A., & Shafer, G. (2005).
  Algorithmic Learning in a Random World.
- EnbPI: Xu, C., & Xie, Y. (2021). Conformal prediction interval for 
  dynamic time-series.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple, Any
from scipy import stats
import warnings


class PredictionIntervals:
    """
    Generate prediction intervals for time series forecasts.
    
    Supports multiple methods for uncertainty quantification:
    - 'conformal': Conformal prediction (distribution-free)
    - 'residual': Based on historical residual distribution
    - 'bootstrap': Bootstrap resampling
    - 'quantile': Quantile regression (requires model support)
    
    Parameters
    ----------
    method : str, default='conformal'
        Method for generating intervals:
        - 'conformal': Conformal prediction intervals
        - 'residual': Normal approximation from residuals
        - 'bootstrap': Bootstrap prediction intervals
        - 'empirical': Empirical quantiles from residuals
        
    coverage : float or list of float, default=0.95
        Target coverage probability. Can be a single value or list
        of values (e.g., [0.80, 0.95] for 80% and 95% intervals).
        
    n_bootstrap : int, default=100
        Number of bootstrap samples (only used for bootstrap method).
        
    conformity_score : str, default='absolute'
        Type of conformity score for conformal prediction:
        - 'absolute': |y - y_hat|
        - 'normalized': |y - y_hat| / sigma (if uncertainty available)
        - 'signed': y - y_hat (asymmetric intervals)
        
    Examples
    --------
    >>> from autotsforecast.uncertainty import PredictionIntervals
    >>> from autotsforecast import AutoForecaster
    >>> 
    >>> # Fit forecaster
    >>> auto = AutoForecaster(candidate_models=models)
    >>> auto.fit(y_train)
    >>> 
    >>> # Create prediction intervals
    >>> pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
    >>> pi.fit(auto, y_train)
    >>> 
    >>> # Generate forecasts with intervals
    >>> forecasts = auto.forecast()
    >>> intervals = pi.predict(forecasts)
    >>> 
    >>> # Access intervals
    >>> print(intervals['lower_95'])
    >>> print(intervals['upper_95'])
    """
    
    def __init__(
        self,
        method: str = 'conformal',
        coverage: Union[float, List[float]] = 0.95,
        n_bootstrap: int = 100,
        conformity_score: str = 'absolute'
    ):
        self.method = method
        self.coverage = [coverage] if isinstance(coverage, (int, float)) else list(coverage)
        self.n_bootstrap = n_bootstrap
        self.conformity_score = conformity_score
        
        # Validate
        for cov in self.coverage:
            if not 0 < cov < 1:
                raise ValueError(f"Coverage must be between 0 and 1, got {cov}")
        
        # State
        self.is_fitted_ = False
        self.residuals_ = None
        self.conformity_scores_ = None
        self.quantiles_ = {}
        self._forecaster_ = None
        
    def fit(
        self,
        forecaster,
        y: pd.DataFrame,
        X: Optional[pd.DataFrame] = None,
        n_cal: Optional[int] = None
    ) -> 'PredictionIntervals':
        """
        Fit the prediction interval estimator.
        
        For conformal prediction, this computes conformity scores on a
        calibration set (last portion of training data).
        
        Parameters
        ----------
        forecaster : BaseForecaster or AutoForecaster
            Fitted forecaster model
        y : pd.DataFrame
            Historical target data
        X : pd.DataFrame, optional
            Historical covariates
        n_cal : int, optional
            Size of calibration set for conformal prediction.
            Default is min(len(y)//4, 100).
            
        Returns
        -------
        self
        """
        self._forecaster_ = forecaster
        
        # Determine calibration size
        if n_cal is None:
            n_cal = min(len(y) // 4, 100)
        n_cal = max(n_cal, 10)  # Minimum calibration size
        
        if n_cal >= len(y):
            warnings.warn(f"Calibration size ({n_cal}) is too large for data size ({len(y)}). "
                         "Using 25% of data.")
            n_cal = len(y) // 4
        
        # Split data
        y_train = y.iloc[:-n_cal]
        y_cal = y.iloc[-n_cal:]
        X_train = X.iloc[:-n_cal] if X is not None else None
        X_cal = X.iloc[-n_cal:] if X is not None else None
        
        if self.method == 'conformal':
            self._fit_conformal(forecaster, y_train, y_cal, X_train, X_cal)
        elif self.method == 'residual':
            self._fit_residual(forecaster, y_train, y_cal, X_train, X_cal)
        elif self.method == 'bootstrap':
            self._fit_bootstrap(forecaster, y, X)
        elif self.method == 'empirical':
            self._fit_empirical(forecaster, y_train, y_cal, X_train, X_cal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted_ = True
        return self
    
    def _fit_conformal(
        self,
        forecaster,
        y_train: pd.DataFrame,
        y_cal: pd.DataFrame,
        X_train: Optional[pd.DataFrame],
        X_cal: Optional[pd.DataFrame]
    ):
        """Fit conformal prediction intervals."""
        import copy
        
        # Clone and refit model on training data only
        try:
            model = copy.deepcopy(forecaster)
        except:
            model = forecaster
        
        # For simplicity, compute one-step-ahead predictions on calibration set
        # by rolling through the calibration period
        predictions_list = []
        actuals_list = []
        
        # Rolling prediction through calibration set
        y_combined = pd.concat([y_train, y_cal])
        
        for i in range(len(y_cal)):
            train_end = len(y_train) + i
            y_fit = y_combined.iloc[:train_end]
            y_actual = y_combined.iloc[train_end:train_end + 1]
            
            X_fit = None
            X_pred = None
            if X_train is not None and X_cal is not None:
                X_combined = pd.concat([X_train, X_cal])
                X_fit = X_combined.iloc[:train_end]
                X_pred = X_combined.iloc[train_end:train_end + 1]
            
            try:
                # Quick refit and predict
                model_copy = copy.deepcopy(forecaster)
                original_horizon = getattr(model_copy, 'horizon', 1)
                if hasattr(model_copy, 'horizon'):
                    model_copy.horizon = 1
                model_copy.fit(y_fit, X_fit)
                pred = model_copy.predict(X_pred)
                
                if isinstance(pred, pd.DataFrame):
                    predictions_list.append(pred.values.flatten())
                else:
                    predictions_list.append(np.array(pred).flatten())
                actuals_list.append(y_actual.values.flatten())
            except Exception as e:
                continue
        
        if len(predictions_list) == 0:
            # Fallback: use simple residual estimation
            warnings.warn("Conformal calibration failed, falling back to residual method")
            self._fit_residual(forecaster, y_train, y_cal, X_train, X_cal)
            return
        
        predictions = np.array(predictions_list)
        actuals = np.array(actuals_list)
        
        # Compute conformity scores
        if self.conformity_score == 'absolute':
            self.conformity_scores_ = np.abs(actuals - predictions)
        elif self.conformity_score == 'signed':
            self.conformity_scores_ = actuals - predictions
        else:
            self.conformity_scores_ = np.abs(actuals - predictions)
        
        # Compute quantiles for each coverage level
        for cov in self.coverage:
            alpha = 1 - cov
            # Conformal quantile with finite sample correction
            n = len(self.conformity_scores_)
            q = np.ceil((n + 1) * (1 - alpha / 2)) / n
            q = min(q, 1.0)
            
            self.quantiles_[cov] = np.quantile(
                self.conformity_scores_, 
                q,
                axis=0
            )
    
    def _fit_residual(
        self,
        forecaster,
        y_train: pd.DataFrame,
        y_cal: pd.DataFrame,
        X_train: Optional[pd.DataFrame],
        X_cal: Optional[pd.DataFrame]
    ):
        """Fit residual-based intervals assuming normal distribution."""
        import copy
        
        # Get fitted values on calibration set
        try:
            model = copy.deepcopy(forecaster)
        except:
            model = forecaster
        
        # Compute residuals using cross-validation style predictions
        all_residuals = []
        
        y_combined = pd.concat([y_train, y_cal])
        X_combined = pd.concat([X_train, X_cal]) if X_train is not None else None
        
        for i in range(len(y_cal)):
            train_end = len(y_train) + i
            y_fit = y_combined.iloc[:train_end]
            y_actual = y_combined.iloc[train_end:train_end + 1]
            
            X_fit = X_combined.iloc[:train_end] if X_combined is not None else None
            X_pred = X_combined.iloc[train_end:train_end + 1] if X_combined is not None else None
            
            try:
                model_copy = copy.deepcopy(forecaster)
                if hasattr(model_copy, 'horizon'):
                    model_copy.horizon = 1
                model_copy.fit(y_fit, X_fit)
                pred = model_copy.predict(X_pred)
                
                if isinstance(pred, pd.DataFrame):
                    pred_vals = pred.values.flatten()
                else:
                    pred_vals = np.array(pred).flatten()
                
                residual = y_actual.values.flatten() - pred_vals
                all_residuals.append(residual)
            except:
                continue
        
        if len(all_residuals) == 0:
            # Fallback: assume some default std
            warnings.warn("Could not compute residuals, using default uncertainty")
            self.residuals_ = np.ones((1, y_train.shape[1])) * y_train.std().values * 0.1
        else:
            self.residuals_ = np.array(all_residuals)
        
        # Compute standard deviation per series
        self._residual_std_ = np.std(self.residuals_, axis=0)
        
        # Store z-scores for each coverage
        for cov in self.coverage:
            z = stats.norm.ppf((1 + cov) / 2)
            self.quantiles_[cov] = z * self._residual_std_
    
    def _fit_bootstrap(self, forecaster, y: pd.DataFrame, X: Optional[pd.DataFrame]):
        """Fit bootstrap prediction intervals."""
        # Store data for bootstrap resampling during prediction
        self._bootstrap_y_ = y
        self._bootstrap_X_ = X
    
    def _fit_empirical(
        self,
        forecaster,
        y_train: pd.DataFrame,
        y_cal: pd.DataFrame,
        X_train: Optional[pd.DataFrame],
        X_cal: Optional[pd.DataFrame]
    ):
        """Fit empirical quantile intervals."""
        # Same as residual but use empirical quantiles
        self._fit_residual(forecaster, y_train, y_cal, X_train, X_cal)
        
        if self.residuals_ is not None and len(self.residuals_) > 0:
            for cov in self.coverage:
                lower_q = (1 - cov) / 2
                upper_q = (1 + cov) / 2
                self.quantiles_[cov] = {
                    'lower': np.quantile(self.residuals_, lower_q, axis=0),
                    'upper': np.quantile(self.residuals_, upper_q, axis=0)
                }
    
    def predict(
        self,
        forecasts: pd.DataFrame,
        X: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate prediction intervals for forecasts.
        
        Parameters
        ----------
        forecasts : pd.DataFrame
            Point forecasts from the model
        X : pd.DataFrame, optional
            Future covariates (for bootstrap method)
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'point': Point forecasts
            - 'lower_{coverage}': Lower bound for each coverage
            - 'upper_{coverage}': Upper bound for each coverage
        """
        if not self.is_fitted_:
            raise ValueError("PredictionIntervals must be fitted first")
        
        result = {'point': forecasts}
        
        for cov in self.coverage:
            cov_pct = int(cov * 100)
            
            if self.method == 'conformal':
                width = self.quantiles_[cov]
                if len(width.shape) == 1:
                    width = np.tile(width, (len(forecasts), 1))
                elif len(width) < len(forecasts):
                    width = np.tile(width[0], (len(forecasts), 1))
                
                lower = forecasts.values - width[:len(forecasts)]
                upper = forecasts.values + width[:len(forecasts)]
                
            elif self.method in ['residual', 'empirical']:
                if isinstance(self.quantiles_.get(cov), dict):
                    # Empirical quantiles
                    lower_q = self.quantiles_[cov]['lower']
                    upper_q = self.quantiles_[cov]['upper']
                    lower = forecasts.values + np.tile(lower_q, (len(forecasts), 1))
                    upper = forecasts.values + np.tile(upper_q, (len(forecasts), 1))
                else:
                    # Normal-based
                    width = self.quantiles_[cov]
                    lower = forecasts.values - np.tile(width, (len(forecasts), 1))
                    upper = forecasts.values + np.tile(width, (len(forecasts), 1))
                    
            elif self.method == 'bootstrap':
                # Bootstrap requires refitting - simplified version
                width = forecasts.std().values * stats.norm.ppf((1 + cov) / 2)
                lower = forecasts.values - width
                upper = forecasts.values + width
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            result[f'lower_{cov_pct}'] = pd.DataFrame(
                lower, index=forecasts.index, columns=forecasts.columns
            )
            result[f'upper_{cov_pct}'] = pd.DataFrame(
                upper, index=forecasts.index, columns=forecasts.columns
            )
        
        return result


class ConformalPredictor:
    """
    Conformal prediction for time series with EnbPI-style updates.
    
    Implements the Ensemble batch Prediction Intervals (EnbPI) approach
    for online conformal prediction with adaptive intervals.
    
    Parameters
    ----------
    coverage : float, default=0.95
        Target coverage probability
        
    update_interval : int, default=1
        How often to update conformity scores with new observations
        
    Examples
    --------
    >>> from autotsforecast.uncertainty import ConformalPredictor
    >>> 
    >>> # Create conformal predictor
    >>> cp = ConformalPredictor(coverage=0.95)
    >>> cp.fit(model, y_train)
    >>> 
    >>> # Get initial prediction with interval
    >>> pred, lower, upper = cp.predict(model, X_future)
    >>> 
    >>> # Update with actual observation
    >>> cp.update(y_actual)
    >>> 
    >>> # Next prediction will have updated intervals
    >>> pred2, lower2, upper2 = cp.predict(model, X_future2)
    """
    
    def __init__(
        self,
        coverage: float = 0.95,
        update_interval: int = 1
    ):
        self.coverage = coverage
        self.update_interval = update_interval
        
        # State
        self.is_fitted_ = False
        self.conformity_scores_ = []
        self._step_count_ = 0
        self._current_quantile_ = None
        
    def fit(
        self,
        forecaster,
        y: pd.DataFrame,
        X: Optional[pd.DataFrame] = None,
        n_cal: Optional[int] = None
    ) -> 'ConformalPredictor':
        """
        Initialize conformal predictor with calibration data.
        
        Parameters
        ----------
        forecaster : fitted forecaster
            The underlying forecasting model
        y : pd.DataFrame
            Historical data for calibration
        X : pd.DataFrame, optional
            Historical covariates
        n_cal : int, optional
            Number of calibration points
            
        Returns
        -------
        self
        """
        import copy
        
        if n_cal is None:
            n_cal = min(len(y) // 4, 50)
        
        n_cal = max(n_cal, 10)
        
        # Compute initial conformity scores
        y_train = y.iloc[:-n_cal]
        y_cal = y.iloc[-n_cal:]
        X_train = X.iloc[:-n_cal] if X is not None else None
        X_cal = X.iloc[-n_cal:] if X is not None else None
        
        for i in range(len(y_cal)):
            train_end = len(y_train) + i
            y_fit = pd.concat([y_train, y_cal.iloc[:i]]) if i > 0 else y_train
            y_actual = y_cal.iloc[i:i+1]
            
            X_fit = None
            X_pred = None
            if X_train is not None:
                X_fit = pd.concat([X_train, X_cal.iloc[:i]]) if i > 0 else X_train
                X_pred = X_cal.iloc[i:i+1]
            
            try:
                model = copy.deepcopy(forecaster)
                if hasattr(model, 'horizon'):
                    model.horizon = 1
                model.fit(y_fit, X_fit)
                pred = model.predict(X_pred)
                
                if isinstance(pred, pd.DataFrame):
                    pred = pred.values.flatten()
                
                score = np.abs(y_actual.values.flatten() - pred)
                self.conformity_scores_.extend(score.tolist())
            except:
                continue
        
        self._update_quantile()
        self.is_fitted_ = True
        return self
    
    def _update_quantile(self):
        """Update the current quantile based on conformity scores."""
        if len(self.conformity_scores_) > 0:
            n = len(self.conformity_scores_)
            q = np.ceil((n + 1) * self.coverage) / n
            q = min(q, 1.0)
            self._current_quantile_ = np.quantile(self.conformity_scores_, q)
        else:
            self._current_quantile_ = 0
    
    def predict(
        self,
        point_forecast: Union[pd.DataFrame, np.ndarray, float]
    ) -> Tuple[Any, Any, Any]:
        """
        Get prediction interval for a point forecast.
        
        Parameters
        ----------
        point_forecast : pd.DataFrame, np.ndarray, or float
            Point forecast from the model
            
        Returns
        -------
        tuple
            (point_forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted_:
            raise ValueError("ConformalPredictor must be fitted first")
        
        width = self._current_quantile_
        
        if isinstance(point_forecast, pd.DataFrame):
            lower = point_forecast - width
            upper = point_forecast + width
        elif isinstance(point_forecast, np.ndarray):
            lower = point_forecast - width
            upper = point_forecast + width
        else:
            lower = point_forecast - width
            upper = point_forecast + width
        
        return point_forecast, lower, upper
    
    def update(
        self,
        y_actual: Union[pd.DataFrame, np.ndarray, float],
        y_predicted: Union[pd.DataFrame, np.ndarray, float]
    ):
        """
        Update conformity scores with new observation.
        
        Parameters
        ----------
        y_actual : actual observed value
        y_predicted : model's prediction for this point
        """
        if isinstance(y_actual, pd.DataFrame):
            actual = y_actual.values.flatten()
        elif isinstance(y_actual, np.ndarray):
            actual = y_actual.flatten()
        else:
            actual = np.array([y_actual])
        
        if isinstance(y_predicted, pd.DataFrame):
            predicted = y_predicted.values.flatten()
        elif isinstance(y_predicted, np.ndarray):
            predicted = y_predicted.flatten()
        else:
            predicted = np.array([y_predicted])
        
        # Add new conformity score
        new_score = np.abs(actual - predicted)
        self.conformity_scores_.extend(new_score.tolist())
        
        self._step_count_ += 1
        
        # Update quantile periodically
        if self._step_count_ % self.update_interval == 0:
            self._update_quantile()
    
    def get_coverage_history(self) -> float:
        """
        Get the empirical coverage achieved so far.
        
        Returns
        -------
        float
            Empirical coverage rate
        """
        if len(self.conformity_scores_) == 0:
            return self.coverage
        
        # Count how many scores are within the current quantile
        within = sum(s <= self._current_quantile_ for s in self.conformity_scores_)
        return within / len(self.conformity_scores_)

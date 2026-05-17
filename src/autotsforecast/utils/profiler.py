"""
Dataset profiler for automatic time series characterisation and model routing.

Quickly analyses a time series DataFrame to detect intermittency, seasonality,
trend strength, data quality issues, and data volume, then recommends the best
AutoForecaster preset and a curated model shortlist.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ProfileResult:
    """Structured result from :class:`DatasetProfiler`.

    Attributes
    ----------
    n_series : int
        Number of time series columns.
    n_obs : int
        Number of observations (rows).
    freq : str or None
        Detected time frequency (e.g., ``'D'``, ``'W'``, ``'M'``).
    has_datetime_index : bool
        Whether the DataFrame has a DatetimeIndex.
    missing_rate : float
        Fraction of missing values across all series (0–1).
    zero_rate : float
        Fraction of zero values across all series (0–1).
        High values (> 0.3) suggest intermittent demand.
    mean_lag1_autocorr : float
        Average lag-1 autocorrelation across series (−1 to 1).
        High values (> 0.5) indicate strong temporal dependence.
    trend_strength : Dict[str, float]
        Per-series trend strength estimate (0–1, where 1 = pure trend).
    seasonality_detected : bool
        Whether significant seasonality was detected in any series.
    is_intermittent : bool
        True when ``zero_rate > 0.3``.
    is_short : bool
        True when ``n_obs < 50`` — limits model choice.
    recommended_preset : str
        One of ``'fast'``, ``'balanced'``, ``'accuracy'``,
        ``'zero_shot'``, ``'intermittent'``.
    recommended_models : List[str]
        Ordered list of suggested model class names.
    notes : List[str]
        Human-readable observations and warnings.
    """

    n_series: int = 0
    n_obs: int = 0
    freq: Optional[str] = None
    has_datetime_index: bool = False
    missing_rate: float = 0.0
    zero_rate: float = 0.0
    mean_lag1_autocorr: float = 0.0
    trend_strength: Dict[str, float] = field(default_factory=dict)
    seasonality_detected: bool = False
    is_intermittent: bool = False
    is_short: bool = False
    recommended_preset: str = "balanced"
    recommended_models: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "DATASET PROFILE",
            "=" * 60,
            f"  Series        : {self.n_series}",
            f"  Observations  : {self.n_obs}",
            f"  Frequency     : {self.freq or 'unknown'}",
            f"  Missing rate  : {self.missing_rate:.1%}",
            f"  Zero rate     : {self.zero_rate:.1%}"
            + (" (intermittent)" if self.is_intermittent else ""),
            f"  Lag-1 autocorr: {self.mean_lag1_autocorr:.3f}",
            f"  Seasonality   : {'detected' if self.seasonality_detected else 'not detected'}",
            "",
            f"  Recommended preset : {self.recommended_preset}",
            f"  Recommended models : {', '.join(self.recommended_models)}",
        ]
        if self.notes:
            lines += ["", "  Notes:"] + [f"    • {n}" for n in self.notes]
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self):
        print(self.summary())


class DatasetProfiler:
    """Analyse a time series DataFrame and recommend forecasting strategies.

    Usage
    -----
    >>> from autotsforecast.utils.profiler import DatasetProfiler
    >>> profiler = DatasetProfiler()
    >>> result = profiler.profile(y_train)
    >>> result.print_summary()
    >>> print(result.recommended_preset)   # e.g., 'balanced'

    The result can be fed directly into ``get_preset_models`` or inspected to
    guide ``AutoForecaster`` configuration.
    """

    # Thresholds used in routing logic
    INTERMITTENT_ZERO_THRESHOLD = 0.30   # > 30 % zeros → intermittent
    SHORT_SERIES_THRESHOLD = 50          # < 50 obs → "short"
    HIGH_AUTOCORR_THRESHOLD = 0.50       # > 0.5 lag-1 → strong AR structure
    SEASONALITY_CORR_THRESHOLD = 0.40    # |corr(y_t, y_{t-s})| > 0.4 → seasonal
    TREND_STRENGTH_THRESHOLD = 0.40      # detrended variance ratio

    # Common seasonal periods to probe
    _FREQ_PERIODS: Dict[str, int] = {
        "D": 7, "W": 52, "M": 12, "MS": 12, "ME": 12,
        "Q": 4, "QS": 4, "QE": 4, "H": 24, "T": 60, "min": 60,
    }

    def profile(self, y: pd.DataFrame) -> ProfileResult:
        """Profile *y* and return a :class:`ProfileResult`.

        Parameters
        ----------
        y : pd.DataFrame
            Time series with one column per variable.  Index must be
            monotonic (DatetimeIndex preferred).

        Returns
        -------
        ProfileResult
        """
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame.")

        result = ProfileResult()
        result.n_series = y.shape[1]
        result.n_obs = y.shape[0]
        result.has_datetime_index = isinstance(y.index, pd.DatetimeIndex)
        result.notes = []

        # ── Frequency ────────────────────────────────────────────────────────
        if result.has_datetime_index:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result.freq = pd.infer_freq(y.index)
        if result.freq is None and result.has_datetime_index:
            result.notes.append(
                "Could not infer frequency — provide a regular DatetimeIndex for best results."
            )

        # ── Missing values ────────────────────────────────────────────────────
        total_cells = y.size
        result.missing_rate = float(y.isna().sum().sum()) / max(total_cells, 1)
        if result.missing_rate > 0.05:
            result.notes.append(
                f"High missing rate ({result.missing_rate:.1%}). "
                "Consider imputation before forecasting."
            )

        # Work on filled values for subsequent stats
        y_filled = y.ffill().bfill().fillna(0.0)

        # ── Zero rate (intermittency) ─────────────────────────────────────────
        result.zero_rate = float((y_filled == 0).sum().sum()) / max(total_cells, 1)
        result.is_intermittent = result.zero_rate > self.INTERMITTENT_ZERO_THRESHOLD
        if result.is_intermittent:
            result.notes.append(
                f"High zero rate ({result.zero_rate:.1%}) — intermittent demand detected. "
                "Consider CrostonForecaster or SBA method."
            )

        # ── Short series ──────────────────────────────────────────────────────
        result.is_short = result.n_obs < self.SHORT_SERIES_THRESHOLD
        if result.is_short:
            result.notes.append(
                f"Short series ({result.n_obs} observations). "
                "ML models need more data; prefer ETS/ARIMA/Theta or Chronos-2 zero-shot."
            )

        # ── Lag-1 autocorrelation ─────────────────────────────────────────────
        lag1_corrs = []
        for col in y_filled.columns:
            s = y_filled[col].values.astype(float)
            if np.std(s) > 0 and len(s) > 2:
                corr = float(np.corrcoef(s[:-1], s[1:])[0, 1])
                if not np.isnan(corr):
                    lag1_corrs.append(corr)
        result.mean_lag1_autocorr = float(np.mean(lag1_corrs)) if lag1_corrs else 0.0

        # ── Trend strength ────────────────────────────────────────────────────
        for col in y_filled.columns:
            s = y_filled[col].values.astype(float)
            strength = self._trend_strength(s)
            result.trend_strength[col] = strength

        avg_trend = float(np.mean(list(result.trend_strength.values()))) if result.trend_strength else 0.0
        if avg_trend > self.TREND_STRENGTH_THRESHOLD:
            result.notes.append(
                f"Strong trend detected (avg strength {avg_trend:.2f}). "
                "Use trend='add' in ETS or include time-index features."
            )

        # ── Seasonality ───────────────────────────────────────────────────────
        result.seasonality_detected = self._detect_seasonality(y_filled, result.freq)
        if result.seasonality_detected:
            result.notes.append(
                "Seasonal patterns detected. Consider ETS with seasonal='add', "
                "Prophet, ThetaForecaster, or calendar features."
            )

        # ── Data volume for ML ────────────────────────────────────────────────
        enough_for_ml = result.n_obs >= 100
        enough_for_neural = result.n_obs >= 300

        if not enough_for_ml:
            result.notes.append(
                f"Only {result.n_obs} observations — ML models may overfit. "
                "Prefer statistical models or Chronos-2."
            )

        # ── Recommend preset & models ─────────────────────────────────────────
        result.recommended_preset, result.recommended_models = self._recommend(
            result,
            avg_trend=avg_trend,
            enough_for_ml=enough_for_ml,
            enough_for_neural=enough_for_neural,
        )

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _trend_strength(self, s: np.ndarray) -> float:
        """Estimate trend strength as fraction of variance explained by a linear fit."""
        n = len(s)
        if n < 3:
            return 0.0
        t = np.arange(n, dtype=float)
        # Least-squares linear fit
        slope, intercept = np.polyfit(t, s, 1)
        fitted = slope * t + intercept
        residuals = s - fitted
        var_res = np.var(residuals)
        var_tot = np.var(s)
        if var_tot < 1e-10:
            return 0.0
        return float(max(0.0, 1.0 - var_res / var_tot))

    def _detect_seasonality(self, y: pd.DataFrame, freq: Optional[str]) -> bool:
        """Return True if seasonal autocorrelation is significant for any series."""
        # Determine candidate seasonal period
        period = None
        if freq:
            base = "".join(c for c in str(freq) if c.isalpha())
            period = self._FREQ_PERIODS.get(base)
        if period is None:
            period = 7  # default probe for daily data

        if period < 2:
            return False

        for col in y.columns:
            s = y[col].values.astype(float)
            if len(s) <= period or np.std(s) < 1e-10:
                continue
            if len(s) > period:
                corr = float(np.corrcoef(s[:-period], s[period:])[0, 1])
                if not np.isnan(corr) and abs(corr) > self.SEASONALITY_CORR_THRESHOLD:
                    return True
        return False

    def _recommend(
        self,
        r: ProfileResult,
        avg_trend: float,
        enough_for_ml: bool,
        enough_for_neural: bool,
    ):
        """Return (preset, model_list) based on profiled characteristics."""

        # ── Intermittent demand ───────────────────────────────────────────────
        if r.is_intermittent:
            return "intermittent", [
                "CrostonForecaster",
                "ETSForecaster",
                "MovingAverageForecaster",
            ]

        # ── Very short series → zero-shot or statistical only ─────────────────
        if r.is_short:
            return "zero_shot", [
                "Chronos2Forecaster",
                "ETSForecaster",
                "ThetaForecaster",
                "ARIMAForecaster",
            ]

        # ── Enough data for ML, not enough for neural ──────────────────────────
        if enough_for_ml and not enough_for_neural:
            preset = "fast" if r.n_obs < 200 else "balanced"
            models = [
                "ETSForecaster",
                "ThetaForecaster",
                "ARIMAForecaster",
                "ElasticNetForecaster",
                "RandomForestForecaster",
            ]
            if r.seasonality_detected:
                models.insert(0, "ProphetForecaster")
            return preset, models

        # ── Enough for neural ─────────────────────────────────────────────────
        if enough_for_neural:
            models = [
                "LightGBMForecaster",
                "XGBoostForecaster",
                "RandomForestForecaster",
                "ETSForecaster",
                "ThetaForecaster",
                "NBEATSForecaster",
                "NHiTSForecaster",
            ]
            if r.seasonality_detected:
                models.insert(0, "ProphetForecaster")
            return "accuracy", models

        # ── Fallback ──────────────────────────────────────────────────────────
        return "balanced", [
            "ETSForecaster",
            "ARIMAForecaster",
            "ThetaForecaster",
            "RandomForestForecaster",
            "LightGBMForecaster",
        ]

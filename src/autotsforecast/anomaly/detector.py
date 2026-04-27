"""
Anomaly detection for time series data.

Detects anomalies before forecasting to improve model robustness.
Supports multiple methods: zscore, IQR, Isolation Forest, forecast residual.

Usage::

    from autotsforecast.anomaly.detector import AnomalyDetector

    detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
    detector.fit(y_train)
    flags = detector.predict(y_train)   # DataFrame[bool]: True = anomaly
    summary = detector.get_summary()    # AnomalyResult (Pydantic)
    detector.plot(y_train)              # matplotlib figure
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class AnomalyDetector:
    """
    Detect anomalies in time series data.

    Parameters
    ----------
    method : str, default='isolation_forest'
        Detection method:
        - ``'zscore'``: Flag points > ``threshold`` standard deviations from mean.
        - ``'iqr'``: IQR fence (Tukey fences): Q1 - k*IQR, Q3 + k*IQR.
        - ``'isolation_forest'``: sklearn IsolationForest (no extra deps needed).
        - ``'forecast_residual'``: Fit a fast ETS model, flag large residuals.
    contamination : float, default=0.05
        Expected fraction of anomalies (0.0 – 0.5). Used by isolation_forest.
    threshold : float, default=3.0
        Number of standard deviations for zscore method, or IQR multiplier for iqr method.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        threshold: float = 3.0,
    ):
        valid_methods = {"zscore", "iqr", "isolation_forest", "forecast_residual"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        self.method = method
        self.contamination = contamination
        self.threshold = threshold
        self._is_fitted = False
        self._anomaly_flags: Optional[pd.DataFrame] = None
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, y: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the anomaly detector on training data.

        Parameters
        ----------
        y : pd.DataFrame
            Historical time series with DatetimeIndex.

        Returns
        -------
        self
        """
        self._feature_names = y.columns.tolist()
        self._y_train = y.copy()
        self._is_fitted = True
        return self

    def predict(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in ``y``.

        Parameters
        ----------
        y : pd.DataFrame
            Time series data to evaluate.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame with same shape as ``y``.
            ``True`` indicates an anomaly.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        if self.method == "zscore":
            flags = self._zscore(y)
        elif self.method == "iqr":
            flags = self._iqr(y)
        elif self.method == "isolation_forest":
            flags = self._isolation_forest(y)
        elif self.method == "forecast_residual":
            flags = self._forecast_residual(y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._anomaly_flags = flags
        return flags

    def fit_predict(self, y: pd.DataFrame) -> pd.DataFrame:
        """Fit and detect anomalies in one step."""
        return self.fit(y).predict(y)

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _zscore(self, y: pd.DataFrame) -> pd.DataFrame:
        mean = self._y_train.mean()
        std = self._y_train.std().replace(0, 1)
        z = (y - mean) / std
        return z.abs() > self.threshold

    def _iqr(self, y: pd.DataFrame) -> pd.DataFrame:
        q1 = self._y_train.quantile(0.25)
        q3 = self._y_train.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        return (y < lower) | (y > upper)

    def _isolation_forest(self, y: pd.DataFrame) -> pd.DataFrame:
        from sklearn.ensemble import IsolationForest

        flags = pd.DataFrame(False, index=y.index, columns=y.columns)
        for col in y.columns:
            values = y[[col]].dropna()
            if len(values) < 10:
                continue
            clf = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )
            preds = clf.fit_predict(values)
            anomaly_mask = pd.Series(preds == -1, index=values.index)
            flags.loc[anomaly_mask.index, col] = anomaly_mask.values
        return flags

    def _forecast_residual(self, y: pd.DataFrame) -> pd.DataFrame:
        from autotsforecast.models.external import ETSForecaster

        flags = pd.DataFrame(False, index=y.index, columns=y.columns)
        for col in y.columns:
            series = y[[col]].dropna()
            if len(series) < 20:
                continue
            try:
                # Use LOO-style: fit on all but last 20%, evaluate residuals on full
                horizon = max(1, len(series) // 5)
                train = series.iloc[:-horizon]
                model = ETSForecaster(horizon=horizon)
                model.fit(train)
                preds = model.predict()
                actuals = series.iloc[-horizon:]
                residuals = (actuals.values.flatten() - preds[col].values) if col in preds else None
                if residuals is None:
                    continue
                std = np.std(residuals)
                if std == 0:
                    continue
                z = np.abs(residuals) / std
                anomaly_idx = actuals.index[z > self.threshold]
                flags.loc[anomaly_idx, col] = True
            except Exception:
                pass
        return flags

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def get_summary(self):
        """
        Return a structured AnomalyResult summary.

        Returns
        -------
        AnomalyResult
            Pydantic model with anomaly counts and dates per series.
        """
        from autotsforecast.schemas import AnomalyResult

        if self._anomaly_flags is None:
            raise RuntimeError("Call predict() before get_summary().")

        n_anomalies = {}
        anomaly_dates = {}
        for col in self._anomaly_flags.columns:
            mask = self._anomaly_flags[col]
            n_anomalies[col] = int(mask.sum())
            anomaly_dates[col] = [str(d) for d in self._anomaly_flags.index[mask].tolist()]

        return AnomalyResult(
            series_names=list(self._anomaly_flags.columns),
            method=self.method,
            n_anomalies=n_anomalies,
            anomaly_dates=anomaly_dates,
            contamination=self.contamination,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, y: pd.DataFrame, figsize: tuple = (14, 4)):
        """
        Plot the time series with anomalies highlighted in red.

        Parameters
        ----------
        y : pd.DataFrame
            Original data (same index as used in predict).
        figsize : tuple
            Figure size per series panel.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if self._anomaly_flags is None:
            self.predict(y)

        n = len(y.columns)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n))
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, y.columns):
            ax.plot(y.index, y[col], label=col, color="steelblue", linewidth=0.8)
            anomalies = y[col][self._anomaly_flags[col]]
            ax.scatter(anomalies.index, anomalies.values,
                       color="red", zorder=5, label="Anomaly", s=30)
            ax.set_title(f"{col}  [{self.method}]")
            ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

"""
Plain-English insight engine for AutoTSForecast.

Generates human-readable summaries of forecasting results without requiring an LLM.
An optional LLM mode (BYOK) is also supported.

Usage — rule-based (no API key needed)::

    from autotsforecast.nlp.insights import InsightEngine

    engine = InsightEngine(mode='rule_based')
    summary = engine.summarize_forecast(result, y_train)
    risks   = engine.flag_risks(result, intervals)
    report  = engine.generate_report(result, y_train, y_test, intervals)

Usage — LLM-enhanced::

    import openai
    client = openai.OpenAI(api_key="...")

    engine = InsightEngine(mode='llm', llm_client=client, model='gpt-4o')
    summary = engine.summarize_forecast(result, y_train)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model → rationale mapping (rule-based)
# ---------------------------------------------------------------------------

_MODEL_RATIONALE = {
    "MovingAverageForecaster": (
        "Moving Average was selected, suggesting the series is stationary "
        "with no strong trend or seasonality."
    ),
    "ARIMAForecaster": (
        "ARIMA was selected, suggesting the series has autoregressive structure "
        "or requires differencing to achieve stationarity."
    ),
    "ETSForecaster": (
        "Exponential Smoothing (ETS) was selected, suggesting the series has "
        "a smooth trend or level with possible exponential dampening."
    ),
    "RandomForestForecaster": (
        "Random Forest was selected, suggesting the series benefits from "
        "non-linear patterns across lagged features."
    ),
    "XGBoostForecaster": (
        "XGBoost was selected, suggesting the series has complex non-linear "
        "dynamics that tree-based gradient boosting captures well."
    ),
    "ProphetForecaster": (
        "Prophet was selected, suggesting the series has strong seasonal "
        "patterns or holiday effects."
    ),
    "LSTMForecaster": (
        "LSTM was selected, suggesting the series has long-range temporal "
        "dependencies that deep learning models capture."
    ),
    "Chronos2Forecaster": (
        "Chronos-2 (foundation model) was selected and used in zero-shot mode — "
        "no task-specific training was required."
    ),
    "VARForecaster": (
        "VAR was selected, capturing cross-series dependencies through "
        "multivariate autoregression."
    ),
}


def _detect_trend(series: pd.Series, recent_n: int = 30) -> str:
    """Return 'increasing', 'decreasing', or 'stable'."""
    if len(series) < 2:
        return "stable"
    recent = series.iloc[-recent_n:] if len(series) >= recent_n else series
    slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
    rel = abs(slope) / (abs(recent.mean()) + 1e-9)
    if rel < 0.001:
        return "stable"
    return "increasing" if slope > 0 else "decreasing"


def _pct_change(series: pd.Series, forecast: pd.Series) -> float:
    """Mean percentage change from last training value to mean forecast."""
    last = series.iloc[-1] if len(series) > 0 else 0
    if abs(last) < 1e-9:
        return 0.0
    return (forecast.mean() - last) / abs(last) * 100


class InsightEngine:
    """
    Generate plain-English insights from AutoTSForecast results.

    Parameters
    ----------
    mode : str, default='rule_based'
        ``'rule_based'`` — no LLM required; uses heuristics.
        ``'llm'``        — uses an LLM client for richer prose.
    llm_client : optional
        OpenAI / Anthropic compatible client (required when ``mode='llm'``).
    model : str, default='gpt-4o'
        LLM model to use (ignored in rule_based mode).
    """

    def __init__(
        self,
        mode: str = "rule_based",
        llm_client: Any = None,
        model: str = "gpt-4o",
    ):
        if mode not in ("rule_based", "llm"):
            raise ValueError("mode must be 'rule_based' or 'llm'")
        self.mode = mode
        self.llm_client = llm_client
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_forecast(
        self,
        result,  # ForecastResult Pydantic model or dict
        y_train: pd.DataFrame,
        y_test: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate a 2-5 sentence plain-English forecast summary.

        Parameters
        ----------
        result : ForecastResult or dict
            Output from ``AutoForecaster.to_structured(forecasts)``.
        y_train : pd.DataFrame
            Historical training data.
        y_test : pd.DataFrame, optional
            Actual test values, used to compute realised error.

        Returns
        -------
        str
        """
        if hasattr(result, "model_dump"):
            r = result.model_dump()
        else:
            r = result

        if self.mode == "llm":
            return self._llm_summarize(r, y_train, y_test)
        return self._rule_summarize(r, y_train, y_test)

    def summarize_forecast_dataframes(
        self,
        y_train: pd.DataFrame,
        forecasts: pd.DataFrame,
        y_test: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Convenience method — accepts raw DataFrames instead of ForecastResult.
        """
        series_names = list(forecasts.columns)
        values = {col: forecasts[col].tolist() for col in series_names}
        dates = [str(d) for d in forecasts.index.tolist()]
        r = {
            "series_names": series_names,
            "horizon": len(forecasts),
            "dates": dates,
            "values": values,
            "best_model": "AutoForecaster",
            "metric": "rmse",
            "metric_value": None,
        }
        return self._rule_summarize(r, y_train, y_test)

    def flag_risks(
        self,
        result,
        intervals=None,
    ) -> List[str]:
        """
        Return a list of risk flags.

        Flags include:
        - Wide prediction intervals (high uncertainty)
        - Forecast direction reversal vs recent trend
        - Negative forecasts where unlikely
        """
        if hasattr(result, "model_dump"):
            r = result.model_dump()
        else:
            r = result

        risks = []
        values: Dict[str, List[float]] = r.get("values", {})

        for series, forecast_vals in values.items():
            forecast = np.array(forecast_vals)

            # Negative forecasts
            if np.any(forecast < 0):
                risks.append(
                    f"'{series}': forecast contains negative values — verify if negatives are meaningful."
                )

            # Flat forecast (all identical)
            if np.std(forecast) < 1e-9:
                risks.append(
                    f"'{series}': forecast is completely flat — the model may be underfitting."
                )

        if intervals and isinstance(intervals, dict):
            for level in [80, 95]:
                lower_key = f"lower_{level}"
                upper_key = f"upper_{level}"
                if lower_key in intervals and upper_key in intervals:
                    for col in intervals[lower_key].columns if hasattr(intervals[lower_key], "columns") else []:
                        try:
                            width = (
                                intervals[upper_key][col].mean()
                                - intervals[lower_key][col].mean()
                            )
                            mid = abs(intervals[lower_key][col].mean()
                                      + intervals[upper_key][col].mean()) / 2
                            if mid > 0 and width / mid > 1.0:
                                risks.append(
                                    f"'{col}': {level}% prediction interval is very wide "
                                    f"(>{100:.0f}% of point forecast) — high uncertainty."
                                )
                        except Exception:
                            pass

        return risks

    def flag_risks_from_dataframes(
        self,
        y_train: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> List[str]:
        """Convenience version accepting raw DataFrames."""
        values = {col: forecasts[col].tolist() for col in forecasts.columns}
        return self.flag_risks({"values": values})

    def generate_report(
        self,
        result,
        y_train: pd.DataFrame,
        y_test: Optional[pd.DataFrame] = None,
        intervals=None,
    ) -> str:
        """
        Generate a full Markdown report.

        Returns
        -------
        str
            Markdown-formatted report suitable for email, Notion, or Slack.
        """
        if hasattr(result, "model_dump"):
            r = result.model_dump()
        else:
            r = result

        summary = self.summarize_forecast(result, y_train, y_test)
        risks = self.flag_risks(result, intervals)
        model_rationale = self._get_model_rationale(r)

        lines = [
            "# AutoTSForecast Report",
            "",
            "## Summary",
            summary,
            "",
            "## Model Selection Rationale",
        ]
        for series, rationale in model_rationale.items():
            lines.append(f"- **{series}**: {rationale}")

        if risks:
            lines += ["", "## Risk Flags ⚠️"]
            for risk in risks:
                lines.append(f"- {risk}")

        lines += ["", "## Forecast Overview"]
        values: Dict = r.get("values", {})
        if values:
            lines.append("| Series | Mean Forecast | Std | Min | Max |")
            lines.append("|--------|:-------------:|:---:|:---:|:---:|")
            for col, vals in values.items():
                arr = np.array(vals)
                lines.append(
                    f"| {col} | {arr.mean():.2f} | {arr.std():.2f} "
                    f"| {arr.min():.2f} | {arr.max():.2f} |"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_summarize(self, r: dict, y_train: pd.DataFrame, y_test) -> str:
        series_names = r.get("series_names", list(y_train.columns))
        horizon = r.get("horizon", "?")
        best_model = r.get("best_model", "AutoForecaster")
        metric = r.get("metric", "rmse")
        metric_value = r.get("metric_value")
        values: Dict = r.get("values", {})

        parts = []

        # Trend summary per series
        trend_parts = []
        for col in series_names:
            if col in values and col in y_train.columns:
                fc = np.array(values[col])
                pct = _pct_change(y_train[col], pd.Series(fc))
                trend = _detect_trend(y_train[col])
                direction = f"{abs(pct):.1f}% {'increase' if pct > 0 else 'decrease'}" if abs(pct) > 0.5 else "stable"
                trend_parts.append(f"**{col}** is forecast to be {direction} over {horizon} steps "
                                   f"(recent trend: {trend})")

        if trend_parts:
            parts.append("; ".join(trend_parts) + ".")

        # Model selection summary
        if isinstance(best_model, dict):
            model_str = ", ".join(f"{k}: {v}" for k, v in best_model.items())
            parts.append(f"Per-series model selection chose: {model_str}.")
        elif isinstance(best_model, str) and best_model not in ("per-series", "AutoForecaster"):
            parts.append(f"The selected model was **{best_model}**.")

        # CV metric
        if metric_value is not None:
            parts.append(f"Cross-validated {metric.upper()}: **{metric_value:.3f}**.")

        # Test error if available
        if y_test is not None and values:
            errors = []
            for col in series_names:
                if col in values and col in y_test.columns:
                    fc = np.array(values[col])
                    actual = y_test[col].values[:len(fc)]
                    if len(actual) == len(fc):
                        rmse = np.sqrt(np.mean((actual - fc) ** 2))
                        errors.append(f"{col}: RMSE={rmse:.2f}")
            if errors:
                parts.append("Test set errors — " + ", ".join(errors) + ".")

        return " ".join(parts) if parts else "Forecast completed successfully."

    def _get_model_rationale(self, r: dict) -> Dict[str, str]:
        best_model = r.get("best_model", {})
        if isinstance(best_model, dict):
            return {
                series: _MODEL_RATIONALE.get(model, f"{model} was selected.")
                for series, model in best_model.items()
            }
        elif isinstance(best_model, str):
            rationale = _MODEL_RATIONALE.get(best_model, f"{best_model} was selected.")
            return {"all series": rationale}
        return {}

    def _llm_summarize(self, r: dict, y_train: pd.DataFrame, y_test) -> str:
        if self.llm_client is None:
            raise ValueError("llm_client is required when mode='llm'")

        rule_summary = self._rule_summarize(r, y_train, y_test)
        prompt = (
            "You are a senior data scientist presenting forecasting results to a business stakeholder. "
            "Rewrite the following technical summary in clear, concise business language (2-4 sentences). "
            "Focus on what the numbers mean for the business, not the methodology.\n\n"
            f"Technical summary:\n{rule_summary}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Graceful fallback
            return rule_summary + f" (LLM enhancement failed: {e})"

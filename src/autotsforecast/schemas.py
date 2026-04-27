"""
Pydantic structured output models for AutoTSForecast.

These schemas enable machine-readable outputs for agent frameworks,
LangChain, OpenAI function calling, and the MCP server.

All existing DataFrame-returning APIs remain unchanged.
Use `.to_structured()` methods on AutoForecaster / BacktestValidator
to get these structured outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items()}

        def model_dump_json(self) -> str:
            import json
            return json.dumps(self.model_dump(), default=str)

    def Field(*args, **kwargs):  # type: ignore
        return None


class ForecastResult(BaseModel):
    """Structured output for a completed forecast."""
    series_names: List[str] = Field(description="Names of all forecasted series")
    horizon: int = Field(description="Number of steps forecasted")
    dates: List[str] = Field(description="Forecast dates in ISO 8601 format")
    values: Dict[str, List[float]] = Field(description="Mapping of series name to list of forecast values")
    best_model: Union[str, Dict[str, str]] = Field(
        description="Best model name (str for global, dict for per-series)"
    )
    metric: str = Field(description="Metric used for model selection")
    metric_value: Optional[float] = Field(default=None, description="Best model's cross-validated metric score")

    def to_dataframe(self):
        """Convert back to a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.values, index=pd.to_datetime(self.dates))


class FoldResult(BaseModel):
    """Single fold result from backtesting."""
    fold: int
    train_size: int
    test_size: int
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None


class BacktestResult(BaseModel):
    """Structured output for a completed backtesting run."""
    model_name: str = Field(description="Name of the model evaluated")
    n_splits: int = Field(description="Number of cross-validation splits")
    metric: str = Field(description="Primary metric")
    mean_score: float = Field(description="Mean cross-validated score")
    std_score: float = Field(description="Standard deviation of fold scores")
    fold_scores: List[float] = Field(description="Score per fold")
    fold_details: List[FoldResult] = Field(default_factory=list, description="Detailed results per fold")


class IntervalResult(BaseModel):
    """Structured output for prediction intervals."""
    series_names: List[str]
    coverage_levels: List[int] = Field(description="Coverage levels e.g. [80, 95]")
    dates: List[str] = Field(description="Forecast dates in ISO 8601")
    point: Dict[str, List[float]] = Field(description="Point forecasts per series")
    lower: Dict[str, Dict[str, List[float]]] = Field(
        description="Lower bounds: {series: {level: values}}"
    )
    upper: Dict[str, Dict[str, List[float]]] = Field(
        description="Upper bounds: {series: {level: values}}"
    )


class ImportanceResult(BaseModel):
    """Structured output for feature importance / driver analysis."""
    method: str = Field(description="Method used: coefficients, permutation, shap, sensitivity")
    feature_importances: Dict[str, float] = Field(description="Feature name to importance score")
    ranked: List[str] = Field(description="Feature names sorted by importance (descending)")


class AnomalyResult(BaseModel):
    """Structured output from anomaly detection."""
    series_names: List[str]
    method: str
    n_anomalies: Dict[str, int] = Field(description="Number of anomalies detected per series")
    anomaly_dates: Dict[str, List[str]] = Field(description="Dates flagged as anomalies per series")
    contamination: float = Field(description="Expected fraction of anomalies used")


class InsightResult(BaseModel):
    """Structured natural-language insight from the InsightEngine."""
    summary: str = Field(description="2-5 sentence plain-English forecast summary")
    trend: Dict[str, str] = Field(description="Trend direction per series: increasing/stable/decreasing")
    risks: List[str] = Field(default_factory=list, description="Risk flags identified")
    model_rationale: Dict[str, str] = Field(
        default_factory=dict, description="Plain-English model selection rationale per series"
    )
    report_markdown: Optional[str] = Field(default=None, description="Full markdown report")


class ModelInfo(BaseModel):
    """Metadata about a single forecaster."""
    name: str
    class_name: str
    supports_covariates: bool
    requires_extra: Optional[str] = None
    description: str


class ModelCatalog(BaseModel):
    """Catalog of all available forecasting models."""
    models: List[ModelInfo]
    total: int


class RegistryEntry(BaseModel):
    """A single entry in the model registry."""
    name: str
    class_name: str
    horizon: int
    metric: Optional[str] = None
    metric_value: Optional[float] = None
    saved_at: str
    tags: Dict[str, Any] = Field(default_factory=dict)
    filepath: str


def _require_pydantic():
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "pydantic is required for structured outputs. "
            "Install with: pip install 'autotsforecast[mcp]'"
        )

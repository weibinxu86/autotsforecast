"""
LangChain tool wrappers for AutoTSForecast.

Usage::

    from autotsforecast.integrations.langchain_tools import get_autotsforecast_tools
    from langchain.agents import initialize_agent, AgentType
    from langchain_openai import ChatOpenAI

    tools = get_autotsforecast_tools()
    llm = ChatOpenAI(model="gpt-4o")
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
    agent.run("Forecast the next 30 days for this sales data: ...")
"""

import json
from typing import Optional, Type

try:
    from langchain.tools import BaseTool
    from pydantic import BaseModel as LCBaseModel, Field as LCField
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def _require_langchain():
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain is required. Install with: pip install 'autotsforecast[langchain]'"
        )


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

if LANGCHAIN_AVAILABLE:
    class FitAndForecastInput(LCBaseModel):
        csv_data: str = LCField(description="CSV string with datetime index")
        horizon: int = LCField(default=30, description="Steps to forecast")
        metric: str = LCField(default="rmse", description="rmse, mae, or mape")
        n_splits: int = LCField(default=3, description="Backtest splits")
        per_series: bool = LCField(default=True, description="Per-series model selection")

    class BacktestInput(LCBaseModel):
        csv_data: str = LCField(description="CSV training data")
        model_name: str = LCField(default="ETSForecaster")
        horizon: int = LCField(default=30)
        n_splits: int = LCField(default=5)
        test_size: int = LCField(default=20)

    class CSVOnlyInput(LCBaseModel):
        csv_data: str = LCField(description="CSV with datetime index")

    class AnomalyInput(LCBaseModel):
        csv_data: str = LCField(description="CSV training data")
        method: str = LCField(default="isolation_forest")
        contamination: float = LCField(default=0.05)

    class ReconcileInput(LCBaseModel):
        forecast_csv: str = LCField(description="CSV of base forecasts")
        hierarchy: dict = LCField(description='{"total": ["a", "b"]}')


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

class _FitAndForecastTool:
    name = "fit_and_forecast"
    description = (
        "Fit an AutoForecaster on time series CSV data and return forecasts. "
        "Automatically selects the best model per series via walk-forward backtesting. "
        "Input must include 'csv_data' (CSV string with datetime index)."
    )
    if LANGCHAIN_AVAILABLE:
        args_schema: Type[LCBaseModel] = FitAndForecastInput

    def _run(self, csv_data: str, horizon: int = 30, metric: str = "rmse",
             n_splits: int = 3, per_series: bool = True, **kwargs) -> str:
        from autotsforecast.mcp.server import _run_fit_and_forecast
        result = _run_fit_and_forecast(csv_data, horizon, metric, n_splits, per_series)
        return json.dumps(result, default=str)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class _BacktestTool:
    name = "backtest"
    description = "Walk-forward backtesting of a named forecasting model on time series CSV data."
    if LANGCHAIN_AVAILABLE:
        args_schema: Type[LCBaseModel] = BacktestInput

    def _run(self, csv_data: str, model_name: str = "ETSForecaster",
             horizon: int = 30, n_splits: int = 5, test_size: int = 20, **kwargs) -> str:
        from autotsforecast.mcp.server import _run_backtest
        result = _run_backtest(csv_data, model_name, horizon, n_splits, test_size)
        return json.dumps(result, default=str)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class _CalendarFeaturesTool:
    name = "extract_calendar_features"
    description = "Extract calendar features from a datetime-indexed CSV time series."
    if LANGCHAIN_AVAILABLE:
        args_schema: Type[LCBaseModel] = CSVOnlyInput

    def _run(self, csv_data: str, **kwargs) -> str:
        from autotsforecast.mcp.server import _run_calendar_features
        return _run_calendar_features(csv_data)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class _AnomalyTool:
    name = "detect_anomalies"
    description = "Detect anomalies in time series CSV data."
    if LANGCHAIN_AVAILABLE:
        args_schema: Type[LCBaseModel] = AnomalyInput

    def _run(self, csv_data: str, method: str = "isolation_forest",
             contamination: float = 0.05, **kwargs) -> str:
        from autotsforecast.mcp.server import _run_anomaly_detection
        result = _run_anomaly_detection(csv_data, method, contamination)
        return json.dumps(result, default=str)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class _ReconcileTool:
    name = "reconcile_hierarchy"
    description = "Reconcile hierarchical forecasts so parent = sum of children."
    if LANGCHAIN_AVAILABLE:
        args_schema: Type[LCBaseModel] = ReconcileInput

    def _run(self, forecast_csv: str, hierarchy: dict, **kwargs) -> str:
        from autotsforecast.mcp.server import _run_reconcile
        return _run_reconcile(forecast_csv, hierarchy)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


def get_autotsforecast_tools() -> list:
    """
    Return all AutoTSForecast tools ready to pass to a LangChain agent.

    Returns
    -------
    list of BaseTool
        Pass to ``initialize_agent(tools=get_autotsforecast_tools(), ...)``

    Raises
    ------
    ImportError
        If langchain is not installed.
    """
    _require_langchain()

    # Dynamically create LangChain BaseTool subclasses
    tool_bases = [
        _FitAndForecastTool, _BacktestTool, _CalendarFeaturesTool,
        _AnomalyTool, _ReconcileTool,
    ]
    tools = []
    for base in tool_bases:
        attrs = {
            "name": base.name,
            "description": base.description,
            "_run": base._run,
            "_arun": base._arun,
        }
        if hasattr(base, "args_schema"):
            attrs["args_schema"] = base.args_schema
        tool_cls = type(base.__name__.lstrip("_"), (BaseTool,), attrs)
        tools.append(tool_cls())
    return tools

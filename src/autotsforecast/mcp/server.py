"""
MCP server for AutoTSForecast.

Launch with:
    autotsforecast-mcp

Or programmatically:
    from autotsforecast.mcp.server import main
    main()

Claude Desktop config (~/.claude_desktop_config.json):
    {
      "mcpServers": {
        "autotsforecast": {
          "command": "autotsforecast-mcp"
        }
      }
    }
"""

import io
import json
from typing import Any, Dict, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _require_mcp():
    if not MCP_AVAILABLE:
        raise ImportError(
            "mcp package is required. Install with: pip install 'autotsforecast[mcp]'"
        )


def _csv_to_df(csv_str: str):
    import pandas as pd
    df = pd.read_csv(io.StringIO(csv_str), index_col=0, parse_dates=True)
    return df


def _df_to_csv(df) -> str:
    return df.to_csv()


def _get_model_catalog() -> list:
    catalog = [
        {"name": "MovingAverageForecaster", "supports_covariates": False,
         "requires_extra": None, "description": "Simple moving average baseline"},
        {"name": "ARIMAForecaster", "supports_covariates": False,
         "requires_extra": None, "description": "Classical ARIMA model"},
        {"name": "ETSForecaster", "supports_covariates": False,
         "requires_extra": None, "description": "Exponential smoothing (ETS)"},
        {"name": "RandomForestForecaster", "supports_covariates": True,
         "requires_extra": None, "description": "Random Forest with lag features"},
        {"name": "VARForecaster", "supports_covariates": False,
         "requires_extra": None, "description": "Vector Autoregression (needs ≥2 series)"},
        {"name": "XGBoostForecaster", "supports_covariates": True,
         "requires_extra": "ml", "description": "XGBoost with lag features"},
        {"name": "ProphetForecaster", "supports_covariates": True,
         "requires_extra": "prophet", "description": "Facebook Prophet"},
        {"name": "LSTMForecaster", "supports_covariates": False,
         "requires_extra": "neural", "description": "LSTM deep learning model"},
        {"name": "Chronos2Forecaster", "supports_covariates": False,
         "requires_extra": "chronos", "description": "Zero-shot foundation model (no training needed)"},
    ]
    return catalog


def _run_fit_and_forecast(csv_data: str, horizon: int, metric: str,
                          n_splits: int, per_series: bool) -> dict:
    import pandas as pd
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster
    from autotsforecast.models.external import ARIMAForecaster, ETSForecaster, RandomForestForecaster

    y = _csv_to_df(csv_data)
    candidates = [
        MovingAverageForecaster(horizon=horizon, window=7),
        ARIMAForecaster(horizon=horizon),
        ETSForecaster(horizon=horizon),
        RandomForestForecaster(horizon=horizon, n_lags=min(14, len(y) // 4)),
    ]
    auto = AutoForecaster(
        candidate_models=candidates,
        metric=metric,
        n_splits=n_splits,
        per_series_models=per_series,
    )
    auto.fit(y)
    forecasts = auto.forecast()

    result = auto.to_structured(forecasts)
    return result.model_dump()


def _run_backtest(csv_data: str, model_name: str, horizon: int,
                  n_splits: int, test_size: int) -> dict:
    from autotsforecast.backtesting.validator import BacktestValidator
    from autotsforecast.models.base import MovingAverageForecaster
    from autotsforecast.models.external import ARIMAForecaster, ETSForecaster, RandomForestForecaster

    model_map = {
        "MovingAverageForecaster": MovingAverageForecaster(horizon=horizon, window=7),
        "ARIMAForecaster": ARIMAForecaster(horizon=horizon),
        "ETSForecaster": ETSForecaster(horizon=horizon),
        "RandomForestForecaster": RandomForestForecaster(horizon=horizon, n_lags=7),
    }
    model = model_map.get(model_name, ETSForecaster(horizon=horizon))
    y = _csv_to_df(csv_data)
    bv = BacktestValidator(model, n_splits=n_splits, test_size=test_size)
    scores = bv.run(y)
    fold_df = bv.get_fold_results()
    return {
        "model_name": model_name,
        "n_splits": n_splits,
        "mean_rmse": scores.get("rmse", None),
        "fold_scores": fold_df["rmse"].tolist() if "rmse" in fold_df.columns else [],
    }


def _run_prediction_intervals(csv_data: str, horizon: int, coverage: list) -> dict:
    from autotsforecast.models.base import MovingAverageForecaster
    from autotsforecast.uncertainty.intervals import PredictionIntervals

    y = _csv_to_df(csv_data)
    model = MovingAverageForecaster(horizon=horizon, window=7)
    model.fit(y)
    point = model.predict()
    pi = PredictionIntervals(method="conformal", coverage=coverage)
    pi.fit(model, y)
    intervals = pi.predict(point)
    result = {}
    for k, v in intervals.items():
        if hasattr(v, "to_csv"):
            result[k] = v.to_csv()
        else:
            result[k] = str(v)
    return result


def _run_reconcile(forecast_csv: str, hierarchy: dict) -> str:
    from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler
    forecasts = _csv_to_df(forecast_csv)
    reconciler = HierarchicalReconciler(forecasts, hierarchy)
    reconciler.reconcile(method="ols")
    return _df_to_csv(reconciler.reconciled_forecasts)


def _run_calendar_features(csv_data: str) -> str:
    from autotsforecast.features.calendar import CalendarFeatures
    y = _csv_to_df(csv_data)
    cal = CalendarFeatures(cyclical_encoding=True)
    features = cal.fit_transform(y)
    return _df_to_csv(features)


def _run_anomaly_detection(csv_data: str, method: str, contamination: float) -> dict:
    from autotsforecast.anomaly.detector import AnomalyDetector
    y = _csv_to_df(csv_data)
    detector = AnomalyDetector(method=method, contamination=contamination)
    detector.fit_predict(y)
    result = detector.get_summary()
    return result.model_dump()


def create_server() -> "Server":
    _require_mcp()
    server = Server("autotsforecast")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="fit_and_forecast",
                description=(
                    "Fit an AutoForecaster on historical time series data and generate forecasts. "
                    "Automatically selects the best model using walk-forward backtesting."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "csv_data": {"type": "string", "description": "CSV string with datetime index and one column per series"},
                        "horizon": {"type": "integer", "description": "Number of steps to forecast", "default": 30},
                        "metric": {"type": "string", "description": "Selection metric: rmse, mae, mape", "default": "rmse"},
                        "n_splits": {"type": "integer", "description": "Number of backtest splits", "default": 3},
                        "per_series": {"type": "boolean", "description": "Select a different model per series", "default": True},
                    },
                    "required": ["csv_data"],
                },
            ),
            mcp_types.Tool(
                name="backtest",
                description="Walk-forward backtesting of a named model on time series data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "csv_data": {"type": "string", "description": "CSV string with datetime index"},
                        "model_name": {"type": "string", "description": "Model to evaluate e.g. ETSForecaster", "default": "ETSForecaster"},
                        "horizon": {"type": "integer", "default": 30},
                        "n_splits": {"type": "integer", "default": 5},
                        "test_size": {"type": "integer", "default": 20},
                    },
                    "required": ["csv_data"],
                },
            ),
            mcp_types.Tool(
                name="get_prediction_intervals",
                description="Generate conformal prediction intervals around a forecast.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "csv_data": {"type": "string", "description": "CSV training data"},
                        "horizon": {"type": "integer", "default": 30},
                        "coverage": {"type": "array", "items": {"type": "integer"}, "default": [80, 95]},
                    },
                    "required": ["csv_data"],
                },
            ),
            mcp_types.Tool(
                name="reconcile_hierarchy",
                description="Reconcile hierarchical forecasts so they sum correctly (e.g. total = sum of regions).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "forecast_csv": {"type": "string", "description": "CSV of base forecasts"},
                        "hierarchy": {"type": "object", "description": "Dict mapping parent to list of children e.g. {\"total\": [\"a\", \"b\"]}"},
                    },
                    "required": ["forecast_csv", "hierarchy"],
                },
            ),
            mcp_types.Tool(
                name="extract_calendar_features",
                description="Extract calendar features (day-of-week, month, seasonality) from a datetime-indexed series.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "csv_data": {"type": "string", "description": "CSV with datetime index"},
                    },
                    "required": ["csv_data"],
                },
            ),
            mcp_types.Tool(
                name="detect_anomalies",
                description="Detect anomalies in time series data before forecasting.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "csv_data": {"type": "string"},
                        "method": {"type": "string", "default": "isolation_forest",
                                   "enum": ["zscore", "iqr", "isolation_forest", "forecast_residual"]},
                        "contamination": {"type": "number", "default": 0.05},
                    },
                    "required": ["csv_data"],
                },
            ),
            mcp_types.Tool(
                name="list_models",
                description="List all available forecasting models and their capabilities.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            if name == "fit_and_forecast":
                result = _run_fit_and_forecast(
                    csv_data=arguments["csv_data"],
                    horizon=arguments.get("horizon", 30),
                    metric=arguments.get("metric", "rmse"),
                    n_splits=arguments.get("n_splits", 3),
                    per_series=arguments.get("per_series", True),
                )
                text = json.dumps(result, indent=2, default=str)

            elif name == "backtest":
                result = _run_backtest(
                    csv_data=arguments["csv_data"],
                    model_name=arguments.get("model_name", "ETSForecaster"),
                    horizon=arguments.get("horizon", 30),
                    n_splits=arguments.get("n_splits", 5),
                    test_size=arguments.get("test_size", 20),
                )
                text = json.dumps(result, indent=2, default=str)

            elif name == "get_prediction_intervals":
                result = _run_prediction_intervals(
                    csv_data=arguments["csv_data"],
                    horizon=arguments.get("horizon", 30),
                    coverage=arguments.get("coverage", [80, 95]),
                )
                text = json.dumps(result, indent=2, default=str)

            elif name == "reconcile_hierarchy":
                text = _run_reconcile(
                    forecast_csv=arguments["forecast_csv"],
                    hierarchy=arguments["hierarchy"],
                )

            elif name == "extract_calendar_features":
                text = _run_calendar_features(arguments["csv_data"])

            elif name == "detect_anomalies":
                result = _run_anomaly_detection(
                    csv_data=arguments["csv_data"],
                    method=arguments.get("method", "isolation_forest"),
                    contamination=arguments.get("contamination", 0.05),
                )
                text = json.dumps(result, indent=2, default=str)

            elif name == "list_models":
                text = json.dumps(_get_model_catalog(), indent=2)

            else:
                text = f"Unknown tool: {name}"

        except Exception as e:
            text = f"Error in {name}: {type(e).__name__}: {e}"

        return [mcp_types.TextContent(type="text", text=text)]

    return server


async def _async_main():
    _require_mcp()
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for the autotsforecast-mcp CLI command."""
    import asyncio
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

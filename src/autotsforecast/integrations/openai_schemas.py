"""
OpenAI / Anthropic function-calling tool schemas for AutoTSForecast.

Usage with OpenAI:
    from autotsforecast.integrations.openai_schemas import get_openai_tools, handle_tool_call
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Forecast the next 30 days from this data: ..."}],
        tools=get_openai_tools(),
        tool_choice="auto",
    )
    # Handle tool calls
    for tool_call in response.choices[0].message.tool_calls or []:
        result = handle_tool_call(tool_call.function.name, tool_call.function.arguments)

Usage with Anthropic Claude:
    from autotsforecast.integrations.openai_schemas import get_anthropic_tools
    # Same schemas, Anthropic-formatted
"""

import json
import io
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Shared schema definitions
# ---------------------------------------------------------------------------

_FIT_AND_FORECAST = {
    "name": "fit_and_forecast",
    "description": (
        "Fit an AutoForecaster on historical time series CSV data and return forecasts "
        "for a specified horizon. Automatically selects the best model per series using "
        "walk-forward backtesting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "csv_data": {
                "type": "string",
                "description": "CSV string with a datetime index column and one column per time series",
            },
            "horizon": {
                "type": "integer",
                "description": "Number of future steps to forecast",
                "default": 30,
            },
            "metric": {
                "type": "string",
                "description": "Model selection metric",
                "enum": ["rmse", "mae", "mape"],
                "default": "rmse",
            },
            "n_splits": {
                "type": "integer",
                "description": "Number of walk-forward backtesting splits",
                "default": 3,
            },
            "per_series": {
                "type": "boolean",
                "description": "If true, selects a different best model for each series",
                "default": True,
            },
        },
        "required": ["csv_data"],
    },
}

_BACKTEST = {
    "name": "backtest",
    "description": "Run walk-forward backtesting of a specified model on time series data.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_data": {"type": "string", "description": "CSV training data"},
            "model_name": {
                "type": "string",
                "description": "Model to evaluate",
                "enum": [
                    "MovingAverageForecaster", "ARIMAForecaster",
                    "ETSForecaster", "RandomForestForecaster",
                ],
                "default": "ETSForecaster",
            },
            "horizon": {"type": "integer", "default": 30},
            "n_splits": {"type": "integer", "default": 5},
            "test_size": {"type": "integer", "default": 20},
        },
        "required": ["csv_data"],
    },
}

_GET_PREDICTION_INTERVALS = {
    "name": "get_prediction_intervals",
    "description": (
        "Generate conformal prediction intervals (80% and 95% coverage by default) "
        "for time series forecasts. Returns lower and upper bounds per series."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "csv_data": {"type": "string", "description": "CSV training data"},
            "horizon": {"type": "integer", "default": 30},
            "coverage": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Coverage levels e.g. [80, 95]",
                "default": [80, 95],
            },
        },
        "required": ["csv_data"],
    },
}

_RECONCILE_HIERARCHY = {
    "name": "reconcile_hierarchy",
    "description": (
        "Reconcile hierarchical forecasts so that parent series equal the sum of child series "
        "(e.g. total = region_a + region_b). Uses OLS reconciliation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "forecast_csv": {
                "type": "string",
                "description": "CSV of base (unreconciled) forecasts with one column per series",
            },
            "hierarchy": {
                "type": "object",
                "description": 'Dict mapping parent to children e.g. {"total": ["region_a", "region_b"]}',
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": ["forecast_csv", "hierarchy"],
    },
}

_EXTRACT_CALENDAR_FEATURES = {
    "name": "extract_calendar_features",
    "description": (
        "Extract calendar features (day of week, month, quarter, is_weekend, "
        "cyclical sine/cosine encodings) from a datetime-indexed time series."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "csv_data": {"type": "string", "description": "CSV with datetime index"},
        },
        "required": ["csv_data"],
    },
}

_DETECT_ANOMALIES = {
    "name": "detect_anomalies",
    "description": "Detect anomalies in time series data before forecasting.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_data": {"type": "string"},
            "method": {
                "type": "string",
                "enum": ["zscore", "iqr", "isolation_forest", "forecast_residual"],
                "default": "isolation_forest",
            },
            "contamination": {
                "type": "number",
                "description": "Expected fraction of anomalies (0.0–0.5)",
                "default": 0.05,
            },
        },
        "required": ["csv_data"],
    },
}

_LIST_MODELS = {
    "name": "list_models",
    "description": "List all available forecasting models with their capabilities and installation requirements.",
    "parameters": {"type": "object", "properties": {}},
}


# ---------------------------------------------------------------------------
# OpenAI format
# ---------------------------------------------------------------------------

def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Return all tool schemas in OpenAI tool format.

    Pass directly to ``openai.chat.completions.create(tools=...)``.

    Example::

        from autotsforecast.integrations.openai_schemas import get_openai_tools, handle_tool_call
        tools = get_openai_tools()
    """
    schemas = [
        _FIT_AND_FORECAST, _BACKTEST, _GET_PREDICTION_INTERVALS,
        _RECONCILE_HIERARCHY, _EXTRACT_CALENDAR_FEATURES,
        _DETECT_ANOMALIES, _LIST_MODELS,
    ]
    return [{"type": "function", "function": s} for s in schemas]


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """
    Return all tool schemas in Anthropic Claude tool format.

    Pass directly to ``anthropic.messages.create(tools=...)``.
    """
    schemas = [
        _FIT_AND_FORECAST, _BACKTEST, _GET_PREDICTION_INTERVALS,
        _RECONCILE_HIERARCHY, _EXTRACT_CALENDAR_FEATURES,
        _DETECT_ANOMALIES, _LIST_MODELS,
    ]
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in schemas
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher — shared by OpenAI and Anthropic handlers
# ---------------------------------------------------------------------------

def handle_tool_call(tool_name: str, arguments: Union[str, dict]) -> str:
    """
    Execute a tool call from an LLM and return the result as a string.

    Parameters
    ----------
    tool_name : str
        Name of the tool (matches ``name`` in the schema).
    arguments : str or dict
        Tool arguments — either a JSON string or already-parsed dict.

    Returns
    -------
    str
        JSON-encoded result or error message.
    """
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments

    from autotsforecast.mcp.server import (
        _run_fit_and_forecast, _run_backtest, _run_prediction_intervals,
        _run_reconcile, _run_calendar_features, _run_anomaly_detection,
        _get_model_catalog,
    )

    try:
        if tool_name == "fit_and_forecast":
            result = _run_fit_and_forecast(
                csv_data=args["csv_data"],
                horizon=args.get("horizon", 30),
                metric=args.get("metric", "rmse"),
                n_splits=args.get("n_splits", 3),
                per_series=args.get("per_series", True),
            )
            return json.dumps(result, default=str)

        elif tool_name == "backtest":
            result = _run_backtest(
                csv_data=args["csv_data"],
                model_name=args.get("model_name", "ETSForecaster"),
                horizon=args.get("horizon", 30),
                n_splits=args.get("n_splits", 5),
                test_size=args.get("test_size", 20),
            )
            return json.dumps(result, default=str)

        elif tool_name == "get_prediction_intervals":
            result = _run_prediction_intervals(
                csv_data=args["csv_data"],
                horizon=args.get("horizon", 30),
                coverage=args.get("coverage", [80, 95]),
            )
            return json.dumps(result, default=str)

        elif tool_name == "reconcile_hierarchy":
            return _run_reconcile(
                forecast_csv=args["forecast_csv"],
                hierarchy=args["hierarchy"],
            )

        elif tool_name == "extract_calendar_features":
            return _run_calendar_features(args["csv_data"])

        elif tool_name == "detect_anomalies":
            result = _run_anomaly_detection(
                csv_data=args["csv_data"],
                method=args.get("method", "isolation_forest"),
                contamination=args.get("contamination", 0.05),
            )
            return json.dumps(result, default=str)

        elif tool_name == "list_models":
            return json.dumps(_get_model_catalog(), indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

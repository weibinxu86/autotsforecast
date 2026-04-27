"""
FastAPI REST service for AutoTSForecast.

Launch with:
    autotsforecast-api
    # or
    autotsforecast-api --host 0.0.0.0 --port 8000

Swagger UI available at: http://localhost:8000/docs
OpenAPI spec at:         http://localhost:8000/openapi.json
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _require_fastapi():
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required. Install with: pip install 'autotsforecast[api]'"
        )


if FASTAPI_AVAILABLE:
    import autotsforecast as _pkg

    app = FastAPI(
        title="AutoTSForecast API",
        description=(
            "REST API for automated time series forecasting. "
            "Upload your data and get forecasts, backtesting results, "
            "prediction intervals, and more."
        ),
        version=_pkg.__version__,
        contact={"name": "AutoTSForecast", "url": "https://github.com/weibinxu86/autotsforecast"},
        license_info={"name": "MIT"},
    )

    # -------------------------------------------------------------------------
    # Request / response bodies
    # -------------------------------------------------------------------------

    class ForecastRequest(BaseModel):
        csv_data: str
        horizon: int = 30
        metric: str = "rmse"
        n_splits: int = 3
        per_series: bool = True

    class BacktestRequest(BaseModel):
        csv_data: str
        model_name: str = "ETSForecaster"
        horizon: int = 30
        n_splits: int = 5
        test_size: int = 20

    class IntervalsRequest(BaseModel):
        csv_data: str
        horizon: int = 30
        coverage: List[int] = [80, 95]

    class ReconcileRequest(BaseModel):
        forecast_csv: str
        hierarchy: Dict[str, List[str]]

    class CalendarRequest(BaseModel):
        csv_data: str

    class AnomalyRequest(BaseModel):
        csv_data: str
        method: str = "isolation_forest"
        contamination: float = 0.05

    class InsightRequest(BaseModel):
        csv_data: str
        forecast_csv: str

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.get("/health", tags=["Utility"])
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "version": _pkg.__version__}

    @app.get("/models", tags=["Utility"])
    async def list_models():
        """List all available forecasting models."""
        from autotsforecast.mcp.server import _get_model_catalog
        return _get_model_catalog()

    @app.post("/forecast", tags=["Forecasting"])
    async def forecast(request: ForecastRequest):
        """
        Fit an AutoForecaster and return forecasts.

        Upload a CSV string with a datetime index and one column per series.
        The API will automatically select the best model using walk-forward backtesting.
        """
        from autotsforecast.mcp.server import _run_fit_and_forecast
        try:
            result = _run_fit_and_forecast(
                csv_data=request.csv_data,
                horizon=request.horizon,
                metric=request.metric,
                n_splits=request.n_splits,
                per_series=request.per_series,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/forecast/upload", tags=["Forecasting"])
    async def forecast_upload(
        file: UploadFile = File(...),
        horizon: int = Form(30),
        metric: str = Form("rmse"),
        n_splits: int = Form(3),
        per_series: bool = Form(True),
    ):
        """
        Forecast from an uploaded CSV file (multipart/form-data).
        """
        from autotsforecast.mcp.server import _run_fit_and_forecast
        try:
            content = await file.read()
            csv_data = content.decode("utf-8")
            result = _run_fit_and_forecast(csv_data, horizon, metric, n_splits, per_series)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/backtest", tags=["Backtesting"])
    async def backtest(request: BacktestRequest):
        """Walk-forward backtesting of a named model."""
        from autotsforecast.mcp.server import _run_backtest
        try:
            return _run_backtest(
                request.csv_data, request.model_name,
                request.horizon, request.n_splits, request.test_size,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/intervals", tags=["Uncertainty"])
    async def prediction_intervals(request: IntervalsRequest):
        """Generate conformal prediction intervals."""
        from autotsforecast.mcp.server import _run_prediction_intervals
        try:
            result = _run_prediction_intervals(request.csv_data, request.horizon, request.coverage)
            # Convert DataFrames to serialisable dicts
            serialisable = {}
            for k, v in result.items():
                if hasattr(v, "to_dict"):
                    serialisable[k] = v.to_dict(orient="list")
                else:
                    serialisable[k] = v
            return serialisable
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reconcile", tags=["Hierarchical"])
    async def reconcile(request: ReconcileRequest):
        """Reconcile hierarchical forecasts (OLS method)."""
        from autotsforecast.mcp.server import _run_reconcile
        try:
            csv_out = _run_reconcile(request.forecast_csv, request.hierarchy)
            return {"reconciled_csv": csv_out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/calendar-features", tags=["Feature Engineering"])
    async def calendar_features(request: CalendarRequest):
        """Extract calendar features from a datetime-indexed series."""
        from autotsforecast.mcp.server import _run_calendar_features
        try:
            csv_out = _run_calendar_features(request.csv_data)
            return {"features_csv": csv_out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/anomalies", tags=["Anomaly Detection"])
    async def detect_anomalies(request: AnomalyRequest):
        """Detect anomalies in time series data."""
        from autotsforecast.mcp.server import _run_anomaly_detection
        try:
            return _run_anomaly_detection(request.csv_data, request.method, request.contamination)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/insights", tags=["Insights"])
    async def generate_insights(request: InsightRequest):
        """Generate plain-English insights from historical + forecast data."""
        from autotsforecast.nlp.insights import InsightEngine
        import pandas as pd
        try:
            y = pd.read_csv(io.StringIO(request.csv_data), index_col=0, parse_dates=True)
            forecasts = pd.read_csv(io.StringIO(request.forecast_csv), index_col=0, parse_dates=True)
            engine = InsightEngine(mode="rule_based")
            summary = engine.summarize_forecast_dataframes(y, forecasts)
            return {"summary": summary, "risks": engine.flag_risks_from_dataframes(y, forecasts)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for the autotsforecast-api CLI command."""
    import argparse
    _require_fastapi()

    parser = argparse.ArgumentParser(description="AutoTSForecast REST API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install 'autotsforecast[api]'")

    uvicorn.run(
        "autotsforecast.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

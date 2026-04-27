"""Tests for MCP server helper functions (mcp/server.py).

We test the internal _run_* helpers directly, without starting the MCP server,
so this works without having mcp installed.
"""
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def series_csv():
    """CSV string (not JSON) for MCP tools — 120 observations."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    df = pd.DataFrame({"series_a": 100 + rng.normal(0, 2, 120)}, index=dates)
    return df.to_csv()


class TestRunFitAndForecast:
    def test_basic_forecast(self, series_csv):
        from autotsforecast.mcp.server import _run_fit_and_forecast
        result = _run_fit_and_forecast(
            csv_data=series_csv,
            horizon=7,
            metric="rmse",
            n_splits=3,
            per_series=False,
        )
        # Returns a dict (already parsed by _run_fit_and_forecast)
        assert isinstance(result, dict)

    def test_returns_dict(self, series_csv):
        from autotsforecast.mcp.server import _run_fit_and_forecast
        result = _run_fit_and_forecast(series_csv, horizon=5, metric="rmse", n_splits=3, per_series=False)
        assert isinstance(result, dict)


class TestRunBacktest:
    def test_basic_backtest(self, series_csv):
        from autotsforecast.mcp.server import _run_backtest
        result = _run_backtest(
            csv_data=series_csv,
            model_name="ETSForecaster",
            horizon=7,
            n_splits=3,
            test_size=10,
        )
        assert isinstance(result, dict)


class TestRunCalendarFeatures:
    def test_calendar_features(self, series_csv):
        from autotsforecast.mcp.server import _run_calendar_features
        result = _run_calendar_features(csv_data=series_csv)
        # Returns CSV string
        assert isinstance(result, str)


class TestRunAnomalyDetection:
    def test_zscore(self, series_csv):
        from autotsforecast.mcp.server import _run_anomaly_detection
        result = _run_anomaly_detection(
            csv_data=series_csv,
            method="zscore",
            contamination=0.05,
        )
        assert isinstance(result, dict)


class TestGetModelCatalog:
    def test_returns_models_list(self):
        from autotsforecast.mcp.server import _get_model_catalog
        data = _get_model_catalog()
        # Returns a list of dicts
        assert isinstance(data, list)
        assert len(data) > 0
        assert "name" in data[0]

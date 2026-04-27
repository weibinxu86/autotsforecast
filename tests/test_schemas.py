"""Tests for the Pydantic structured output schemas (schemas.py)."""
import pytest
import pandas as pd


def test_forecast_result_basic():
    """ForecastResult should create and serialise without error."""
    try:
        from autotsforecast.schemas import ForecastResult
    except ImportError:
        pytest.skip("pydantic not installed")

    result = ForecastResult(
        series_names=["a", "b"],
        horizon=3,
        dates=["2024-01-01", "2024-01-02", "2024-01-03"],
        values={"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
        best_model="MovingAverageForecaster",
        metric="rmse",
        metric_value=0.5,
    )
    assert result.horizon == 3
    assert len(result.dates) == 3
    assert "a" in result.values


def test_forecast_result_serialise():
    try:
        from autotsforecast.schemas import ForecastResult
    except ImportError:
        pytest.skip("pydantic not installed")

    result = ForecastResult(
        series_names=["x"],
        horizon=2,
        dates=["2024-01-01", "2024-01-02"],
        values={"x": [1.0, 2.0]},
        best_model="ARIMA",
        metric="mae",
    )
    data = result.model_dump()
    assert data["best_model"] == "ARIMA"
    assert data.get("metric_value") is None  # None or absent when pydantic not installed


def test_backtest_result():
    try:
        from autotsforecast.schemas import BacktestResult
    except ImportError:
        pytest.skip("pydantic not installed")

    r = BacktestResult(
        best_model="XGBoost",
        metric="rmse",
        metric_value=0.8,
        n_splits=5,
        all_scores={"XGBoost": {"rmse": 0.8}, "ARIMA": {"rmse": 1.2}},
    )
    assert r.n_splits == 5


def test_anomaly_result():
    try:
        from autotsforecast.schemas import AnomalyResult
    except ImportError:
        pytest.skip("pydantic not installed")

    r = AnomalyResult(
        method="zscore",
        n_anomalies={"series_a": 3},
        anomaly_dates={"series_a": ["2024-01-05"]},
    )
    assert r.method == "zscore"


def test_registry_entry():
    try:
        from autotsforecast.schemas import RegistryEntry
    except ImportError:
        pytest.skip("pydantic not installed")

    entry = RegistryEntry(
        name="my_model",
        class_name="AutoForecaster",
        saved_at="2024-01-01T00:00:00+00:00",
        filepath="/tmp/my_model.joblib",
    )
    assert entry.name == "my_model"


def test_model_catalog():
    try:
        from autotsforecast.schemas import ModelCatalog, ModelInfo
    except ImportError:
        pytest.skip("pydantic not installed")

    catalog = ModelCatalog(models=[
        ModelInfo(name="ARIMA", class_name="ARIMAForecaster",
                  requires_extra=None, description="ARIMA model")
    ])
    assert len(catalog.models) == 1

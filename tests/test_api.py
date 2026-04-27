"""Tests for FastAPI REST service (api/app.py)."""
import json
import io
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from autotsforecast.api.app import app
        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"fastapi or httpx not installed: {e}")


@pytest.fixture
def sample_series_json():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({"series_a": 100 + rng.normal(0, 2, 60)}, index=dates)
    df.index = df.index.astype(str)
    return df.to_json()


class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or data


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))


class TestForecastEndpoint:
    def test_forecast_json(self, client, sample_series_json):
        payload = {
            "data_json": sample_series_json,
            "horizon": 7,
            "metric": "rmse",
            "n_splits": 3,
            "test_size": 10,
        }
        response = client.post("/forecast", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_forecast_missing_data_returns_422(self, client):
        response = client.post("/forecast", json={"horizon": 7})
        assert response.status_code == 422


class TestBacktestEndpoint:
    def test_backtest(self, client, sample_series_json):
        payload = {
            "data_json": sample_series_json,
            "n_splits": 3,
            "test_size": 10,
            "metric": "rmse",
        }
        response = client.post("/backtest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestAnomalyEndpoint:
    def test_anomaly_detection(self, client, sample_series_json):
        payload = {
            "data_json": sample_series_json,
            "method": "zscore",
            "threshold": 3.0,
        }
        response = client.post("/anomalies", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestCalendarFeaturesEndpoint:
    def test_calendar_features(self, client, sample_series_json):
        payload = {
            "data_json": sample_series_json,
        }
        response = client.post("/calendar-features", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

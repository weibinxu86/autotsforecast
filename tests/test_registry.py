"""Tests for ModelRegistry (registry/store.py)."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_registry(tmp_path):
    from autotsforecast.registry.store import ModelRegistry
    return ModelRegistry(registry_dir=tmp_path / "registry")


@pytest.fixture
def fitted_auto(tmp_path):
    """Return a fitted AutoForecaster for registry save/load tests."""
    from autotsforecast import (
        AutoForecaster,
        MovingAverageForecaster,
        ARIMAForecaster,
    )
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    y = pd.DataFrame({"series_a": 100 + rng.normal(0, 2, 80)}, index=dates)
    candidates = [
        MovingAverageForecaster(window=7, horizon=7),
        ARIMAForecaster(order=(1, 0, 0), horizon=7),
    ]
    auto = AutoForecaster(candidates, metric="rmse", n_splits=3, test_size=10, verbose=False)
    auto.fit(y)
    auto.forecast()
    return auto


class TestModelRegistryImport:
    def test_import(self):
        from autotsforecast.registry.store import ModelRegistry
        assert ModelRegistry is not None


class TestRegistryEmpty:
    def test_list_empty(self, tmp_registry):
        df = tmp_registry.list()
        assert len(df) == 0

    def test_load_missing_raises(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.load("nonexistent_model")

    def test_delete_missing_raises(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.delete("nonexistent_model")


class TestSaveLoad:
    def test_save_returns_name(self, tmp_registry, fitted_auto):
        name = tmp_registry.save(fitted_auto, name="test_model")
        assert name == "test_model"

    def test_list_after_save(self, tmp_registry, fitted_auto):
        tmp_registry.save(fitted_auto, name="model_a", tags={"version": "1"})
        df = tmp_registry.list()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "model_a"
        assert df.iloc[0]["tags"] == {"version": "1"}

    def test_load_returns_autoforecaster(self, tmp_registry, fitted_auto):
        from autotsforecast import AutoForecaster
        tmp_registry.save(fitted_auto, name="loadable")
        loaded = tmp_registry.load("loadable")
        assert isinstance(loaded, AutoForecaster)

    def test_loaded_model_can_forecast(self, tmp_registry, fitted_auto):
        tmp_registry.save(fitted_auto, name="forecast_test")
        loaded = tmp_registry.load("forecast_test")
        fc = loaded.forecast()
        assert isinstance(fc, pd.DataFrame)
        assert len(fc) > 0


class TestSaveMultipleModels:
    def test_multiple_models(self, tmp_registry, fitted_auto):
        tmp_registry.save(fitted_auto, name="model_1")
        tmp_registry.save(fitted_auto, name="model_2", tags={"env": "prod"})
        df = tmp_registry.list()
        assert len(df) == 2
        assert set(df["name"]) == {"model_1", "model_2"}


class TestDelete:
    def test_delete_model(self, tmp_registry, fitted_auto):
        tmp_registry.save(fitted_auto, name="to_delete")
        tmp_registry.delete("to_delete")
        df = tmp_registry.list()
        assert len(df) == 0

    def test_file_removed_after_delete(self, tmp_path, fitted_auto):
        from autotsforecast.registry.store import ModelRegistry
        reg = ModelRegistry(registry_dir=tmp_path / "reg2")
        reg.save(fitted_auto, name="deleteme")
        entry_path = reg._index["deleteme"]["filepath"]
        assert os.path.exists(entry_path)
        reg.delete("deleteme")
        assert not os.path.exists(entry_path)


class TestPersistence:
    def test_index_persists_across_instances(self, tmp_path, fitted_auto):
        from autotsforecast.registry.store import ModelRegistry
        reg_dir = tmp_path / "persist_reg"
        r1 = ModelRegistry(registry_dir=reg_dir)
        r1.save(fitted_auto, name="persistent_model")
        # Create a second instance pointing to same dir
        r2 = ModelRegistry(registry_dir=reg_dir)
        df = r2.list()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "persistent_model"


class TestToStructured:
    def test_to_structured_returns_forecast_result(self, fitted_auto):
        try:
            from autotsforecast.schemas import ForecastResult
        except ImportError:
            pytest.skip("pydantic not installed")
        result = fitted_auto.to_structured()
        assert isinstance(result, ForecastResult)
        assert result.horizon > 0
        assert len(result.dates) == result.horizon

    def test_to_structured_with_explicit_df(self, fitted_auto):
        try:
            from autotsforecast.schemas import ForecastResult
        except ImportError:
            pytest.skip("pydantic not installed")
        fc = fitted_auto.forecasts_
        result = fitted_auto.to_structured(fc)
        assert isinstance(result, ForecastResult)
        assert result.best_model == fitted_auto.best_model_name_

    def test_to_structured_no_forecasts_raises(self):
        from autotsforecast import AutoForecaster, MovingAverageForecaster
        auto = AutoForecaster(
            [MovingAverageForecaster(window=3, horizon=5)], verbose=False
        )
        with pytest.raises(ValueError):
            auto.to_structured()

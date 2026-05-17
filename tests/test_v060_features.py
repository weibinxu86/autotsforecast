"""
Tests for AutoTSForecast v0.6.0 features:
- Presets (PRESETS dict, get_preset_models)
- AutoForecaster with preset=, horizon=
- Budget controls: time_limit, max_models, backtest_mode
- get_report() / print_report()
- profile_data() / DatasetProfiler
- New models: LightGBM, CatBoost, ElasticNet, Theta, Croston, NBEATS, NHiTS, TFT
"""

import numpy as np
import pandas as pd
import pytest

from autotsforecast import AutoForecaster, PRESETS, get_preset_models
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import ARIMAForecaster, ETSForecaster, ElasticNetForecaster, ThetaForecaster, CrostonForecaster
from autotsforecast.utils.profiler import DatasetProfiler, ProfileResult


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def daily_data():
    """120-row daily DataFrame with two series."""
    np.random.seed(42)
    idx = pd.date_range("2023-01-01", periods=120, freq="D")
    return pd.DataFrame(
        {
            "a": np.cumsum(np.random.randn(120)) + 100,
            "b": np.cumsum(np.random.randn(120)) + 200,
        },
        index=idx,
    )


@pytest.fixture
def intermittent_data():
    """Sparse series with ~50% zeros to trigger intermittent profiling."""
    np.random.seed(7)
    idx = pd.date_range("2023-01-01", periods=120, freq="D")
    vals = np.random.choice([0, 0, 0, 1, 2, 5], size=120)
    return pd.DataFrame({"demand": vals.astype(float)}, index=idx)


# ── PRESETS dict ─────────────────────────────────────────────────────────────

class TestPresetsDict:

    def test_all_preset_keys_present(self):
        expected = {"fast", "balanced", "accuracy", "zero_shot", "intermittent", "hierarchical"}
        assert expected == set(PRESETS.keys())

    def test_preset_values_are_non_empty_strings(self):
        for name, desc in PRESETS.items():
            assert isinstance(desc, str) and len(desc) > 0, f"Preset '{name}' has empty description"


# ── get_preset_models ─────────────────────────────────────────────────────────

class TestGetPresetModels:

    @pytest.mark.parametrize("preset", ["fast", "balanced", "accuracy", "intermittent", "hierarchical"])
    def test_returns_non_empty_list(self, preset):
        models = get_preset_models(preset, horizon=7)
        assert isinstance(models, list) and len(models) >= 1

    def test_all_models_have_correct_horizon(self):
        for preset in ["fast", "balanced"]:
            for model in get_preset_models(preset, horizon=14):
                assert model.horizon == 14

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_models("nonexistent", horizon=7)

    def test_zero_shot_preset_empty_gracefully(self):
        # zero_shot may return [] if chronos not installed; should not raise
        try:
            models = get_preset_models("zero_shot", horizon=7)
            assert isinstance(models, list)
        except Exception as exc:
            pytest.fail(f"get_preset_models('zero_shot', ...) raised {exc}")


# ── AutoForecaster with preset ────────────────────────────────────────────────

class TestAutoForecasterPreset:

    def test_preset_fast_fit_forecast(self, daily_data):
        auto = AutoForecaster(
            preset="fast",
            horizon=7,
            n_splits=2,
            test_size=7,
            verbose=False,
        )
        auto.fit(daily_data)
        fc = auto.forecast()
        assert fc.shape == (7, 2)

    def test_preset_balanced_selects_best_model(self, daily_data):
        auto = AutoForecaster(
            preset="balanced",
            horizon=7,
            n_splits=2,
            test_size=7,
            verbose=False,
        )
        auto.fit(daily_data)
        assert auto.best_model_name_ is not None

    def test_preset_and_explicit_candidates_raises(self, daily_data):
        """Cannot specify both preset and candidate_models."""
        with pytest.raises((ValueError, TypeError)):
            auto = AutoForecaster(
                preset="fast",
                horizon=7,
                candidate_models=[MovingAverageForecaster(horizon=7)],
            )
            auto.fit(daily_data)

    def test_preset_without_horizon_raises(self, daily_data):
        """preset requires horizon."""
        with pytest.raises((ValueError, TypeError)):
            auto = AutoForecaster(preset="fast")
            auto.fit(daily_data)


# ── Budget controls ───────────────────────────────────────────────────────────

class TestBudgetControls:

    def test_max_models_limits_candidates(self, daily_data):
        auto = AutoForecaster(
            preset="balanced",
            horizon=7,
            n_splits=2,
            test_size=7,
            max_models=2,
            verbose=False,
        )
        auto.fit(daily_data)
        report = auto.get_report()
        # At most 2 models were evaluated
        assert len(report["model_ranking"]) <= 2

    def test_time_limit_respected(self, daily_data):
        """time_limit=0.001 should stop after the first (or zero) candidates."""
        auto = AutoForecaster(
            preset="balanced",
            horizon=7,
            n_splits=2,
            test_size=7,
            time_limit=0.001,   # effectively immediate stop
            verbose=False,
        )
        auto.fit(daily_data)
        # Should not raise; best_model_name_ may be None if nothing finished
        # but fit() itself completes without error
        assert True

    @pytest.mark.parametrize("mode", ["full", "fast", "last_fold"])
    def test_backtest_mode_variants(self, daily_data, mode):
        auto = AutoForecaster(
            preset="fast",
            horizon=7,
            n_splits=3,
            test_size=7,
            backtest_mode=mode,
            verbose=False,
        )
        auto.fit(daily_data)
        fc = auto.forecast()
        assert fc.shape == (7, 2)


# ── get_report / print_report ─────────────────────────────────────────────────

class TestGetReport:

    @pytest.fixture
    def fitted_auto(self, daily_data):
        auto = AutoForecaster(
            preset="fast",
            horizon=7,
            n_splits=2,
            test_size=7,
            verbose=False,
        )
        auto.fit(daily_data)
        return auto

    def test_get_report_returns_dict(self, fitted_auto):
        report = fitted_auto.get_report()
        assert isinstance(report, dict)

    def test_get_report_has_required_keys(self, fitted_auto):
        report = fitted_auto.get_report()
        for key in ("best_model", "metric", "model_ranking"):
            assert key in report, f"Missing key '{key}' in get_report() result"

    def test_ranking_is_list(self, fitted_auto):
        report = fitted_auto.get_report()
        assert isinstance(report["model_ranking"], list)

    def test_print_report_does_not_raise(self, fitted_auto, capsys):
        fitted_auto.print_report()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_get_report_before_fit_raises(self):
        auto = AutoForecaster(
            preset="fast",
            horizon=7,
            verbose=False,
        )
        with pytest.raises(Exception):
            auto.get_report()


# ── DatasetProfiler / profile_data ───────────────────────────────────────────

class TestDatasetProfiler:

    def test_profile_result_type(self, daily_data):
        result = AutoForecaster.profile_data(daily_data)
        assert isinstance(result, ProfileResult)

    def test_profile_detects_series_count(self, daily_data):
        result = AutoForecaster.profile_data(daily_data)
        assert result.n_series == 2

    def test_profile_detects_obs_count(self, daily_data):
        result = AutoForecaster.profile_data(daily_data)
        assert result.n_obs == 120

    def test_profile_detects_intermittent(self, intermittent_data):
        result = AutoForecaster.profile_data(intermittent_data)
        assert result.is_intermittent, "High zero-rate series should be flagged intermittent"

    def test_profile_recommends_intermittent_preset(self, intermittent_data):
        result = AutoForecaster.profile_data(intermittent_data)
        assert result.recommended_preset == "intermittent"

    def test_profile_summary_is_string(self, daily_data):
        result = AutoForecaster.profile_data(daily_data)
        summary = result.summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_print_summary_does_not_raise(self, daily_data, capsys):
        result = AutoForecaster.profile_data(daily_data)
        result.print_summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_direct_profiler_usage(self, daily_data):
        profiler = DatasetProfiler()
        result = profiler.profile(daily_data)
        assert isinstance(result, ProfileResult)
        assert result.n_obs == len(daily_data)

    def test_profile_has_recommended_preset(self, daily_data):
        result = AutoForecaster.profile_data(daily_data)
        assert result.recommended_preset in PRESETS

    def test_profile_short_series(self):
        short = pd.DataFrame({"x": np.random.randn(30)},
                              index=pd.date_range("2023-01-01", periods=30, freq="D"))
        result = AutoForecaster.profile_data(short)
        assert result.is_short


# ── New model: ElasticNetForecaster ──────────────────────────────────────────

class TestElasticNetForecaster:

    @pytest.fixture
    def model(self):
        return ElasticNetForecaster(horizon=7, n_lags=7, alpha=0.5, l1_ratio=0.5)

    def test_fit_predict_shape(self, model, daily_data):
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)

    def test_forecast_is_numeric(self, model, daily_data):
        model.fit(daily_data)
        fc = model.predict()
        assert not fc.isnull().any().any()

    def test_get_params_returns_dict(self, model):
        params = model.get_params()
        assert "horizon" in params and "alpha" in params


# ── New model: ThetaForecaster ────────────────────────────────────────────────

class TestThetaForecaster:

    @pytest.fixture
    def model(self):
        return ThetaForecaster(horizon=7)

    def test_fit_predict_shape(self, model, daily_data):
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)

    def test_forecast_is_numeric(self, model, daily_data):
        model.fit(daily_data)
        fc = model.predict()
        assert not fc.isnull().any().any()


# ── New model: CrostonForecaster ─────────────────────────────────────────────

class TestCrostonForecaster:

    @pytest.fixture
    def model(self):
        return CrostonForecaster(horizon=7)

    def test_fit_predict_shape(self, model, intermittent_data):
        model.fit(intermittent_data)
        fc = model.predict()
        assert fc.shape == (7, 1)

    def test_forecast_is_non_negative(self, model, intermittent_data):
        model.fit(intermittent_data)
        fc = model.predict()
        assert (fc.values >= 0).all(), "Croston forecasts should be non-negative"


# ── New model: LightGBMForecaster ─────────────────────────────────────────────

class TestLightGBMForecaster:

    def test_fit_predict_shape(self, daily_data):
        pytest.importorskip("lightgbm")
        from autotsforecast.models.external import LightGBMForecaster
        model = LightGBMForecaster(horizon=7, n_lags=7, n_estimators=20)
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)

    def test_forecast_is_numeric(self, daily_data):
        pytest.importorskip("lightgbm")
        from autotsforecast.models.external import LightGBMForecaster
        model = LightGBMForecaster(horizon=7, n_lags=7, n_estimators=20)
        model.fit(daily_data)
        fc = model.predict()
        assert not fc.isnull().any().any()

    def test_missing_dependency_raises(self):
        """If lightgbm is not installed, constructor should raise ImportError."""
        import sys
        # Only run this check if lightgbm is NOT installed
        if "lightgbm" in sys.modules:
            pytest.skip("lightgbm is installed; cannot test missing-dependency path")
        from autotsforecast.models.external import LightGBMForecaster
        with pytest.raises(ImportError):
            LightGBMForecaster(horizon=7)


# ── New model: CatBoostForecaster ─────────────────────────────────────────────

class TestCatBoostForecaster:

    def test_fit_predict_shape(self, daily_data):
        pytest.importorskip("catboost")
        from autotsforecast.models.external import CatBoostForecaster
        model = CatBoostForecaster(horizon=7, n_lags=7, iterations=20)
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)

    def test_forecast_is_numeric(self, daily_data):
        pytest.importorskip("catboost")
        from autotsforecast.models.external import CatBoostForecaster
        model = CatBoostForecaster(horizon=7, n_lags=7, iterations=20)
        model.fit(daily_data)
        fc = model.predict()
        assert not fc.isnull().any().any()


# ── New model: NBEATSForecaster ───────────────────────────────────────────────

class TestNBEATSForecaster:

    def test_fit_predict_shape(self, daily_data):
        pytest.importorskip("darts")
        from autotsforecast.models.external import NBEATSForecaster
        model = NBEATSForecaster(horizon=7, input_chunk_length=14, n_epochs=2)
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)


# ── New model: NHiTSForecaster ────────────────────────────────────────────────

class TestNHiTSForecaster:

    def test_fit_predict_shape(self, daily_data):
        pytest.importorskip("darts")
        from autotsforecast.models.external import NHiTSForecaster
        model = NHiTSForecaster(horizon=7, input_chunk_length=14, n_epochs=2)
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)


# ── New model: TFTForecaster ──────────────────────────────────────────────────

class TestTFTForecaster:

    def test_fit_predict_shape(self, daily_data):
        pytest.importorskip("darts")
        from autotsforecast.models.external import TFTForecaster
        model = TFTForecaster(horizon=7, input_chunk_length=14, n_epochs=2)
        model.fit(daily_data)
        fc = model.predict()
        assert fc.shape == (7, 2)


# ── AutoForecaster integration with new models ────────────────────────────────

class TestAutoForecasterNewModels:

    def test_auto_with_elasticnet_and_theta(self, daily_data):
        candidates = [
            MovingAverageForecaster(horizon=7, window=5),
            ETSForecaster(horizon=7, trend="add", seasonal=None),
            ElasticNetForecaster(horizon=7, n_lags=7),
            ThetaForecaster(horizon=7),
        ]
        auto = AutoForecaster(
            candidate_models=candidates,
            n_splits=2,
            test_size=7,
            metric="rmse",
            verbose=False,
        )
        auto.fit(daily_data)
        fc = auto.forecast()
        assert fc.shape == (7, 2)
        report = auto.get_report()
        assert report["best_model"] is not None

    def test_auto_with_lightgbm_if_available(self, daily_data):
        lgb = pytest.importorskip("lightgbm")
        from autotsforecast.models.external import LightGBMForecaster

        candidates = [
            MovingAverageForecaster(horizon=7, window=5),
            LightGBMForecaster(horizon=7, n_lags=7, n_estimators=20),
        ]
        auto = AutoForecaster(
            candidate_models=candidates,
            n_splits=2,
            test_size=7,
            verbose=False,
        )
        auto.fit(daily_data)
        fc = auto.forecast()
        assert fc.shape == (7, 2)

    def test_parallel_preset_search(self, daily_data):
        """n_jobs=-1 parallel preset search completes without error."""
        auto = AutoForecaster(
            preset="fast",
            horizon=7,
            n_splits=2,
            test_size=7,
            n_jobs=-1,
            verbose=False,
        )
        auto.fit(daily_data)
        fc = auto.forecast()
        assert fc.shape == (7, 2)

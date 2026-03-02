"""
Comprehensive tests for AutoForecaster: fitting, forecasting, model selection,
covariates, summary, edge cases, and contract guarantees.
"""

import numpy as np
import pandas as pd
import pytest
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster, LinearForecaster
from autotsforecast.models.external import ARIMAForecaster, ETSForecaster, RandomForestForecaster


# ── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def daily_data():
    np.random.seed(0)
    dates = pd.date_range('2021-01-01', periods=120, freq='D')
    return pd.DataFrame({
        'a': np.cumsum(np.random.randn(120)) + 100,
        'b': np.cumsum(np.random.randn(120)) + 200,
    }, index=dates)


@pytest.fixture
def daily_covariates(daily_data):
    return pd.DataFrame({
        'promo': np.random.choice([0, 1], len(daily_data)),
        'temp':  np.random.uniform(10, 30, len(daily_data)),
    }, index=daily_data.index)


@pytest.fixture
def candidates():
    return [
        MovingAverageForecaster(horizon=7, window=5),
        ETSForecaster(horizon=7, trend='add', seasonal=None),
    ]


@pytest.fixture
def auto_global(candidates):
    return AutoForecaster(
        candidate_models=candidates,
        per_series_models=False,
        n_splits=2, test_size=7, metric='rmse', verbose=False,
    )


@pytest.fixture
def auto_per_series(candidates):
    return AutoForecaster(
        candidate_models=candidates,
        per_series_models=True,
        n_splits=2, test_size=7, metric='rmse', verbose=False,
    )


# ── Fit / forecast shape contracts ─────────────────────────────────────────

class TestFitForecastContracts:

    def test_forecast_shape_global(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        fc = auto_global.forecast()
        assert fc.shape == (7, 2), "Forecast should have (horizon, n_series) shape"

    def test_forecast_shape_per_series(self, auto_per_series, daily_data):
        auto_per_series.fit(daily_data)
        fc = auto_per_series.forecast()
        assert fc.shape == (7, 2)

    def test_forecast_columns_match_input(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        fc = auto_global.forecast()
        assert list(fc.columns) == list(daily_data.columns)

    def test_forecast_index_is_datetime(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        fc = auto_global.forecast()
        assert pd.api.types.is_datetime64_any_dtype(fc.index)

    def test_forecast_no_nans(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        fc = auto_global.forecast()
        assert not fc.isnull().any().any(), "Forecast should contain no NaN values"

    def test_forecast_follows_training_dates(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        fc = auto_global.forecast()
        assert fc.index[0] > daily_data.index[-1], "Forecast must start after training end"

    def test_univariate_input(self, candidates):
        np.random.seed(1)
        dates = pd.date_range('2021-01-01', periods=80, freq='D')
        y = pd.DataFrame({'x': np.cumsum(np.random.randn(80)) + 50}, index=dates)
        auto = AutoForecaster(
            candidate_models=candidates, per_series_models=False,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(y)
        fc = auto.forecast()
        assert fc.shape == (7, 1)


# ── Covariates ───────────────────────────────────────────────────────────────

class TestCovariates:

    def test_shared_X_global(self, candidates, daily_data, daily_covariates):
        ml = [RandomForestForecaster(horizon=7, n_lags=7, n_estimators=20)]
        auto = AutoForecaster(
            candidate_models=ml, per_series_models=False,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(daily_data, X=daily_covariates)
        fc = auto.forecast(X=daily_covariates.tail(7))
        assert fc.shape == (7, 2)
        assert not fc.isnull().any().any()

    def test_shared_X_per_series(self, daily_data, daily_covariates):
        ml = [RandomForestForecaster(horizon=7, n_lags=7, n_estimators=20)]
        auto = AutoForecaster(
            candidate_models=ml, per_series_models=True,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(daily_data, X=daily_covariates)
        fc = auto.forecast(X=daily_covariates.tail(7))
        assert fc.shape == (7, 2)

    def test_per_series_X_dict(self, daily_data, daily_covariates):
        """Each series can receive a different feature set."""
        ml = [RandomForestForecaster(horizon=7, n_lags=7, n_estimators=20)]
        X_train_dict = {
            'a': daily_covariates[['promo']].iloc[:-7],
            'b': daily_covariates[['temp']].iloc[:-7],
        }
        X_test_dict = {
            'a': daily_covariates[['promo']].tail(7),
            'b': daily_covariates[['temp']].tail(7),
        }
        y_train = daily_data.iloc[:-7]
        auto = AutoForecaster(
            candidate_models=ml, per_series_models=True,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(y_train, X=X_train_dict)
        fc = auto.forecast(X=X_test_dict)
        assert fc.shape == (7, 2)
        assert not fc.isnull().any().any()

    def test_per_series_X_different_column_counts(self, daily_data, daily_covariates):
        """Series can have different numbers of features."""
        ml = [RandomForestForecaster(horizon=7, n_lags=7, n_estimators=20)]
        y_train = daily_data.iloc[:-7]
        X_train_dict = {
            'a': daily_covariates[['promo', 'temp']].iloc[:-7],  # 2 features
            'b': daily_covariates[['temp']].iloc[:-7],            # 1 feature
        }
        X_test_dict = {
            'a': daily_covariates[['promo', 'temp']].tail(7),
            'b': daily_covariates[['temp']].tail(7),
        }
        auto = AutoForecaster(
            candidate_models=ml, per_series_models=True,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(y_train, X=X_train_dict)
        fc = auto.forecast(X=X_test_dict)
        assert fc.shape == (7, 2)


# ── Model selection attributes ───────────────────────────────────────────────

class TestModelSelection:

    def test_best_model_name_set_global(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        assert hasattr(auto_global, 'best_model_name_')
        assert isinstance(auto_global.best_model_name_, str)

    def test_best_models_dict_per_series(self, auto_per_series, daily_data):
        auto_per_series.fit(daily_data)
        assert hasattr(auto_per_series, 'best_models_')
        assert set(auto_per_series.best_models_.keys()) == set(daily_data.columns)

    def test_cv_results_populated(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        assert hasattr(auto_global, 'cv_results_')
        assert len(auto_global.cv_results_) > 0

    def test_selected_model_is_from_candidates(self, candidates, daily_data):
        candidate_names = {type(c).__name__ for c in candidates}
        auto = AutoForecaster(
            candidate_models=candidates, per_series_models=False,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(daily_data)
        assert auto.best_model_name_ in candidate_names

    def test_per_series_each_series_best_model_in_candidates(self, candidates, daily_data):
        candidate_names = {type(c).__name__ for c in candidates}
        auto = AutoForecaster(
            candidate_models=candidates, per_series_models=True,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto.fit(daily_data)
        for s, m in auto.best_models_.items():
            assert type(m).__name__ in candidate_names

    def test_different_metrics(self, daily_data):
        for metric in ['rmse', 'mae', 'mape']:
            candidates = [
                MovingAverageForecaster(horizon=7, window=5),
                ETSForecaster(horizon=7, trend='add', seasonal=None),
            ]
            auto = AutoForecaster(
                candidate_models=candidates, per_series_models=False,
                n_splits=2, test_size=7, metric=metric, verbose=False,
            )
            auto.fit(daily_data)
            assert auto.best_model_name_ is not None


# ── Summary / print_summary ──────────────────────────────────────────────────

class TestSummary:

    def test_get_summary_global_returns_dict(self, auto_global, daily_data):
        auto_global.fit(daily_data)
        summary = auto_global.get_summary()
        assert isinstance(summary, dict)
        assert 'best_model' in summary

    def test_get_summary_per_series_returns_dict(self, auto_per_series, daily_data):
        auto_per_series.fit(daily_data)
        summary = auto_per_series.get_summary()
        assert isinstance(summary, dict)

    def test_print_summary_global_does_not_raise(self, auto_global, daily_data, capsys):
        auto_global.fit(daily_data)
        auto_global.print_summary()  # should not raise
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_summary_per_series_does_not_raise(self, auto_per_series, daily_data, capsys):
        auto_per_series.fit(daily_data)
        auto_per_series.print_summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ── Error handling ───────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_forecast_before_fit_raises(self, auto_global):
        with pytest.raises(Exception):
            auto_global.forecast()

    def test_empty_candidate_list_raises(self):
        with pytest.raises(Exception):
            AutoForecaster(candidate_models=[], per_series_models=False,
                           n_splits=2, test_size=7, verbose=False)

    def test_insufficient_data_raises(self, candidates):
        auto = AutoForecaster(
            candidate_models=candidates, per_series_models=False,
            n_splits=5, test_size=20, metric='rmse', verbose=False,
        )
        tiny = pd.DataFrame({'x': np.random.randn(10)},
                            index=pd.date_range('2021-01-01', periods=10, freq='D'))
        with pytest.raises(Exception):
            auto.fit(tiny)


# ── Model isolation — no shared state across calls ───────────────────────────

class TestModelIsolation:

    def test_refit_with_different_data(self, auto_global, daily_data):
        """Fitting a second time should overwrite previous state."""
        auto_global.fit(daily_data)
        first_name = auto_global.best_model_name_

        np.random.seed(99)
        dates = pd.date_range('2022-01-01', periods=120, freq='D')
        data2 = pd.DataFrame({
            'a': np.cumsum(np.random.randn(120)) + 50,
            'b': np.cumsum(np.random.randn(120)) + 80,
        }, index=dates)
        auto_global.fit(data2)
        fc = auto_global.forecast()
        assert fc.shape == (7, 2)

    def test_two_instances_independent(self, candidates, daily_data):
        """Two AutoForecaster instances should be fully independent."""
        auto1 = AutoForecaster(
            candidate_models=candidates, per_series_models=False,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto2 = AutoForecaster(
            candidate_models=candidates, per_series_models=False,
            n_splits=2, test_size=7, metric='rmse', verbose=False,
        )
        auto1.fit(daily_data)
        auto2.fit(daily_data)
        fc1 = auto1.forecast()
        fc2 = auto2.forecast()
        # Both should produce valid forecasts; instances don't share state
        assert fc1.shape == fc2.shape

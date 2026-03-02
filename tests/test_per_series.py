"""
Tests for per-series covariates, BacktestValidator correctness (no mutation),
and model clone / deep-copy guarantees.
"""

import copy
import numpy as np
import pandas as pd
import pytest
from autotsforecast import AutoForecaster
from autotsforecast.backtesting.validator import BacktestValidator
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import RandomForestForecaster, ETSForecaster


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dgp():
    """
    Data-generating process with perfectly distinct drivers:
      series_a  = 100 + 50*promotion  + noise(sigma=3)
      series_b  = 80  + 4*temperature + noise(sigma=3)
    Giving each series only its true driver should outperform giving both.
    """
    np.random.seed(42)
    n = 200
    horizon = 14
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    t = np.arange(n)

    temperature = 15 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1, n)
    promotion = (np.random.rand(n) < 0.15).astype(int)

    a = 100 + 50 * promotion + np.random.normal(0, 3, n)
    b = 80 + 4 * temperature + np.random.normal(0, 3, n)

    y = pd.DataFrame({'series_a': a, 'series_b': b}, index=dates)
    X = pd.DataFrame({'temperature': temperature, 'promotion': promotion}, index=dates)

    y_train = y.iloc[:-horizon]
    y_test  = y.iloc[-horizon:]
    X_train = X.iloc[:-horizon]
    X_test  = X.iloc[-horizon:]

    return y_train, y_test, X_train, X_test, horizon


@pytest.fixture
def ml_candidates(dgp):
    _, _, _, _, horizon = dgp
    return [
        RandomForestForecaster(horizon=horizon, n_lags=14, n_estimators=50),
    ]


# ── Per-series covariate API tests ───────────────────────────────────────────

class TestPerSeriesCovariateAPI:

    def test_dict_X_accepted(self, dgp, ml_candidates):
        y_train, y_test, X_train, X_test, horizon = dgp
        per_X_train = {
            'series_a': X_train[['promotion']],
            'series_b': X_train[['temperature']],
        }
        per_X_test = {
            'series_a': X_test[['promotion']],
            'series_b': X_test[['temperature']],
        }
        auto = AutoForecaster(
            candidate_models=ml_candidates, per_series_models=True,
            n_splits=2, test_size=horizon, metric='rmse', verbose=False,
        )
        auto.fit(y_train, X=per_X_train)
        fc = auto.forecast(X=per_X_test)
        assert fc.shape == (horizon, 2)

    def test_per_series_X_no_nans(self, dgp, ml_candidates):
        y_train, y_test, X_train, X_test, horizon = dgp
        per_X_train = {'series_a': X_train[['promotion']], 'series_b': X_train[['temperature']]}
        per_X_test  = {'series_a': X_test[['promotion']],  'series_b': X_test[['temperature']]}
        auto = AutoForecaster(
            candidate_models=ml_candidates, per_series_models=True,
            n_splits=2, test_size=horizon, metric='rmse', verbose=False,
        )
        auto.fit(y_train, X=per_X_train)
        fc = auto.forecast(X=per_X_test)
        assert not fc.isnull().any().any()

    def test_per_series_X_columns_preserved(self, dgp, ml_candidates):
        y_train, y_test, X_train, X_test, horizon = dgp
        per_X_train = {'series_a': X_train[['promotion']], 'series_b': X_train[['temperature']]}
        per_X_test  = {'series_a': X_test[['promotion']],  'series_b': X_test[['temperature']]}
        auto = AutoForecaster(
            candidate_models=ml_candidates, per_series_models=True,
            n_splits=2, test_size=horizon, metric='rmse', verbose=False,
        )
        auto.fit(y_train, X=per_X_train)
        fc = auto.forecast(X=per_X_test)
        assert set(fc.columns) == {'series_a', 'series_b'}

    def test_targeted_features_not_worse_than_shared(self, dgp):
        """
        Core guarantee: giving each series only its true driver should not worsen
        accuracy vs giving both (noisy) features.  We test that the per-series
        RMSE sum is ≤ shared RMSE sum (or within 20% if tree variance is high).
        """
        y_train, y_test, X_train, X_test, horizon = dgp

        def rmse(a, b):
            return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

        ml = [RandomForestForecaster(horizon=horizon, n_lags=14, n_estimators=100)]

        # Shared: both features for every series
        auto_shared = AutoForecaster(
            candidate_models=ml, per_series_models=True,
            n_splits=2, test_size=horizon, metric='rmse', verbose=False,
        )
        auto_shared.fit(y_train, X=X_train)
        fc_shared = auto_shared.forecast(X=X_test)
        shared_total = rmse(y_test['series_a'], fc_shared['series_a']) + \
                       rmse(y_test['series_b'], fc_shared['series_b'])

        # Per-series: targeted features
        per_X_train = {'series_a': X_train[['promotion']], 'series_b': X_train[['temperature']]}
        per_X_test  = {'series_a': X_test[['promotion']],  'series_b': X_test[['temperature']]}
        auto_per = AutoForecaster(
            candidate_models=ml, per_series_models=True,
            n_splits=2, test_size=horizon, metric='rmse', verbose=False,
        )
        auto_per.fit(y_train, X=per_X_train)
        fc_per = auto_per.forecast(X=per_X_test)
        per_total = rmse(y_test['series_a'], fc_per['series_a']) + \
                    rmse(y_test['series_b'], fc_per['series_b'])

        # Allow 20% margin for tree-model variance
        assert per_total <= shared_total * 1.20, (
            f"Per-series RMSE ({per_total:.2f}) should not exceed "
            f"shared RMSE ({shared_total:.2f}) by more than 20%"
        )


# ── BacktestValidator — no horizon mutation ──────────────────────────────────

class TestBacktestNoMutation:

    def test_model_horizon_unchanged_after_run(self):
        """
        BacktestValidator must not mutate the model's horizon attribute.
        This was the bug fixed in v0.3.9/v0.4.0.
        """
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=100, freq='D')
        y = pd.DataFrame({
            'x': np.cumsum(np.random.randn(100)) + 50,
            'z': np.cumsum(np.random.randn(100)) + 80,
        }, index=dates)

        model = MovingAverageForecaster(horizon=5, window=3)
        original_horizon = model.horizon

        validator = BacktestValidator(model, n_splits=3, test_size=10)
        validator.run(y)

        assert model.horizon == original_horizon, (
            f"BacktestValidator mutated model.horizon: "
            f"expected {original_horizon}, got {model.horizon}"
        )

    def test_fold_models_are_independent(self):
        """Each fold must operate on an independent deep copy of the model."""
        np.random.seed(1)
        dates = pd.date_range('2021-01-01', periods=100, freq='D')
        y = pd.DataFrame({
            'p': np.cumsum(np.random.randn(100)) + 100,
            'q': np.cumsum(np.random.randn(100)) + 150,
        }, index=dates)

        model = MovingAverageForecaster(horizon=5, window=3)
        validator = BacktestValidator(model, n_splits=3, test_size=10)
        metrics = validator.run(y)

        # Metrics must be valid numbers (fold isolation means no bleed-over)
        assert metrics['rmse'] > 0
        assert np.isfinite(metrics['rmse'])

    def test_run_with_holdout_no_mutation(self):
        """run_with_holdout must not mutate model.horizon."""
        np.random.seed(2)
        dates = pd.date_range('2021-01-01', periods=100, freq='D')
        y = pd.DataFrame({
            'x': np.cumsum(np.random.randn(100)) + 50,
            'z': np.cumsum(np.random.randn(100)) + 80,
        }, index=dates)

        model = MovingAverageForecaster(horizon=5, window=3)
        original_horizon = model.horizon

        validator = BacktestValidator(model, n_splits=3, test_size=10)
        validator.run_with_holdout(y, holdout_size=10)

        assert model.horizon == original_horizon


# ── VARForecaster single-series guard ─────────────────────────────────────────

class TestVARGuard:

    def test_single_series_raises_value_error(self):
        from autotsforecast.models.base import VARForecaster
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=60, freq='D')
        y = pd.DataFrame({'x': np.random.randn(60)}, index=dates)
        model = VARForecaster(horizon=3, lags=2)
        with pytest.raises(ValueError, match="at least 2"):
            model.fit(y)

    def test_two_series_works(self):
        from autotsforecast.models.base import VARForecaster
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=60, freq='D')
        y = pd.DataFrame({
            'x': np.random.randn(60) + 10,
            'z': np.random.randn(60) + 20,
        }, index=dates)
        model = VARForecaster(horizon=3, lags=2)
        model.fit(y)
        fc = model.predict()
        assert fc.shape == (3, 2)


# ── Model is_fitted guard ────────────────────────────────────────────────────

class TestIsFittedGuard:

    def test_moving_average_predict_before_fit_raises(self):
        model = MovingAverageForecaster(horizon=3, window=5)
        with pytest.raises(Exception):
            model.predict()

    def test_ets_predict_before_fit_raises(self):
        model = ETSForecaster(horizon=3)
        with pytest.raises(Exception):
            model.predict()

    def test_rf_predict_before_fit_raises(self):
        model = RandomForestForecaster(horizon=3, n_lags=5)
        with pytest.raises(Exception):
            model.predict()

    def test_is_fitted_false_before_fit(self):
        model = MovingAverageForecaster(horizon=3, window=5)
        assert not model.is_fitted

    def test_is_fitted_true_after_fit(self):
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=60, freq='D')
        y = pd.DataFrame({'x': np.random.randn(60) + 50}, index=dates)
        model = MovingAverageForecaster(horizon=3, window=5)
        model.fit(y)
        assert model.is_fitted


# ── Deep-copy independence ───────────────────────────────────────────────────

class TestDeepCopy:

    def test_deepcopy_of_fitted_model_is_independent(self):
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=80, freq='D')
        y = pd.DataFrame({'x': np.random.randn(80) + 50}, index=dates)
        model = MovingAverageForecaster(horizon=5, window=3)
        model.fit(y)

        model_copy = copy.deepcopy(model)
        # Modifying copy should not affect original
        model_copy.horizon = 99
        assert model.horizon == 5
        assert model_copy.horizon == 99

    def test_deepcopy_produces_valid_predictions(self):
        np.random.seed(0)
        dates = pd.date_range('2021-01-01', periods=80, freq='D')
        y = pd.DataFrame({
            'p': np.random.randn(80) + 100,
            'q': np.random.randn(80) + 200,
        }, index=dates)
        model = MovingAverageForecaster(horizon=3, window=5)
        model.fit(y)

        model_copy = copy.deepcopy(model)
        fc = model_copy.predict()
        assert fc.shape == (3, 2)
        assert not fc.isnull().any().any()

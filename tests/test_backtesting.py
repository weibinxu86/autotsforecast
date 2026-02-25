import unittest

import numpy as np
import pandas as pd

from autotsforecast.backtesting.validator import BacktestValidator
from autotsforecast.models.base import VARForecaster


class TestBacktestValidator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        index = pd.date_range("2020-01-01", periods=80, freq="D")
        self.y = pd.DataFrame(
            {
                "sales_A": rng.normal(loc=100, scale=2, size=len(index)).cumsum(),
                "sales_B": rng.normal(loc=150, scale=3, size=len(index)).cumsum(),
            },
            index=index,
        )
        model = VARForecaster(horizon=1, lags=1)
        self.validator = BacktestValidator(model, n_splits=3, test_size=10, window_type="expanding")

    def test_run_returns_metrics(self):
        metrics = self.validator.run(self.y)

        for key in ["rmse", "mae", "mape", "smape", "r2"]:
            self.assertIn(key, metrics)

    def test_run_returns_mse(self):
        """MSE must be present so that validate_results() works correctly."""
        metrics = self.validator.run(self.y)
        self.assertIn("mse", metrics)
        self.assertGreaterEqual(metrics["mse"], 0)

    def test_validate_results_returns_true_for_valid_metrics(self):
        """validate_results() should return True when given a valid metrics dict."""
        metrics = self.validator.run(self.y)
        self.assertTrue(self.validator.validate_results(metrics))

    def test_validate_results_returns_false_for_invalid_metrics(self):
        """validate_results() should return False for negative or missing metrics."""
        self.assertFalse(self.validator.validate_results({"mse": -1, "mae": 0.5}))
        self.assertFalse(self.validator.validate_results({"mae": 0.5}))  # mse missing


if __name__ == "__main__":
    unittest.main()
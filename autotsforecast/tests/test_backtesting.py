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


if __name__ == "__main__":
    unittest.main()
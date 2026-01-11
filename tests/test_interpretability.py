import unittest

import numpy as np
import pandas as pd

from autotsforecast.interpretability.drivers import DriversAnalyzer
from autotsforecast.models.base import LinearForecaster


class TestDriversAnalyzer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 50

        self.X = pd.DataFrame(
            {
                "price": rng.normal(loc=10, scale=1, size=n),
                "promotion": rng.integers(0, 2, size=n),
                "temperature": rng.normal(loc=20, scale=3, size=n),
            }
        )
        self.y = pd.DataFrame({"sales": 100 + 2 * self.X["price"] + 5 * self.X["promotion"]})

        model = LinearForecaster(horizon=1)
        model.fit(self.y, self.X)
        self.analyzer = DriversAnalyzer(model)

    def test_analyze_drivers_returns_sensitivity(self):
        results = self.analyzer.analyze_drivers(
            self.X,
            self.y,
            numerical_features=["price", "temperature"],
            categorical_features=["promotion"],
        )

        self.assertIn("sensitivity", results)
        self.assertIsInstance(results["sensitivity"], pd.DataFrame)
        self.assertIn("sales", results["sensitivity"].columns)


if __name__ == "__main__":
    unittest.main()
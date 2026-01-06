import unittest

import pandas as pd

from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler


class TestHierarchicalReconciler(unittest.TestCase):
    def test_reconcile_bottom_up_is_coherent(self):
        forecasts = pd.DataFrame(
            {
                "Total": [100, 110, 120],
                "A": [60, 65, 70],
                "B": [40, 45, 50],
                "A1": [35, 38, 40],
                "A2": [25, 27, 30],
                "B1": [22, 25, 28],
                "B2": [18, 20, 22],
            }
        )
        hierarchy = {"Total": ["A", "B"], "A": ["A1", "A2"], "B": ["B1", "B2"]}

        reconciler = HierarchicalReconciler(forecasts, hierarchy)
        reconciler.reconcile(method="bottom_up")

        self.assertTrue(reconciler.validate_coherency())

    def test_invalid_hierarchy_raises(self):
        forecasts = pd.DataFrame({"Total": [100, 110], "A": [60, 65]})
        hierarchy = {"Total": ["A", "B"]}  # B is missing

        with self.assertRaises(ValueError):
            HierarchicalReconciler(forecasts, hierarchy)


if __name__ == "__main__":
    unittest.main()
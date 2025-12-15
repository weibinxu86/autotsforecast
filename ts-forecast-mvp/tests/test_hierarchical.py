import unittest
from ts_forecast.hierarchical.reconciliation import HierarchicalReconciler

class TestHierarchicalReconciler(unittest.TestCase):

    def setUp(self):
        self.reconciler = HierarchicalReconciler()

    def test_reconcile_forecasts(self):
        # Sample forecasts and hierarchy
        forecasts = {
            'level_1': [100, 200, 300],
            'level_2': [50, 100, 150]
        }
        hierarchy = {
            'level_1': ['level_2']
        }
        reconciled_forecasts = self.reconciler.reconcile_forecasts(forecasts, hierarchy)
        expected_forecasts = {
            'level_1': [100, 200, 300],
            'level_2': [50, 100, 150]
        }
        self.assertEqual(reconciled_forecasts, expected_forecasts)

    def test_invalid_hierarchy(self):
        forecasts = {
            'level_1': [100, 200, 300]
        }
        hierarchy = {
            'level_1': ['level_2']
        }
        with self.assertRaises(ValueError):
            self.reconciler.reconcile_forecasts(forecasts, hierarchy)

    def test_empty_forecasts(self):
        forecasts = {}
        hierarchy = {}
        reconciled_forecasts = self.reconciler.reconcile_forecasts(forecasts, hierarchy)
        self.assertEqual(reconciled_forecasts, {})

if __name__ == '__main__':
    unittest.main()
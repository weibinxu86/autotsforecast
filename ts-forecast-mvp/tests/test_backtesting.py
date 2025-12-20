import unittest
from autotsforecast.backtesting.validator import BacktestValidator

class TestBacktestValidator(unittest.TestCase):

    def setUp(self):
        self.validator = BacktestValidator()

    def test_validate_results(self):
        # Example test case for validating backtest results
        results = {
            'model': 'test_model',
            'mse': 0.1,
            'mae': 0.05
        }
        is_valid = self.validator.validate_results(results)
        self.assertTrue(is_valid)

    def test_invalid_results(self):
        # Example test case for invalid backtest results
        results = {
            'model': 'test_model',
            'mse': -0.1,  # Invalid MSE
            'mae': 0.05
        }
        is_valid = self.validator.validate_results(results)
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()
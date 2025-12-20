import unittest
from autotsforecast.interpretability.drivers import DriversAnalyzer

class TestDriversAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = DriversAnalyzer()

    def test_analyze_drivers(self):
        # Sample data for testing
        covariates = {
            'temperature': [30, 32, 31, 29, 28],
            'humidity': [70, 65, 68, 72, 75]
        }
        predictions = [100, 110, 105, 95, 90]
        
        # Test the analyze_drivers method
        results = self.analyzer.analyze_drivers(covariates, predictions)
        self.assertIsNotNone(results)
        self.assertIn('importance', results)

    def test_plot_driver_importance(self):
        covariates = {
            'temperature': [30, 32, 31, 29, 28],
            'humidity': [70, 65, 68, 72, 75]
        }
        predictions = [100, 110, 105, 95, 90]
        
        # Test the plot_driver_importance method
        self.analyzer.analyze_drivers(covariates, predictions)  # Ensure analysis is done first
        plot = self.analyzer.plot_driver_importance()
        self.assertIsNotNone(plot)

if __name__ == '__main__':
    unittest.main()
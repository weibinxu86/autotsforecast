"""TS-Forecast: Multivariate time series forecasting toolkit

A comprehensive package for multivariate time series forecasting with:
- Model selection and comparison
- Backtesting and cross-validation
- Hierarchical forecast reconciliation
- Interpretability and driver analysis
"""

__version__ = "0.1.0"

from ts_forecast.models.base import BaseForecaster, VARForecaster, LinearForecaster, MovingAverageForecaster
from ts_forecast.models.selection import ModelSelector
from ts_forecast.backtesting.validator import BacktestValidator
from ts_forecast.hierarchical.reconciliation import HierarchicalReconciler
from ts_forecast.interpretability.drivers import DriverAnalyzer

__all__ = [
    "BaseForecaster",
    "VARForecaster",
    "LinearForecaster",
    "MovingAverageForecaster",
    "ModelSelector",
    "BacktestValidator",
    "HierarchicalReconciler",
    "DriverAnalyzer",
]
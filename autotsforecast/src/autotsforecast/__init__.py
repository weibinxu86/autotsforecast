"""TS-Forecast: Multivariate time series forecasting toolkit

A comprehensive package for multivariate time series forecasting with:
- Model selection and comparison
- Backtesting and cross-validation
- Hierarchical forecast reconciliation
- Interpretability and driver analysis
"""

__version__ = "0.1.0"

from autotsforecast.models.base import BaseForecaster, VARForecaster, LinearForecaster, MovingAverageForecaster
from autotsforecast.models.selection import ModelSelector
from autotsforecast.models.external import (
    RandomForestForecaster,
    XGBoostForecaster,
    ProphetForecaster,
    ARIMAForecaster,
    ETSForecaster,
    LSTMForecaster
)
from autotsforecast.backtesting.validator import BacktestValidator
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler
from autotsforecast.interpretability.drivers import DriverAnalyzer
from autotsforecast.forecaster import AutoForecaster, get_default_candidate_models

__all__ = [
    "BaseForecaster",
    "VARForecaster",
    "LinearForecaster",
    "MovingAverageForecaster",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "ProphetForecaster",
    "ARIMAForecaster",
    "ETSForecaster",
    "LSTMForecaster",
    "ModelSelector",
    "BacktestValidator",
    "HierarchicalReconciler",
    "AutoForecaster",
    "get_default_candidate_models",
    "DriverAnalyzer",
]
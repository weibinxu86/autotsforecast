"""AutoTSForecast: Automated Multivariate Time Series Forecasting Toolkit

A comprehensive package for multivariate time series forecasting with:
- Automatic model selection and comparison
- Backtesting and cross-validation
- Hierarchical forecast reconciliation
- Interpretability and driver analysis
- Prediction intervals and uncertainty quantification
- Calendar feature engineering
- Interactive visualization
- Parallel processing for speed
"""

__version__ = "0.3.3"

# Core models
from autotsforecast.models.base import BaseForecaster, VARForecaster, LinearForecaster, MovingAverageForecaster
from autotsforecast.models.selection import ModelSelector
from autotsforecast.models.external import (
    Chronos2Forecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    ProphetForecaster,
    ARIMAForecaster,
    ETSForecaster,
    LSTMForecaster
)

# Core components
from autotsforecast.backtesting.validator import BacktestValidator
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler
from autotsforecast.interpretability.drivers import DriverAnalyzer
from autotsforecast.forecaster import AutoForecaster, get_default_candidate_models

# New features: Calendar & Feature Engineering
from autotsforecast.features.calendar import CalendarFeatures, add_calendar_features
from autotsforecast.features.engine import FeatureEngine

# New features: Prediction Intervals
from autotsforecast.uncertainty.intervals import PredictionIntervals, ConformalPredictor

# New features: Visualization & Progress
from autotsforecast.visualization.plots import (
    plot_forecast,
    plot_forecast_interactive,
    plot_model_comparison,
    plot_feature_importance,
    plot_residuals,
    plot_components,
    ForecastPlotter
)
from autotsforecast.visualization.progress import ProgressTracker, progress_bar

# New features: Parallel Processing
from autotsforecast.utils.parallel import ParallelForecaster, parallel_map, batch_forecast

__all__ = [
    # Core Models
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
    "Chronos2Forecaster",
    # Core Components
    "ModelSelector",
    "BacktestValidator",
    "HierarchicalReconciler",
    "AutoForecaster",
    "get_default_candidate_models",
    "DriverAnalyzer",
    # Feature Engineering
    "CalendarFeatures",
    "add_calendar_features",
    "FeatureEngine",
    # Uncertainty Quantification
    "PredictionIntervals",
    "ConformalPredictor",
    # Visualization
    "plot_forecast",
    "plot_forecast_interactive",
    "plot_model_comparison",
    "plot_feature_importance",
    "plot_residuals",
    "plot_components",
    "ForecastPlotter",
    "ProgressTracker",
    "progress_bar",
    # Parallel Processing
    "ParallelForecaster",
    "parallel_map",
    "batch_forecast",
]
# AutoTSForecast

**Automated Time Series Forecasting with Per-Series Model Selection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/autotsforecast)](https://pypi.org/project/autotsforecast/)

AutoTSForecast automatically finds the best forecasting model for each of your time series. No more guessing whether Prophet, ARIMA, XGBoost, or **Chronos-2 foundation model** works best — let the algorithm decide. **New: Zero-shot forecasting with Chronos-2 — no training required, just pass your data and get state-of-the-art predictions!**

## 🚀 Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Chronos-2 Foundation Model** 🆕 | Zero-shot forecasting with pre-trained models (9M-710M params) | **No training needed** — just pass your data! |
| **Per-Series Model Selection** | Automatically pick the best model for *each* series | Different series, different patterns → optimal accuracy |
| **Per-Series Covariates** 🆕 | Pass different features to different series | Products driven by different factors get custom features |
| **Prediction Intervals** 🆕 | Conformal prediction with coverage guarantees | Quantify uncertainty without assumptions |
| **Calendar Features** 🆕 | Auto-extract day-of-week, month, holidays | Handle seasonality automatically |
| **Hierarchical Reconciliation** | Ensure forecasts add up (total = sum of parts) | Coherent forecasts across organizational levels |
| **Parallel Processing** 🆕 | Fit many series simultaneously | Scale to thousands of series |
| **Interpretability** | Sensitivity analysis & SHAP | Understand what drives your forecasts |

## ✨ What's New in v0.3.8+

- **🚀 Chronos-2 Foundation Model** — Zero-shot forecasting with state-of-the-art pre-trained models (no training needed!)
- **🎯 Per-Series Covariates** — Pass different features to different series via `X={series: df}`
- **📊 Prediction Intervals** — Conformal prediction for uncertainty quantification
- **📅 Calendar Features** — Automatic time-based feature extraction with cyclical encoding
- **🖼️ Better Visualization** — Static (matplotlib) and interactive (Plotly) forecast plots
- **⚡ Parallel Processing** — Speed up multi-series forecasting with joblib
- **📈 Progress Tracking** — Rich progress bars for long-running operations

## Installation

### 🚀 Recommended: Install Everything

```bash
pip install "autotsforecast[all]"
```

This installs **all 10 models** plus visualization, interpretability, and new features.

### Basic Install (Core Models Only)

```bash
pip install autotsforecast
```

This gives you 6 models **out of the box**:
| Model | Description |
|-------|-------------|
| `ARIMAForecaster` | Classical ARIMA |
| `ETSForecaster` | Exponential smoothing |
| `LinearForecaster` | Linear regression with lags |
| `MovingAverageForecaster` | Simple baseline |
| `RandomForestForecaster` | ML with covariates ✓ |
| `VARForecaster` | Vector autoregression |

### Install Specific Optional Models

Some models require additional dependencies:

```bash
# Add XGBoost (gradient boosting with covariates)
pip install "autotsforecast[ml]"

# Add Prophet (Facebook's forecasting library)
pip install "autotsforecast[prophet]"

# Add LSTM (deep learning)
pip install "autotsforecast[neural]"

# Add Chronos-2 (foundation model - state-of-the-art zero-shot forecasting)
pip install "autotsforecast[chronos]"

# Add SHAP (interpretability)
pip install "autotsforecast[interpret]"

# Add visualization tools (Plotly, progress bars)
pip install "autotsforecast[viz]"
```

### Model Availability Summary

| Model | Basic Install | Extra Required |
|-------|:-------------:|----------------|
| ARIMA, ETS, Linear, MovingAverage, RandomForest, VAR | ✅ | — |
| XGBoostForecaster | ❌ | `pip install "autotsforecast[ml]"` |
| ProphetForecaster | ❌ | `pip install "autotsforecast[prophet]"` |
| LSTMForecaster | ❌ | `pip install "autotsforecast[neural]"` |
| Chronos2Forecaster | ❌ | `pip install "autotsforecast[chronos]"` |
| SHAP Analysis | ❌ | `pip install "autotsforecast[interpret]"` |
| Interactive Plots | ❌ | `pip install "autotsforecast[viz]"` |

## Quick Start

### 1. AutoForecaster — Let the Algorithm Choose

```python
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import ARIMAForecaster, ProphetForecaster, RandomForestForecaster, Chronos2Forecaster

# Your time series data (pandas DataFrame)
# y = pd.DataFrame({'series_a': [...], 'series_b': [...]})

# Define candidate models (including Chronos-2 foundation model)
candidates = [
    ARIMAForecaster(horizon=14),
    ProphetForecaster(horizon=14),
    RandomForestForecaster(horizon=14, n_lags=7),
    MovingAverageForecaster(horizon=14, window=7),
    Chronos2Forecaster(horizon=14, model_name='autogluon/chronos-2-small'),  # Zero-shot foundation model
]

# AutoForecaster picks the best model across all series (default)
auto = AutoForecaster(candidate_models=candidates, metric='rmse')
auto.fit(y_train)
forecasts = auto.forecast()

# See which model was selected
print(auto.best_model_name_)  # e.g., 'Chronos2Forecaster'

# OR: Pick the best model for EACH series separately
auto = AutoForecaster(candidate_models=candidates, metric='rmse', per_series_models=True)
auto.fit(y_train)
forecasts = auto.forecast()

# See which models were selected per series
print(auto.best_model_names_)  # e.g., {'series_a': 'Chronos2Forecaster', 'series_b': 'ARIMAForecaster'}
```

### 2. Using Covariates (External Features)

```python
from autotsforecast.models.external import XGBoostForecaster

# X contains external features (temperature, promotions, etc.)
model = XGBoostForecaster(horizon=14, n_lags=7)
model.fit(y_train, X=X_train)
forecasts = model.predict(X=X_test)
```

**Models supporting covariates:** Prophet, XGBoost, RandomForest, Linear

### 2.1 Calendar Features

Automatic time-based feature extraction:

```python
from autotsforecast.features.calendar import CalendarFeatures

# Auto-detect features with cyclical encoding
cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)

# Generate future features for forecasting
future_features = cal.transform_future(horizon=30)
```

### 2.2 Per-Series Covariates — Different Features for Each Series

**Use Case:** When different time series are driven by different external factors.

```python
from autotsforecast import AutoForecaster
from autotsforecast.models.base import LinearForecaster
from autotsforecast.models.external import RandomForestForecaster, XGBoostForecaster

# Example: Forecasting sales for different products
# Product A: Summer product (driven by weather and advertising)
X_product_a = pd.DataFrame({
    'temperature': [...],      # Weather matters for Product A
    'advertising_spend': [...] # Marketing campaigns
}, index=dates)

# Product B: Everyday product (driven by pricing and promotions)
X_product_b = pd.DataFrame({
    'competitor_price': [...],  # Price competition matters for Product B
    'promotion_active': [...]   # Promotional events
}, index=dates)

# Create dictionary mapping each series to its covariates
X_train_dict = {
    'product_a_sales': X_product_a_train,
    'product_b_sales': X_product_b_train
}

X_test_dict = {
    'product_a_sales': X_product_a_test,
    'product_b_sales': X_product_b_test
}

# Define candidate models
candidates = [
    LinearForecaster(horizon=14),
    RandomForestForecaster(horizon=14, n_lags=7),
    XGBoostForecaster(horizon=14, n_lags=7)
]

# AutoForecaster with per-series model selection
auto = AutoForecaster(
    candidate_models=candidates,
    per_series_models=True,  # Select best model for each series
    metric='rmse'
)

# Fit: Each series uses its own covariates
auto.fit(y_train, X=X_train_dict)

# Forecast: Provide future covariates for each series
forecasts = auto.forecast(X=X_test_dict)

# See which model was selected for each series
print(auto.best_model_names_)
# Output: {'product_a_sales': 'RandomForestForecaster', 
#          'product_b_sales': 'XGBoostForecaster'}
```

**Key Benefits:**
- ✅ Each series uses only relevant features (reduces noise)
- ✅ Better accuracy through targeted feature engineering
- ✅ Handle heterogeneous products with different drivers
- ✅ Scalable to large portfolios with diverse characteristics
- ✅ Backward compatible: still works with single DataFrame for all series

### 3. Hierarchical Reconciliation

Ensure forecasts add up correctly (e.g., `total = region_a + region_b`):

```python
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

hierarchy = {'total': ['region_a', 'region_b']}
reconciler = HierarchicalReconciler(forecasts=base_forecasts, hierarchy=hierarchy)
reconciler.reconcile(method='ols')
coherent_forecasts = reconciler.reconciled_forecasts
```

### 4. Backtesting (Cross-Validation)

```python
from autotsforecast.backtesting.validator import BacktestValidator

validator = BacktestValidator(model=my_model, n_splits=5, test_size=14)
validator.run(y_train, X=X_train)

# Get results
results = validator.get_fold_results()  # RMSE per fold
print(f"Average RMSE: {results['rmse'].mean():.2f}")
```

### 5. Interpretability (Feature Importance)

```python
from autotsforecast.interpretability.drivers import DriverAnalyzer

analyzer = DriverAnalyzer(model=fitted_model, feature_names=['temperature', 'promotion'])
importance = analyzer.calculate_feature_importance(X_test, y_test, method='sensitivity')
```

### 6. Prediction Intervals

Generate prediction intervals with conformal prediction:

```python
from autotsforecast.uncertainty.intervals import PredictionIntervals

# After fitting a model
pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model, y_train)
intervals = pi.predict(forecasts)

# Access intervals
print(intervals['lower_95'], intervals['upper_95'])
```

### 7. Chronos-2 Foundation Model (Zero-Shot Forecasting)

State-of-the-art pretrained model - **no training needed!**

```python
from autotsforecast.models.external import Chronos2Forecaster

# Initialize with default model (120M params, best accuracy)
model = Chronos2Forecaster(
    horizon=30,
    model_name="amazon/chronos-2"  # or "autogluon/chronos-2-small" for faster inference
)

# Fit (just stores context, no training!)
model.fit(y_train)

# Generate point forecasts (median)
forecasts = model.predict()

# Generate probabilistic forecasts with uncertainty quantification
quantile_forecasts = model.predict_quantiles(quantile_levels=[0.1, 0.5, 0.9])
# Returns: value_q10, value_q50, value_q90 columns
```

**Available Model Sizes:**
- `amazon/chronos-2` - 120M params (best accuracy)
- `autogluon/chronos-2-small` - 28M params (balanced, **tested: 0.63% MAPE**)
- `amazon/chronos-bolt-tiny` - 9M params (ultra fast)
- `amazon/chronos-bolt-small` - 48M params (balanced speed/accuracy)
- `amazon/chronos-bolt-base` - 205M params (high accuracy + fast)

**Why Chronos-2?**
- ✅ Zero-shot: No training required
- ✅ State-of-the-art accuracy on multiple benchmarks
- ✅ Built-in uncertainty quantification
- ✅ Multiple model sizes for different use cases

### 8. Visualization

Create publication-ready plots:

```python
from autotsforecast.visualization.plots import plot_forecast, plot_forecast_interactive

# Static matplotlib plot
fig = plot_forecast(y_train, y_test, forecast, lower=lower_95, upper=upper_95)

# Interactive Plotly plot
fig = plot_forecast_interactive(y_train, y_test, forecast)
fig.show()
```

### 9. Parallel Processing

Speed up multi-series forecasting:

```python
from autotsforecast.utils.parallel import ParallelForecaster, parallel_map

# Create parallel forecaster
pf = ParallelForecaster(n_jobs=4)

# Fit each series in parallel
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=14),
    y=y_train,
    X=X_train
)
```

## Requirements

- Python ≥ 3.8
- Core: numpy, pandas, scikit-learn, statsmodels, scipy, joblib

## License

MIT License

## Contributing

Contributions welcome! Visit the GitHub repository to get started.

```bibtex
@software{autotsforecast2026,
  title={AutoTSForecast: Automated Time Series Forecasting},
  author={Weibin Xu},
  year={2026},
  url={https://github.com/weibinxu86/autotsforecast}
}
```


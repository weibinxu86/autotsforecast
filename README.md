# AutoTSForecast

**Automated Time Series Forecasting with Per-Series Model Selection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/autotsforecast)](https://pypi.org/project/autotsforecast/)

AutoTSForecast automatically finds the best forecasting model for each of your time series. No more guessing whether Prophet, ARIMA, or XGBoost works best — let the algorithm decide.

## Installation

### 🚀 Recommended: Install Everything

```bash
pip install "autotsforecast[all]"
```

This installs **all 9 models** plus visualization and interpretability tools.

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

# Add SHAP (interpretability)
pip install "autotsforecast[interpret]"

# Add visualization tools
pip install "autotsforecast[viz]"
```

### Model Availability Summary

| Model | Basic Install | Extra Required |
|-------|:-------------:|----------------|
| ARIMA, ETS, Linear, MovingAverage, RandomForest, VAR | ✅ | — |
| XGBoostForecaster | ❌ | `pip install "autotsforecast[ml]"` |
| ProphetForecaster | ❌ | `pip install "autotsforecast[prophet]"` |
| LSTMForecaster | ❌ | `pip install "autotsforecast[neural]"` |
| SHAP Analysis | ❌ | `pip install "autotsforecast[interpret]"` |

## Quick Start

### 1. AutoForecaster — Let the Algorithm Choose

```python
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import ARIMAForecaster, ProphetForecaster, RandomForestForecaster

# Your time series data (pandas DataFrame)
# y = pd.DataFrame({'series_a': [...], 'series_b': [...]})

# Define candidate models
candidates = [
    ARIMAForecaster(horizon=14),
    ProphetForecaster(horizon=14),
    RandomForestForecaster(horizon=14, n_lags=7),
    MovingAverageForecaster(horizon=14, window=7),
]

# AutoForecaster picks the best model across all series (default)
auto = AutoForecaster(candidate_models=candidates, metric='rmse')
auto.fit(y_train)
forecasts = auto.forecast()

# See which model was selected
print(auto.best_model_name_)  # e.g., 'ProphetForecaster'

# OR: Pick the best model for EACH series separately
auto = AutoForecaster(candidate_models=candidates, metric='rmse', per_series_models=True)
auto.fit(y_train)
forecasts = auto.forecast()

# See which models were selected per series
print(auto.best_model_names_)  # e.g., {'series_a': 'ProphetForecaster', 'series_b': 'ARIMAForecaster'}
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

### 2.1 Per-Series Covariates — Different Features for Each Series

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

## Documentation

- [Quick Start Guide](QUICKSTART.md) — fastest overview
- [API Reference](API_REFERENCE.md) — detailed parameter documentation
- [Parameter Guide](PARAMETER_GUIDE.md) — model parameter recommendations
- [Tutorial Notebook](examples/autotsforecast_tutorial.ipynb) — comprehensive examples

## Requirements

- Python ≥ 3.8
- Core: numpy, pandas, scikit-learn, statsmodels, scipy

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


# AutoTSForecast

**Automated Time Series Forecasting with Per-Series Model Selection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoTSForecast automatically finds the best forecasting model for each of your time series. No more guessing whether Prophet, ARIMA, or XGBoost works best — let the algorithm decide.

## Installation

### Basic Install (Core Models)

```bash
pip install autotsforecast
```

This gives you these models **out of the box**:
| Model | Description |
|-------|-------------|
| `ARIMAForecaster` | Classical ARIMA |
| `ETSForecaster` | Exponential smoothing |
| `LinearForecaster` | Linear regression with lags |
| `MovingAverageForecaster` | Simple baseline |
| `RandomForestForecaster` | ML with covariates ✓ |
| `VARForecaster` | Vector autoregression |

### Install with Optional Models

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

# Install EVERYTHING (recommended for full functionality)
pip install "autotsforecast[all]"
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
from autotsforecast.models.base import RandomForestForecaster, MovingAverageForecaster
from autotsforecast.models.external import ARIMAForecaster, ProphetForecaster

# Your time series data (pandas DataFrame)
# y = pd.DataFrame({'series_a': [...], 'series_b': [...]})

# Define candidate models
candidates = [
    ARIMAForecaster(horizon=14),
    ProphetForecaster(horizon=14),
    RandomForestForecaster(horizon=14, n_lags=7),
    MovingAverageForecaster(horizon=14, window=7),
]

# AutoForecaster picks the best model for EACH series
auto = AutoForecaster(candidate_models=candidates, metric='rmse')
auto.fit(y_train)
forecasts = auto.forecast()

# See which models were selected
print(auto.best_model_name_)  # e.g., {'series_a': 'Prophet', 'series_b': 'ARIMA'}
```

### 2. Using Covariates (External Features)

```python
from autotsforecast.models.external import XGBoostForecaster

# X contains external features (temperature, promotions, etc.)
model = XGBoostForecaster(horizon=14, n_lags=7)
model.fit(y_train, X=X_train)
forecasts = model.predict(X=X_test)
```

**Models supporting covariates:** Prophet, ARIMA, XGBoost, RandomForest, LSTM

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

## Full Tutorial

See **[examples/autotsforecast_tutorial.ipynb](examples/autotsforecast_tutorial.ipynb)** for a complete walkthrough covering:

1. **AutoForecaster vs Individual Models** — Proof that per-series selection wins
2. **Hierarchical Reconciliation** — Coherent forecasts with regional improvements
3. **Interpretability** — Understand which features drive predictions

## API Reference

See **[API_REFERENCE.md](API_REFERENCE.md)** for complete parameter documentation.

See **[PARAMETER_GUIDE.md](PARAMETER_GUIDE.md)** for quick parameter lookup.

## Requirements

- Python ≥ 3.8
- Core: numpy, pandas, scikit-learn, statsmodels, scipy

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please fork the repo and submit a pull request.

## Citation

```bibtex
@software{autotsforecast2025,
  title={AutoTSForecast: Automated Time Series Forecasting},
  author={Weibin Xu},
  year={2025},
  url={https://github.com/weibinxu86/autotsforecast}
}
```

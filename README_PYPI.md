# AutoTSForecast

**Automated Time Series Forecasting with Per-Series Model Selection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoTSForecast automatically finds the best forecasting model for each of your time series. No more guessing whether Prophet, ARIMA, or XGBoost works best — let the algorithm decide.

## Installation

```bash
pip install "autotsforecast[all]"
```

## Key Features

### 🤖 AutoForecaster — Automatic Per-Series Model Selection
Evaluates multiple models via cross-validation and automatically selects the best one for each time series.

```python
from autotsforecast import AutoForecaster

auto = AutoForecaster(candidate_models=[...], metric='rmse')
auto.fit(y_train)
forecasts = auto.forecast()
```

### 📊 Hierarchical Reconciliation
Ensure forecasts are coherent across aggregation levels (e.g., regions sum to total).

```python
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

reconciler = HierarchicalReconciler(forecasts=base_forecasts, hierarchy={'total': ['region_a', 'region_b']})
reconciler.reconcile(method='ols')
```

### ✅ Backtesting & Cross-Validation
Robust time series cross-validation to evaluate model performance.

```python
from autotsforecast.backtesting.validator import BacktestValidator

validator = BacktestValidator(model=my_model, n_splits=5, test_size=14)
validator.run(y_train)
```

### 🔍 Interpretability
Understand which features drive your forecasts using sensitivity analysis and SHAP values.

```python
from autotsforecast.interpretability.drivers import DriverAnalyzer

analyzer = DriverAnalyzer(model=fitted_model, feature_names=['temperature', 'promotion'])
importance = analyzer.calculate_feature_importance(X_test, y_test)
```

## Available Models

**Core Models** (included): ARIMA, ETS, MovingAverage, RandomForest, VAR

> `LinearForecaster` is also included but **requires covariates `X`** — it is not part of the default candidate pool.

**Optional Models**: XGBoost, Prophet, LSTM (install with extras: `[ml]`, `[prophet]`, `[neural]`), Chronos-2 (install with `[chronos]`)

**Covariate Support**: Prophet, ARIMA, XGBoost, RandomForest, LSTM, Linear

## Documentation

**Full documentation, tutorials, and examples:**  
[https://github.com/weibinxu86/autotsforecast](https://github.com/weibinxu86/autotsforecast)

## License

MIT License

# AutoTSForecast Quick Start

## Installation

```bash
pip install autotsforecast           # Core
pip install "autotsforecast[all]"    # All features
```

## 1. Basic Forecasting

```python
import pandas as pd
from autotsforecast.models.base import MovingAverageForecaster

# Your data
y_train, y_test = df.iloc[:150], df.iloc[150:]

model = MovingAverageForecaster(horizon=30, window=7)
model.fit(y_train)
predictions = model.predict()
```

## 2. AutoForecaster (Auto Model Selection)

```python
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import RandomForestForecaster, ARIMAForecaster

candidates = [
    MovingAverageForecaster(horizon=30, window=7),
    RandomForestForecaster(horizon=30, n_lags=7),
    ARIMAForecaster(horizon=30),
]

auto = AutoForecaster(candidate_models=candidates, metric='rmse', n_splits=3)
auto.fit(y_train)
forecasts = auto.forecast()
print(f"Best model: {auto.best_model_name_}")
```

## 3. Using Covariates

```python
from autotsforecast.models.external import RandomForestForecaster

model = RandomForestForecaster(horizon=30, n_lags=7)
model.fit(y_train, X=X_train)
predictions = model.predict(X=X_test)
```

## 4. Per-Series Covariates

Different features for different series:

```python
# Different covariates per series
X_train_dict = {
    'series_a': X_train[['temperature']],   # Series A uses temperature
    'series_b': X_train[['promotion']],     # Series B uses promotion
}

auto = AutoForecaster(candidates, per_series_models=True)
auto.fit(y_train, X=X_train_dict)
forecasts = auto.forecast(X=X_test_dict)
```

## 5. Prediction Intervals

```python
from autotsforecast.uncertainty.intervals import PredictionIntervals

pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model, y_train)
intervals = pi.predict(forecasts)
# intervals['lower_95'], intervals['upper_95']
```

## 6. Calendar Features

```python
from autotsforecast.features.calendar import CalendarFeatures

cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)
```

## 7. Hierarchical Reconciliation

```python
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

hierarchy = {'total': ['region_a', 'region_b']}
reconciler = HierarchicalReconciler(forecasts, hierarchy)
reconciler.reconcile(method='ols')
coherent = reconciler.reconciled_forecasts
```

## 8. Backtesting

```python
from autotsforecast.backtesting.validator import BacktestValidator

validator = BacktestValidator(model=model, n_splits=5, test_size=14)
validator.run(y_train, X=X_train)
results = validator.get_fold_results()
```

## 9. Parallel Processing

```python
from autotsforecast.utils.parallel import ParallelForecaster

pf = ParallelForecaster(n_jobs=4)
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=14, n_lags=7),
    y=y_train, X=X_train
)
```

## 10. Interpretability

```python
from autotsforecast.interpretability.drivers import DriverAnalyzer

analyzer = DriverAnalyzer(model, feature_names=['temp', 'promo'])
importance = analyzer.calculate_feature_importance(X_test, y_test, method='sensitivity')
```

## Model Comparison

| Model | Covariates | Speed | Best For |
|-------|-----------|-------|----------|
| MovingAverage | ❌ | ⭐⭐⭐ | Simple baselines |
| ARIMA/ETS | ❌ | ⭐⭐ | Classical time series |
| RandomForest | ✅ | ⭐⭐ | General purpose |
| XGBoost | ✅ | ⭐⭐ | High accuracy |
| Prophet | ✅ | ⭐⭐ | Seasonality + holidays |

## Common Parameters

| Parameter | Typical Values | Description |
|-----------|---------------|-------------|
| `horizon` | 7-30 | Forecast steps |
| `n_lags` | 7-30 | Lag features |
| `n_estimators` | 100-500 | Trees (RF/XGB) |
| `n_splits` | 3-5 | CV folds |
| `metric` | `'rmse'` | Selection metric |

## More Resources

- [API Reference](API_REFERENCE.md) — Complete parameter docs
- [Tutorial](examples/autotsforecast_tutorial.ipynb) — Hands-on examples

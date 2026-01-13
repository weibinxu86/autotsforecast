# Parameter Guide

Quick reference for finding the right parameters.

## Parameter Lookup

| I want to... | Parameter | Values | Location |
|--------------|-----------|--------|----------|
| Set forecast length | `horizon` | 7-90 | All models |
| Use lag features | `n_lags` | 7-30 | RF, XGBoost, Linear |
| Set CV folds | `n_splits` | 2-5 | AutoForecaster |
| Use per-series models | `per_series_models` | `True` | AutoForecaster |
| Pass per-series covariates | `X` | `Dict[str, DataFrame]` | `fit()` / `forecast()` |
| Parallel processing | `n_jobs` | -1 (all CPUs) | AutoForecaster, ParallelForecaster |
| Prediction intervals | `coverage` | `[0.80, 0.95]` | PredictionIntervals |
| Reconciliation method | `method` | `'ols'`, `'bottom_up'` | HierarchicalReconciler |

## Model Parameters

### RandomForestForecaster / XGBoostForecaster

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `horizon` | required | 7-30 | Forecast steps |
| `n_lags` | 7 | 7-30 | More lags = more history |
| `n_estimators` | 100 | 100-500 | More = better, slower |
| `max_depth` | None | 5-15 | Controls overfitting |

### ARIMAForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `order` | (1,1,1) | (p, d, q) - AR, diff, MA orders |
| `seasonal_order` | None | (P, D, Q, s) for seasonality |

### ProphetForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `yearly_seasonality` | 'auto' | Yearly patterns |
| `weekly_seasonality` | 'auto' | Weekly patterns |
| `daily_seasonality` | 'auto' | Daily patterns |

## AutoForecaster Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `candidate_models` | required | List of models to compare |
| `metric` | `'rmse'` | `'rmse'`, `'mae'`, `'mape'`, `'mse'` |
| `n_splits` | 5 | CV folds (2-10) |
| `test_size` | 20 | Validation window size |
| `per_series_models` | False | Different model per series |
| `n_jobs` | 1 | Parallel workers (-1 = all CPUs) |

## Per-Series Covariates

```python
# Single DataFrame: same features for all series
auto.fit(y_train, X=X_train)

# Dict: different features per series
X_dict = {
    'series_a': X_train[['feature1']],
    'series_b': X_train[['feature2', 'feature3']],
}
auto.fit(y_train, X=X_dict)
```

## Tips

1. **Start simple**: MovingAverage or ARIMA first, then ML models
2. **n_lags**: Match your seasonal period (7 for weekly, 30 for monthly)
3. **n_estimators**: 100 for testing, 300-500 for production
4. **n_splits**: 3 for small data, 5 for medium, 10 for large
5. **Per-series covariates**: Use when series have different drivers

## Full Documentation

- [API Reference](API_REFERENCE.md) — All parameters with complete details
- [Quick Start](QUICKSTART.md) — Working examples
- [Tutorial](examples/autotsforecast_tutorial.ipynb) — Comprehensive guide

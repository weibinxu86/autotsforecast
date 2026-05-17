# Parameter Guide

Quick reference for finding the right parameters.

## Parameter Lookup

| I want to... | Parameter | Values | Location |
|--------------|-----------|--------|----------|
| Use a preset | `preset` | `'fast'`, `'balanced'`, `'accuracy'`, `'zero_shot'`, `'intermittent'`, `'hierarchical'` | AutoForecaster |
| Set forecast length | `horizon` | 7-90 | All models, required with `preset` |
| Use lag features | `n_lags` | 7-30 | RF, XGBoost, LightGBM, CatBoost, ElasticNet |
| Set CV folds | `n_splits` | 2-5 | AutoForecaster |
| Speed up backtesting | `backtest_mode` | `'fast'`, `'last_fold'`, `'full'` | AutoForecaster |
| Limit model search | `max_models` | 1-20 | AutoForecaster |
| Add time budget | `time_limit` | seconds (float) | AutoForecaster |
| Use per-series models | `per_series_models` | `True` | AutoForecaster |
| Pass per-series covariates | `X` | `Dict[str, DataFrame]` | `fit()` / `forecast()` |
| Parallel processing | `n_jobs` | -1 (all CPUs) | AutoForecaster, ParallelForecaster |
| Prediction intervals | `coverage` | `[0.80, 0.95]` | PredictionIntervals |
| Reconciliation method | `method` | `'ols'`, `'bottom_up'` | HierarchicalReconciler |

## AutoForecaster Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `candidate_models` | `None` | List of models to compare (required unless `preset` is set) |
| `preset` | `None` | Auto-populate candidates: `'fast'`, `'balanced'`, `'accuracy'`, `'zero_shot'`, `'intermittent'`, `'hierarchical'` |
| `horizon` | `None` | Forecast horizon — required when using `preset` |
| `metric` | `'rmse'` | `'rmse'`, `'mae'`, `'mape'`, `'mse'` |
| `n_splits` | 5 | CV folds (2-10) |
| `test_size` | 20 | Validation window size |
| `backtest_mode` | `'full'` | `'full'` = n_splits folds; `'fast'` = 2 folds; `'last_fold'` = 1 fold |
| `time_limit` | `None` | Stop search after this many seconds |
| `max_models` | `None` | Evaluate at most this many candidates |
| `per_series_models` | `False` | Different model per series |
| `n_jobs` | 1 | Parallel workers across candidates (-1 = all CPUs) |

## Model Parameters

### RandomForestForecaster / XGBoostForecaster / LightGBMForecaster / CatBoostForecaster

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `horizon` | required | 7-30 | Forecast steps |
| `n_lags` | 7-14 | 7-30 | More lags = more history |
| `n_estimators` / `iterations` | 100-300 | 100-500 | More = better, slower |
| `max_depth` / `depth` | None-6 | 5-15 | Controls overfitting |

### ElasticNetForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `n_lags` | 14 | Lag feature count |
| `alpha` | 1.0 | Regularisation strength |
| `l1_ratio` | 0.5 | 0 = Ridge, 1 = Lasso |

### ThetaForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `period` | auto | Seasonal period (auto-detected from freq) |
| `deseasonalize` | `True` | Remove seasonality before fitting |

### CrostonForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `alpha` | 0.1 | Smoothing parameter |
| `method` | `'sba'` | `'sba'` (Syntetos-Boylan, bias-corrected) or `'croston'` (original) |

### ARIMAForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `order` | (1,1,1) | (p, d, q) — AR, diff, MA orders |
| `seasonal_order` | None | (P, D, Q, s) for seasonality |

### ProphetForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `yearly_seasonality` | 'auto' | Yearly patterns |
| `weekly_seasonality` | 'auto' | Weekly patterns |
| `daily_seasonality` | 'auto' | Daily patterns |

### NBEATSForecaster / NHiTSForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `input_chunk_length` | 2×horizon | Look-back window |
| `n_epochs` | 50 | Training epochs |

### TFTForecaster

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | required | Forecast steps |
| `input_chunk_length` | 2×horizon | Look-back window |
| `hidden_size` | 64 | Hidden layer size |
| `n_epochs` | 50 | Training epochs |

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

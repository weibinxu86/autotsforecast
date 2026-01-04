# API Reference

Complete parameter documentation for all models and classes in AutoTSForecast.

## Table of Contents
- [AutoForecaster](#autoforecaster)
- [Backtesting (Standalone Feature)](#backtesting)
- [Forecasting Models](#forecasting-models)
- [Hierarchical Reconciliation](#hierarchical-reconciliation)
- [Interpretability Tools](#interpretability-tools)
- [Preprocessing](#preprocessing)

---

## AutoForecaster

Automatic model selection with cross-validation.

### Parameters

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `candidate_models` | list | required | List of forecaster objects | Models to evaluate during selection |
| `metric` | str | `'rmse'` | `'rmse'`, `'mae'`, `'mape'`, `'mse'` | Metric for model selection |
| `n_splits` | int | `5` | 2-10 | Number of cross-validation folds |
| `test_size` | int | `20` | 1 to dataset_size/2 | Size of each validation window |
| `window_type` | str | `'expanding'` | `'expanding'`, `'rolling'` | CV window type |
| `verbose` | bool | `True` | `True`, `False` | Print progress messages |
| `per_series_models` | bool | `False` | `True`, `False` | Select different model per series |
| `n_jobs` | int | `-1` | -1, 1 to CPU_count | Parallel processing (-1 = all cores) |

### Methods

```python
fit(y: pd.DataFrame, X: Optional[pd.DataFrame] = None)
```
- `y`: Target time series (DatetimeIndex, multiple columns)
- `X`: Optional covariates (same index as y)

```python
forecast(X: Optional[pd.DataFrame] = None) -> pd.DataFrame
```
- `X`: Future covariates for forecast period
- Returns: Forecasts with same columns as training data

---

## Backtesting (Standalone Feature)

### BacktestValidator

**Standalone time series cross-validation tool that works with ANY forecasting model.**

BacktestValidator is an independent feature that can be used separately from AutoForecaster. While AutoForecaster uses backtesting internally for model selection, BacktestValidator allows you to:
- Validate any single model's performance
- Get detailed fold-by-fold metrics and statistics
- Visualize backtesting results automatically
- Extract predictions and actuals for custom analysis
- Compare models manually with full transparency

**Key Advantages:**
- ✅ Works independently with ANY forecaster
- ✅ Detailed fold-level insights (not just averages)
- ✅ Automated visualization of performance
- ✅ Access to raw predictions for custom analysis
- ✅ Comprehensive metrics: RMSE, MAE, MAPE, SMAPE, R²

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `model` | BaseForecaster | required | Any forecaster | Model to validate |
| `n_splits` | int | `5` | 2-10 | Number of CV splits |
| `test_size` | int | `20` | 1 to dataset_size/2 | Validation window size |
| `window_type` | str | `'expanding'` | `'expanding'`, `'rolling'` | CV window type (expanding recommended) |

**Methods:**

```python
run(y: pd.DataFrame, X: pd.DataFrame = None) -> dict
```
Run backtesting and return overall metrics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | pd.DataFrame | required | Target time series |
| `X` | pd.DataFrame | `None` | Optional covariates |
| **Returns** | dict | - | Overall metrics: `{'rmse': float, 'mae': float, 'mape': float, 'smape': float, 'r2': float}` |

```python
get_fold_results() -> pd.DataFrame
```
Get detailed results for each fold (train/test sizes, metrics per fold).

```python
get_summary() -> pd.DataFrame
```
Get summary statistics (mean, std, min, max) across all folds for each metric.

```python
plot_results(figsize: Tuple[int, int] = (15, 10))
```
Automatically visualize backtesting results:
- Metrics by fold (line charts)
- R² by fold
- Actual vs Predicted for last fold

```python
get_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]
```
Extract all actuals and predictions from all folds for custom analysis.
- **Returns**: `(actuals_df, predictions_df)` - concatenated DataFrames from all folds

**Complete Example:**
```python
from autotsforecast.backtesting import BacktestValidator
from autotsforecast import RandomForestForecaster

# Create any forecasting model
model = RandomForestForecaster(horizon=14, n_lags=7)

# Create validator
validator = BacktestValidator(
    model=model,
    n_splits=5,           # 5 CV folds
    test_size=14,         # 14-day test windows
    window_type='expanding'  # Expanding window (recommended)
)

# Run backtesting
overall_metrics = validator.run(y_train, X_train)
print(f"Overall RMSE: {overall_metrics['rmse']:.2f}")
print(f"Overall MAPE: {overall_metrics['mape']:.2f}%")

# Get fold-by-fold results
fold_results = validator.get_fold_results()
print(fold_results[['fold', 'train_size', 'test_size', 'rmse', 'mape']])

# Get summary statistics
summary = validator.get_summary()
print(summary)  # Shows mean, std, min, max for each metric

# Visualize results
validator.plot_results()

# Extract predictions for custom analysis
actuals, predictions = validator.get_predictions()
errors = actuals - predictions
print(f"Mean error: {errors.mean()}")
```

**Use Cases:**
1. **Validate a single model**: Understand performance across different time periods
2. **Manual model comparison**: Compare multiple models with full fold-level transparency
3. **Performance analysis**: Identify periods where model performs well/poorly
4. **Custom metrics**: Extract predictions and compute your own metrics
5. **Reporting**: Generate detailed performance reports with visualizations

---

## Forecasting Models

### VARForecaster

Vector Autoregression for multivariate time series.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `lags` | int | `1` | 1-21 | Number of lag terms |
| `trend` | str | `'c'` | `'c'`, `'ct'`, `'ctt'`, `'n'` | Trend type: constant, constant+trend, constant+trend+trend², none |

**Covariate Support:** Yes (exogenous variables)

**Example:**
```python
VARForecaster(horizon=14, lags=7, trend='c')
```

---

### LinearForecaster

Linear regression with automatic lag feature creation.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |

**Covariate Support:** Yes (automatically included as features)

**Example:**
```python
LinearForecaster(horizon=14)
```

---

### MovingAverageForecaster

Simple moving average forecaster.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `window` | int | `7` | 1-100 | Window size for averaging |

**Covariate Support:** No

**Example:**
```python
MovingAverageForecaster(horizon=14, window=7)
```

---

### RandomForestForecaster

Random Forest with automatic lag and covariate handling.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `n_lags` | int | `7` | 1-60 | Number of lag features |
| `n_estimators` | int | `100` | 10-1000 | Number of trees |
| `max_depth` | int | `None` | None, 2-50 | Maximum tree depth (None = unlimited) |
| `random_state` | int | `42` | Any integer | Random seed for reproducibility |
| `preprocess_covariates` | bool | `True` | `True`, `False` | Auto-encode categorical features |
| `**rf_params` | dict | `{}` | sklearn params | Additional RandomForestRegressor parameters |

**Covariate Support:** Yes (with automatic categorical encoding)

**Categorical Encoding:** Automatic (one-hot or label encoding)

**Example:**
```python
RandomForestForecaster(
    horizon=14, 
    n_lags=14, 
    n_estimators=400, 
    max_depth=10,
    random_state=42
)
```

---

### XGBoostForecaster

XGBoost with automatic lag and covariate handling.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `n_lags` | int | `7` | 1-60 | Number of lag features |
| `n_estimators` | int | `100` | 10-1000 | Number of boosting rounds |
| `max_depth` | int | `6` | 2-20 | Maximum tree depth |
| `learning_rate` | float | `0.1` | 0.001-0.5 | Learning rate (lower = slower, more accurate) |
| `random_state` | int | `42` | Any integer | Random seed |
| `preprocess_covariates` | bool | `True` | `True`, `False` | Auto-encode categorical features |
| `**xgb_params` | dict | `{}` | XGBoost params | Additional XGBRegressor parameters |

**Covariate Support:** Yes (with automatic categorical encoding)

**Categorical Encoding:** Automatic (one-hot or label encoding)

**Example:**
```python
XGBoostForecaster(
    horizon=14,
    n_lags=14,
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
```

---

### ProphetForecaster

Facebook Prophet with automatic seasonality detection.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `**prophet_params` | dict | `{}` | Prophet params | Additional Prophet parameters |

**Covariate Support:** Yes (as regressors)

**Example:**
```python
ProphetForecaster(horizon=14)
```

---

### ARIMAForecaster

ARIMA/SARIMA forecasting.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `order` | tuple | `(1,1,1)` | (p, d, q) where p,q: 0-5, d: 0-2 | ARIMA order |
| `seasonal_order` | tuple | `(0,0,0,0)` | (P, D, Q, s) where s = seasonal period | Seasonal ARIMA order |
| `**arima_params` | dict | `{}` | statsmodels params | Additional SARIMAX parameters |

**Covariate Support:** Yes (as exogenous variables)

**Example:**
```python
ARIMAForecaster(
    horizon=14,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
```

---

### ETSForecaster

Exponential Smoothing State Space Model.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `seasonal_periods` | int | `None` | None, 2-365 | Seasonal period (7=weekly, 12=monthly) |
| `trend` | str | `None` | None, `'add'`, `'mul'` | Trend component |
| `seasonal` | str | `None` | None, `'add'`, `'mul'` | Seasonal component |
| `**ets_params` | dict | `{}` | statsmodels params | Additional ETSModel parameters |

**Covariate Support:** No

**Example:**
```python
ETSForecaster(
    horizon=14,
    seasonal_periods=7,
    trend=None,
    seasonal='add'
)
```

---

### LSTMForecaster

Long Short-Term Memory neural network.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `horizon` | int | required | ≥ 1 | Number of steps to forecast |
| `n_lags` | int | `21` | 10-100 | Number of lag features |
| `hidden_size` | int | `32` | 8-256 | LSTM hidden layer size |
| `num_layers` | int | `1` | 1-5 | Number of LSTM layers |
| `dropout` | float | `0.0` | 0.0-0.5 | Dropout rate (0.0 = no dropout) |
| `epochs` | int | `20` | 5-200 | Training epochs |
| `batch_size` | int | `32` | 8-256 | Batch size for training |
| `learning_rate` | float | `0.001` | 0.0001-0.1 | Learning rate |
| `random_state` | int | `42` | Any integer | Random seed |

**Covariate Support:** Yes

**Example:**
```python
LSTMForecaster(
    horizon=14,
    n_lags=21,
    hidden_size=32,
    num_layers=1,
    dropout=0.0,
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    random_state=42
)
```

---

## Hierarchical Reconciliation

### HierarchicalReconciler

Ensures hierarchical forecast coherence.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `forecasts` | pd.DataFrame | required | DataFrame | Base forecasts to reconcile |
| `hierarchy` | dict | required | Dict[str, List[str]] | Hierarchy structure (parent: [children]) |

**Methods:**

```python
reconcile(method: str = 'bottom_up') -> HierarchicalReconciler
```

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `method` | str | `'bottom_up'` | `'bottom_up'`, `'top_down'`, `'middle_out'`, `'mint_ols'`, `'mint_shrink'`, `'mint_cov'` | Reconciliation method |

**Methods Explained:**
- `'bottom_up'`: Aggregate from bottom level
- `'top_down'`: Disaggregate from top using proportions
- `'middle_out'`: Start from middle level
- `'mint_ols'`: MinT optimal reconciliation (OLS)
- `'mint_shrink'`: MinT with shrinkage
- `'mint_cov'`: MinT with covariance estimation

**Example:**
```python
from autotsforecast.hierarchical import HierarchicalReconciler

hierarchy = {'Total': ['North', 'South', 'East']}
reconciler = HierarchicalReconciler(forecasts, hierarchy)
result = reconciler.reconcile(method='mint_shrink')
reconciled_forecasts = result.reconciled_forecasts
```

---

## Interpretability Tools

### DriverAnalyzer

Analyze covariate impact on forecasts.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `model` | BaseForecaster | required | Any fitted forecaster | Model to analyze |
| `feature_names` | list | `None` | List of strings | Feature names for interpretability |

**Methods:**

```python
calculate_feature_importance(X: pd.DataFrame, y: pd.DataFrame, method: str = 'coefficients')
```

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `X` | pd.DataFrame | required | DataFrame | Covariate data |
| `y` | pd.DataFrame | required | DataFrame | Target data |
| `method` | str | `'coefficients'` | `'coefficients'`, `'permutation'`, `'shap'` | Importance method |

**Methods Explained:**
- `'coefficients'`: Linear model coefficients (LinearForecaster only)
- `'permutation'`: Permutation importance (any model)
- `'shap'`: SHAP values (tree-based models)

```python
calculate_shap_values(X: pd.DataFrame, background_samples: pd.DataFrame = None, max_samples: int = 100)
```

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `X` | pd.DataFrame | required | DataFrame | Feature data |
| `background_samples` | pd.DataFrame | `None` | DataFrame or None | Background for SHAP (None = auto-sample from X) |
| `max_samples` | int | `100` | 50-500 | Maximum samples for SHAP calculation |

**Example:**
```python
from autotsforecast.interpretability import DriverAnalyzer

analyzer = DriverAnalyzer(model, feature_names=['temp', 'promo'])
importance = analyzer.calculate_feature_importance(X_train, y_train, method='shap')
```

---

## Preprocessing

### CovariatePreprocessor

Automatic preprocessing for categorical and numerical features.

**Parameters:**

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `categorical_features` | list | `None` | List of strings or None | Categorical column names (None = auto-detect) |
| `numerical_features` | list | `None` | List of strings or None | Numerical column names (None = auto-detect) |
| `encoding` | str | `'onehot'` | `'onehot'`, `'label'` | Categorical encoding method |
| `scale_numerical` | bool | `False` | `True`, `False` | Standardize numerical features |
| `handle_missing` | str | `'forward_fill'` | `'forward_fill'`, `'backward_fill'`, `'mean'`, `'drop'` | Missing value strategy |
| `max_categories` | int | `50` | 10-1000 | Max categories for one-hot (use label encoding beyond) |

**Methods:**

```python
fit(X: pd.DataFrame) -> CovariatePreprocessor
transform(X: pd.DataFrame) -> pd.DataFrame
fit_transform(X: pd.DataFrame) -> pd.DataFrame
```

**Example:**
```python
from autotsforecast.utils.preprocessing import CovariatePreprocessor

preprocessor = CovariatePreprocessor(
    encoding='onehot',
    scale_numerical=False,
    handle_missing='forward_fill',
    max_categories=50
)
X_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

---

## Common Patterns

### Basic Forecasting
```python
model = RandomForestForecaster(horizon=14, n_lags=7)
model.fit(y_train, X_train)
forecasts = model.predict(X_test)
```

### Automatic Model Selection
```python
auto = AutoForecaster(
    candidate_models=[model1, model2, model3],
    n_splits=3,
    per_series_models=True,
    n_jobs=-1
)
auto.fit(y_train, X_train)
forecasts = auto.forecast(X_test)
```

### With Hierarchical Reconciliation
```python
auto.fit(y_train, X_train)
base_forecasts = auto.forecast(X_test)

hierarchy = {'Total': ['Region_A', 'Region_B']}
reconciler = HierarchicalReconciler(base_forecasts, hierarchy)
result = reconciler.reconcile(method='mint_ols')
reconciled = result.reconciled_forecasts
```

### With Feature Importance
```python
model.fit(y_train, X_train)
analyzer = DriverAnalyzer(model, feature_names=list(X_train.columns))
importance = analyzer.calculate_feature_importance(X_train, y_train, method='shap')
```

---

## Tips and Best Practices

### Choosing Parameters

**Horizon:**
- Start small (7-14 days) for better accuracy
- Longer horizons (30+ days) are less accurate but more useful for planning

**Lags (n_lags):**
- Weekly data: 7-14 lags
- Monthly data: 12-24 lags
- Daily data: 14-30 lags
- LSTMs benefit from longer lags (30-60)

**CV Splits (n_splits):**
- More splits (5+) = more robust selection but slower
- Fewer splits (2-3) = faster but less reliable
- Ensure sufficient data: need at least `(n_splits + 1) × test_size` points

**Tree-Based Model Parameters:**
- More trees (`n_estimators=500`) = better but slower
- Deeper trees (`max_depth=10+`) = risk overfitting on small data
- Lower learning rate (`learning_rate=0.01-0.05`) = slower but more accurate for XGBoost

**Categorical Encoding:**
- Use `'onehot'` (default) for low-cardinality features (< 10 categories)
- Use `'label'` for high-cardinality features (50+ categories)
- Automatic detection works well in most cases

---

## See Also

- **[README.md](README.md)**: Package overview and quick start
- **[Tutorial](examples/autotsforecast_tutorial.ipynb)**: Comprehensive hands-on guide
- **[QUICKSTART.md](QUICKSTART.md)**: 5-minute getting started guide
- **[GitHub Issues](https://github.com/weibinxu86/autotsforecast/issues)**: Bug reports and feature requests

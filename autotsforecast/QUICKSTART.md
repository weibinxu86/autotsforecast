# AutoTSForecast Quick Reference

## Installation

```bash
# Basic installation
pip install autotsforecast

# With optional dependencies (XGBoost, SHAP, visualization)
pip install -r requirements-optional.txt
```

## Basic Usage

### 1. Simple Multivariate Forecasting

```python
from autotsforecast import MovingAverageForecaster
import pandas as pd

# Your data: DataFrame with multiple columns (multivariate)
df = pd.DataFrame({
    'North': [...],
    'South': [...],
    'East': [...]
}, index=pd.date_range('2023-01-01', periods=200, freq='D'))

# Split data
y_train, y_test = df.iloc[:150], df.iloc[150:]

# Create and fit model
model = MovingAverageForecaster(horizon=30, window=7)
model.fit(y_train)

# Predict
predictions = model.predict()
```

### 2. AutoForecaster (Automatic Model Selection)

```python
from autotsforecast import (
    AutoForecaster,
    MovingAverageForecaster,
    VARForecaster,
    RandomForestForecaster,
    XGBoostForecaster
)

# Define candidate models
candidate_models = [
    MovingAverageForecaster(horizon=30, window=7),
    VARForecaster(horizon=30, lags=7),
    RandomForestForecaster(horizon=30, n_estimators=100),
    XGBoostForecaster(horizon=30, n_estimators=100)
]

# Automatic selection via backtesting
auto = AutoForecaster(
    candidate_models=candidate_models,
    metric='rmse',
    n_splits=3,
    test_size=10
)

auto.fit(y_train)
forecasts = auto.forecast()

print(f"Best model: {auto.best_model_name_}")
```

### 3. Using Covariates (External Features)

```python
# Covariates with categorical and numerical features
X = pd.DataFrame({
    'day_name': dates.day_name(),  # CATEGORICAL
    'temperature': [...],           # NUMERICAL
    'is_weekend': [...]             # NUMERICAL
}, index=dates)

# Split covariates
X_train, X_test = X.iloc[:150], X.iloc[150:]

# Fit with covariates (automatic preprocessing)
model = RandomForestForecaster(horizon=50)
model.fit(y_train, X_train)

# Predict with future covariates
predictions = model.predict(X_test)
```

## Advanced Features

### 4. Hierarchical Reconciliation

For time series with hierarchical structure (e.g., Total = Region1 + Region2):

```python
from autotsforecast.hierarchical import HierarchicalReconciler

# Get base forecasts for all levels
base_forecasts = pd.DataFrame({
    'Total': [...],
    'North': [...],
    'South': [...],
    'East': [...]
})

# Define hierarchy
hierarchy = {
    'Total': ['North', 'South', 'East']
}

# Create reconciler
reconciler = HierarchicalReconciler(base_forecasts, hierarchy)

# Reconcile using different methods
reconciled_bu = reconciler.reconcile(method='bottom_up')
reconciled_td = reconciler.reconcile(method='top_down', proportions='forecast')
reconciled_opt = reconciler.reconcile(method='mint_shrink')

print("Reconciliation methods:")
print("- bottom_up: Aggregate from lowest level")
print("- top_down: Disaggregate from highest level")
print("- mint_cov: MinTrace with covariance weighting")
print("- mint_shrink: MinTrace with shrinkage (recommended)")
```

### 5. SHAP Model Interpretability

Explain model predictions using SHAP values:

```python
from autotsforecast.interpretability import DriverAnalyzer

# Fit a model first
model = XGBoostForecaster(horizon=30)
model.fit(y_train, X_train)

# Create analyzer
analyzer = DriverAnalyzer(model, feature_names=X_train.columns.tolist())

# Calculate SHAP values
shap_values = analyzer.calculate_shap_values(
    X_test,
    background_samples=X_train.sample(100),
    max_samples=100
)

# Visualize SHAP values
analyzer.plot_shap_summary(
    X_test,
    shap_values,
    target_name='North',
    plot_type='dot'  # or 'bar', 'violin'
)

# Get feature importance from SHAP
shap_importance = analyzer.get_shap_feature_importance(shap_values)
print(shap_importance)
```

### 6. Traditional Feature Importance

For comparison with SHAP:

```python
# Coefficient-based (linear models only)
importance = analyzer.calculate_feature_importance(
    X_train, y_train,
    method='coefficients'
)

# Permutation importance (model-agnostic)
importance = analyzer.calculate_feature_importance(
    X_train, y_train,
    method='permutation'
)

# Sensitivity analysis
importance = analyzer.calculate_feature_importance(
    X_train, y_train,
    method='sensitivity'
)

# Visualize
analyzer.plot_importance(importance, top_n=10)
```

## Model Comparison

| Model | Supports Multivariate | Supports Covariates | Speed | Interpretability |
|-------|----------------------|---------------------|-------|------------------|
| MovingAverage | ✅ | ❌ | ⭐⭐⭐ | ⭐⭐⭐ |
| VAR | ✅ | ❌ | ⭐⭐ | ⭐⭐ |
| Linear | ✅ | ✅ | ⭐⭐⭐ | ⭐⭐⭐ |
| RandomForest | ✅ | ✅ | ⭐⭐ | ⭐⭐ (SHAP) |
| XGBoost | ✅ | ✅ | ⭐⭐ | ⭐⭐ (SHAP) |
| Prophet | ✅ | ✅ | ⭐⭐ | ⭐⭐ |
| ARIMA | ✅ | ✅ | ⭐ | ⭐⭐ |
| ETS | ✅ | ❌ | ⭐⭐ | ⭐⭐ |
| LSTM | ✅ | ✅ | ⭐ | ⭐ |

## Common Parameters

For complete parameter documentation, see **[API_REFERENCE.md](API_REFERENCE.md)**.

### Model Initialization
- `horizon`: Number of steps to forecast (required)
- `window`: 3-30 for moving average (MovingAverage only)
- `lags`: 1-21 for VAR, 7-30 for RF/XGBoost
- `n_lags`: Same as lags (used by RF, XGBoost, LSTM)
- `n_estimators`: 100-500 trees (RF, XGBoost)
- `max_depth`: 5-10 or None for unlimited (RF, XGBoost)
- `learning_rate`: 0.01-0.3 (XGBoost only)
- `seasonal_periods`: 7 (weekly), 12 (monthly) for ETS
- `order`: (p, d, q) where p,q: 1-5, d: 0-2 (ARIMA)
- `seasonal_order`: (P, D, Q, s) where s = seasonal period (ARIMA)

### AutoForecaster
- `candidate_models`: List of model instances to compare
- `metric`: Selection metric (`'rmse'`, `'mae'`, `'mape'`, `'mse'`)
- `n_splits`: 2-5 cross-validation splits (more = better selection, slower)
- `test_size`: Validation window size (typically = horizon)
- `window_type`: `'expanding'` (recommended) or `'rolling'`
- `per_series_models`: `True` (per-series selection) or `False` (global model)
- `n_jobs`: `-1` for all CPU cores, or specific number (e.g., `4`)

### HierarchicalReconciler
- `method`: 
  - `'bottom_up'`: Aggregate from bottom level
  - `'top_down'`: Disaggregate from top (requires proportions)
  - `'mint_ols'`: MinTrace with OLS
  - `'mint_shrink'`: MinTrace with shrinkage (recommended)
  - `'mint_cov'`: MinTrace with covariance

### CovariatePreprocessor
- `encoding`: `'onehot'` (default) or `'label'` for categorical features
- `scale_numerical`: `True` or `False` (standardize numerical features)
- `handle_missing`: `'forward_fill'`, `'backward_fill'`, `'mean'`, `'drop'`
- `max_categories`: 50 (threshold for switching to label encoding)

## Tips

1. **Start simple**: Try MovingAverage or VAR first before complex models
2. **Use AutoForecaster**: Let it find the best model for your data
3. **Covariates**: Add external features to improve accuracy
4. **SHAP for understanding**: Use SHAP to understand which features drive predictions
5. **Hierarchical reconciliation**: Ensures forecasts are coherent across levels
6. **Horizon**: Balance between forecast accuracy (shorter) and planning needs (longer)

## Troubleshooting

### ImportError: shap not installed
```bash
pip install shap
```

### ImportError: xgboost not installed
```bash
pip install xgboost
```

### Data size issues with XGBoost
- XGBoost creates lagged features, reducing effective sample size
- Use smaller horizon or more training data
- Consider RandomForest as alternative

### Hierarchical reconciliation errors
- Ensure all hierarchy nodes exist in forecasts
- Check for cycles in hierarchy definition
- Verify bottom-level series are properly identified

## Support

For issues and questions:
- **[API Reference](API_REFERENCE.md)**: Complete parameter documentation
- **[Tutorial](examples/autotsforecast_tutorial.ipynb)**: Comprehensive hands-on guide
- **[GitHub Issues](https://github.com/weibinxu86/autotsforecast/issues)**: Bug reports and questions
- **[Installation Guide](INSTALL.md)**: Detailed setup instructions

# Parameter Guide - Where to Find What You Need

This guide helps you quickly find the parameter information you need.

## Quick Navigation

### I want to... → Go to...

| Goal | Document | Section |
|------|----------|---------|
| Use per-series covariates (different features per series) | [API_REFERENCE.md](API_REFERENCE.md) | "Per-Series Covariates" |
| See per-series covariates examples | [Tutorial](examples/autotsforecast_tutorial.ipynb) | Part 1: "Per-Series Covariates" section |
| See ALL possible parameter values for every function | [API_REFERENCE.md](API_REFERENCE.md) | Entire document |
| Learn how to use standalone backtesting | [API_REFERENCE.md](API_REFERENCE.md) | "Backtesting (Standalone Feature)" |
| See backtesting examples in action | [QUICKSTART.md](QUICKSTART.md) | "Standalone Backtesting" section |
| Get practical parameter recommendations | [API_REFERENCE.md](API_REFERENCE.md) | "Tips and Best Practices" section |
| See quick parameter examples for common tasks | [QUICKSTART.md](QUICKSTART.md) | "Common Parameters" section |
| Understand what each model does | [README.md](README.md) | "Available Models" table |
| Learn how AutoForecaster works | [README.md](README.md) | "How It Works" section |
| See categorical feature encoding options | [API_REFERENCE.md](API_REFERENCE.md) | "CovariatePreprocessor" section |
| Find hierarchical reconciliation methods | [API_REFERENCE.md](API_REFERENCE.md) | "HierarchicalReconciler" section |

---

## Complete Parameter Reference by Component

### 1. AutoForecaster Class
**→ [API_REFERENCE.md - AutoForecaster Section](API_REFERENCE.md#autoforecaster)**

All parameters with allowed values and defaults:
- `candidate_models`, `metric`, `n_splits`, `test_size`, `window_type`, `verbose`, `per_series_models`, `n_jobs`

**Per-Series Covariates:**
- Pass `X` as `Dict[str, pd.DataFrame]` to use different covariates per series
- Each key = series name, value = that series' covariate DataFrame
- See [API_REFERENCE.md - Per-Series Covariates](API_REFERENCE.md#per-series-covariates)

### 2. BacktestValidator (Standalone Backtesting)
**→ [API_REFERENCE.md - Backtesting Section](API_REFERENCE.md#backtesting)**

Standalone time series cross-validation tool:
- `model`, `n_splits`, `test_size`, `window_type`
- Methods: `run()`, `get_fold_results()`, `get_summary()`, `plot_results()`, `get_predictions()`
- Works independently with ANY forecasting model
- Get fold-level insights and visualizations

### 3. Forecasting Models
**→ [API_REFERENCE.md - Forecasting Models Section](API_REFERENCE.md#forecasting-models)**

Complete parameter tables for all 9 models:
- VARForecaster
- LinearForecaster
- MovingAverageForecaster
- RandomForestForecaster
- XGBoostForecaster
- ProphetForecaster
- ARIMAForecaster
- ETSForecaster
- LSTMForecaster

Each model includes:
- All parameters with allowed value ranges
- Default values
- Covariate support status
- Working code examples

### 4. Hierarchical Reconciliation
**→ [API_REFERENCE.md - Hierarchical Reconciliation Section](API_REFERENCE.md#hierarchical-reconciliation)**

Parameters and methods:
- Reconciliation methods: `'bottom_up'`, `'top_down'`, `'middle_out'`, `'ols'`
- Hierarchy structure format
- Usage examples

### 5. Interpretability Tools
**→ [API_REFERENCE.md - Interpretability Tools Section](API_REFERENCE.md#interpretability-tools)**

DriverAnalyzer parameters:
- Feature importance methods: `'coefficients'`, `'permutation'`, `'shap'`
- SHAP calculation parameters
- Model compatibility guide

### 6. Preprocessing
**→ [API_REFERENCE.md - Preprocessing Section](API_REFERENCE.md#preprocessing)**

CovariatePreprocessor parameters:
- Encoding options: `'onehot'`, `'label'`
- Missing value handling: `'forward_fill'`, `'backward_fill'`, `'mean'`, `'drop'`
- Feature detection settings
- Scaling options

---

## Practical Parameter Recommendations

### For Beginners
**→ [API_REFERENCE.md - Tips and Best Practices](API_REFERENCE.md#tips-and-best-practices)**

Practical value ranges with explanations:
- "What values should I use for `n_lags`?" → 7-30 for most cases
- "How many trees for RandomForest?" → 100-500 (more = better, slower)
- "What learning rate for XGBoost?" → 0.01-0.3 (lower = slower, more accurate)

### For Quick Reference
**→ [QUICKSTART.md - Common Parameters](QUICKSTART.md#common-parameters)**

Most frequently used parameters with typical values for:
- Model initialization
- AutoForecaster configuration
- Hierarchical reconciliation
- Preprocessing options

---

## Parameter Selection Tips

### Choosing `n_lags` (Lag Features)
- **Weekly data**: 7-14 lags
- **Daily data**: 14-30 lags
- **Monthly data**: 12-24 lags
- **LSTMs**: Benefit from longer lags (30-60)

### Choosing `n_splits` (Cross-Validation)
- **Small datasets** (< 500 points): 2-3 splits
- **Medium datasets** (500-2000 points): 3-5 splits
- **Large datasets** (> 2000 points): 5-10 splits
- Rule: Need at least `(n_splits + 1) × test_size` training points

### Choosing `n_estimators` (Tree Models)
- **Quick testing**: 50-100 trees
- **Production use**: 200-500 trees
- **High accuracy needed**: 500-1000 trees (slower)

### Choosing `learning_rate` (XGBoost)
- **Fast experimentation**: 0.1-0.3
- **Good accuracy**: 0.05-0.1
- **Best accuracy**: 0.01-0.05 (increase `n_estimators` to compensate)

### Choosing Categorical Encoding
- **Low cardinality** (< 10 categories): `encoding='onehot'`
- **Medium cardinality** (10-50 categories): Either works, default `'onehot'` is fine
- **High cardinality** (> 50 categories): `encoding='label'`

---

## Example Workflows

### 1. "I want to use RandomForestForecaster - what parameters do I need?"

**Step 1:** See required parameters in [API_REFERENCE.md - RandomForestForecaster](API_REFERENCE.md#randomforestforecaster)
```python
RandomForestForecaster(
    horizon=14,           # Required: forecast length
    n_lags=14,           # Optional: default 7
    n_estimators=400,    # Optional: default 100
    max_depth=10,        # Optional: default None
    random_state=42      # Optional: for reproducibility
)
```

**Step 2:** See recommended values in [API_REFERENCE.md - Tips and Best Practices](API_REFERENCE.md#tips-and-best-practices)
- `n_lags`: 7-30 (more = more history, slower)
- `n_estimators`: 100-500 (more = better accuracy, slower)
- `max_depth`: None (unlimited) or 5-20 (controls overfitting)

### 2. "What reconciliation method should I use?"

**Step 1:** See all methods in [API_REFERENCE.md - HierarchicalReconciler](API_REFERENCE.md#hierarchical-reconciliation)

**Step 2:** Choose based on your needs:
- **Trust bottom-level forecasts**: `method='bottom_up'`
- **Trust top-level forecast**: `method='top_down'`
- **Balanced/optimal approach**: `method='ols'`

### 3. "How do I handle categorical features?"

**Step 1:** See encoding options in [API_REFERENCE.md - CovariatePreprocessor](API_REFERENCE.md#covariatepreprocessor)

**Step 2:** For RandomForest/XGBoost, it's automatic:
```python
model = RandomForestForecaster(
    horizon=14,
    preprocess_covariates=True  # Default: automatic encoding
)
model.fit(y_train, X_train)  # Categorical features auto-detected and encoded
```

**Step 3 (optional):** Customize encoding:
```python
from autotsforecast.utils.preprocessing import CovariatePreprocessor

preprocessor = CovariatePreprocessor(
    encoding='label',      # or 'onehot' (default)
    max_categories=50      # threshold
)
X_processed = preprocessor.fit_transform(X_train)
```

### 4. "How do I use different covariates for different series?"

**Step 1:** See complete documentation in [API_REFERENCE.md - Per-Series Covariates](API_REFERENCE.md#per-series-covariates)

**Step 2:** Create a dictionary mapping each series to its covariates:
```python
from autotsforecast import AutoForecaster
from autotsforecast.models.external import ProphetForecaster, RandomForestForecaster

# Different features for different products
X_train_dict = {
    'product_a': pd.DataFrame({  # Weather-sensitive product
        'temperature': [...],
        'advertising_spend': [...]
    }, index=dates_train),
    'product_b': pd.DataFrame({  # Price-sensitive product
        'competitor_price': [...],
        'promotion_active': [...]
    }, index=dates_train)
}

X_test_dict = {
    'product_a': X_a_test,
    'product_b': X_b_test
}

# Train with per-series covariates
auto = AutoForecaster(
    candidate_models=[
        ProphetForecaster(horizon=14),
        RandomForestForecaster(horizon=14, n_lags=7)
    ],
    per_series_models=True
)
auto.fit(y_train, X=X_train_dict)
forecasts = auto.forecast(X=X_test_dict)
```

**Step 3:** See practical examples in [Tutorial - Part 1](examples/autotsforecast_tutorial.ipynb)

### 5. "How do I validate my model's performance using backtesting?"

**Step 1:** See complete documentation in [API_REFERENCE.md - BacktestValidator](API_REFERENCE.md#backtesting)

**Step 2:** Use BacktestValidator (works with ANY model independently):
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
    window_type='expanding'  # or 'rolling'
)

# Run backtesting
metrics = validator.run(y_train, X_train)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")

# Get fold-by-fold details
fold_results = validator.get_fold_results()
summary = validator.get_summary()

# Visualize
validator.plot_results()
```

**Step 3:** See practical examples in [QUICKSTART.md](QUICKSTART.md)

---

## Full Documentation Index

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [API_REFERENCE.md](API_REFERENCE.md) | **Complete** parameter tables for all functions | Looking up exact allowed values |
| [API_REFERENCE.md - Per-Series Covariates](API_REFERENCE.md#per-series-covariates) | Per-series covariate documentation | Using different features per series |
| [Tutorial](examples/autotsforecast_tutorial.ipynb) | Hands-on guide with **practical** parameter recommendations | Learning how to use the package |
| [QUICKSTART.md](QUICKSTART.md) | Quick examples with **common** parameter values | Getting started quickly |
| [README.md](README.md) | Package overview and **high-level** features | Understanding what the package does |
| [INSTALL.md](INSTALL.md) | Installation instructions | Setting up the package |
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | Architecture and design details | Understanding how it works internally |

---

## Still Can't Find What You Need?

1. **Search API_REFERENCE.md** for the specific parameter name
2. **Check the Tutorial** for practical examples of that parameter in use
3. **Look at QUICKSTART.md** for common use cases
4. **Open an issue** on [GitHub](https://github.com/weibinxu86/autotsforecast/issues) with your specific question

**Pro tip:** Use Ctrl+F (or Cmd+F) to search within documents for specific parameter names!

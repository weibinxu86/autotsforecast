# Changelog

## Version 0.1.0 - Package Rename and Feature Enhancements

### Breaking Changes
- **Package renamed from `ts_forecast` to `autotsforecast`**
  - All imports now use `from autotsforecast import ...`
  - Directory structure: `src/autotsforecast/`
  - PyPI package name: `autotsforecast`

### New Features

#### 1. Enhanced Hierarchical Reconciliation
- **Location:** `autotsforecast/hierarchical/reconciliation.py`
- **Features:**
  - Bottom-up aggregation
  - Top-down disaggregation
  - MinTrace optimal reconciliation (OLS, WLS, Shrinkage)
  - Support for complex hierarchical structures
  - Automatic validation of hierarchy consistency

**Usage Example:**
```python
from autotsforecast.hierarchical import HierarchicalReconciler

# Define hierarchy
hierarchy = {
    'Total': ['North', 'South', 'East'],
    'North': ['North_A', 'North_B']
}

reconciler = HierarchicalReconciler(forecasts, hierarchy)
reconciled_forecasts = reconciler.reconcile(method='mint_shrink')
```

#### 2. SHAP-based Model Interpretability
- **Location:** `autotsforecast/interpretability/drivers.py`
- **Features:**
  - SHAP value calculation for all model types
  - TreeExplainer for RandomForest/XGBoost
  - LinearExplainer for linear models
  - KernelExplainer for model-agnostic interpretation
  - Summary plots (dot, bar, violin)
  - Feature importance via mean absolute SHAP values

**Usage Example:**
```python
from autotsforecast.interpretability import DriverAnalyzer

# Create analyzer
analyzer = DriverAnalyzer(fitted_model)

# Calculate SHAP values
shap_values = analyzer.calculate_shap_values(X_test)

# Visualize
analyzer.plot_shap_summary(X_test, shap_values, target_name='North')

# Get feature importance
importance_df = analyzer.get_shap_feature_importance(shap_values)
```

### Dependencies
- **New optional dependency:** `shap>=0.42.0` for model interpretability
  - Install with: `pip install shap` or `pip install -r requirements-optional.txt`

### Migration Guide

#### Updating Existing Code
Replace all occurrences of `ts_forecast` with `autotsforecast`:

**Before:**
```python
from ts_forecast import AutoForecaster, XGBoostForecaster
from ts_forecast.models.base import BaseForecaster
```

**After:**
```python
from autotsforecast import AutoForecaster, XGBoostForecaster
from autotsforecast.models.base import BaseForecaster
```

#### Updating Installation
```bash
# Uninstall old package
pip uninstall ts-forecast

# Install new package
pip install autotsforecast

# Install with optional dependencies
pip install autotsforecast[viz]
pip install -r requirements-optional.txt  # For SHAP, XGBoost, etc.
```

### Documentation Updates
- Updated tutorial notebook: `examples/forecasting_tutorial.ipynb`
- Added SHAP examples in summary cell
- Added hierarchical reconciliation examples
- Updated all README references

### Bug Fixes
- Fixed indexing issue in RandomForestForecaster and XGBoostForecaster when using lagged features
- Fixed AutoForecaster to use correct BacktestValidator method (`run` instead of `validate`)
- Fixed metric extraction from backtesting results

### Known Issues
- SHAP integration requires manual installation: `pip install shap`
- Hierarchical reconciliation is optional and not required for basic multivariate forecasting
- XGBoost with large horizons may require more training data due to lagged feature creation

### Testing
All existing tests have been updated to use the new package name:
- `tests/test_backtesting.py`
- `tests/test_hierarchical.py`
- `tests/test_interpretability.py`
- `tests/test_models.py`

Run tests with:
```bash
pytest tests/
```

### Contributors
- Package restructuring and SHAP integration
- Enhanced hierarchical reconciliation algorithms
- Bug fixes and stability improvements

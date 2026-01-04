# AutoTSForecast

**Automated Multivariate Time Series Forecasting with Model Selection, Hierarchical Reconciliation, and Covariate Interpretability (SHAP)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AutoTSForecast is a comprehensive Python package for multivariate time series forecasting that provides automatic model selection, hierarchical reconciliation, and interpretability tools.

🤖 **Automatic Model Selection** • 📊 **9 Forecasting Algorithms** • 🎯 **Hierarchical Reconciliation** • 🔍 **Covariate Interpretability**

> 💡 **Looking for parameter values?** See [Parameter Guide](PARAMETER_GUIDE.md) for quick navigation or [API Reference](API_REFERENCE.md) for complete documentation.

## Key Features

### Core Capabilities
- ✅ **9 Forecasting Algorithms**: VAR, Linear, Moving Average, Random Forest, XGBoost, Prophet, ARIMA, ETS, LSTM
- ✅ **AutoForecaster**: Automatically selects the best model per series using time-respecting cross-validation
- ✅ **Per-Series Model Selection**: Each time series can have its own optimally-selected model
- ✅ **Parallel Processing**: Fast model selection using all CPU cores (`n_jobs=-1`) via joblib
- ✅ **Flexible Covariate Support**: Different covariates for different series, or mix with/without covariates
- ✅ **Automatic Categorical Handling**: One-hot or label encoding for categorical features - no manual preprocessing needed
- ✅ **Independent Backtesting**: Standalone time series cross-validation with expanding/rolling windows

### Advanced Features
- ✅ **Hierarchical Reconciliation**: Ensures forecasts are coherent (e.g., Total = Region A + Region B)
  - Methods: Bottom-up, Top-down, MinTrace (OLS, shrinkage, covariance)
  - Automatically enforces aggregation constraints
  
- ✅ **SHAP Interpretability**: Understand which external covariates drive predictions
  - Model-agnostic design - works with any AutoForecaster-selected model
  - Automatically filters out lag features, focuses on business drivers
  - Visualizations: summary plots, feature importance rankings

- ✅ **Standalone Backtesting Module**: Independent time series cross-validation
  - Use with ANY forecasting model (not just AutoForecaster)
  - Expanding or rolling window validation
  - Comprehensive metrics: RMSE, MAE, MAPE, SMAPE, R²
  - Automated fold-by-fold analysis with visualizations
  - Get predictions and actuals for custom analysis
  - Time-respecting splits (no data leakage)

### Data Handling
- ✅ **Automatic Covariate Preprocessing**: Handles categorical and numerical features
  - Categorical: One-hot encoding (default) or label encoding
  - Numerical: Optional scaling with StandardScaler
  - Auto-detection: Automatically identifies feature types from data
  - Missing values: Forward fill, backward fill, mean imputation, or drop
- ✅ **Multivariate Forecasting**: Model multiple related time series simultaneously
- ✅ **External Drivers**: Include promotions, weather, holidays, macroeconomic indicators

### Performance Optimization
- ✅ **Parallel Model Selection**: Uses joblib for multi-core processing
  - Set `n_jobs=-1` to use all CPU cores
  - Significantly speeds up AutoForecaster with multiple candidate models
  - Works with both per-series and global model selection
- ✅ **Efficient Feature Engineering**: Vectorized lag creation and covariate alignment
- ✅ **Memory-Efficient**: Processes data in chunks where applicable

## Installation

### Via PyPI (Coming Soon)

```bash
pip install autotsforecast
```

### Development Installation

```bash
git clone https://github.com/weibinxu86/autotsforecast.git
cd autotsforecast
pip install -e .[all]
```

## Quick Start

### 1. Basic Forecasting

```python
import pandas as pd
from autotsforecast import RandomForestForecaster

# Your multivariate time series data
data = pd.DataFrame({
    'sales_north': [...],
    'sales_south': [...],
    'sales_east': [...]
})

# Fit and forecast
model = RandomForestForecaster(n_lags=7, horizon=30)
model.fit(data)
forecasts = model.predict()
```

### 2. Automatic Model Selection

```python
from autotsforecast import AutoForecaster, VARForecaster, RandomForestForecaster

# Define candidates
candidates = [
    VARForecaster(lags=3, horizon=30),
    VARForecaster(lags=7, horizon=30),
    RandomForestForecaster(n_lags=7, horizon=30),
    # ... add more models
]

# AutoForecaster picks the best one using parallel processing
auto = AutoForecaster(
    candidate_models=candidates, 
    metric='rmse',
    n_jobs=-1  # Use all CPU cores for fast parallel selection
)
auto.fit(data)
forecasts = auto.forecast()

print(f"Best model: {auto.best_model_name_}")
```

### 3. Using Covariates with Automatic Categorical Handling

```python
from autotsforecast import RandomForestForecaster
import pandas as pd

# Your data with both numerical and categorical features
covariates = pd.DataFrame({
    'temperature': [20.5, 21.0, 19.8, ...],     # Numerical
    'promotion': [0, 1, 1, 0, ...],              # Binary
    'day_of_week': ['Mon', 'Tue', 'Wed', ...],   # Categorical
    'region': ['North', 'South', 'East', ...]    # Categorical
})

# Model automatically detects and encodes categorical features
model = RandomForestForecaster(n_lags=7, horizon=30, preprocess_covariates=True)
model.fit(data, X=covariates)
forecasts = model.predict(X_test=covariates_future)

# Categorical features are automatically one-hot encoded
# No manual preprocessing needed!
```

### 3. Hierarchical Reconciliation

```python
from autotsforecast.hierarchical import HierarchicalReconciler

# Define hierarchy
hierarchy = {'Total': ['North', 'South', 'East']}

# Ensure forecasts are coherent
reconciler = HierarchicalReconciler(hierarchy=hierarchy)
reconciled = reconciler.reconcile(forecasts, method='mint_shrink')
```

### 4. Standalone Backtesting (Independent Feature)

```python
from autotsforecast.backtesting import BacktestValidator
from autotsforecast import RandomForestForecaster

# Create any forecasting model
model = RandomForestForecaster(horizon=14, n_lags=7)

# Backtest with expanding window CV
validator = BacktestValidator(
    model=model,
    n_splits=5,           # 5 cross-validation folds
    test_size=14,         # 14-day test windows
    window_type='expanding'  # or 'rolling'
)

# Run backtesting
metrics = validator.run(y_train, X_train)
print(f"Average RMSE: {metrics['rmse']:.2f}")
print(f"Average MAPE: {metrics['mape']:.2f}%")

# Get detailed fold-by-fold results
fold_results = validator.get_fold_results()
print(fold_results)

# Get summary statistics
summary = validator.get_summary()
print(summary)

# Visualize results
validator.plot_results()

# Get all predictions and actuals for custom analysis
actuals, predictions = validator.get_predictions()
```

### 5. Covariate Interpretability (SHAP)

```python
from autotsforecast.interpretability import DriverAnalyzer

# Analyze impact of external covariates on predictions
interpreter = DriverAnalyzer(model)
shap_values = interpreter.calculate_shap_values(X_covariates)  # X should contain only covariates
importance = interpreter.get_shap_feature_importance(shap_values)

# Note: SHAP analysis focuses on external covariates only (e.g., marketing, weather)
# Lag features are excluded from interpretability analysis
```

## Complete Tutorial

See **[examples/autotsforecast_tutorial.ipynb](examples/autotsforecast_tutorial.ipynb)** for a comprehensive guide covering:

1. Basic forecasting with multiple models
2. AutoForecaster for automatic selection
3. Using covariates to improve accuracy
4. Hierarchical reconciliation
5. Covariate interpretability with SHAP (external drivers only)

## How It Works

### Parallel Processing Architecture

AutoForecaster achieves fast model selection through parallel processing:

1. **Joblib Backend**: Uses `joblib.Parallel` to distribute work across CPU cores
2. **Per-Series Parallelism**: When `per_series_models=True`, each time series is processed independently in parallel
3. **Model Evaluation**: Within each series, candidate models are evaluated sequentially (CV folds can be parallelized)
4. **Resource Management**: Set `n_jobs=-1` to use all available cores, or specify a number (e.g., `n_jobs=4`)

**Performance Gains:**
- With 3 time series and 9 candidate models: up to 3x speedup on multi-core systems
- Larger datasets benefit more from parallelization
- No code changes needed - just set `n_jobs=-1`

### Categorical Feature Handling

Models automatically process categorical features without manual encoding:

1. **Auto-Detection**: Inspects data types and unique value counts
   - String columns → categorical
   - Numeric columns with few unique values (< max_categories) → categorical
   - All others → numerical

2. **Encoding Methods**:
   - **One-Hot Encoding** (default): Creates binary columns for each category
     - Example: `['Mon', 'Tue', 'Wed']` → `[Mon_1, Tue_1, Wed_1]`
   - **Label Encoding**: Maps categories to integers (0, 1, 2, ...)
     - Use `encoding='label'` for high-cardinality features

3. **Preprocessing Pipeline**:
   ```python
   from autotsforecast.utils.preprocessing import CovariatePreprocessor
   
   # Customize preprocessing (optional - models do this automatically)
   preprocessor = CovariatePreprocessor(
       encoding='onehot',           # or 'label'
       scale_numerical=False,       # Optional: standardize numerical features
       handle_missing='forward_fill', # or 'backward_fill', 'mean', 'drop'
       max_categories=50            # Threshold for categorical detection
   )
   X_processed = preprocessor.fit_transform(X)
   ```

4. **Integration with Models**:
   - RandomForest and XGBoost: Set `preprocess_covariates=True` (default)
   - Preprocessing happens automatically during `fit()`
   - Transformations are saved and applied to test data during `predict()`

**Example Workflow:**
```python
# Your raw data with mixed types
X = pd.DataFrame({
    'temp': [20, 21, 19],           # Numerical
    'promo': [0, 1, 0],              # Binary (treated as categorical or numerical)
    'day': ['Mon', 'Tue', 'Wed'],   # Categorical
    'store': ['A', 'B', 'A']         # Categorical
})

# Model handles everything automatically
model = RandomForestForecaster(n_lags=7, horizon=14)
model.fit(y_train, X)  # Categorical features auto-encoded internally
forecasts = model.predict(X_test)  # Same encoding applied to test data
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `VARForecaster` | Vector AutoRegression | Capturing cross-variable dependencies |
| `LinearForecaster` | Linear regression with lags | Simple linear patterns |
| `MovingAverageForecaster` | Simple moving average | Stable baseline |
| `RandomForestForecaster` | Ensemble with lags + covariates | Non-linear patterns with external factors |
| `XGBoostForecaster` | Gradient boosting with lags | High-performance ML forecasting |
| `ProphetForecaster` | Facebook Prophet | Robust forecasting with holidays and seasonality |
| `ARIMAForecaster` | ARIMA/SARIMA | Classical statistical forecasting |
| `ETSForecaster` | Error-Trend-Seasonality (Exponential Smoothing) | Data with trends and seasonality |
| `LSTMForecaster` | Long Short-Term Memory neural network | Complex temporal patterns and sequences |

## Documentation

- 📘 **[Parameter Guide](PARAMETER_GUIDE.md)**: Quick navigation to find any parameter you need
- 📕 **[API Reference](API_REFERENCE.md)**: Complete parameter documentation for all models and functions
- 📗 **[Tutorial](examples/autotsforecast_tutorial.ipynb)**: Comprehensive hands-on guide
- 📙 **[Quick Start](QUICKSTART.md)**: 5-minute getting started guide
- 📄 **[Installation Guide](INSTALL.md)**: Detailed setup instructions
- 📋 **[Changelog](CHANGELOG.md)**: Version history
- 🔧 **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)**: Architecture and design details
- 📦 **[Publishing Guide](PUBLISHING.md)**: For maintainers (PyPI distribution)

## Publishing to PyPI

For maintainers, the package is ready for PyPI distribution:

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Test on TestPyPI
python -m twine upload --repository testpypi dist/*

# Publish to PyPI
python -m twine upload dist/*
```

After publishing, users can install via:

```bash
pip install autotsforecast              # Basic
pip install autotsforecast[all]         # All features
pip install autotsforecast[viz]         # Visualization only
pip install autotsforecast[interpret]   # SHAP only
```

See [PUBLISHING.md](PUBLISHING.md) for complete instructions.

## Requirements

**Core Dependencies:**
- Python >= 3.8
- numpy, pandas, scikit-learn, statsmodels, scipy

**Optional Dependencies:**
- matplotlib, seaborn (visualization)
- shap (interpretability)
- xgboost (XGBoost model)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{autotsforecast2025,
  title={AutoTSForecast: Automated Multivariate Time Series Forecasting},
  author={Weibin Xu},
  year={2025},
  url={https://github.com/weibinxu86/autotsforecast}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/weibinxu86/autotsforecast/issues)
- **Discussions**: [GitHub Discussions](https://github.com/weibinxu86/autotsforecast/discussions)
- **Tutorial**: [examples/autotsforecast_tutorial.ipynb](examples/autotsforecast_tutorial.ipynb)

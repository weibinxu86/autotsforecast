# AutoTSForecast

**Automated Multivariate Time Series Forecasting with Model Selection, Hierarchical Reconciliation, and SHAP Interpretability**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AutoTSForecast is a comprehensive Python package for multivariate time series forecasting that provides automatic model selection, hierarchical reconciliation, and interpretability tools.

🤖 **Automatic Model Selection** • 📊 **8+ Forecasting Algorithms** • 🎯 **Hierarchical Reconciliation** • 🔍 **SHAP Interpretability**

## Key Features

- ✅ **AutoForecaster**: Automatically selects the best model using backtesting
- ✅ **Multiple Models**: VAR, Linear, Moving Average, Random Forest, XGBoost
- ✅ **Covariate Support**: Include external variables (promotions, weather, holidays)
- ✅ **Hierarchical Methods**: Bottom-up, top-down, MinTrace reconciliation
- ✅ **SHAP Values**: Understand feature importance in predictions

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

# AutoForecaster picks the best one
auto = AutoForecaster(candidate_models=candidates, metric='rmse')
auto.fit(data)
forecasts = auto.forecast()

print(f"Best model: {auto.best_model_name_}")
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

### 4. Model Interpretability (SHAP)

```python
from autotsforecast.interpretability import DriverAnalyzer

interpreter = DriverAnalyzer(model)
shap_values = interpreter.calculate_shap_values(X)
importance = interpreter.get_shap_feature_importance(shap_values)
```

## Complete Tutorial

See **[examples/autotsforecast_tutorial.ipynb](examples/autotsforecast_tutorial.ipynb)** for a comprehensive guide covering:

1. Basic forecasting with multiple models
2. AutoForecaster for automatic selection
3. Using covariates to improve accuracy
4. Hierarchical reconciliation
5. SHAP interpretability

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `VARForecaster` | Vector AutoRegression | Capturing cross-variable dependencies |
| `LinearForecaster` | Linear regression with lags | Simple linear patterns |
| `MovingAverageForecaster` | Simple moving average | Stable baseline |
| `RandomForestForecaster` | Ensemble with lags + covariates | Non-linear patterns with external factors |
| `XGBoostForecaster` | Gradient boosting with lags | High-performance ML forecasting |

## Documentation

- **[Installation Guide](INSTALL_GUIDE.md)**: Detailed setup instructions
- **[Quick Start](QUICKSTART.md)**: 5-minute getting started guide
- **[Changelog](CHANGELOG.md)**: Version history
- **[Publishing Guide](PUBLISHING.md)**: For maintainers (PyPI distribution)

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

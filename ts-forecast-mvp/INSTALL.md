# Installation and Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Option 1: Install in Development Mode (Recommended for development)

```bash
# Navigate to the package directory
cd ts-forecast-mvp

# Install in editable mode with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with visualization dependencies
pip install -e ".[viz]"
```

### Option 2: Install from requirements.txt

```bash
cd ts-forecast-mvp
pip install -r requirements.txt
```

### Option 3: Install development requirements

```bash
pip install -r requirements-dev.txt
```

## Verify Installation

```python
# Test import
import autotsforecast
print(f"TS-Forecast version: {autotsforecast.__version__}")

# List available classes
from autotsforecast import (
    VARForecaster,
    LinearForecaster,
    ModelSelector,
    BacktestValidator,
    HierarchicalReconciler,
    DriverAnalyzer
)
print("✅ All modules imported successfully!")
```

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=autotsforecast --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests for specific module
pytest tests/test_backtesting_new.py::TestBacktestValidator -v
```

## Run Examples

```bash
# Run the quickstart example
python examples/quickstart.py
```

## Quick Usage Example

```python
import numpy as np
import pandas as pd
from autotsforecast import ModelSelector, VARForecaster

# Create sample data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'sales_A': np.random.randn(100).cumsum() + 100,
    'sales_B': np.random.randn(100).cumsum() + 150
}, index=dates)

# Method 1: Automatic model selection
selector = ModelSelector(metric='rmse')
selector.fit(data, cv_folds=3)
best_name, best_model = selector.get_best_model()
print(f"Best model: {best_name}")

# Generate forecasts
forecasts = selector.predict()
print(forecasts)

# Method 2: Use specific model
model = VARForecaster(horizon=5, lags=2)
model.fit(data)
predictions = model.predict()
print(predictions)
```

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Ensure the package is installed
pip list | grep ts-forecast

# Reinstall if needed
pip install -e . --force-reinstall
```

### Missing Dependencies

```bash
# Install all required dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Test Failures

```bash
# Ensure all dependencies are installed
pip install -r requirements-dev.txt

# Run tests with verbose output
pytest -v

# Run specific failing test
pytest tests/test_models.py::TestVARForecaster::test_fit_predict -v
```

## Next Steps

1. **Explore Examples**: Check out `examples/quickstart.py` for comprehensive examples
2. **Read Documentation**: Review docstrings in source files
3. **Run Tests**: Execute test suite to understand expected behavior
4. **Customize**: Extend `BaseForecaster` to add your own models

## Common Tasks

### Add a New Model

```python
from autotsforecast.models.base import BaseForecaster
import pandas as pd

class MyCustomModel(BaseForecaster):
    def __init__(self, horizon=1, **kwargs):
        super().__init__(horizon)
        self.kwargs = kwargs
    
    def fit(self, y, X=None):
        # Your fitting logic
        self.is_fitted = True
        return self
    
    def predict(self, X=None):
        # Your prediction logic
        return pd.DataFrame(...)
```

### Customize Model Selection

```python
from autotsforecast import ModelSelector, VARForecaster, LinearForecaster

# Create custom model list
my_models = [
    VARForecaster(horizon=1, lags=1),
    VARForecaster(horizon=1, lags=2),
    VARForecaster(horizon=1, lags=3),
]

# Use with selector
selector = ModelSelector(models=my_models, metric='mape')
selector.fit(data, cv_folds=5)
```

## Support

For issues, questions, or contributions:
- Check the test files for usage examples
- Review the quickstart guide in `examples/`
- Read the IMPLEMENTATION_SUMMARY.md for architecture details

---

**Happy Forecasting! 📈**

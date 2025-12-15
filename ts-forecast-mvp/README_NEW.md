# TS-Forecast MVP

A comprehensive Python package for **multivariate time series forecasting** with model selection, backtesting, hierarchical forecasting, and interpretability.

## 🎯 Features

- **📊 Model Selection**: Automatically compare and select the best forecasting model based on performance metrics
- **🔄 Backtesting**: Robust time series cross-validation with expanding/rolling windows
- **🌳 Hierarchical Forecasting**: Reconcile forecasts across hierarchical structures (bottom-up, top-down, MinT)
- **🔍 Interpretability**: Analyze the impact of covariates (drivers) on predictions
- **🛠️ Utility Functions**: Data preprocessing, feature engineering, and seasonality detection

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ts-forecast-mvp.git
cd ts-forecast-mvp

# Install the package
pip install -e .

# Or install dependencies separately
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Forecasting with Model Selection

```python
import pandas as pd
import numpy as np
from ts_forecast import ModelSelector

# Generate sample data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'sales_A': np.random.randn(100).cumsum() + 100,
    'sales_B': np.random.randn(100).cumsum() + 150
}, index=dates)

# Model selection
selector = ModelSelector(metric='rmse')
selector.fit(data, cv_folds=3)

# Get best model
best_name, best_model = selector.get_best_model()
print(f"Best model: {best_name}")

# Generate forecasts
forecasts = selector.predict(steps=10)
```

### Backtesting

```python
from ts_forecast import BacktestValidator, VARForecaster

# Create model
model = VARForecaster(horizon=1, lags=2)

# Run backtesting
validator = BacktestValidator(model, n_splits=5, test_size=10)
metrics = validator.run(data)
print("Metrics:", metrics)

# Plot results
validator.plot_results()
```

### Hierarchical Forecasting

```python
from ts_forecast import HierarchicalReconciler

# Define hierarchy
hierarchy = {
    'Total': ['A', 'B'],
    'A': ['A1', 'A2'],
    'B': ['B1', 'B2']
}

# Create forecasts
forecasts = pd.DataFrame({
    'Total': [100, 110, 120],
    'A': [60, 65, 70],
    'B': [40, 45, 50],
    'A1': [35, 38, 40],
    'A2': [25, 27, 30],
    'B1': [22, 25, 28],
    'B2': [18, 20, 22]
})

# Reconcile
reconciler = HierarchicalReconciler(forecasts, hierarchy)
reconciler.reconcile(method='bottom_up')
reconciled = reconciler.get_reconciled_forecasts()
```

### Driver Analysis

```python
from ts_forecast import DriverAnalyzer, LinearForecaster

# Prepare data
X = pd.DataFrame({'price': [10, 12, 11, 13], 'promo': [0, 1, 0, 1]})
y = pd.DataFrame({'sales': [100, 150, 110, 160]})

# Fit and analyze
model = LinearForecaster(horizon=1)
model.fit(y, X)

analyzer = DriverAnalyzer(model)
importance = analyzer.calculate_feature_importance(X, y, method='coefficients')
print(importance)
```

## 📚 Modules

- **Models**: `VARForecaster`, `LinearForecaster`, `MovingAverageForecaster`, `ModelSelector`
- **Backtesting**: `BacktestValidator`
- **Hierarchical**: `HierarchicalReconciler`
- **Interpretability**: `DriverAnalyzer`
- **Utils**: Data preprocessing and feature engineering functions

## 🧪 Testing

```bash
pytest
pytest --cov=ts_forecast
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a Pull Request.

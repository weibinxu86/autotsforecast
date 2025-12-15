# ts-forecast-mvp

## Overview
`ts-forecast-mvp` is a Python package designed for forecasting multivariate time series data. It provides tools for model selection, backtesting, hierarchical forecasting, and interpretability with drivers (covariates for categorical and numerical data).

## Features
- **Model Selection**: Choose the best forecasting model based on performance metrics and cross-validation.
- **Backtesting**: Validate forecasting models and ensure robustness through backtesting techniques.
- **Hierarchical Forecasting**: Reconcile forecasts at different levels of a hierarchy for improved accuracy.
- **Interpretability**: Analyze the impact of covariates on model predictions to understand driving factors.

## Installation
To install the package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/ts-forecast-mvp.git
cd ts-forecast-mvp
pip install -r requirements.txt
```

## Usage
A quickstart guide is available in the `examples/quickstart.py` file. Here’s a brief example of how to use the package:

```python
from ts_forecast.models.selection import ModelSelector
from ts_forecast.backtesting.validator import BacktestValidator

# Initialize model selector and backtest validator
model_selector = ModelSelector()
backtest_validator = BacktestValidator()

# Your code for fitting models and validating results goes here
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
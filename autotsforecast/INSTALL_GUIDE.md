# Installation Guide

## Using pip with Python

Since `pip` might not be in your PATH, use Python module syntax:

```powershell
# Basic installation (from source, development mode)
python -m pip install -e .

# Or if using virtual environment
C:/forecasting/.venv/Scripts/python.exe -m pip install -e .
```

## Installation Steps

### 1. Navigate to the project directory
```powershell
cd c:\forecasting\ts-forecast-mvp
```

### 2. Install the package
```powershell
python -m pip install -e .
```

### 3. Install optional dependencies
```powershell
# For SHAP interpretability
python -m pip install shap

# For visualization
python -m pip install matplotlib seaborn

# For XGBoost (if not already installed)
python -m pip install xgboost

# Or install all optional dependencies at once
python -m pip install -r requirements-optional.txt
```

### 4. Verify installation
```powershell
python test_installation.py
```

You should see:
```
✓ ALL TESTS PASSED - AutoTSForecast is ready to use!
```

## Quick Test

```powershell
python -c "from autotsforecast import AutoForecaster; print('Success!')"
```

## Using the Package

After installation, you can import in any Python script or notebook:

```python
from autotsforecast import (
    AutoForecaster,
    MovingAverageForecaster,
    VARForecaster,
    RandomForestForecaster,
    XGBoostForecaster
)
```

## Troubleshooting

### "pip is not recognized"
Use `python -m pip` instead of `pip` directly.

### Import errors
Make sure you're using the same Python that installed the package:
```powershell
python -c "import sys; print(sys.executable)"
```

### Missing dependencies
```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

## Next Steps

1. Run the installation test: `python test_installation.py`
2. Open the tutorial: `examples/forecasting_tutorial.ipynb`
3. Read the quick start: `QUICKSTART.md`

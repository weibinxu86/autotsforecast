# Release Notes

## v0.3.4 (January 2026)

### 🚀 Chronos-2 Foundation Model (NEW)

**Major new feature**: Integration with Amazon's **Chronos-2** state-of-the-art foundation model for zero-shot time series forecasting.

```python
from autotsforecast.models.external import Chronos2Forecaster

# Use pre-trained foundation model - no training required!
model = Chronos2Forecaster(model_name='autogluon/chronos-2-small', horizon=10)
model.fit(y_train)  # Just stores context
forecasts = model.predict()  # Zero-shot forecasting
```

**Why Chronos-2?**
- 🎯 **Zero-shot learning**: Pre-trained on 100+ billion tokens of time series data
- 🌍 **Domain agnostic**: Works across retail, finance, weather, energy, etc. without retraining
- 📊 **Multiple model sizes**: Choose from 6 models (9M to 710M parameters) based on your accuracy/speed tradeoff
- 🔮 **Probabilistic forecasts**: Get full prediction distributions, not just point forecasts
- 🏆 **SOTA performance**: Achieves state-of-the-art results on multiple benchmarks

**Available Models:**
| Model | Parameters | Best For |
|-------|-----------|----------|
| `chronos-bolt-tiny` | 9M | Fastest inference |
| `chronos-bolt-mini` | 21M | Speed/accuracy balance |
| `autogluon/chronos-2-small` | 28M | **Recommended** |
| `chronos-bolt-small` | 47M | Higher accuracy |
| `amazon/chronos-2` | 120M | Best quality |
| `chronos-bolt-base` | 205M | Production systems |

**Installation:**
```bash
pip install autotsforecast[chronos]
```

**Documentation:** See [README Section 7](README.md#7-chronos-2-foundation-model-support-new) for complete examples and model comparison.

---

## v0.3.3 (January 2026)

### 🎯 Per-Series Covariates

The headline feature of this release: **pass different covariates to different series**.

```python
# Different series have different drivers
per_series_X = {
    'product_a': X_train[['weather', 'advertising']],  # Product A drivers
    'product_b': X_train[['price', 'promotion']],      # Product B drivers
}

auto = AutoForecaster(candidate_models=candidates, per_series_models=True)
auto.fit(y_train, X=per_series_X)  # Pass dict to X parameter
forecasts = auto.forecast(X=per_series_X_test)
```

**Benefits:**
- Each series uses only its relevant features (reduces noise)
- Better accuracy through targeted feature engineering
- Handles heterogeneous products with different drivers
- Backward compatible: single DataFrame still works for shared covariates

### 📊 Prediction Intervals

Generate prediction intervals using conformal prediction:

```python
from autotsforecast.uncertainty.intervals import PredictionIntervals

pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model, y_train)
intervals = pi.predict(forecasts)
# intervals['lower_95'], intervals['upper_95']
```

### 📅 Calendar Features

Automatic time-based feature extraction with cyclical encoding:

```python
from autotsforecast.features.calendar import CalendarFeatures

cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)
# Auto-detects dayofweek, month, etc. with sin/cos encoding
```

**Features:**
- Auto-detection of relevant time components
- Cyclical encoding (sin/cos) for periodic features
- Fourier terms for seasonal patterns
- Holiday detection (with `holidays` package)

### 🖼️ Visualization

Publication-ready plots with matplotlib and Plotly:

```python
from autotsforecast.visualization.plots import plot_forecast, plot_model_comparison

# Static matplotlib plot with prediction intervals
fig = plot_forecast(
    y_train=y_train,
    forecasts=predictions,
    y_test=y_test,
    intervals={'lower_95': lower, 'upper_95': upper}
)

# Model comparison bar chart
fig = plot_model_comparison(results, metric='rmse')
```

### ⚡ Parallel Processing

Speed up multi-series forecasting:

```python
from autotsforecast.utils.parallel import ParallelForecaster

pf = ParallelForecaster(n_jobs=4)
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=14),
    y=y_train,
    X=X_train
)
```

### 📈 Progress Tracking

Rich progress bars for long-running operations (requires `rich` package):

```python
# Progress bars automatically appear during:
# - Cross-validation
# - Model fitting
# - Parallel operations
```

---

## v0.2.2 (December 2025)

### Features
- Per-series model selection (`per_series_models=True`)
- Hierarchical reconciliation (OLS, MinT methods)
- Backtesting with time series cross-validation
- Interpretability with sensitivity analysis

### Models
- ARIMA, ETS, Prophet
- XGBoost, RandomForest
- LSTM (via Darts)
- Linear, MovingAverage, VAR

---

## v0.1.0 (November 2025, yanked)

- Initial release
- AutoForecaster with automatic model selection
- Support for covariates (external regressors)
- Basic forecasting models

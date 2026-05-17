# Release Notes

## v0.6.0 (May 2026) — 16 Models, Smart Presets, Faster Search

### 🚀 New Models (8 added, 16 total)

- **`LightGBMForecaster`** — LightGBM gradient boosting with direct multi-step forecasting. Fast, accurate, supports covariates. `pip install "autotsforecast[lightgbm]"`
- **`CatBoostForecaster`** — CatBoost gradient boosting. Strong on categorical-adjacent tabular patterns. `pip install "autotsforecast[catboost]"`
- **`ElasticNetForecaster`** — Elastic-Net regularised regression with lag features. Fast baseline, no extra install.
- **`ThetaForecaster`** — Theta method (statsmodels). Reliable on seasonal series with long history.
- **`CrostonForecaster`** — Croston/SBA method for intermittent (sparse) demand. No extra install.
- **`NBEATSForecaster`** — Neural Basis Expansion Analysis via Darts. `pip install "autotsforecast[neural]"`
- **`NHiTSForecaster`** — Neural Hierarchical Interpolation via Darts. `pip install "autotsforecast[neural]"`
- **`TFTForecaster`** — Temporal Fusion Transformer via Darts. `pip install "autotsforecast[neural]"`

### 📦 Smart Presets

One-word model selection:

```python
auto = AutoForecaster(preset="balanced", horizon=14)
```

| Preset | Models | Use case |
|--------|--------|----------|
| `fast` | Linear, MA, ElasticNet, Theta, ARIMA, ETS | < 60 s budget |
| `balanced` | + RF, XGBoost, LightGBM | Default recommendation |
| `accuracy` | All ML + NBEATS, NHiTS, TFT | Overnight runs |
| `zero_shot` | Chronos-2 | Cold-start / no training data |
| `intermittent` | Croston, ElasticNet, LightGBM | Sparse demand |
| `hierarchical` | VAR, RF, XGBoost, LightGBM | Multi-level org hierarchies |

### ⚡ Parallel & Budget-Aware Search

```python
auto = AutoForecaster(
    preset="accuracy",
    horizon=30,
    n_jobs=-1,           # parallel across candidates
    time_limit=120,      # stop after 2 minutes
    max_models=8,        # try at most 8 models
    backtest_mode="fast" # 2 folds instead of 5
)
```

### 🔍 Dataset Profiler

```python
result = AutoForecaster.profile_data(y_train)
result.print_summary()
# → recommended_preset: 'balanced'
```

### 📊 Structured Report

```python
auto.fit(y)
auto.print_report()   # ranked leaderboard + selection rationale
report = auto.get_report()  # machine-readable dict
```

### 📦 New Optional Extras

| Extra | Install | Contents |
|-------|---------|----------|
| `lightgbm` | `pip install "autotsforecast[lightgbm]"` | `lightgbm>=3.0` |
| `catboost` | `pip install "autotsforecast[catboost]"` | `catboost>=1.0` |

`[ml]` extra now includes LightGBM and CatBoost. `[all]` includes everything.

### 🔄 Backwards Compatibility

All existing `candidate_models=[...]` usage is fully backwards compatible.

---

## v0.5.0 (June 2026) — Agentic AI Edition

### 🤖 New Agentic AI Features

- **MCP Server** (`autotsforecast.mcp.server`) — Expose 7 forecasting tools to Claude Desktop, Cursor, Windsurf, and any MCP-compatible AI assistant via the Model Context Protocol. Start with `autotsforecast-mcp` CLI.
- **FastAPI REST Service** (`autotsforecast.api.app`) — Full HTTP API with 9 endpoints. Start with `autotsforecast-api` CLI. Accepts CSV data as JSON strings or multipart file uploads.
- **OpenAI / Anthropic Tool Schemas** (`autotsforecast.integrations.openai_schemas`) — Drop-in function-calling schemas for GPT and Claude. `handle_tool_call()` dispatcher handles everything.
- **LangChain Tools** (`autotsforecast.integrations.langchain_tools`) — `BaseTool` wrappers for all forecasting operations. Pass to any LangChain ReAct or LCEL agent.
- **AnomalyDetector** (`autotsforecast.anomaly.detector`) — Four detection methods: `zscore`, `iqr`, `isolation_forest`, `forecast_residual`. Returns boolean DataFrame + Pydantic summary.
- **InsightEngine** (`autotsforecast.nlp.insights`) — Rule-based or LLM-powered plain-English summaries and risk flags for any forecast.
- **ModelRegistry** (`autotsforecast.registry.store`) — Local model persistence: `save()`, `load()`, `list()`, `delete()`. Default storage at `~/.autotsforecast/registry/`.
- **Structured Outputs** (`autotsforecast.schemas`) — Pydantic v2 result models (`ForecastResult`, `BacktestResult`, `AnomalyResult`, etc.) with graceful fallback when pydantic is not installed. `AutoForecaster.to_structured()` added.

### 📦 New Optional Extras

| Extra | Command | Contents |
|-------|---------|---------|
| `mcp` | `pip install "autotsforecast[mcp]"` | `mcp>=1.0` |
| `api` | `pip install "autotsforecast[api]"` | `fastapi`, `uvicorn`, `python-multipart` |
| `langchain` | `pip install "autotsforecast[langchain]"` | `langchain` |
| `nlp` | `pip install "autotsforecast[nlp]"` | `openai` |
| `agentic` | `pip install "autotsforecast[agentic]"` | All of the above |

### 🆕 CLI Entry Points

- `autotsforecast-mcp` — Start MCP server (stdio transport)
- `autotsforecast-api` — Start FastAPI server (HTTP)

### ✅ Testing

- 50 tests pass, 7 skipped (expected — extras not installed in CI)
- 7 new test files: `test_schemas.py`, `test_anomaly.py`, `test_insights.py`, `test_registry.py`, `test_mcp_tools.py`, `test_openai_schemas.py`, `test_api.py`

### 🔄 Backwards Compatibility

All existing APIs are fully backwards compatible. New features are opt-in extras.

---

## v0.4.0 (March 2026)

### 🚀 New & Improved

- **Rewritten tutorial notebook** — `examples/autotsforecast_tutorial.ipynb` now uses a carefully designed synthetic DGP that **guarantees measurable improvements** for both showcase features:
  - *Per-series covariates*: `region_a` is driven by promotion; `region_b` by temperature. Giving each series only its true driver removes cross-noise and produces clear RMSE gains vs. the shared-features baseline.
  - *Hierarchical reconciliation*: OLS reconciliation enforces `total = region_a + region_b`, exploiting the accurate component forecasts to improve total-level accuracy.
- **Portable notebook** — Added an installation cell (`pip install autotsforecast[ml]`) so the notebook runs out-of-the-box anywhere, not just inside this repo.
- **Corrected all documentation** — `README.md`, `API_REFERENCE.md`, `TECHNICAL_DOCUMENTATION.md`, `README_PYPI.md`, `QUICKSTART.md` all updated to v0.4.0 with accurate model tables, covariate support flags, and Chronos-2 details.

### 🐛 Bug Fixes (carried from v0.3.9 development)

- **`get_summary()` / `print_summary()` crash in per-series mode** — Both methods now have a dedicated per-series branch returning `per_series_models` dict and per-series CV results.
- **`VARForecaster` silent failure on single series** — Raises a clear `ValueError` when fewer than 2 columns are passed.
- **`BacktestValidator` mutates shared model instance** — Each CV fold now deep-copies the model before fitting (fixed in both `run()` and `run_with_holdout()`).
- **Absolute import in `validator.py`** — Changed to relative import, fixing `ImportError` when running from source before installation.
- **`LinearForecaster` silently skipped in default candidates** — Removed from `get_default_candidate_models()`; it requires covariates and was silently dropped on every plain `fit(y)` call.
- **Fragile `_clone_model` fallback** — Removed 15-attribute hardcoded list; now falls back to `get_params()` only, with a clear error if that also fails.

### ⚙️ Internals & Tooling

- **Single-source version** — `__version__` read from `importlib.metadata`, eliminating `__init__.py` / `pyproject.toml` drift.
- **GitHub Actions CI** — Added `.github/workflows/tests.yml` running the full test suite on Python 3.9–3.12 on every push and PR.

---

## v0.3.9 (March 2026)

### 🐛 Bug Fixes

- **`get_summary()` / `print_summary()` crash in per-series mode** — Both methods now have a dedicated per-series branch. In per-series mode `get_summary()` returns `per_series_models` (dict) and `all_results` (per-series CV results) instead of crashing with a `KeyError`.
- **`VARForecaster` silent failure on single series** — `VARForecaster.fit()` now raises a clear `ValueError` when fewer than 2 columns are passed, instead of letting statsmodels emit a confusing error.
- **`BacktestValidator` mutates shared model instance** — Each CV fold now deep-copies the model before fitting, eliminating any risk of shared state corrupting results when running parallel or sequential folds.
- **Absolute import in `validator.py`** — Changed `from autotsforecast.models.base import BaseForecaster` to a relative import, fixing `ImportError` when running tests from source before installation.
- **`LinearForecaster` silently skipped in `get_default_candidate_models()`** — Removed `LinearForecaster` from the default candidate pool. It requires covariates `X` and was being silently dropped on every plain `AutoForecaster.fit(y)` call.
- **Fragile `_clone_model` fallback** — Removed the hardcoded 15-attribute fallback list; `_clone_model` now falls back only to `get_params()` (which all built-in models expose), with a clear error message if that also fails.

### ⚙️ Internals & Tooling

- **Single-source version** — `__version__` is now read from package metadata via `importlib.metadata`, eliminating the drift that caused `__init__.py` to report `0.3.3` while `pyproject.toml` said `0.3.9`.
- **CI/CD** — Added GitHub Actions workflow (`.github/workflows/tests.yml`) that runs the full test suite against Python 3.9–3.12 on every push and pull request.

---

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

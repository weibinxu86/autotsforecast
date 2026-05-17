# AutoTSForecast

**Automated Time Series Forecasting — 16+ Models, Smart Presets, AI-Native**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoTSForecast evaluates every model — statistical, ML, and deep learning — and picks the winner for your data.

## Installation

```bash
# Core + all 16 models
pip install "autotsforecast[all]"

# Agentic AI features (MCP, FastAPI, LangChain, OpenAI tools)
pip install "autotsforecast[agentic]"

# New v0.6.0 boosting models
pip install "autotsforecast[lightgbm]"
pip install "autotsforecast[catboost]"
```

## Quick Start

```python
from autotsforecast import AutoForecaster

# Profile your data, get preset recommendation
result = AutoForecaster.profile_data(y_train)
result.print_summary()  # → recommended_preset: 'balanced'

# One-line auto-selection
auto = AutoForecaster(preset="balanced", horizon=14)
auto.fit(y_train)
forecasts = auto.forecast()
auto.print_report()  # ranked leaderboard
```

## All 16 Available Models

| Model | Type | Covariates |
|-------|------|------------|
| `MovingAverageForecaster` | Statistical | ❌ |
| `ARIMAForecaster` | Statistical | ❌ |
| `ETSForecaster` | Statistical | ❌ |
| `ThetaForecaster` 🆕 | Statistical | ❌ |
| `CrostonForecaster` 🆕 | Statistical (intermittent) | ❌ |
| `VARForecaster` | Statistical | ❌ |
| `LinearForecaster` | ML | ✅ |
| `ElasticNetForecaster` 🆕 | ML | ✅ |
| `RandomForestForecaster` | ML | ✅ |
| `XGBoostForecaster` | ML | ✅ |
| `LightGBMForecaster` 🆕 | ML | ✅ |
| `CatBoostForecaster` 🆕 | ML | ✅ |
| `LSTMForecaster` | Deep learning | ❌ |
| `NBEATSForecaster` 🆕 | Deep learning | ❌ |
| `NHiTSForecaster` 🆕 | Deep learning | ❌ |
| `TFTForecaster` 🆕 | Deep learning | ❌ |
| `Chronos2Forecaster` | Foundation | ❌ |

## Key Features

### 🤖 AutoForecaster — Automatic Per-Series Model Selection

```python
from autotsforecast import AutoForecaster

# Preset-based (v0.6.0)
auto = AutoForecaster(preset="balanced", horizon=14)

# Budget-aware (v0.6.0)
auto = AutoForecaster(
    preset="accuracy",
    horizon=30,
    n_jobs=-1,           # parallel candidates
    time_limit=120,      # stop after 2 minutes
    max_models=8,
    backtest_mode="fast"
)

auto.fit(y_train)
result = auto.to_structured()   # JSON-serialisable Pydantic model
```

### 🌐 MCP Server — Claude Desktop / Cursor / Windsurf (v0.5.0)
```bash
pip install "autotsforecast[mcp]"
autotsforecast-mcp   # 7 tools: forecast, backtest, anomaly detection, and more
```

### 🔧 OpenAI / Anthropic Tool Calling (v0.5.0)
```python
from autotsforecast.integrations.openai_schemas import get_openai_tools, handle_tool_call

tools = get_openai_tools()   # pass to openai.chat.completions.create(tools=...)
result = handle_tool_call("fit_and_forecast", {"csv_data": "...", "horizon": 14})
```

### 🦜 LangChain Tools (v0.5.0)
```python
from autotsforecast.integrations.langchain_tools import get_autotsforecast_tools
tools = get_autotsforecast_tools()   # BaseTool list for any LangChain agent
```

### 🌐 FastAPI REST Service (v0.5.0)
```bash
pip install "autotsforecast[api]"
autotsforecast-api   # http://0.0.0.0:8000 — POST /forecast, /anomalies, /backtest, ...
```

### 📡 Anomaly Detection (v0.5.0)
```python
from autotsforecast.anomaly.detector import AnomalyDetector

detector = AnomalyDetector(method='zscore')
anomalies = detector.fit_predict(y_train)
```

### 💬 NLP Insight Engine (v0.5.0)
```python
from autotsforecast.nlp.insights import InsightEngine

engine = InsightEngine(mode='rule_based')
summary = engine.summarize_forecast_dataframes(y_train, forecasts, y_test)
```

### 📦 Model Registry (v0.5.0)
```python
from autotsforecast.registry.store import ModelRegistry

registry = ModelRegistry()
registry.save(auto, name='production_v1')
auto2 = registry.load('production_v1')
```

### 📊 Hierarchical Reconciliation
Ensure forecasts are coherent across aggregation levels (e.g., regions sum to total).

```python
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

reconciler = HierarchicalReconciler(forecasts=base_forecasts, hierarchy={'total': ['region_a', 'region_b']})
reconciler.reconcile(method='ols')
```

### ✅ Backtesting & Cross-Validation
Robust time series cross-validation to evaluate model performance.

```python
from autotsforecast.backtesting.validator import BacktestValidator

validator = BacktestValidator(model=my_model, n_splits=5, test_size=14)
validator.run(y_train)
```

### 🔍 Interpretability
Understand which features drive your forecasts using sensitivity analysis and SHAP values.

```python
from autotsforecast.interpretability.drivers import DriverAnalyzer

analyzer = DriverAnalyzer(model=fitted_model, feature_names=['temperature', 'promotion'])
importance = analyzer.calculate_feature_importance(X_test, y_test)
```

## Available Models

**Core Models** (included): ARIMA, ETS, MovingAverage, RandomForest, VAR

> `LinearForecaster` is also included but **requires covariates `X`** — it is not part of the default candidate pool.

**Optional Models**: XGBoost, Prophet, LSTM (install with extras: `[ml]`, `[prophet]`, `[neural]`), Chronos-2 (install with `[chronos]`)

**Covariate Support**: Prophet, ARIMA, XGBoost, RandomForest, LSTM, Linear

## v0.5.0 Extras Summary

| Extra | Command | Adds |
|-------|---------|------|
| `mcp` | `pip install "autotsforecast[mcp]"` | MCP server for Claude/Cursor/Windsurf |
| `api` | `pip install "autotsforecast[api]"` | FastAPI REST service |
| `langchain` | `pip install "autotsforecast[langchain]"` | LangChain BaseTool wrappers |
| `agentic` | `pip install "autotsforecast[agentic]"` | All of the above |

## Documentation

**Full documentation, tutorials, and examples:**  
[https://github.com/weibinxu86/autotsforecast](https://github.com/weibinxu86/autotsforecast)

## License

MIT License

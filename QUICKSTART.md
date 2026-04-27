# AutoTSForecast Quick Start

## Installation

```bash
pip install autotsforecast           # Core
pip install "autotsforecast[all]"    # All features
```

## 1. Basic Forecasting

```python
import pandas as pd
from autotsforecast.models.base import MovingAverageForecaster

# Your data
y_train, y_test = df.iloc[:150], df.iloc[150:]

model = MovingAverageForecaster(horizon=30, window=7)
model.fit(y_train)
predictions = model.predict()
```

## 2. AutoForecaster (Auto Model Selection)

```python
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import RandomForestForecaster, ARIMAForecaster

candidates = [
    MovingAverageForecaster(horizon=30, window=7),
    RandomForestForecaster(horizon=30, n_lags=7),
    ARIMAForecaster(horizon=30),
]

auto = AutoForecaster(candidate_models=candidates, metric='rmse', n_splits=3)
auto.fit(y_train)
forecasts = auto.forecast()
print(f"Best model: {auto.best_model_name_}")
```

## 3. Using Covariates

```python
from autotsforecast.models.external import RandomForestForecaster

model = RandomForestForecaster(horizon=30, n_lags=7)
model.fit(y_train, X=X_train)
predictions = model.predict(X=X_test)
```

## 4. Per-Series Covariates

Different features for different series:

```python
# Different covariates per series
X_train_dict = {
    'series_a': X_train[['temperature']],   # Series A uses temperature
    'series_b': X_train[['promotion']],     # Series B uses promotion
}

auto = AutoForecaster(candidates, per_series_models=True)
auto.fit(y_train, X=X_train_dict)
forecasts = auto.forecast(X=X_test_dict)
```

## 5. Prediction Intervals

```python
from autotsforecast.uncertainty.intervals import PredictionIntervals

pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model, y_train)
intervals = pi.predict(forecasts)
# intervals['lower_95'], intervals['upper_95']
```

## 6. Calendar Features

```python
from autotsforecast.features.calendar import CalendarFeatures

cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)
```

## 7. Hierarchical Reconciliation

```python
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

hierarchy = {'total': ['region_a', 'region_b']}
reconciler = HierarchicalReconciler(forecasts, hierarchy)
reconciler.reconcile(method='ols')
coherent = reconciler.reconciled_forecasts
```

## 8. Backtesting

```python
from autotsforecast.backtesting.validator import BacktestValidator

validator = BacktestValidator(model=model, n_splits=5, test_size=14)
validator.run(y_train, X=X_train)
results = validator.get_fold_results()
```

## 9. Parallel Processing

```python
from autotsforecast.utils.parallel import ParallelForecaster

pf = ParallelForecaster(n_jobs=4)
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=14, n_lags=7),
    y=y_train, X=X_train
)
```

## 10. Interpretability

```python
from autotsforecast.interpretability.drivers import DriverAnalyzer

analyzer = DriverAnalyzer(model, feature_names=['temp', 'promo'])
importance = analyzer.calculate_feature_importance(X_test, y_test, method='sensitivity')
```

## 11. Chronos-2 Foundation Model (Zero-Shot Forecasting)

**State-of-the-art pretrained model** — no training needed!

```python
from autotsforecast.models.external import Chronos2Forecaster

# Default: amazon/chronos-2 (120M params, best accuracy)
model = Chronos2Forecaster(horizon=24, model_name="amazon/chronos-2")
model.fit(y_train)  # Just stores context, no training!
forecasts = model.predict()

# Fast variant: Chronos-Bolt (up to 250x faster)
model_fast = Chronos2Forecaster(horizon=24, model_name="amazon/chronos-bolt-small")
model_fast.fit(y_train)
forecasts_fast = model_fast.predict()

# Probabilistic forecasts with quantiles
quantile_forecasts = model.predict_quantiles(quantile_levels=[0.1, 0.5, 0.9])
```

**Available Model Sizes:**
- `"amazon/chronos-2"` — 120M params (best accuracy)
- `"autogluon/chronos-2-small"` — 28M params (good balance)
- `"amazon/chronos-bolt-tiny"` — 9M params (ultra fast)
- `"amazon/chronos-bolt-small"` — 48M params (balanced)
- `"amazon/chronos-bolt-base"` — 205M params (accurate + fast)

**Installation:**
```bash
pip install "autotsforecast[chronos]"
```

## Model Comparison

| Model | Covariates | Speed | Best For |
|-------|-----------|-------|----------|
| MovingAverage | ❌ | ⭐⭐⭐ | Simple baselines |
| ARIMA/ETS | ❌ | ⭐⭐ | Classical time series |
| RandomForest | ✅ | ⭐⭐ | General purpose |
| XGBoost | ✅ | ⭐⭐ | High accuracy |
| Prophet | ✅ | ⭐⭐ | Seasonality + holidays |
| LSTM | ✅ | ⭐ | Deep learning |
| **Chronos-2** | ❌ | ⭐⭐⭐ | **Zero-shot state-of-the-art** |

## Common Parameters

| Parameter | Typical Values | Description |
|-----------|---------------|-------------|
| `horizon` | 7-30 | Forecast steps |
| `n_lags` | 7-30 | Lag features |
| `n_estimators` | 100-500 | Trees (RF/XGB) |
| `n_splits` | 3-5 | CV folds |
| `metric` | `'rmse'` | Selection metric |

## More Resources

- [API Reference](API_REFERENCE.md) — Complete parameter docs
- [Tutorial](examples/autotsforecast_tutorial.ipynb) — Core forecasting tutorial
- [Agentic AI Tutorial](examples/agentic_tutorial.ipynb) — MCP, OpenAI tools, anomaly detection, registry

---

## 🤖 Agentic AI (v0.5.0)

### 12. Anomaly Detection

Detect outliers before forecasting to protect model accuracy:

```python
from autotsforecast.anomaly.detector import AnomalyDetector

# Available methods: 'zscore', 'iqr', 'isolation_forest', 'forecast_residual'
detector = AnomalyDetector(method='zscore', contamination=0.05)
anomalies = detector.fit_predict(y_train)   # bool DataFrame, True = anomaly
summary = detector.get_summary()            # AnomalyResult Pydantic model

print(f"Method: {summary.method}")
print(f"Total anomalies: {summary.total_anomalies}")
print(summary.per_series)  # dict[series_name, count]
```

**Installation:** core package (no extras needed)

### 13. Structured Outputs (Pydantic)

Get JSON-serialisable, agent-ready output from AutoForecaster:

```python
from autotsforecast import AutoForecaster

auto = AutoForecaster(candidate_models=candidates, metric='rmse')
auto.fit(y_train)
forecasts = auto.forecast()

result = auto.to_structured()         # ForecastResult Pydantic model
print(result.best_model)              # "ARIMAForecaster"
print(result.metric)                  # "rmse"
print(result.model_dump_json())       # JSON string — pass to any agent
```

**Installation:** `pip install pydantic>=2.0` (optional — works without it too)

### 14. Natural Language Insights

Generate plain-English summaries of forecasts for agents and reports:

```python
from autotsforecast.nlp.insights import InsightEngine

engine = InsightEngine(mode='rule_based')
summary = engine.summarize_forecast_dataframes(y_train, forecasts, y_test)
risks   = engine.flag_risks_from_dataframes(y_train, forecasts)
report  = engine.generate_report(result, y_train, y_test)

# Or use with your LLM client (OpenAI, Anthropic, etc.)
engine_llm = InsightEngine(mode='llm', llm_client=openai_client, model='gpt-4o')
narrative = engine_llm.summarize_forecast_dataframes(y_train, forecasts, y_test)
```

**Installation:** core package (`mode='rule_based'`); any OpenAI-compatible client for `mode='llm'`

### 15. Model Registry

Persist fitted models locally, reload them across processes or deployments:

```python
from autotsforecast.registry.store import ModelRegistry

registry = ModelRegistry()                  # Default: ~/.autotsforecast/registry/

# Save
registry.save(auto, name='production_v1', tags={'env': 'prod', 'version': '1.0'})

# List all saved models
df = registry.list()
print(df[['name', 'model_class', 'saved_at']])

# Load and use
auto_loaded = registry.load('production_v1')
new_forecasts = auto_loaded.forecast()

# Clean up
registry.delete('production_v1')
```

**Installation:** core package (no extras needed)

### 16. MCP Server (Claude Desktop / Cursor / Windsurf)

Expose all forecasting tools to AI assistants via the Model Context Protocol:

```bash
# Install
pip install "autotsforecast[mcp]"

# Run server (stdio transport, compatible with all MCP clients)
autotsforecast-mcp
```

**Configure Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "autotsforecast": {
      "command": "autotsforecast-mcp"
    }
  }
}
```

**Available MCP tools:**
| Tool | Description |
|------|-------------|
| `fit_and_forecast` | AutoForecaster: select best model and forecast |
| `run_backtest` | Time-series cross-validation for any model |
| `prediction_intervals` | Conformal prediction intervals |
| `anomaly_detection` | Detect outliers in CSV data |
| `calendar_features` | Extract time-based features |
| `reconcile_hierarchy` | Hierarchical forecast reconciliation |
| `model_catalog` | List all available models |

Once configured, simply ask Claude: *"Forecast the next 30 days of my sales data"* and attach a CSV.

### 17. FastAPI REST Service

Start an HTTP server that any language, agent, or microservice can call:

```bash
# Install
pip install "autotsforecast[api]"

# Start server (default: http://0.0.0.0:8000)
autotsforecast-api

# Custom host/port
autotsforecast-api --host 127.0.0.1 --port 9000
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/forecast` | Fit + forecast from JSON |
| POST | `/forecast/upload` | Fit + forecast from CSV file upload |
| POST | `/backtest` | Run cross-validation |
| POST | `/intervals` | Compute prediction intervals |
| POST | `/reconcile` | Hierarchical reconciliation |
| POST | `/calendar-features` | Calendar feature extraction |
| POST | `/anomalies` | Anomaly detection |

**Example request:**
```python
import httpx, json

data = {
    "csv_data": y_train.to_csv(),
    "horizon": 14,
    "metric": "rmse"
}
resp = httpx.post("http://localhost:8000/forecast", json=data)
print(resp.json()["forecasts"])
```

### 18. OpenAI / Anthropic Tool Calling

Use autotsforecast as function-calling tools with any LLM:

```python
from autotsforecast.integrations.openai_schemas import (
    get_openai_tools,
    get_anthropic_tools,
    handle_tool_call
)

# --- OpenAI GPT-4o ---
import openai
client = openai.OpenAI()

tools = get_openai_tools()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Forecast next 14 days from this CSV: ..."}],
    tools=tools
)

# Dispatch the tool call and get the result
for call in response.choices[0].message.tool_calls:
    result = handle_tool_call(call.function.name, call.function.arguments)
    print(result)

# --- Anthropic Claude ---
import anthropic
client = anthropic.Anthropic()

tools = get_anthropic_tools()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Detect anomalies in this data: ..."}]
)
```

**Installation:** `pip install openai` or `pip install anthropic` (no autotsforecast extras needed)

### 19. LangChain Integration

Use autotsforecast tools with any LangChain or LCEL agent:

```python
from autotsforecast.integrations.langchain_tools import get_autotsforecast_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

tools = get_autotsforecast_tools()
llm = ChatOpenAI(model="gpt-4o")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({
    "input": "Analyze my sales data for anomalies, then forecast the next 14 days."
})
```

**Installation:** `pip install "autotsforecast[langchain]"`

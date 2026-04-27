# Technical Documentation

AutoTSForecast v0.5.0 — Architecture and Implementation Details

## Package Structure

```
autotsforecast/
├── __init__.py              # Package exports and version
├── forecaster.py            # AutoForecaster class (+ to_structured())
├── schemas.py               # Pydantic result models (ForecastResult, etc.)
├── models/
│   ├── base.py              # BaseForecaster, VAR, Linear, MovingAverage
│   ├── external.py          # RandomForest, XGBoost, Prophet, ARIMA, ETS, LSTM, Chronos2
│   └── selection.py         # ModelSelector for CV-based selection
├── backtesting/
│   └── validator.py         # BacktestValidator for time series CV
├── hierarchical/
│   └── reconciliation.py    # HierarchicalReconciler (OLS, bottom-up, top-down)
├── interpretability/
│   └── drivers.py           # DriverAnalyzer (SHAP, sensitivity, permutation)
├── features/
│   ├── calendar.py          # CalendarFeatures (time-based feature extraction)
│   └── engine.py            # FeatureEngine (general feature engineering)
├── uncertainty/
│   └── intervals.py         # PredictionIntervals, ConformalPredictor
├── anomaly/                 # NEW v0.5.0
│   └── detector.py          # AnomalyDetector (zscore, iqr, isolation_forest, residual)
├── nlp/                     # NEW v0.5.0
│   └── insights.py          # InsightEngine (rule-based + LLM narrative)
├── registry/                # NEW v0.5.0
│   └── store.py             # ModelRegistry (local pickle + JSON index)
├── mcp/                     # NEW v0.5.0
│   └── server.py            # MCP server with 7 forecasting tools
├── api/                     # NEW v0.5.0
│   └── app.py               # FastAPI REST service
├── integrations/            # NEW v0.5.0
│   ├── openai_schemas.py    # OpenAI/Anthropic function-calling schemas
│   └── langchain_tools.py   # LangChain BaseTool wrappers
├── visualization/
│   ├── plots.py             # plot_forecast, plot_model_comparison, etc.
│   └── progress.py          # ProgressTracker for long operations
└── utils/
    ├── data.py              # Data utilities
    ├── parallel.py          # ParallelForecaster, parallel_map
    └── preprocessing.py     # CovariatePreprocessor
```

## Core Components

### AutoForecaster (`forecaster.py`)

Main entry point for automatic model selection.

**Key Features:**
- Cross-validation based model selection
- Per-series model selection (`per_series_models=True`)
- Per-series covariates (pass `X` as `Dict[str, DataFrame]`)
- Parallel fitting (`n_jobs=-1`)

**Flow:**
1. `fit(y, X)` → Runs CV for each candidate model
2. Selects best model(s) based on metric
3. Refits best model(s) on full training data
4. `forecast(X)` → Generates predictions

**Per-Series Covariates Implementation:**
```python
# In fit():
if isinstance(X, dict):
    self._per_series_covariates_ = True
    # Each series uses its own X[series_name]
```

### BaseForecaster (`models/base.py`)

Abstract base class for all forecasters.

**Interface:**
```python
class BaseForecaster(ABC):
    def fit(self, y: pd.DataFrame, X: pd.DataFrame = None) -> 'BaseForecaster'
    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame
    @property
    def horizon(self) -> int
```

**Implementations:**
- `VARForecaster` — statsmodels VAR (requires ≥ 2 series; does not accept `X`)
- `LinearForecaster` — sklearn LinearRegression; **requires covariates `X`** at both fit and predict time
- `MovingAverageForecaster` — Simple moving average

### External Models (`models/external.py`)

ML and statistical models with optional dependencies.

| Model | Backend | Covariates | Notes |
|-------|---------|-----------|-------|
| `RandomForestForecaster` | sklearn | ✅ | Auto lag features + categorical encoding |
| `XGBoostForecaster` | xgboost | ✅ | Gradient boosting |
| `ProphetForecaster` | prophet | ✅ | Seasonality + holidays |
| `ARIMAForecaster` | statsmodels | ✅ | Classical ARIMA |
| `ETSForecaster` | statsmodels | ❌ | Exponential smoothing |
| `LSTMForecaster` | darts/torch | ✅ | Deep learning |
| `Chronos2Forecaster` | chronos-forecasting | ❌ | Zero-shot foundation model (no training) |

### BacktestValidator (`backtesting/validator.py`)

Time series cross-validation.

**Window Types:**
- `expanding`: Training window grows each fold
- `rolling`: Fixed-size training window

**Methods:**
- `run(y, X)` → Returns overall metrics
- `get_fold_results()` → Per-fold DataFrame
- `get_summary()` → Mean/std/min/max statistics
- `plot_results()` → Visualization
- `get_predictions()` → Raw actuals and predictions

### HierarchicalReconciler (`hierarchical/reconciliation.py`)

Ensures hierarchical coherence (e.g., total = sum of parts).

**Methods:**
- `bottom_up`: Aggregate from bottom level
- `top_down`: Disaggregate using proportions
- `ols`: Optimal reconciliation (minimizes squared error)

### PredictionIntervals (`uncertainty/intervals.py`)

Uncertainty quantification.

**Methods:**
- `conformal`: Distribution-free coverage guarantees
- `residual`: Gaussian intervals from residual std
- `empirical`: Empirical quantile intervals

**Usage:**
```python
pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model, y_train)
intervals = pi.predict(forecasts)
# intervals['lower_95'], intervals['upper_95']
```

### CalendarFeatures (`features/calendar.py`)

Time-based feature extraction.

**Features:**
- Auto-detection of relevant time components
- Cyclical encoding (sin/cos) for periodic features
- Fourier terms for complex seasonality
- Holiday detection (with `holidays` package)

### ParallelForecaster (`utils/parallel.py`)

Parallel multi-series forecasting.

**Key Function:**
```python
pf = ParallelForecaster(n_jobs=4)
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=14),
    y=y_train, X=X_train
)
```

## Data Flow

### Training Flow

```
y (DataFrame) ──┬──> AutoForecaster.fit()
                │         │
X (DataFrame    │    ┌────┴────┐
   or Dict)  ───┘    │ For each candidate model:
                     │   BacktestValidator.run()
                     │   → CV metrics
                     └────┬────┘
                          │
                     Select best model(s)
                          │
                     Refit on full data
                          │
                     Store in best_models_
```

### Prediction Flow

```
X_future ──> AutoForecaster.forecast()
                    │
             ┌──────┴──────┐
             │ For each series:
             │   best_models_[series].predict(X)
             └──────┬──────┘
                    │
             Combine into DataFrame
                    │
             Return forecasts
```

## Dependencies

### Core (Required)
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 1.0.0
- statsmodels ≥ 0.13.0
- scipy ≥ 1.7.0
- joblib ≥ 1.1.0

### Optional
- `[ml]`: xgboost ≥ 2.0.0
- `[prophet]`: prophet ≥ 1.1.0
- `[neural]`: darts, torch, lightning
- `[viz]`: matplotlib, seaborn, plotly
- `[interpret]`: shap ≥ 0.42.0

## API Conventions

### Input Data

- `y`: `pd.DataFrame` with `DatetimeIndex`, columns = series names
- `X`: `pd.DataFrame` (shared) or `Dict[str, pd.DataFrame]` (per-series)
- Index must align between `y` and `X`

### Output Data

- Forecasts: `pd.DataFrame` with same columns as `y`
- Intervals: `dict` with keys like `'lower_95'`, `'upper_95'`

### Naming Conventions

- Classes: `CamelCase` (e.g., `AutoForecaster`)
- Methods: `snake_case` (e.g., `fit`, `predict`, `get_fold_results`)
- Parameters: `snake_case` (e.g., `n_splits`, `per_series_models`)

## Extending the Package

### Adding a New Model

1. Inherit from `BaseForecaster`
2. Implement `fit(y, X=None)` and `predict(X=None)`
3. Set `self.horizon` in constructor

```python
class MyForecaster(BaseForecaster):
    def __init__(self, horizon: int, my_param: float = 1.0):
        self.horizon = horizon
        self.my_param = my_param
    
    def fit(self, y: pd.DataFrame, X: pd.DataFrame = None):
        # Training logic
        self._fitted = True
        return self
    
    def predict(self, X: pd.DataFrame = None) -> pd.DataFrame:
        # Prediction logic
        return forecasts
```

### Adding a New Reconciliation Method

Add method to `HierarchicalReconciler.reconcile()`:

```python
def reconcile(self, method='ols'):
    if method == 'my_method':
        return self._my_reconciliation()
```

---

## 🤖 Agentic AI Components (v0.5.0)

### AnomalyDetector (`anomaly/detector.py`)

Wraps multiple outlier detection algorithms with a unified interface.

**Design:**
- `fit(y)` stores fitted detector state per series
- `predict(y)` returns a boolean DataFrame (`True` = anomaly)
- `fit_predict(y)` is shortcut for fit then predict
- `get_summary()` returns `AnomalyResult` Pydantic model

**Method dispatch:**

```python
_METHODS = {
    'zscore': _detect_zscore,
    'iqr': _detect_iqr,
    'isolation_forest': _detect_isolation_forest,
    'forecast_residual': _detect_residual,
}
```

Each method operates column-by-column (per series), allowing different series to have different anomaly patterns.

### InsightEngine (`nlp/insights.py`)

**Two modes:**
- `rule_based` — deterministic text based on statistical summaries (trend direction, MAPE, risk flags). No external dependencies.
- `llm` — passes the same summaries to an OpenAI-compatible `llm_client` for a polished narrative.

**Flow:**
1. `summarize_forecast_dataframes()` — computes per-series statistics → formats prompt → returns text
2. `flag_risks_from_dataframes()` — checks for high volatility, long flat series, large forecast jumps → returns warning list
3. `generate_report()` — combines `ForecastResult` metadata + summary + risks into a full report

### ModelRegistry (`registry/store.py`)

**Storage layout** (`~/.autotsforecast/registry/` by default):
```
registry/
├── index.json              # List of RegistryEntry dicts
├── production_v1.pkl       # Pickled AutoForecaster
└── staging_v2.pkl
```

**Operations:**
- `save()` — pickle → write to `{name}.pkl`, append `RegistryEntry` to `index.json`
- `load()` — read `index.json`, unpickle `{name}.pkl`
- `list()` — read `index.json` → return as pandas DataFrame
- `delete()` — remove `.pkl`, remove entry from `index.json`
- `clear()` — remove all `.pkl` files, reset `index.json` to `[]`

### Schemas (`schemas.py`)

Pydantic v2 result models with graceful fallback (custom `BaseModel` stub if pydantic not installed).

**Inheritance hierarchy:**
```
BaseModel (pydantic or stub)
├── ForecastResult
├── BacktestResult
├── IntervalResult
├── ImportanceResult
├── AnomalyResult
├── InsightResult
├── ModelInfo → ModelCatalog
└── RegistryEntry
```

All models support `.model_dump()` and `.model_dump_json()` (or `dict()` and `json()` as aliases in the stub).

### MCP Server (`mcp/server.py`)

Implements the [Model Context Protocol](https://modelcontextprotocol.io) for Claude Desktop, Cursor, and Windsurf.

**Architecture:**
- `create_server()` — instantiates `mcp.Server("autotsforecast")` and registers 7 tools via `@server.call_tool()`
- Each tool takes CSV string input (not file paths) for stateless operation
- `main()` — runs `mcp.run(server, stdio_transport)` as CLI entry point

**Tool handler pattern:**
```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "fit_and_forecast":
        return await _run_fit_and_forecast(**arguments)
    ...
```

### FastAPI Service (`api/app.py`)

Wraps all forecasting operations as HTTP endpoints.

**Architecture:**
- Pydantic request/response models for each endpoint
- CSV data accepted as JSON string fields or multipart file upload (`/forecast/upload`)
- All handlers are thin wrappers around the same helpers used by the MCP server

### Integration Layer (`integrations/`)

**`openai_schemas.py`:**
- `get_openai_tools()` — returns tool defs in OpenAI `tools` format (list of dicts with `type`, `function`, `parameters`)
- `get_anthropic_tools()` — same, Anthropic schema format
- `handle_tool_call(name, arguments)` — dispatch table that parses arguments and calls the correct helper

**`langchain_tools.py`:**
- Each forecasting operation is a `BaseTool` subclass with `name`, `description`, and `_run()` method
- `get_autotsforecast_tools()` — returns list of all tool instances
- Graceful ImportError if langchain not installed

---

## Dependency Architecture

```
Core (always installed)
├── autotsforecast.*         # All base functionality
├── anomaly.detector         # AnomalyDetector (no extras)
├── nlp.insights             # InsightEngine rule_based (no extras)
├── registry.store           # ModelRegistry (no extras)
└── schemas                  # Pydantic models (pydantic optional)

Optional extras
├── [mcp]    → mcp>=1.0         → mcp.server
├── [api]    → fastapi, uvicorn → api.app
├── [langchain] → langchain     → integrations.langchain_tools
├── [nlp]    → openai           → InsightEngine mode='llm'
└── [agentic] → all of the above
```

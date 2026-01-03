# AutoTSForecast: Technical Documentation

## Overview

This document provides a detailed technical explanation of how **AutoTSForecast** performs automatic model selection and interpretability analysis.

---

## Table of Contents

1. [Automatic Model Selection](#1-automatic-model-selection)
2. [Model Interpretability & Explanation](#2-model-interpretability--explanation)
3. [Hierarchical Reconciliation](#3-hierarchical-reconciliation)
4. [Implementation Details](#4-implementation-details)

---

## 1. Automatic Model Selection

### 1.1 Overview

The `AutoForecaster` class implements automated model selection using **time series cross-validation** (backtesting). It evaluates multiple candidate models and selects the best performer based on a specified metric.

### 1.2 Algorithm

```
INPUT: 
  - candidate_models: List of forecasting models
  - metric: Evaluation metric (RMSE, MAE, MAPE, R²)
  - n_splits: Number of backtesting folds
  - test_size: Size of each test set
  - window_type: 'expanding' or 'rolling'

PROCESS:
  1. For each candidate model:
     a. Perform time series cross-validation
     b. Calculate performance metric on each fold
     c. Average metrics across folds
  
  2. Select model with best average performance
  
  3. Retrain best model on full dataset
  
  4. Use for final predictions

OUTPUT:
  - best_model_: Fitted best performing model
  - cv_results_: Cross-validation scores for all models
  - forecasts_: Final predictions
```

### 1.3 Time Series Cross-Validation

#### Expanding Window

Each training set grows progressively:

```
Fold 1: |████████████████████|→→→→→| test
Fold 2: |████████████████████████|→→→→→| test
Fold 3: |████████████████████████████|→→→→→| test
```

**Advantages:**
- Uses all available historical data
- Mimics real-world scenario where data accumulates over time
- More stable for small datasets

#### Rolling Window

Fixed-size training window slides forward:

```
Fold 1: |████████████████████|→→→→→| test
Fold 2:     |████████████████████|→→→→→| test
Fold 3:         |████████████████████|→→→→→| test
```

**Advantages:**
- Focuses on recent patterns
- Better for non-stationary data with changing dynamics
- Prevents over-reliance on old data

### 1.4 Implementation (`AutoForecaster.fit()`)

**File:** `src/autotsforecast/forecaster.py`

```python
def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None):
    """
    Evaluate candidate models and select the best one.
    """
    best_score = float('inf') if self.metric != 'r2' else float('-inf')
    
    for model in self.candidate_models:
        # Create backtesting validator
        validator = BacktestValidator(
            model=model,
            n_splits=self.n_splits,
            test_size=self.test_size,
            window_type=self.window_type
        )
        
        # Run cross-validation
        results = validator.run(y, X)
        
        # Calculate average metric
        avg_score = np.mean([r['metrics'][self.metric] for r in results])
        
        # Update best model if better performance
        if self._is_better(avg_score, best_score):
            best_score = avg_score
            self.best_model_ = model
            self.cv_results_[model_name] = results
    
    # Retrain best model on full dataset
    self.best_model_.fit(y, X)
    return self
```

### 1.5 Supported Metrics

| Metric | Formula | Best Value | Use Case |
|--------|---------|------------|----------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Lower | General purpose, penalizes large errors |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Lower | Robust to outliers |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Lower | Scale-independent, percentage errors |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Higher (→1) | Explained variance |

### 1.6 Model Candidates

The AutoForecaster typically evaluates these models:

#### Statistical Models

**1. Vector AutoRegression (VAR)**
- **Type:** Multivariate linear model
- **Parameters:** `lags` (1, 3, 5, 7)
- **Best For:** Capturing cross-variable dependencies
- **Equation:** $y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + \epsilon_t$

**2. Linear Forecaster**
- **Type:** Linear regression with lags
- **Parameters:** `n_lags` (3, 5, 7)
- **Best For:** Simple linear patterns
- **Equation:** $\hat{y}_t = \beta_0 + \sum_{i=1}^p \beta_i y_{t-i}$

**3. Moving Average**
- **Type:** Simple average of recent values
- **Parameters:** `window` (5, 7, 10)
- **Best For:** Stable baseline, smoothing noise
- **Equation:** $\hat{y}_t = \frac{1}{w}\sum_{i=1}^w y_{t-i}$

#### Machine Learning Models

**4. Random Forest**
- **Type:** Ensemble of decision trees
- **Parameters:** `n_lags`, `n_estimators`, `max_depth`
- **Best For:** Non-linear patterns, feature interactions
- **Features:** Lags + covariates
- **Supports:** Covariate preprocessing

**5. XGBoost**
- **Type:** Gradient boosting
- **Parameters:** `n_lags`, `n_estimators`, `learning_rate`
- **Best For:** High-performance ML forecasting
- **Features:** Lags + covariates
- **Supports:** Covariate preprocessing

**6. Prophet**
- **Type:** Facebook's time series forecasting model
- **Parameters:** `horizon`, `growth`, `seasonality_mode`
- **Best For:** Business time series with strong seasonal effects and holidays
- **Features:** Automatic handling of trends, seasonality, and special events

**7. ARIMA**
- **Type:** AutoRegressive Integrated Moving Average
- **Parameters:** `order` (p,d,q), `seasonal_order`
- **Best For:** Classical statistical forecasting, stationary data
- **Features:** Time-tested approach for univariate and multivariate series

**8. ETS (Error-Trend-Seasonality)**
- **Type:** Exponential Smoothing State Space
- **Parameters:** `horizon`, `seasonal`, `seasonal_periods`
- **Best For:** Time series with trends and seasonal patterns
- **Features:** Automatic decomposition of trend, seasonality, and error

**9. LSTM (Long Short-Term Memory)**
- **Type:** Recurrent Neural Network
- **Parameters:** `n_lags`, `hidden_dim`, `num_layers`
- **Best For:** Complex temporal dependencies and long-range patterns
- **Features:** Deep learning architecture for sequence modeling

### 1.7 Backtesting Implementation

**File:** `src/autotsforecast/backtesting/validator.py`

```python
class BacktestValidator:
    def run(self, y, X=None):
        results = []
        
        for split_idx in range(self.n_splits):
            # Calculate split boundaries
            if self.window_type == 'expanding':
                train_end = len(y) - (self.n_splits - split_idx) * self.test_size
                train_start = 0
            else:  # rolling
                train_end = len(y) - (self.n_splits - split_idx) * self.test_size
                train_start = train_end - self.train_size
            
            test_start = train_end
            test_end = test_start + self.test_size
            
            # Split data
            y_train = y.iloc[train_start:train_end]
            y_test = y.iloc[test_start:test_end]
            
            # Fit and predict
            self.model.fit(y_train, X_train)
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions)
            results.append({'split': split_idx, 'metrics': metrics})
        
        return results
```

---

## 2. Model Interpretability & Explanation

### 2.1 Overview

The `DriverAnalyzer` class provides interpretability for forecasting models using **SHAP (SHapley Additive exPlanations)** values. This helps understand which **external covariates** drive predictions.

**Important:** SHAP analysis focuses exclusively on external covariates (e.g., marketing spend, weather, holidays). Lag features are excluded from interpretability analysis as they are internal time series components.

### 2.2 What are SHAP Values?

SHAP values are based on **cooperative game theory** (Shapley values) and provide:

- **Feature Importance:** How much each feature contributes to predictions
- **Direction:** Whether the feature pushes predictions higher or lower
- **Model-Agnostic:** Works with any model (with appropriate explainer)
- **Additive:** Sum of SHAP values = prediction - baseline

**Mathematical Foundation:**

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{i\}) - f(S)]$$

Where:
- $\phi_i$ = SHAP value for feature $i$
- $F$ = set of all features
- $S$ = subset of features
- $f(S)$ = model prediction using features in $S$

### 2.3 Models Supporting Explanation

#### Models with SHAP Support

| Model | Explainer Type | Speed | Accuracy |
|-------|----------------|-------|----------|
| **RandomForest** | TreeExplainer | ⚡ Fast | ✅ Exact |
| **XGBoost** | TreeExplainer | ⚡ Fast | ✅ Exact |
| **Linear** | LinearExplainer | ⚡⚡ Very Fast | ✅ Exact |
| **VAR** | LinearExplainer | ⚡⚡ Very Fast | ✅ Exact |
| **Moving Average** | KernelExplainer | 🐌 Slow | ⚠️ Approximate |

#### Explainer Selection Logic

**File:** `src/autotsforecast/interpretability/drivers.py`

```python
def calculate_shap_values(self, X, background_samples=None, max_samples=100):
    """
    Automatically selects appropriate SHAP explainer based on model type.
    """
    model = self.model.models[0]  # First horizon model
    
    # 1. TreeExplainer for tree-based models
    if hasattr(model, 'estimators_') or 'XGB' in str(type(model)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    
    # 2. LinearExplainer for linear models
    elif hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    
    # 3. KernelExplainer for other models (model-agnostic)
    else:
        if background_samples is None:
            background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X.iloc[:max_samples])
    
    return shap_values
```

### 2.4 Feature Engineering for Explanation

Before calculating SHAP values, covariate features must be prepared. **Note:** SHAP analysis excludes lag features and only explains the impact of external covariates.

#### Feature Matrix for Model Training

For a multivariate time series with $k$ variables and $n$ lags:

**Without Covariates:**
```
Features = [y1_lag1, y1_lag2, ..., y1_lagn, 
            y2_lag1, y2_lag2, ..., y2_lagn,
            ...,
            yk_lag1, yk_lag2, ..., yk_lagn]

Total features = k × n
```

**With Covariates:**
```
Features = [y1_lag1, ..., y1_lagn,
            y2_lag1, ..., y2_lagn,
            ...,
            yk_lag1, ..., yk_lagn,
            cov1, cov2, ..., covm]

Total features = k × n + m
```

**For SHAP Analysis (Covariates Only):**
```
SHAP Features = [cov1, cov2, ..., covm]

Total SHAP features = m
```

**Example:**
- 3 target variables (North, South, East)
- 7 lags
- 2 covariates (marketing_spend, temperature)
- **Total model features:** 3 × 7 + 2 = **23 features**
- **SHAP analysis features:** **2 features** (marketing_spend, temperature only)

### 2.5 Interpretation Workflow

```
1. Train Model
   ↓
2. Extract Covariate Features
   - Filter out lag features
   - Keep only external covariates
   ↓
3. Select SHAP Explainer
   - TreeExplainer (RF, XGBoost)
   - LinearExplainer (Linear models)
   - KernelExplainer (Others)
   ↓
4. Calculate SHAP Values (Covariates Only)
   - For each prediction
   - For each covariate feature
   ↓
5. Aggregate & Visualize
   - Covariate importance ranking
   - Summary plots
   - Dependence plots
```

### 2.6 Implementation Example

**File:** `src/autotsforecast/interpretability/drivers.py`

```python
class DriverAnalyzer:
    """Model interpretability using SHAP values."""
    
    def __init__(self, model):
        self.model = model
    
    def get_shap_feature_importance(self, shap_values_dict):
        """
        Aggregate SHAP values to get feature importance.
        """
        importance_scores = {}
        
        for target_name, shap_vals in shap_values_dict.items():
            # Mean absolute SHAP value = importance
            importance = np.abs(shap_vals).mean(axis=0)
            importance_scores[target_name] = importance
        
        # Create DataFrame
        df = pd.DataFrame(importance_scores)
        df['feature'] = feature_names
        df['avg_importance'] = df.mean(axis=1)
        
        return df.sort_values('avg_importance', ascending=False)
    
    def plot_shap_summary(self, X, shap_values_dict, target_name, plot_type='dot'):
        """
        Visualize SHAP values.
        
        plot_type:
          - 'dot': Each point is a sample, color = feature value
          - 'bar': Average importance across all samples
          - 'violin': Distribution of SHAP values
        """
        shap.summary_plot(
            shap_values_dict[target_name],
            X,
            plot_type=plot_type,
            show=False
        )
        plt.title(f'SHAP Summary: {target_name}')
        plt.tight_layout()
        plt.show()
```

### 2.7 Interpretation Outputs

#### Feature Importance Ranking

Shows which covariates contribute most to predictions:

```
Covariate               | Avg Importance
------------------------|---------------
marketing_spend        | 12.45
temperature            | 8.32
holiday_indicator      | 5.67
promotion_active       | 4.23
...
```

**Note:** Lag features (e.g., North_lag1, South_lag2) are excluded from this analysis.

**Interpretation:**
- `North_lag1`: Most recent North value is most important
- `marketing_spend`: Marketing has significant impact
- External factors matter more than deeper lags

#### SHAP Summary Plot (Dot)

Each row = covariate feature, each dot = sample:
- **X-axis:** SHAP value (impact on prediction)
- **Color:** Feature value (red=high, blue=low)
- **Pattern:** How feature value affects prediction

**Example Insights:**
- High marketing spend → higher predictions (red dots on right)
- Temperature has non-linear effect on sales
- Holiday indicators show strong positive impact

#### SHAP Summary Plot (Bar)

Simple bar chart of average absolute SHAP values:
- Easy to see which features matter most
- Good for presentations and reports

### 2.8 Use Cases for Explanation

1. **Feature Selection**
   - Identify unimportant features to remove
   - Reduce model complexity
   - Speed up training

2. **Domain Validation**
   - Verify model learns sensible patterns
   - Check if business logic is captured
   - Detect data leakage

3. **Stakeholder Communication**
   - Explain "why" to non-technical users
   - Build trust in model predictions
   - Support decision-making

4. **Model Debugging**
   - Understand poor predictions
   - Identify biases
   - Compare models

---

## 3. Hierarchical Reconciliation

### 3.1 Overview

Hierarchical reconciliation ensures forecasts are **coherent** across aggregation levels. For example:

```
Total Sales = North + South + East
```

Base forecasts may violate this constraint. Reconciliation methods enforce coherence while minimizing information loss.

### 3.2 Methods Implemented

#### Bottom-Up Reconciliation

**Concept:** Aggregate from lowest level

```python
Total_reconciled = North_base + South_base + East_base
```

**Advantages:**
- Simple and intuitive
- Preserves bottom-level forecasts
- No optimization needed

**Disadvantages:**
- Ignores top-level information
- Errors accumulate upward

#### Top-Down Reconciliation

**Concept:** Disaggregate from highest level using proportions

```python
p_north = historical_avg(North / Total)
North_reconciled = Total_base × p_north
```

**Advantages:**
- Uses aggregate-level information
- Good when top level is more predictable

**Disadvantages:**
- Ignores bottom-level patterns
- Fixed proportions may be unrealistic

#### MinTrace (Optimal Reconciliation)

**Concept:** Find optimal weights that minimize forecast error variance

**Mathematical Formulation:**

$$\tilde{y} = S(S'W^{-1}S)^{-1}S'W^{-1}\hat{y}$$

Where:
- $\tilde{y}$ = reconciled forecasts
- $\hat{y}$ = base forecasts
- $S$ = summing matrix (hierarchy structure)
- $W$ = error covariance matrix

**Variants:**

1. **MinTrace-OLS:** $W = I$ (identity)
2. **MinTrace-WLS:** $W = \text{diag}(\hat{\sigma}^2)$ (weighted)
3. **MinTrace-Shrink:** $W = \lambda\Sigma + (1-\lambda)I$ (shrinkage)

**File:** `src/autotsforecast/hierarchical/reconciliation.py`

```python
def reconcile(self, forecasts, method='mint_shrink'):
    """
    Reconcile forecasts to ensure hierarchical coherence.
    """
    S = self._create_summing_matrix()
    base_forecasts = forecasts[self.all_series].values
    
    if method == 'bottom_up':
        # Simple aggregation
        bottom_level = forecasts[self.bottom_series]
        reconciled = S @ bottom_level.T
    
    elif method == 'mint_shrink':
        # Optimal reconciliation with shrinkage
        W = self._estimate_error_covariance(forecasts)
        W_shrink = self.shrinkage * W + (1 - self.shrinkage) * np.eye(len(W))
        
        # MinTrace formula
        W_inv = np.linalg.inv(W_shrink)
        G = np.linalg.inv(S.T @ W_inv @ S) @ S.T @ W_inv
        reconciled = (S @ G @ base_forecasts.T).T
    
    return pd.DataFrame(reconciled, columns=self.all_series)
```

### 3.3 When to Use Each Method

| Method | Use When | Pros | Cons |
|--------|----------|------|------|
| **Bottom-Up** | Bottom level is reliable | Simple, intuitive | Ignores top-level info |
| **Top-Down** | Aggregate is reliable | Uses total forecast | Fixed proportions |
| **MinTrace-OLS** | Equal reliability | Statistically optimal | May be unstable |
| **MinTrace-Shrink** | General case | Best performance | More complex |

---

## 4. Implementation Details

### 4.1 Package Structure

```
src/autotsforecast/
├── __init__.py                    # Package exports
├── forecaster.py                  # AutoForecaster (model selection)
├── models/
│   ├── base.py                   # VAR, Linear, MovingAverage
│   ├── external.py               # RandomForest, XGBoost
│   └── selection.py              # ModelSelector helper
├── backtesting/
│   └── validator.py              # BacktestValidator (CV)
├── hierarchical/
│   └── reconciliation.py         # HierarchicalReconciler
├── interpretability/
│   └── drivers.py                # DriverAnalyzer (SHAP)
└── utils/
    ├── data.py                   # CovariatePreprocessor
    └── preprocessing.py          # Data utilities
```

### 4.2 Key Design Patterns

#### 1. Base Model Interface

All models inherit from `BaseForecaster`:

```python
class BaseForecaster:
    def fit(self, y: pd.DataFrame, X: Optional[pd.DataFrame] = None):
        """Fit the model to data."""
        pass
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts."""
        pass
```

#### 2. Covariate Preprocessing

Automatic handling of categorical/numerical features:

```python
class CovariatePreprocessor:
    def fit_transform(self, X):
        # Separate numerical and categorical
        num_features = X.select_dtypes(include=[np.number])
        cat_features = X.select_dtypes(exclude=[np.number])
        
        # Standardize numerical
        num_scaled = self.scaler.fit_transform(num_features)
        
        # One-hot encode categorical
        cat_encoded = self.encoder.fit_transform(cat_features)
        
        return np.hstack([num_scaled, cat_encoded])
```

#### 3. Recursive Multi-Step Forecasting

For ML models, predictions use a growing lag buffer:

```python
def predict(self, X=None):
    last_values = self.last_values_.copy()
    predictions = []
    
    for h, model in enumerate(self.models):
        # Create features from current lag buffer
        X_pred = self._create_features_for_prediction(
            last_values, X, step=h
        )
        
        # Predict
        pred = model.predict(X_pred)
        predictions.append(pred)
        
        # Update lag buffer with new prediction
        new_row = pd.DataFrame([pred], columns=self.feature_names)
        last_values = pd.concat([last_values, new_row], ignore_index=True)
    
    return pd.DataFrame(predictions, columns=self.feature_names)
```

### 4.3 Performance Considerations

#### Model Selection
- **Time Complexity:** O(M × K × N)
  - M = number of candidate models
  - K = number of CV folds
  - N = dataset size

- **Optimization:** 
  - Parallel evaluation possible (future)
  - Early stopping for poor models
  - Cache preprocessed features

#### SHAP Calculation
- **TreeExplainer:** O(TLD²) 
  - T = trees, L = leaves, D = depth
  - Very fast for RF/XGBoost

- **KernelExplainer:** O(2^F × S)
  - F = features, S = samples
  - Use sampling: `max_samples=100`

#### Memory Usage
- Lag features: O(N × V × L)
  - N = samples, V = variables, L = lags
- SHAP values: O(N × F)
  - Store per sample, per feature

---

## 5. Best Practices

### 5.1 Model Selection

1. **Start with diverse candidates**
   - Include statistical and ML models
   - Try different lag lengths (3, 5, 7)
   - Mix simple and complex models

2. **Choose appropriate metric**
   - RMSE: General purpose
   - MAPE: Scale-independent comparison
   - MAE: Robust to outliers

3. **Use expanding window for small data**
   - More stable with <500 samples
   - Rolling for >1000 samples

### 5.2 Interpretability

1. **Feature engineering matters**
   - Create meaningful lag features
   - Name features descriptively
   - Document covariate sources

2. **Sample wisely**
   - Use representative samples
   - Balance computational cost vs accuracy
   - 100-500 samples usually sufficient

3. **Validate insights**
   - Check if SHAP values match domain knowledge
   - Compare across models
   - Use multiple plot types

### 5.3 Hierarchical Reconciliation

1. **Choose method based on data**
   - MinTrace-Shrink: default choice
   - Bottom-Up: when bottom is reliable
   - Top-Down: when aggregate is stable

2. **Verify coherence**
   - Always check sum constraints
   - Monitor reconciliation impact on accuracy
   - Compare base vs reconciled forecasts

---

## 6. References

### Academic Papers

1. **SHAP Values:**
   - Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.

2. **Hierarchical Reconciliation:**
   - Wickramasuriya et al. (2019). "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization." JASA.

3. **Time Series Cross-Validation:**
   - Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice" (3rd ed).

### Software Libraries

- **SHAP:** https://github.com/slundberg/shap
- **scikit-learn:** https://scikit-learn.org/
- **statsmodels:** https://www.statsmodels.org/
- **XGBoost:** https://xgboost.readthedocs.io/

---

## Summary

**AutoTSForecast** provides a complete pipeline for multivariate time series forecasting:

1. **Model Selection**: Automated evaluation of 9 models using backtesting
2. **Interpretability**: SHAP-based explanation for all model types
3. **Hierarchical Reconciliation**: Optimal methods for forecast coherence

The package emphasizes:
- ✅ Ease of use (AutoForecaster one-liner)
- ✅ Flexibility (multiple models and methods)
- ✅ Transparency (SHAP explanations)
- ✅ Robustness (proper CV and reconciliation)

For usage examples, see [examples/autotsforecast_tutorial.ipynb](examples/autotsforecast_tutorial.ipynb).

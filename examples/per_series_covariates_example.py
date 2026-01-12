"""
Example: Per-Series Covariates in AutoForecaster

This example demonstrates how to use different covariates (external features)
for different time series in AutoTSForecast.
"""

import pandas as pd
import numpy as np
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster, LinearForecaster
from autotsforecast.models.external import RandomForestForecaster

# Set random seed
np.random.seed(42)

print("="*80)
print("Per-Series Covariates Example")
print("="*80)

# ============================================================================
# Scenario: Forecasting sales for two products with different drivers
# ============================================================================

dates = pd.date_range('2024-01-01', periods=150, freq='D')

# Product A: Summer product (driven by temperature and advertising)
y = pd.DataFrame({
    'product_a_sales': 1000 + np.cumsum(np.random.randn(150) * 10),
    'product_b_sales': 500 + np.cumsum(np.random.randn(150) * 5)
}, index=dates)

# Product A is influenced by weather and marketing
X_product_a = pd.DataFrame({
    'temperature': np.random.uniform(60, 95, 150),
    'advertising_spend': np.random.uniform(1000, 5000, 150),
    'day_of_week': dates.dayofweek
}, index=dates)

# Product B is influenced by competitor pricing and promotions
X_product_b = pd.DataFrame({
    'competitor_price': np.random.uniform(20, 50, 150),
    'promotion_active': np.random.choice([0, 1], 150, p=[0.75, 0.25]),
    'inventory_level': np.random.uniform(50, 200, 150)
}, index=dates)

# Split data
y_train = y.iloc[:120]
y_test = y.iloc[120:]

X_a_train = X_product_a.iloc[:120]
X_a_test = X_product_a.iloc[120:134]  # 14 days forecast

X_b_train = X_product_b.iloc[:120]
X_b_test = X_product_b.iloc[120:134]

print("\nDataset Information:")
print(f"  Training period: {len(y_train)} days")
print(f"  Forecast horizon: 14 days")
print(f"\nProduct A covariates: {list(X_a_train.columns)}")
print(f"Product B covariates: {list(X_b_train.columns)}")

# ============================================================================
# Setup per-series covariates
# ============================================================================

# Create dictionary mapping each series to its specific covariates
X_train_dict = {
    'product_a_sales': X_a_train,
    'product_b_sales': X_b_train
}

X_test_dict = {
    'product_a_sales': X_a_test,
    'product_b_sales': X_b_test
}

print("\n" + "="*80)
print("Training AutoForecaster with Per-Series Covariates")
print("="*80)

# Define candidate models
candidates = [
    MovingAverageForecaster(horizon=14, window=7),
    LinearForecaster(horizon=14),
    RandomForestForecaster(horizon=14, n_lags=7, n_estimators=50)
]

# Create AutoForecaster with per-series model selection
auto = AutoForecaster(
    candidate_models=candidates,
    metric='rmse',
    n_splits=3,
    test_size=14,
    per_series_models=True,  # Enable per-series selection
    verbose=True
)

# Fit with per-series covariates
print("\nFitting models...")
auto.fit(y_train, X=X_train_dict)

print("\n" + "="*80)
print("Selected Models:")
print("="*80)
for series_name, model_name in auto.best_model_names_.items():
    print(f"  {series_name:20s} → {model_name}")

# ============================================================================
# Generate Forecasts
# ============================================================================

print("\n" + "="*80)
print("Generating Forecasts")
print("="*80)

forecasts = auto.forecast(X=X_test_dict)

print(f"\nForecast shape: {forecasts.shape}")
print(f"Forecast period: {forecasts.index[0]} to {forecasts.index[-1]}")

print("\nFirst 5 forecast values:")
print(forecasts.head())

# ============================================================================
# Compare with Actual
# ============================================================================

print("\n" + "="*80)
print("Forecast vs Actual")
print("="*80)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

for col in forecasts.columns:
    mae = mean_absolute_error(y_test[col], forecasts[col])
    rmse = np.sqrt(mean_squared_error(y_test[col], forecasts[col]))
    print(f"\n{col}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

# ============================================================================
# Key Benefits
# ============================================================================

print("\n" + "="*80)
print("Key Benefits of Per-Series Covariates")
print("="*80)
print("""
1. Different Features per Series
   - Each product uses only relevant external factors
   - Product A: weather-dependent (temperature)
   - Product B: competition-dependent (competitor price)

2. Better Accuracy
   - Models trained on relevant features only
   - Reduces noise from irrelevant covariates

3. Flexibility
   - Add/remove features per series
   - Different feature engineering per product

4. Interpretability
   - Clear understanding of what drives each series
   - Easier to explain to stakeholders

5. Scalability
   - Handle heterogeneous product portfolios
   - Each series can have unique data sources
""")

print("="*80)
print("Example Complete!")
print("="*80)

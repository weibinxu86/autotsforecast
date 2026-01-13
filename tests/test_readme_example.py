"""Test README per-series covariates example"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

print("Testing README Per-Series Covariates Example")
print("=" * 60)

# Setup dates
np.random.seed(42)
n = 100
dates = pd.date_range('2024-01-01', periods=n, freq='D')

# Product A: Summer product (driven by weather and advertising)
X_product_a = pd.DataFrame({
    'temperature': np.random.uniform(60, 90, n),
    'advertising_spend': np.random.uniform(100, 1000, n)
}, index=dates)

# Product B: Everyday product (driven by pricing and promotions)
X_product_b = pd.DataFrame({
    'competitor_price': np.random.uniform(10, 30, n),
    'promotion_active': np.random.choice([0, 1], n)
}, index=dates)

# Sales data
y = pd.DataFrame({
    'product_a_sales': 100 + 2*X_product_a['temperature'] + 0.1*X_product_a['advertising_spend'] + np.random.normal(0, 10, n),
    'product_b_sales': 500 - 10*X_product_b['competitor_price'] + 50*X_product_b['promotion_active'] + np.random.normal(0, 20, n)
}, index=dates)

# Train/test split
train_end = 80
y_train = y.iloc[:train_end]

X_product_a_train = X_product_a.iloc[:train_end]
X_product_b_train = X_product_b.iloc[:train_end]
X_product_a_test = X_product_a.iloc[train_end:train_end+10]
X_product_b_test = X_product_b.iloc[train_end:train_end+10]

# ---- README Example Code ----
from autotsforecast import AutoForecaster
from autotsforecast.models.base import LinearForecaster
from autotsforecast.models.external import RandomForestForecaster, XGBoostForecaster

# Create dictionary mapping each series to its covariates
X_train_dict = {
    'product_a_sales': X_product_a_train,
    'product_b_sales': X_product_b_train
}

X_test_dict = {
    'product_a_sales': X_product_a_test,
    'product_b_sales': X_product_b_test
}

# Define candidate models
candidates = [
    LinearForecaster(horizon=10),
    RandomForestForecaster(horizon=10, n_lags=7),
]

# AutoForecaster with per-series model selection
auto = AutoForecaster(
    candidate_models=candidates,
    per_series_models=True,  # Select best model for each series
    metric='rmse',
    n_splits=2,
    verbose=False
)

# Fit: Each series uses its own covariates
auto.fit(y_train, X=X_train_dict)

# Forecast: Provide future covariates for each series
forecasts = auto.forecast(X=X_test_dict)

# See which model was selected for each series
print("Best models per series:")
print(auto.best_model_names_)

print(f"\nForecast shape: {forecasts.shape}")
print(f"Columns: {list(forecasts.columns)}")

print("\n" + "=" * 60)
print("README EXAMPLE PASSED!")
print("=" * 60)

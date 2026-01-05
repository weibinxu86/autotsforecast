"""Test to verify ARIMA no longer uses covariates"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r"c:\forecasting\autotsforecast\src")

from autotsforecast.models.external import ARIMAForecaster

# Create simple test data
np.random.seed(42)
n = 100
idx = pd.date_range("2023-01-01", periods=n, freq="D")

y = pd.DataFrame({
    'series': 100 + np.cumsum(np.random.randn(n))
}, index=idx)

X = pd.DataFrame({
    'temp': 20 + np.random.randn(n),
    'promo': np.random.choice([0, 1], n)
}, index=idx)

print("="*80)
print("TESTING ARIMA COVARIATE BUG FIX")
print("="*80)

# Check class attribute
print(f"\n1. ARIMAForecaster.supports_covariates = {ARIMAForecaster.supports_covariates}")
assert ARIMAForecaster.supports_covariates == False, "❌ ARIMA should NOT support covariates!"
print("   ✅ Correctly set to False")

# Test that fit/predict ignore X
print("\n2. Testing fit() ignores covariates...")
model = ARIMAForecaster(horizon=5, order=(1,1,1))
model.fit(y, X)  # X should be ignored
print("   ✅ fit() completed (X ignored)")

print("\n3. Testing predict() ignores covariates...")
pred_with_X = model.predict(X.iloc[-5:])
pred_without_X = model.predict(None)
print(f"   Predictions with X: {pred_with_X['series'].values[:3]}")
print(f"   Predictions without X: {pred_without_X['series'].values[:3]}")

# They should be identical since X is ignored
if np.allclose(pred_with_X['series'].values, pred_without_X['series'].values):
    print("   ✅ Predictions identical - X is properly ignored!")
else:
    print("   ❌ ERROR: Predictions differ - X is still being used!")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - ARIMA BUG IS FIXED!")
print("="*80)
print("\nARIMA now correctly:")
print("  • Ignores exogenous variables (X)")
print("  • Uses only past values of the series")
print("  • Is a pure autoregressive time-series model")
print("  • Won't be confused with ARIMAX in comparisons")

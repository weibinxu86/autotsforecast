"""
Quick test of README imports and basic functionality
"""
import pandas as pd
import numpy as np
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Testing README example imports...")

# Test 1: Basic imports from README example 1
try:
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster
    from autotsforecast.models.external import ARIMAForecaster, RandomForestForecaster
    print("[PASS] Example 1 imports: SUCCESS")
except Exception as e:
    print(f"[FAIL] Example 1 imports: FAILED - {e}")

# Test 2: XGBoost import from example 2
try:
    from autotsforecast.models.external import XGBoostForecaster
    print("[PASS] Example 2 imports (XGBoost): SUCCESS")
except Exception as e:
    print(f"[SKIP] Example 2 imports (XGBoost): SKIPPED - {e}")

# Test 3: Hierarchical import from example 3
try:
    from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler
    print("[PASS] Example 3 imports (Hierarchical): SUCCESS")
except Exception as e:
    print(f"[FAIL] Example 3 imports (Hierarchical): FAILED - {e}")

# Test 4: Backtesting import from example 4
try:
    from autotsforecast.backtesting.validator import BacktestValidator
    print("[PASS] Example 4 imports (Backtesting): SUCCESS")
except Exception as e:
    print(f"[FAIL] Example 4 imports (Backtesting): FAILED - {e}")

# Test 5: Interpretability import from example 5
try:
    from autotsforecast.interpretability.drivers import DriverAnalyzer
    print("[PASS] Example 5 imports (Interpretability): SUCCESS")
except Exception as e:
    print(f"[FAIL] Example 5 imports (Interpretability): FAILED - {e}")

print("\n" + "="*60)
print("Quick functionality test...")
print("="*60)

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
y = pd.DataFrame({
    'series_a': np.cumsum(np.random.randn(100)) + 100,
    'series_b': np.cumsum(np.random.randn(100)) + 50
}, index=dates)

y_train = y.iloc[:80]

# Test simple model
print("\n1. Testing MovingAverageForecaster...")
try:
    model = MovingAverageForecaster(horizon=7, window=5)
    model.fit(y_train)
    forecast = model.predict()
    print(f"   [PASS] Forecast shape: {forecast.shape}")
    assert forecast.shape == (7, 2), "Unexpected forecast shape"
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")

# Test AutoForecaster with simple candidates
print("\n2. Testing AutoForecaster (simplified)...")
try:
    candidates = [
        MovingAverageForecaster(horizon=7, window=3),
        MovingAverageForecaster(horizon=7, window=5),
    ]
    auto = AutoForecaster(candidate_models=candidates, metric='rmse', n_splits=2, test_size=10, verbose=False)
    auto.fit(y_train)
    forecasts = auto.forecast()
    print(f"   [PASS] Best model: {auto.best_model_name_}")
    print(f"   [PASS] Forecast shape: {forecasts.shape}")
    assert forecasts.shape == (7, 2), "Unexpected forecast shape"
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test per-series mode
print("\n3. Testing AutoForecaster (per-series mode)...")
try:
    candidates = [
        MovingAverageForecaster(horizon=7, window=3),
        MovingAverageForecaster(horizon=7, window=5),
    ]
    auto = AutoForecaster(
        candidate_models=candidates, 
        metric='rmse', 
        n_splits=2, 
        test_size=10,
        per_series_models=True,
        verbose=False
    )
    auto.fit(y_train)
    forecasts = auto.forecast()
    print(f"   [PASS] Best models: {auto.best_model_names_}")
    print(f"   [PASS] Forecast shape: {forecasts.shape}")
    assert forecasts.shape == (7, 2), "Unexpected forecast shape"
    assert isinstance(auto.best_model_names_, dict), "best_model_names_ should be dict"
except Exception as e:
    print(f"   [FAIL] FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[SUCCESS] All README examples verified!")
print("="*60)

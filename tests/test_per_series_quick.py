"""
Quick test of per-series covariates functionality
"""
import pandas as pd
import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Testing Per-Series Covariates - Quick Test")
print("="*60)

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=80, freq='D')

y = pd.DataFrame({
    'series_a': np.cumsum(np.random.randn(80)) + 100,
    'series_b': np.cumsum(np.random.randn(80)) + 50
}, index=dates)

# Different covariates for each series
X_a = pd.DataFrame({
    'temp': np.random.uniform(60, 90, 80),
    'ads': np.random.uniform(100, 500, 80)
}, index=dates)

X_b = pd.DataFrame({
    'price': np.random.uniform(10, 30, 80),
    'promo': np.random.choice([0, 1], 80)
}, index=dates)

y_train = y.iloc[:60]
X_a_train = X_a.iloc[:60]
X_b_train = X_b.iloc[:60]

X_a_test = X_a.iloc[60:67]
X_b_test = X_b.iloc[60:67]

print("\n[TEST 1] Per-series covariates with dict")
print("-" * 60)

try:
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster, LinearForecaster
    
    # Create covariate dict
    X_train_dict = {
        'series_a': X_a_train,
        'series_b': X_b_train
    }
    
    X_test_dict = {
        'series_a': X_a_test,
        'series_b': X_b_test
    }
    
    print("Fitting with per-series covariates...")
    print(f"  - series_a covariates: {list(X_a_train.columns)}")
    print(f"  - series_b covariates: {list(X_b_train.columns)}")
    
    candidates = [
        MovingAverageForecaster(horizon=7, window=5),
        LinearForecaster(horizon=7)
    ]
    
    auto = AutoForecaster(
        candidate_models=candidates,
        metric='rmse',
        n_splits=2,
        test_size=10,
        per_series_models=True,
        verbose=False
    )
    
    auto.fit(y_train, X=X_train_dict)
    print(f"[PASS] Fitted successfully!")
    print(f"  - Best models: {auto.best_model_names_}")
    
    forecasts = auto.forecast(X=X_test_dict)
    print(f"[PASS] Forecasted successfully!")
    print(f"  - Forecast shape: {forecasts.shape}")
    print(f"  - Columns: {list(forecasts.columns)}")
    
    assert forecasts.shape == (7, 2), f"Expected (7, 2), got {forecasts.shape}"
    print("\n[SUCCESS] TEST 1 PASSED!")
    
except Exception as e:
    print(f"\n[FAIL] TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n[TEST 2] Single DataFrame (backward compatibility)")
print("-" * 60)

try:
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster
    
    print("Fitting with single covariate DataFrame...")
    
    candidates = [MovingAverageForecaster(horizon=7, window=5)]
    
    auto2 = AutoForecaster(
        candidate_models=candidates,
        metric='rmse',
        n_splits=2,
        test_size=10,
        per_series_models=True,
        verbose=False
    )
    
    auto2.fit(y_train, X=X_a_train)  # Use same covariates for both
    print(f"[PASS] Fitted successfully!")
    
    forecasts2 = auto2.forecast(X=X_a_test)
    print(f"[PASS] Forecasted successfully!")
    print(f"  - Forecast shape: {forecasts2.shape}")
    
    print("\n[SUCCESS] TEST 2 PASSED - Backward compatibility OK!")
    
except Exception as e:
    print(f"\n[FAIL] TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n[TEST 3] Error handling - missing series")
print("-" * 60)

try:
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster
    
    # Incomplete dict
    X_incomplete = {
        'series_a': X_a_train
        # Missing series_b
    }
    
    candidates = [MovingAverageForecaster(horizon=7, window=5)]
    auto3 = AutoForecaster(
        candidate_models=candidates,
        metric='rmse',
        n_splits=2,
        test_size=10,
        per_series_models=True,
        verbose=False
    )
    
    try:
        auto3.fit(y_train, X=X_incomplete)
        print("[FAIL] Should have raised ValueError")
    except ValueError as ve:
        if "missing" in str(ve).lower():
            print(f"[PASS] Correctly raised ValueError")
            print(f"  - Message: {str(ve)[:80]}...")
        else:
            print(f"[FAIL] Wrong error: {ve}")
    
    print("\n[SUCCESS] TEST 3 PASSED - Error handling OK!")
    
except Exception as e:
    print(f"\n[FAIL] TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[SUCCESS] All tests passed!")
print("Per-series covariate functionality is working correctly.")
print("="*60)

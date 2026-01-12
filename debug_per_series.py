"""
Debug per-series covariates - check individual model behavior
"""
import pandas as pd
import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Debugging Per-Series Covariates")
print("="*60)

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=80, freq='D')

y = pd.DataFrame({
    'series_a': np.cumsum(np.random.randn(80)) + 100
}, index=dates)

X_a = pd.DataFrame({
    'temp': np.random.uniform(60, 90, 80),
    'ads': np.random.uniform(100, 500, 80)
}, index=dates)

y_train = y.iloc[:60]
X_a_train = X_a.iloc[:60]
X_a_test = X_a.iloc[60:67]

print("\nTest data shapes:")
print(f"  y_train: {y_train.shape}")
print(f"  X_a_train: {X_a_train.shape}")
print(f"  X_a_test: {X_a_test.shape}")

# Test 1: Direct Linear Forecaster
print("\n[TEST 1] LinearForecaster directly")
print("-" * 60)
try:
    from autotsforecast.models.base import LinearForecaster
    
    model = LinearForecaster(horizon=7)
    model.fit(y_train, X=X_a_train)
    print("[PASS] Fitted")
    
    forecast = model.predict(X=X_a_test)
    print(f"[PASS] Predicted: {forecast.shape}")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Direct RandomForest
print("\n[TEST 2] RandomForestForecaster directly")
print("-" * 60)
try:
    from autotsforecast.models.external import RandomForestForecaster
    
    model = RandomForestForecaster(horizon=7, n_lags=7)
    model.fit(y_train, X=X_a_train)
    print("[PASS] Fitted")
    
    forecast = model.predict(X=X_a_test)
    print(f"[PASS] Predicted: {forecast.shape}")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 3: BacktestValidator with LinearForecaster
print("\n[TEST 3] BacktestValidator with LinearForecaster")
print("-" * 60)
try:
    from autotsforecast.backtesting.validator import BacktestValidator
    from autotsforecast.models.base import LinearForecaster
    
    model = LinearForecaster(horizon=7)
    validator = BacktestValidator(
        model=model,
        n_splits=2,
        test_size=10,
        window_type='expanding'
    )
    
    results = validator.run(y_train, X=X_a_train)
    print(f"[PASS] Backtesting completed")
    print(f"  RMSE: {results['rmse']:.4f}")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 4: AutoForecaster with per-series covariates (single series)
print("\n[TEST 4] AutoForecaster with per-series covariates (single series)")
print("-" * 60)
try:
    from autotsforecast import AutoForecaster
    from autotsforecast.models.base import MovingAverageForecaster, LinearForecaster
    
    X_dict = {'series_a': X_a_train}
    X_test_dict = {'series_a': X_a_test}
    
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
        verbose=True,  # Enable verbose to see what's happening
        n_jobs=1  # Use single job to avoid parallel issues
    )
    
    auto.fit(y_train, X=X_dict)
    print(f"[PASS] Fitted: {auto.best_model_names_}")
    
    forecast = auto.forecast(X=X_test_dict)
    print(f"[PASS] Forecasted: {forecast.shape}")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug complete!")
print("="*60)

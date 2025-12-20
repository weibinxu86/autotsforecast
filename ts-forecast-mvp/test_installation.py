"""
Quick test to verify autotsforecast installation and basic functionality
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("AutoTSForecast Installation Test")
print("=" * 60)

# Test 1: Import main components
print("\n1. Testing imports...")
try:
    from autotsforecast import (
        AutoForecaster,
        MovingAverageForecaster,
        VARForecaster,
        RandomForestForecaster,
        XGBoostForecaster
    )
    print("   ✓ Core models imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create sample data
print("\n2. Creating sample multivariate data...")
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'North': 100 + np.cumsum(np.random.randn(100)),
    'South': 120 + np.cumsum(np.random.randn(100)),
    'East': 90 + np.cumsum(np.random.randn(100))
}, index=dates)
print(f"   ✓ Created data with shape: {df.shape}")

# Test 3: Fit a simple model
print("\n3. Testing MovingAverageForecaster...")
try:
    model = MovingAverageForecaster(horizon=10, window=7)
    model.fit(df.iloc[:80])
    predictions = model.predict()
    print(f"   ✓ Model fitted and predicted. Forecast shape: {predictions.shape}")
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    exit(1)

# Test 4: Test AutoForecaster
print("\n4. Testing AutoForecaster...")
try:
    candidate_models = [
        MovingAverageForecaster(horizon=10, window=5),
        VARForecaster(horizon=10, lags=5)
    ]
    auto = AutoForecaster(
        candidate_models=candidate_models,
        metric='rmse',
        n_splits=2,
        test_size=5,
        verbose=False
    )
    auto.fit(df.iloc[:80])
    forecasts = auto.forecast()
    print(f"   ✓ AutoForecaster worked. Best model: {auto.best_model_name_}")
except Exception as e:
    print(f"   ✗ AutoForecaster test failed: {e}")
    exit(1)

# Test 5: Test with covariates
print("\n5. Testing with covariates...")
try:
    X = pd.DataFrame({
        'temperature': 20 + np.random.randn(100) * 5,
        'is_weekend': np.random.choice([0, 1], 100)
    }, index=dates)
    
    model_rf = RandomForestForecaster(horizon=10)
    model_rf.fit(df.iloc[:80], X.iloc[:80])
    pred_rf = model_rf.predict(X.iloc[80:90])
    print(f"   ✓ RandomForest with covariates. Forecast shape: {pred_rf.shape}")
except Exception as e:
    print(f"   ✗ Covariate test failed: {e}")
    exit(1)

# Test 6: Check optional features
print("\n6. Checking optional features...")
try:
    from autotsforecast.hierarchical import HierarchicalReconciler
    print("   ✓ Hierarchical reconciliation available")
except ImportError:
    print("   ⚠ Hierarchical reconciliation not available")

try:
    from autotsforecast.interpretability import DriverAnalyzer
    print("   ✓ Interpretability (DriverAnalyzer) available")
except ImportError:
    print("   ⚠ Interpretability not available")

try:
    import shap
    print("   ✓ SHAP installed for advanced interpretability")
except ImportError:
    print("   ⚠ SHAP not installed (optional)")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - AutoTSForecast is ready to use!")
print("=" * 60)
print("\nNext steps:")
print("  - Open examples/forecasting_tutorial.ipynb for full tutorial")
print("  - See QUICKSTART.md for quick examples")
print("  - Check CHANGELOG.md for new features")

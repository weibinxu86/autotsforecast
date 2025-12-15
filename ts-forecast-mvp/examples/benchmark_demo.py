"""
Simple Benchmarking Demo for TS-Forecast Package

This script demonstrates the benchmarking capabilities with synthetic data.
Run after installing: pip install -e .
"""

print("="*70)
print("  TS-FORECAST PACKAGE BENCHMARKING DEMO")
print("="*70)
print("\nNote: This is a demonstration script.")
print("To run full benchmarks, install dependencies:")
print("  pip install -r requirements.txt")
print("\nThen run: python examples/benchmark.py")
print("="*70)

# Check if dependencies are installed
try:
    import numpy as np
    import pandas as pd
    from ts_forecast import VARForecaster, ModelSelector, BacktestValidator
    print("\n✅ All dependencies installed!")
    print("\nRunning quick benchmark demo...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    data = pd.DataFrame({
        'sales_A': np.random.randn(200).cumsum() + 100,
        'sales_B': np.random.randn(200).cumsum() + 150,
        'sales_C': np.random.randn(200).cumsum() + 120
    }, index=dates)
    
    print(f"\n📊 Sample Data: {data.shape[0]} observations, {data.shape[1]} series")
    
    # Quick model test
    print("\n1️⃣  Testing VAR Model...")
    model = VARForecaster(horizon=1, lags=3)
    train = data.iloc[:160]
    test = data.iloc[160:]
    model.fit(train)
    pred = model.predict()
    print(f"   ✓ Prediction shape: {pred.shape}")
    
    # Quick model selection
    print("\n2️⃣  Testing Model Selector...")
    selector = ModelSelector(metric='rmse')
    selector.fit(data.iloc[:180], cv_folds=2)
    best_name, _ = selector.get_best_model()
    print(f"   ✓ Best model: {best_name}")
    
    # Quick backtesting
    print("\n3️⃣  Testing Backtesting...")
    validator = BacktestValidator(model, n_splits=3, test_size=15)
    metrics = validator.run(data)
    print(f"   ✓ RMSE: {metrics['rmse']:.4f}")
    print(f"   ✓ MAE: {metrics['mae']:.4f}")
    print(f"   ✓ R²: {metrics['r2']:.4f}")
    
    print("\n" + "="*70)
    print("✨ DEMO COMPLETE!")
    print("="*70)
    print("\nFor full benchmarking results, run:")
    print("  python examples/benchmark.py")
    print("\nThis will compare against:")
    print("  • Naive models (persistence, seasonal)")
    print("  • Statistical models (VAR with different lags)")
    print("  • Multiple real-world datasets")
    print("="*70 + "\n")
    
except ImportError as e:
    print(f"\n❌ Missing dependencies: {e}")
    print("\nPlease install the package first:")
    print("\n  cd ts-forecast-mvp")
    print("  pip install -r requirements.txt")
    print("  pip install -e .")
    print("\nThen run:")
    print("  python examples/benchmark_demo.py")
    print("  python examples/benchmark.py  # Full benchmarks")
    print("\n" + "="*70 + "\n")

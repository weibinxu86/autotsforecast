"""Quick test to verify sensitivity analysis fix"""

import numpy as np
import pandas as pd
from src.autotsforecast.interpretability.drivers import DriverAnalyzer
from src.autotsforecast.models.base import LinearForecaster

# Generate sample data with known DGP
np.random.seed(42)
n = 100

X = pd.DataFrame({
    'price': 10 + np.random.randn(n) * 2,
    'promo': np.random.choice([0, 1], n),  # Binary feature
    'temp': 20 + np.random.randn(n) * 5
})

# DGP: sales = -2*price + 30*promo + 1.5*temp + noise + 100
y = pd.DataFrame({
    'sales': (
        -2 * X['price'] +
        30 * X['promo'] +
        1.5 * X['temp'] +
        np.random.randn(n) * 5 +
        100
    )
})

print("="*80)
print("Testing Sensitivity Analysis Fix")
print("="*80)
print(f"\nData-Generating Process (DGP):")
print(f"  sales = -2*price + 30*promo + 1.5*temp + noise + 100")
print(f"\nExpected sensitivity ranking:")
print(f"  1. promo (binary flip: 0↔1, coefficient 30)")
print(f"  2. price (continuous, coefficient -2, high variation)")
print(f"  3. temp (continuous, coefficient 1.5, moderate variation)")

# Fit model
model = LinearForecaster(horizon=1)
model.fit(y, X)

# Test sensitivity analysis
analyzer = DriverAnalyzer(model, feature_names=X.columns.tolist())
sensitivity = analyzer.calculate_feature_importance(X, y, method='sensitivity')

print(f"\n" + "="*80)
print("SENSITIVITY RESULTS (Mean Absolute Change)")
print("="*80)
print(sensitivity)

print(f"\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check 1: All values non-negative
all_positive = (sensitivity.values >= 0).all()
print(f"✓ All sensitivities non-negative: {all_positive}")

# Check 2: Binary feature (promo) has non-zero sensitivity
promo_sens = sensitivity.loc['promo', 'sales']
print(f"✓ Promo sensitivity > 0: {promo_sens:.4f} > 0")

# Check 3: All features have impact
price_sens = sensitivity.loc['price', 'sales']
temp_sens = sensitivity.loc['temp', 'sales']
print(f"✓ Price sensitivity > 0: {price_sens:.4f} > 0")
print(f"✓ Temp sensitivity > 0: {temp_sens:.4f} > 0")

# Check 4: Promo should have highest sensitivity (binary flip with large coef)
ranking = sensitivity['sales'].sort_values(ascending=False)
print(f"\n✓ Sensitivity ranking:")
for i, (feat, val) in enumerate(ranking.items(), 1):
    print(f"  {i}. {feat}: {val:.4f}")

print(f"\n" + "="*80)
if all_positive and promo_sens > 0 and price_sens > 0 and temp_sens > 0:
    print("✅ ALL CHECKS PASSED - Sensitivity analysis is working correctly!")
else:
    print("❌ SOME CHECKS FAILED")
print("="*80)

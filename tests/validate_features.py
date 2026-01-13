"""Quick validation of all v0.3.0 features"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

print("AutoTSForecast v0.3.0 Feature Validation")
print("=" * 60)

np.random.seed(42)
n, h = 100, 5
dates = pd.date_range('2023-01-01', periods=n, freq='D')

# Simple test data
y = pd.DataFrame({
    'series_a': 100 + np.cumsum(np.random.randn(n)),
    'series_b': 50 + np.cumsum(np.random.randn(n))
}, index=dates)

X = pd.DataFrame({
    'temp': np.random.uniform(60, 90, n),
    'promo': np.random.choice([0, 1], n)
}, index=dates)

y_train, y_test = y.iloc[:-h], y.iloc[-h:]
X_train, X_test = X.iloc[:-h], X.iloc[-h:]

# TEST 1: Per-Series Covariates
print("\n[1] Per-Series Covariates...")
from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster

per_series_X = {
    'series_a': X_train[['temp']],
    'series_b': X_train[['promo']]
}
per_series_X_test = {
    'series_a': X_test[['temp']],
    'series_b': X_test[['promo']]
}

auto = AutoForecaster(
    candidate_models=[MovingAverageForecaster(horizon=h, window=3)],
    per_series_models=True, n_splits=2, verbose=False
)
auto.fit(y_train, X=per_series_X)
fcst = auto.forecast(X=per_series_X_test)
assert fcst.shape == (h, 2), f"Bad shape: {fcst.shape}"
print("    PASS - Per-series covariates work")

# TEST 2: Prediction Intervals
print("[2] Prediction Intervals...")
from autotsforecast.uncertainty.intervals import PredictionIntervals

model = MovingAverageForecaster(horizon=h, window=3)
model.fit(y_train[['series_a']])
preds = model.predict()

pi = PredictionIntervals(method='conformal', coverage=[0.90])
pi.fit(model, y_train[['series_a']])
intervals = pi.predict(preds)
assert 'lower_90' in intervals and 'upper_90' in intervals
print("    PASS - Prediction intervals work")

# TEST 3: Calendar Features
print("[3] Calendar Features...")
from autotsforecast.features.calendar import CalendarFeatures

cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)
assert features.shape[0] == len(y_train)
print(f"    PASS - Calendar features work ({features.shape[1]} features)")

# TEST 4: Visualization
print("[4] Visualization...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from autotsforecast.visualization.plots import plot_forecast, plot_model_comparison

fig = plot_forecast(y_train=y_train[['series_a']], forecasts=preds, title='Test')
plt.close(fig)
fig2 = plot_model_comparison({'M1': {'rmse': 10}, 'M2': {'rmse': 12}}, metric='rmse')
plt.close(fig2)
print("    PASS - Visualization works")

# TEST 5: Parallel Processing
print("[5] Parallel Processing...")
from autotsforecast.utils.parallel import ParallelForecaster

pf = ParallelForecaster(n_jobs=2, verbose=False)
fitted = pf.parallel_series_fit(
    model_factory=lambda: MovingAverageForecaster(horizon=h, window=3),
    y=y_train
)
assert len(fitted) == 2
print("    PASS - Parallel processing works")

# TEST 6: Hierarchical Reconciliation
print("[6] Hierarchical Reconciliation...")
from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

base_fc = pd.DataFrame({
    'a': [10, 11, 12, 13, 14],
    'b': [20, 21, 22, 23, 24],
    'total': [28, 30, 32, 34, 36]  # incoherent
})
hierarchy = {'total': ['a', 'b']}
reconciler = HierarchicalReconciler(forecasts=base_fc, hierarchy=hierarchy)
reconciler.reconcile(method='ols')
rec = reconciler.reconciled_forecasts
# Check coherence
diff = abs(rec['total'].iloc[0] - (rec['a'].iloc[0] + rec['b'].iloc[0]))
assert diff < 0.01, f"Incoherent: {diff}"
print("    PASS - Hierarchical reconciliation works")

# TEST 7: Interpretability
print("[7] Interpretability...")
from autotsforecast.interpretability.drivers import DriverAnalyzer
from autotsforecast.models.base import LinearForecaster

lf = LinearForecaster(horizon=h)
lf.fit(y_train[['series_a']], X=X_train)
analyzer = DriverAnalyzer(model=lf, feature_names=['temp', 'promo'])
importance = analyzer.calculate_feature_importance(X_test, y_test[['series_a']], method='sensitivity')
assert len(importance) == 2
print("    PASS - Interpretability works")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED - v0.3.0 features validated!")
print("=" * 60)

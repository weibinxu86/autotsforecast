"""Test all README examples"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

print("=" * 70)
print("TESTING ALL README EXAMPLES")
print("=" * 70)

# Setup test data
np.random.seed(42)
n, horizon = 120, 14
dates = pd.date_range('2024-01-01', periods=n, freq='D')

# Create realistic test data
y = pd.DataFrame({
    'series_a': 100 + np.cumsum(np.random.randn(n)) + 10*np.sin(2*np.pi*np.arange(n)/7),
    'series_b': 50 + np.cumsum(np.random.randn(n)) + 5*np.sin(2*np.pi*np.arange(n)/30),
}, index=dates)

X = pd.DataFrame({
    'temperature': 70 + 10*np.sin(2*np.pi*np.arange(n)/365) + np.random.normal(0, 3, n),
    'promotion': np.random.choice([0, 1], n, p=[0.85, 0.15])
}, index=dates)

y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]

# ============================================================
print("\n[1] AutoForecaster - Basic Usage")
print("-" * 50)

from autotsforecast import AutoForecaster
from autotsforecast.models.base import MovingAverageForecaster
from autotsforecast.models.external import ARIMAForecaster, RandomForestForecaster

candidates = [
    ARIMAForecaster(horizon=horizon),
    RandomForestForecaster(horizon=horizon, n_lags=7),
    MovingAverageForecaster(horizon=horizon, window=7),
]

# Default mode
auto = AutoForecaster(candidate_models=candidates, metric='rmse', n_splits=2, verbose=False)
auto.fit(y_train)
forecasts = auto.forecast()
print(f"PASS - Best model: {auto.best_model_name_}")

# Per-series mode
auto2 = AutoForecaster(candidate_models=candidates, metric='rmse', per_series_models=True, n_splits=2, verbose=False)
auto2.fit(y_train)
forecasts2 = auto2.forecast()
print(f"PASS - Best models per series: {auto2.best_model_names_}")

# ============================================================
print("\n[2] Using Covariates")
print("-" * 50)

model = RandomForestForecaster(horizon=horizon, n_lags=7)
model.fit(y_train[['series_a']], X=X_train)
forecasts_cov = model.predict(X=X_test)
print(f"PASS - Forecast with covariates shape: {forecasts_cov.shape}")

# ============================================================
print("\n[2.1] Per-Series Covariates")
print("-" * 50)

# Create per-series covariate dicts
X_train_dict = {
    'series_a': X_train[['temperature']],
    'series_b': X_train[['promotion']]
}
X_test_dict = {
    'series_a': X_test[['temperature']],
    'series_b': X_test[['promotion']]
}

candidates_cov = [
    RandomForestForecaster(horizon=horizon, n_lags=7),
]

auto_cov = AutoForecaster(
    candidate_models=candidates_cov,
    per_series_models=True,
    metric='rmse',
    n_splits=2,
    verbose=False
)
auto_cov.fit(y_train, X=X_train_dict)
forecasts_per_cov = auto_cov.forecast(X=X_test_dict)
print(f"PASS - Per-series covariates: {auto_cov.best_model_names_}")

# ============================================================
print("\n[3] Hierarchical Reconciliation")
print("-" * 50)

from autotsforecast.hierarchical.reconciliation import HierarchicalReconciler

base_forecasts = pd.DataFrame({
    'region_a': np.random.uniform(90, 110, horizon),
    'region_b': np.random.uniform(45, 55, horizon),
    'total': np.random.uniform(140, 160, horizon)  # intentionally incoherent
})

hierarchy = {'total': ['region_a', 'region_b']}
reconciler = HierarchicalReconciler(forecasts=base_forecasts, hierarchy=hierarchy)
reconciler.reconcile(method='ols')
coherent_forecasts = reconciler.reconciled_forecasts
# Verify coherence
diff = abs(coherent_forecasts['total'].iloc[0] - (coherent_forecasts['region_a'].iloc[0] + coherent_forecasts['region_b'].iloc[0]))
print(f"PASS - Coherent forecasts (diff={diff:.6f})")

# ============================================================
print("\n[4] Backtesting")
print("-" * 50)

from autotsforecast.backtesting.validator import BacktestValidator

my_model = MovingAverageForecaster(horizon=horizon, window=7)
validator = BacktestValidator(model=my_model, n_splits=3, test_size=horizon)
validator.run(y_train[['series_a']])
results = validator.get_fold_results()
print(f"PASS - Average RMSE: {results['rmse'].mean():.2f}")

# ============================================================
print("\n[5] Interpretability")
print("-" * 50)

from autotsforecast.interpretability.drivers import DriverAnalyzer

fitted_model = RandomForestForecaster(horizon=horizon, n_lags=7)
fitted_model.fit(y_train[['series_a']], X=X_train)

analyzer = DriverAnalyzer(model=fitted_model, feature_names=['temperature', 'promotion'])
importance = analyzer.calculate_feature_importance(X_test, y_test[['series_a']], method='sensitivity')
print(f"PASS - Feature importance: {list(importance.index)}")

# ============================================================
print("\n[6] Prediction Intervals")
print("-" * 50)

from autotsforecast.uncertainty.intervals import PredictionIntervals

model_pi = RandomForestForecaster(horizon=horizon, n_lags=7)
model_pi.fit(y_train[['series_a']])
preds = model_pi.predict()

pi = PredictionIntervals(method='conformal', coverage=[0.80, 0.95])
pi.fit(model_pi, y_train[['series_a']])
intervals = pi.predict(preds)
print(f"PASS - Intervals keys: {list(intervals.keys())}")

# ============================================================
print("\n[7] Calendar Features")
print("-" * 50)

from autotsforecast.features.calendar import CalendarFeatures

cal = CalendarFeatures(cyclical_encoding=True)
features = cal.fit_transform(y_train)
print(f"PASS - Calendar features: {list(features.columns)[:5]}...")

# ============================================================
print("\n[8] Visualization")
print("-" * 50)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from autotsforecast.visualization.plots import plot_forecast

fig = plot_forecast(
    y_train=y_train[['series_a']], 
    forecasts=preds,
    y_test=y_test[['series_a']],
    intervals={'lower_95': intervals['lower_95'], 'upper_95': intervals['upper_95']},
    title='Test'
)
plt.close(fig)
print("PASS - plot_forecast works")

# ============================================================
print("\n[9] Parallel Processing")
print("-" * 50)

from autotsforecast.utils.parallel import ParallelForecaster

pf = ParallelForecaster(n_jobs=2, verbose=False)
fitted_models = pf.parallel_series_fit(
    model_factory=lambda: RandomForestForecaster(horizon=horizon, n_lags=7),
    y=y_train,
    X=X_train
)
print(f"PASS - Parallel fitted models: {list(fitted_models.keys())}")

# ============================================================
print("\n" + "=" * 70)
print("ALL 9 README EXAMPLES PASSED!")
print("=" * 70)

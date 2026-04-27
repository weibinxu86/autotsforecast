"""Tests for AnomalyDetector (anomaly/detector.py)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_series():
    """Daily series with no anomalies."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {"series_a": 100 + rng.normal(0, 2, 120),
         "series_b": 50 + rng.normal(0, 1, 120)},
        index=dates,
    )
    return data


@pytest.fixture
def series_with_anomalies():
    """Series with known spike anomalies."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    values = 100 + rng.normal(0, 2, 120)
    # Inject obvious spikes
    values[30] = 999
    values[60] = -999
    values[90] = 888
    data = pd.DataFrame({"series_a": values}, index=dates)
    return data


class TestAnomalyDetectorImport:
    def test_import(self):
        from autotsforecast.anomaly.detector import AnomalyDetector
        assert AnomalyDetector is not None


class TestZScore:
    def test_fit_predict_clean(self, clean_series):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="zscore", threshold=3.0)
        mask = detector.fit_predict(clean_series)
        assert isinstance(mask, pd.DataFrame)
        assert mask.shape == clean_series.shape
        assert mask.dtypes.iloc[0] == bool

    def test_detects_spikes(self, series_with_anomalies):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="zscore", threshold=3.0)
        mask = detector.fit_predict(series_with_anomalies)
        # Known spikes at indices 30, 60, 90 should all be flagged
        assert mask.iloc[30, 0]
        assert mask.iloc[60, 0]
        assert mask.iloc[90, 0]

    def test_get_summary(self, series_with_anomalies):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="zscore", threshold=3.0)
        detector.fit_predict(series_with_anomalies)
        summary = detector.get_summary()
        assert summary is not None
        assert hasattr(summary, "method") or isinstance(summary, dict)


class TestIQR:
    def test_fit_predict(self, series_with_anomalies):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="iqr")
        mask = detector.fit_predict(series_with_anomalies)
        assert isinstance(mask, pd.DataFrame)
        # Should catch at least the extreme spikes
        assert mask["series_a"].sum() >= 2


class TestIsolationForest:
    def test_fit_predict(self, clean_series):
        from autotsforecast.anomaly.detector import AnomalyDetector
        try:
            detector = AnomalyDetector(method="isolation_forest", contamination=0.05)
            mask = detector.fit_predict(clean_series)
            assert isinstance(mask, pd.DataFrame)
        except ImportError:
            pytest.skip("sklearn not installed for isolation_forest")


class TestForecastResidual:
    def test_fit_predict(self, series_with_anomalies):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="forecast_residual", threshold=3.0)
        mask = detector.fit_predict(series_with_anomalies)
        assert isinstance(mask, pd.DataFrame)


class TestFitSeparately:
    def test_fit_then_predict(self, series_with_anomalies):
        from autotsforecast.anomaly.detector import AnomalyDetector
        detector = AnomalyDetector(method="zscore", threshold=3.0)
        detector.fit(series_with_anomalies)
        mask = detector.predict(series_with_anomalies)
        assert isinstance(mask, pd.DataFrame)

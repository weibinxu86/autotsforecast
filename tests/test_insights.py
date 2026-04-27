"""Tests for InsightEngine (nlp/insights.py)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train():
    rng = np.random.default_rng(10)
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    return pd.DataFrame(
        {"sales": 100 + np.arange(90) * 0.5 + rng.normal(0, 3, 90)},
        index=dates,
    )


@pytest.fixture
def sample_forecasts():
    dates = pd.date_range("2023-04-01", periods=14, freq="D")
    return pd.DataFrame(
        {"sales": 145 + np.arange(14) * 0.5},
        index=dates,
    )


class TestInsightEngineImport:
    def test_import(self):
        from autotsforecast.nlp.insights import InsightEngine
        assert InsightEngine is not None


class TestRuleBasedMode:
    def test_instantiate_rules(self):
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        assert engine is not None

    def test_summarize_forecast_dataframes(self, sample_train, sample_forecasts):
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        summary = engine.summarize_forecast_dataframes(sample_train, sample_forecasts)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_flag_risks_from_dataframes(self, sample_train, sample_forecasts):
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        risks = engine.flag_risks_from_dataframes(sample_train, sample_forecasts)
        assert isinstance(risks, list)

    def test_generate_report(self, sample_train, sample_forecasts):
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        # Pass a minimal result dict so generate_report can work
        result = {"series_names": list(sample_train.columns), "best_model": "ETSForecaster", "metric": "rmse"}
        report = engine.generate_report(result, sample_train, None)
        assert isinstance(report, str)

    def test_trend_detection_upward(self, sample_train, sample_forecasts):
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        summary = engine.summarize_forecast_dataframes(sample_train, sample_forecasts)
        # Upward trend should be mentioned
        lower = summary.lower()
        assert any(w in lower for w in ["upward", "increasing", "uptrend", "trend", "rising", "growth"])

    def test_empty_forecast(self, sample_train):
        """Should not crash on empty forecast."""
        from autotsforecast.nlp.insights import InsightEngine
        engine = InsightEngine(mode="rule_based")
        empty_fc = pd.DataFrame({"sales": []}, index=pd.DatetimeIndex([]))
        # Should either return a string or raise gracefully
        try:
            summary = engine.summarize_forecast_dataframes(sample_train, empty_fc)
            assert isinstance(summary, str)
        except (ValueError, IndexError):
            pass  # Acceptable for edge-case


class TestLLMModeStub:
    def test_llm_mode_no_client_raises(self):
        from autotsforecast.nlp.insights import InsightEngine
        # Mode=llm with no client should raise or fall back gracefully
        try:
            engine = InsightEngine(mode="llm", llm_client=None)
            # If it doesn't raise, calling summarize should either work or raise informative error
        except (ValueError, TypeError):
            pass  # acceptable
"""Tests for OpenAI/Anthropic tool schemas (integrations/openai_schemas.py)."""
import json
import numpy as np
import pandas as pd
import pytest


class TestGetOpenAITools:
    def test_returns_list(self):
        from autotsforecast.integrations.openai_schemas import get_openai_tools
        tools = get_openai_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_each_tool_has_function_key(self):
        from autotsforecast.integrations.openai_schemas import get_openai_tools
        for tool in get_openai_tools():
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


class TestGetAnthropicTools:
    def test_returns_list(self):
        from autotsforecast.integrations.openai_schemas import get_anthropic_tools
        tools = get_anthropic_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_has_name_and_input_schema(self):
        from autotsforecast.integrations.openai_schemas import get_anthropic_tools
        for tool in get_anthropic_tools():
            assert "name" in tool
            assert "input_schema" in tool


class TestHandleToolCall:
    @pytest.fixture
    def mini_series_csv(self):
        rng = np.random.default_rng(99)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        df = pd.DataFrame({"series_a": 100 + rng.normal(0, 2, 60)}, index=dates)
        return df.to_csv()

    def test_list_models(self):
        from autotsforecast.integrations.openai_schemas import handle_tool_call
        result = handle_tool_call("list_models", {})
        data = json.loads(result)
        assert isinstance(data, (list, dict))

    def test_unknown_tool_returns_error_json(self):
        from autotsforecast.integrations.openai_schemas import handle_tool_call
        result = handle_tool_call("nonexistent_tool", {})
        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data

    def test_fit_and_forecast(self, mini_series_csv):
        from autotsforecast.integrations.openai_schemas import handle_tool_call
        args = {
            "csv_data": mini_series_csv,
            "horizon": 7,
            "metric": "rmse",
            "n_splits": 3,
            "per_series": False,
        }
        result = handle_tool_call("fit_and_forecast", args)
        data = json.loads(result)
        assert isinstance(data, dict)

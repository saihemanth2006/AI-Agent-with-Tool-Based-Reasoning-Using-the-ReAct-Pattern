"""
Tests for the ReAct Agent tools and agent loop.

Run with: python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    get_weather,
    search_web,
    calculate,
    run_python,
    read_file,
    execute_tool,
    TOOL_DEFINITIONS,
    TOOL_REGISTRY,
)


# =========================================================================
# Tool Definition Tests
# =========================================================================

class TestToolDefinitions:
    """Tests for tool schema definitions."""

    def test_at_least_three_tools_defined(self):
        """Verify at least three tool schemas are defined."""
        assert len(TOOL_DEFINITIONS) >= 3, (
            f"Expected at least 3 tool definitions, found {len(TOOL_DEFINITIONS)}"
        )

    def test_at_least_three_tool_functions_registered(self):
        """Verify at least three tool functions are registered."""
        assert len(TOOL_REGISTRY) >= 3, (
            f"Expected at least 3 registered tool functions, found {len(TOOL_REGISTRY)}"
        )

    def test_each_schema_has_required_fields(self):
        """Each schema must have name, description, and parameters."""
        for tool_def in TOOL_DEFINITIONS:
            assert "name" in tool_def, f"Tool schema missing 'name': {tool_def}"
            assert "description" in tool_def, f"Tool '{tool_def.get('name')}' missing 'description'"
            assert "parameters" in tool_def, f"Tool '{tool_def.get('name')}' missing 'parameters'"

    def test_parameters_have_properties_and_required(self):
        """Each schema's parameters must have properties and required fields."""
        for tool_def in TOOL_DEFINITIONS:
            params = tool_def["parameters"]
            assert params.get("type") == "object", (
                f"Tool '{tool_def['name']}' parameters type should be 'object'"
            )
            assert "properties" in params, (
                f"Tool '{tool_def['name']}' parameters missing 'properties'"
            )
            assert "required" in params, (
                f"Tool '{tool_def['name']}' parameters missing 'required'"
            )

    def test_all_registered_tools_have_schemas(self):
        """Every registered tool function should have a corresponding schema."""
        schema_names = {t["name"] for t in TOOL_DEFINITIONS}
        for fn_name in TOOL_REGISTRY:
            assert fn_name in schema_names, (
                f"Registered tool '{fn_name}' has no corresponding schema"
            )

    def test_tools_are_distinct(self):
        """Tools should have unique names."""
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"


# =========================================================================
# Individual Tool Tests
# =========================================================================

class TestCalculateTool:
    """Tests for the calculate tool."""

    def test_basic_addition(self):
        result = calculate("2 + 2")
        assert "4" in result

    def test_exponentiation(self):
        result = calculate("2 ** 10")
        assert "1024" in result

    def test_complex_expression(self):
        result = calculate("(10 + 5) * 3")
        assert "45" in result

    def test_empty_expression(self):
        result = calculate("")
        assert "Error" in result

    def test_invalid_expression(self):
        result = calculate("not_a_number + xyz")
        assert "Error" in result

    def test_returns_string(self):
        result = calculate("1 + 1")
        assert isinstance(result, str)


class TestRunPythonTool:
    """Tests for the run_python tool."""

    def test_simple_print(self):
        result = run_python("print('hello world')")
        assert "hello world" in result

    def test_calculation(self):
        result = run_python("print(sum(range(1, 11)))")
        assert "55" in result

    def test_empty_code(self):
        result = run_python("")
        assert "Error" in result

    def test_syntax_error(self):
        result = run_python("def incomplete(")
        assert "stderr" in result or "Error" in result

    def test_returns_string(self):
        result = run_python("print(1)")
        assert isinstance(result, str)


class TestReadFileTool:
    """Tests for the read_file tool."""

    def test_read_existing_file(self):
        # Create a temporary file within a controlled directory
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content here")
            temp_path = f.name

        try:
            # Temporarily update the allowed dir
            import agent.tools as tools_module
            original_dir = tools_module.ALLOWED_READ_DIR
            tools_module.ALLOWED_READ_DIR = os.path.dirname(temp_path)

            result = read_file(temp_path)
            assert "test content here" in result

            tools_module.ALLOWED_READ_DIR = original_dir
        finally:
            os.unlink(temp_path)

    def test_empty_path(self):
        result = read_file("")
        assert "Error" in result

    def test_nonexistent_file(self):
        result = read_file("/nonexistent/path/file.txt")
        assert "Error" in result

    def test_returns_string(self):
        result = read_file("some_file.txt")
        assert isinstance(result, str)


class TestExecuteTool:
    """Tests for the execute_tool dispatcher."""

    def test_execute_known_tool(self):
        result = execute_tool("calculate", {"expression": "1 + 1"})
        assert "2" in result

    def test_execute_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "Unknown tool" in result

    def test_execute_with_wrong_args(self):
        result = execute_tool("calculate", {"wrong_arg": "value"})
        assert "Error" in result


# =========================================================================
# Agent Loop Tests (mocked)
# =========================================================================

class TestAgentLoop:
    """Tests for the agent loop structure."""

    def test_max_steps_enforcement(self):
        """Agent should stop after max_steps and return a graceful message."""
        from unittest.mock import patch, MagicMock

        # Mock the LLM to always request a tool call (never finish)
        mock_response = {
            "stop_reason": "tool_use",
            "text": "Let me think...",
            "tool_calls": [{"name": "calculate", "arguments": {"expression": "1+1"}, "id": "test"}],
            "raw_content": [{"type": "text", "text": "thinking"}],
        }

        with patch("agent.react_agent.call_llm", return_value=mock_response):
            from agent.react_agent import run_agent
            result = run_agent(
                task="test task",
                provider="gemini",
                max_steps=3,
                verbose=False,
            )
            assert "Max steps reached" in result

    def test_agent_returns_final_answer(self):
        """Agent should return the LLM's text when stop_reason is end_turn."""
        from unittest.mock import patch

        mock_response = {
            "stop_reason": "end_turn",
            "text": "The answer is 42.",
            "tool_calls": [],
            "raw_content": [],
        }

        with patch("agent.react_agent.call_llm", return_value=mock_response):
            from agent.react_agent import run_agent
            result = run_agent(
                task="test",
                provider="gemini",
                max_steps=5,
                verbose=False,
            )
            assert "42" in result


# =========================================================================
# Weather and Search tools — integration tests (network-dependent)
# =========================================================================

class TestWeatherTool:
    """Tests for the get_weather tool (basic validation only)."""

    def test_empty_city(self):
        result = get_weather("")
        assert "Error" in result

    def test_returns_string(self):
        result = get_weather("InvalidCityXYZ123")
        assert isinstance(result, str)


class TestSearchWebTool:
    """Tests for the search_web tool (basic validation only)."""

    def test_empty_query(self):
        result = search_web("")
        assert "Error" in result

    def test_returns_string(self):
        result = search_web("test")
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

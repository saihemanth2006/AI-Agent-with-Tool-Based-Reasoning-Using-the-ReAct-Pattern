"""
Tool implementations for the ReAct agent.

Each tool is a Python function that accepts arguments and returns a string result.
Tools are registered with JSON schema definitions so the LLM knows how to call them.
"""

import os
import subprocess
import sys
import tempfile
import logging
from typing import Any

import requests
import numexpr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: get_weather
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """
    Get the current weather for a specific city using the wttr.in API.

    Args:
        city: The name of the city, e.g., 'Tokyo' or 'San Francisco'.

    Returns:
        A string summary of the current weather conditions.
    """
    if not city or not city.strip():
        return "Error: City name cannot be empty."

    try:
        url = f"https://wttr.in/{city.strip()}?format=j1"
        response = requests.get(url, timeout=10, headers={"User-Agent": "ReAct-Agent/1.0"})
        response.raise_for_status()

        data = response.json()
        current = data.get("current_condition", [{}])[0]

        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        description = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        wind_speed_kmph = current.get("windspeedKmph", "N/A")
        feels_like_c = current.get("FeelsLikeC", "N/A")

        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)
        country = area.get("country", [{}])[0].get("value", "Unknown")

        return (
            f"Weather in {area_name}, {country}:\n"
            f"  Condition: {description}\n"
            f"  Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  Feels Like: {feels_like_c}°C\n"
            f"  Humidity: {humidity}%\n"
            f"  Wind Speed: {wind_speed_kmph} km/h"
        )

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while fetching weather for '{city}'."
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the weather service for '{city}'."
    except requests.exceptions.HTTPError as e:
        return f"Error: Weather service returned HTTP {e.response.status_code} for '{city}'."
    except (KeyError, IndexError, ValueError) as e:
        return f"Error: Failed to parse weather data for '{city}': {e}"


# ---------------------------------------------------------------------------
# Tool 2: search_web
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """
    Search the web using the DuckDuckGo Instant Answer API.

    Args:
        query: The search query string.

    Returns:
        A string with search result snippets.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query.strip(), "format": "json", "no_html": 1, "skip_disambig": 1}
        response = requests.get(url, params=params, timeout=10, headers={"User-Agent": "ReAct-Agent/1.0"})
        response.raise_for_status()

        data = response.json()
        results = []

        # Check for Abstract (instant answer)
        if data.get("AbstractText"):
            source = data.get("AbstractSource", "Unknown")
            results.append(f"[{source}] {data['AbstractText']}")

        # Check for Answer (calculated/direct answer)
        if data.get("Answer"):
            results.append(f"[Answer] {data['Answer']}")

        # Check for Related Topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if not results:
            return f"No results found for query: '{query}'. Try rephrasing."

        return f"Search results for '{query}':\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return f"Error: Search request timed out for query '{query}'."
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the search service."
    except (ValueError, KeyError) as e:
        return f"Error: Failed to parse search results: {e}"


# ---------------------------------------------------------------------------
# Tool 3: calculate
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Uses the numexpr library for safe evaluation. Supports standard
    arithmetic operations, powers, and common math functions.

    Args:
        expression: A mathematical expression string, e.g., '2 + 2' or '5**3 + sin(1.5)'.

    Returns:
        A string with the result of the calculation.
    """
    if not expression or not expression.strip():
        return "Error: Expression cannot be empty."

    expr = expression.strip()

    try:
        result = numexpr.evaluate(expr)
        return f"{expr} = {result}"
    except (SyntaxError, TypeError, KeyError) as e:
        return f"Error: Invalid expression '{expr}': {e}"
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expr}'."
    except Exception as e:
        return f"Error: Could not evaluate '{expr}': {e}"


# ---------------------------------------------------------------------------
# Tool 4: run_python
# ---------------------------------------------------------------------------

def run_python(code: str) -> str:
    """
    Execute a string of Python code in a sandboxed subprocess and return its stdout.

    The code runs with a 30-second timeout to prevent long-running scripts.

    Args:
        code: A string of Python code to execute.

    Returns:
        The standard output of the executed code, or an error message.
    """
    if not code or not code.strip():
        return "Error: Code cannot be empty."

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tempfile.gettempdir(),
        )

        output_parts = []
        if result.stdout.strip():
            output_parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            output_parts.append(f"stderr:\n{result.stderr.strip()}")

        if not output_parts:
            return "Code executed successfully with no output."

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except OSError as e:
        return f"Error: Failed to execute code: {e}"


# ---------------------------------------------------------------------------
# Tool 5: read_file
# ---------------------------------------------------------------------------

# Configurable allowed directory for file reading (default: current working directory)
ALLOWED_READ_DIR = os.environ.get("ALLOWED_READ_DIR", os.getcwd())


def read_file(path: str) -> str:
    """
    Read the contents of a local text file.

    Validates the path to prevent directory traversal attacks.
    Only files within the allowed directory can be read.

    Args:
        path: The path to the file to read.

    Returns:
        The file contents as a string, or an error message.
    """
    if not path or not path.strip():
        return "Error: File path cannot be empty."

    try:
        # Resolve the absolute path and prevent directory traversal
        resolved = os.path.realpath(os.path.abspath(path.strip()))
        allowed = os.path.realpath(os.path.abspath(ALLOWED_READ_DIR))

        if not resolved.startswith(allowed + os.sep) and resolved != allowed:
            return (
                f"Error: Access denied. Path '{path}' is outside the allowed "
                f"directory '{ALLOWED_READ_DIR}'."
            )

        if not os.path.exists(resolved):
            return f"Error: File not found: '{path}'."

        if not os.path.isfile(resolved):
            return f"Error: '{path}' is not a regular file."

        # Limit file size to 100KB to prevent reading very large files
        file_size = os.path.getsize(resolved)
        if file_size > 100 * 1024:
            return f"Error: File is too large ({file_size} bytes). Maximum allowed size is 100KB."

        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return f"Contents of '{path}' ({len(content)} characters):\n{content}"

    except PermissionError:
        return f"Error: Permission denied when reading '{path}'."
    except OSError as e:
        return f"Error: Could not read file '{path}': {e}"


# ---------------------------------------------------------------------------
# Tool Schema Definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a specific city. Returns a summary "
            "including temperature, conditions, humidity, and wind speed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g., 'Tokyo' or 'San Francisco'.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for information using DuckDuckGo. Returns snippets "
            "from top results for the given query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g., 'capital of France' or 'Python programming language'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Safely evaluate a mathematical expression. Supports arithmetic "
            "operations (+, -, *, /, **), and common math functions like "
            "sin(), cos(), sqrt(), log(), exp(). Example: '2**10 + sqrt(144)'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 2' or '5**3 + sin(1.5)'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a string of Python code and return its standard output. "
            "The code runs in a sandboxed subprocess with a 30-second timeout. "
            "Use print() to produce output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Use print() to produce output.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a local text file. Only files within the "
            "allowed directory can be read. Maximum file size is 100KB."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Registry — maps tool names to their callable functions
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate,
    "run_python": run_python,
    "read_file": read_file,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a registered tool by name with the given arguments.

    Args:
        name: The name of the tool to execute.
        arguments: A dictionary of keyword arguments to pass to the tool.

    Returns:
        The string result from the tool, or an error message if the tool
        is not found or execution fails.
    """
    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'. Available tools: {list(TOOL_REGISTRY.keys())}"

    tool_fn = TOOL_REGISTRY[name]

    try:
        result = tool_fn(**arguments)
        return result
    except TypeError as e:
        return f"Error: Invalid arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error: Tool '{name}' failed with unexpected error: {e}"

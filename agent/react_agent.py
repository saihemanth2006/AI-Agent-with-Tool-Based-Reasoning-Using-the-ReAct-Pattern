"""
ReAct Agent — Reasoning + Acting Loop.

This module implements the core agent loop that orchestrates the conversation
between a Large Language Model (LLM) and external tools using the ReAct pattern.

Supported LLM providers:
  - Google Gemini (default)
  - OpenAI (gpt-4o, gpt-4o-mini, etc.)
  - Anthropic (claude-3.5-sonnet, etc.)
"""

import json
import logging
import os
from typing import Optional

from agent.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour codes for console output
# ---------------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

# ---------------------------------------------------------------------------
# System prompt for the ReAct agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant that solves problems step-by-step using the ReAct (Reasoning + Acting) pattern.

For each step you MUST:
1. **Think**: Reason about the current state, what information you have, and what you need to do next.
2. **Act**: If you need external information or computation, call the appropriate tool.
3. **Observe**: Review the tool's output and use it to plan your next step.

Guidelines:
- Always think before acting. Explain your reasoning clearly.
- Use tools when you need real-time data, calculations, or external information.
- If a tool returns an error, analyze the error and try a different approach (e.g., fix a typo, use a different tool, or rephrase the query).
- When you have gathered all the information needed, provide a clear and comprehensive final answer to the user.
- Combine information from multiple tool calls into a coherent response.
- Be precise with numbers and data from tool outputs.
"""


# ============================================================================
# Provider-specific LLM callers
# ============================================================================

def _call_gemini(messages: list[dict], tools: list[dict], model: str) -> dict:
    """
    Call Google Gemini API with tool support using the google-genai SDK.

    Returns a dict with keys: 'stop_reason', 'text', 'tool_calls', 'raw_content'.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    # Convert tool definitions to Gemini function declarations
    function_declarations = []
    for tool_def in tools:
        properties = {}
        for prop_name, prop_schema in tool_def["parameters"]["properties"].items():
            properties[prop_name] = types.Schema(
                type=prop_schema.get("type", "STRING").upper(),
                description=prop_schema.get("description", ""),
            )

        fn_decl = types.FunctionDeclaration(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties=properties,
                required=tool_def["parameters"].get("required", []),
            ),
        )
        function_declarations.append(fn_decl)

    gemini_tool = types.Tool(function_declarations=function_declarations)

    # Build contents list for the Gemini API
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        parts = []

        if isinstance(msg["content"], str):
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, str):
                    parts.append(types.Part.from_text(text=item))
                elif isinstance(item, dict):
                    if item.get("type") == "function_response":
                        parts.append(types.Part.from_function_response(
                            name=item["name"],
                            response={"result": item["content"]},
                        ))
                    elif item.get("type") == "function_call":
                        parts.append(types.Part.from_function_call(
                            name=item["name"],
                            args=item["args"],
                        ))
                    elif "text" in item:
                        parts.append(types.Part.from_text(text=item["text"]))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    # Make the API call with retry for rate limiting
    import time
    import re as _re

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[gemini_tool],
                ),
            )
            break  # Success — exit retry loop
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                # Extract retry delay from error if available
                delay_match = _re.search(r"retry in (\d+)", error_str)
                wait_time = int(delay_match.group(1)) + 2 if delay_match else (attempt + 1) * 15
                logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                print(f"   \033[93m⏳ Rate limited. Waiting {wait_time}s before retry...\033[0m")
                time.sleep(wait_time)
            else:
                raise  # Re-raise non-429 errors or if out of retries

    # Parse response
    result = {"stop_reason": "end_turn", "text": "", "tool_calls": [], "raw_content": []}

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                result["tool_calls"].append({
                    "name": fc.name,
                    "arguments": args,
                })
                result["raw_content"].append({
                    "type": "function_call",
                    "name": fc.name,
                    "args": args,
                })
                result["stop_reason"] = "tool_use"
            elif part.text:
                result["text"] += part.text
                result["raw_content"].append({"type": "text", "text": part.text})

    return result


def _call_openai(messages: list[dict], tools: list[dict], model: str) -> dict:
    """
    Call OpenAI API with tool/function calling support.

    Returns a dict with keys: 'stop_reason', 'text', 'tool_calls', 'raw_content'.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    # Convert tool definitions to OpenAI format
    openai_tools = []
    for tool_def in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": tool_def["parameters"],
            },
        })

    # Build OpenAI messages
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            openai_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            content = msg.get("content", "")
            tool_calls_data = msg.get("tool_calls")
            if tool_calls_data:
                openai_messages.append({
                    "role": "assistant",
                    "content": content if isinstance(content, str) else None,
                    "tool_calls": tool_calls_data,
                })
            else:
                if isinstance(content, str):
                    openai_messages.append({"role": "assistant", "content": content})
        elif msg["role"] == "tool":
            openai_messages.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            })

    response = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        tools=openai_tools if openai_tools else None,
    )

    choice = response.choices[0]
    result = {"stop_reason": "end_turn", "text": "", "tool_calls": [], "raw_content": []}

    if choice.finish_reason == "tool_calls":
        result["stop_reason"] = "tool_use"

    if choice.message.content:
        result["text"] = choice.message.content
        result["raw_content"].append({"type": "text", "text": choice.message.content})

    if choice.message.tool_calls:
        result["stop_reason"] = "tool_use"
        raw_tool_calls = []
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result["tool_calls"].append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })
            raw_tool_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        result["raw_tool_calls"] = raw_tool_calls

    return result


def _call_anthropic(messages: list[dict], tools: list[dict], model: str) -> dict:
    """
    Call Anthropic API with tool use support.

    Returns a dict with keys: 'stop_reason', 'text', 'tool_calls', 'raw_content'.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # Convert tool definitions to Anthropic format
    anthropic_tools = []
    for tool_def in tools:
        anthropic_tools.append({
            "name": tool_def["name"],
            "description": tool_def["description"],
            "input_schema": tool_def["parameters"],
        })

    # Build Anthropic messages
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                anthropic_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            anthropic_messages.append({"role": "assistant", "content": msg["content"]})

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=anthropic_messages,
        tools=anthropic_tools,
    )

    result = {
        "stop_reason": "end_turn",
        "text": "",
        "tool_calls": [],
        "raw_content": list(response.content),
    }

    if response.stop_reason == "tool_use":
        result["stop_reason"] = "tool_use"

    for block in response.content:
        if block.type == "text":
            result["text"] += block.text
        elif block.type == "tool_use":
            result["tool_calls"].append({
                "id": block.id,
                "name": block.name,
                "arguments": block.input,
            })
            result["stop_reason"] = "tool_use"

    return result


# ============================================================================
# Unified LLM caller
# ============================================================================

def call_llm(
    messages: list[dict],
    tools: list[dict],
    provider: str = "gemini",
    model: Optional[str] = None,
) -> dict:
    """
    Call an LLM provider with tool support.

    Args:
        messages: Conversation history in a normalized format.
        tools: List of tool JSON schema definitions.
        provider: One of 'gemini', 'openai', 'anthropic'.
        model: Model name override. If None, uses a sensible default.

    Returns:
        Normalized response dict with keys:
        - stop_reason: 'end_turn' or 'tool_use'
        - text: Any text content from the response
        - tool_calls: List of dicts with 'name', 'arguments', and optionally 'id'
        - raw_content: Provider-specific raw content for message history
    """
    provider = provider.lower()

    defaults = {
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
    }

    if not model:
        model = defaults.get(provider, "gemini-2.0-flash")

    if provider == "gemini":
        return _call_gemini(messages, tools, model)
    elif provider == "openai":
        return _call_openai(messages, tools, model)
    elif provider == "anthropic":
        return _call_anthropic(messages, tools, model)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'gemini', 'openai', or 'anthropic'.")


# ============================================================================
# Message history helpers (provider-specific formatting)
# ============================================================================

def _append_assistant_and_tool_results_gemini(
    messages: list[dict],
    response: dict,
    tool_results: list[dict],
) -> list[dict]:
    """Append assistant response and tool results in Gemini format."""
    # Assistant turn with function call(s)
    messages.append({"role": "assistant", "content": response["raw_content"]})

    # Tool results as a user message with function responses
    parts = []
    for tr in tool_results:
        parts.append({
            "type": "function_response",
            "name": tr["name"],
            "content": tr["result"],
        })
    messages.append({"role": "user", "content": parts})

    return messages


def _append_assistant_and_tool_results_openai(
    messages: list[dict],
    response: dict,
    tool_results: list[dict],
) -> list[dict]:
    """Append assistant response and tool results in OpenAI format."""
    # Assistant turn
    assistant_msg = {
        "role": "assistant",
        "content": response.get("text", ""),
    }
    if response.get("raw_tool_calls"):
        assistant_msg["tool_calls"] = response["raw_tool_calls"]
    messages.append(assistant_msg)

    # Each tool result as a separate "tool" message
    for tr in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tr["id"],
            "content": tr["result"],
        })

    return messages


def _append_assistant_and_tool_results_anthropic(
    messages: list[dict],
    response: dict,
    tool_results: list[dict],
) -> list[dict]:
    """Append assistant response and tool results in Anthropic format."""
    # Assistant turn — use raw content blocks
    assistant_content = []
    for block in response["raw_content"]:
        if hasattr(block, "type"):
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        elif isinstance(block, dict):
            assistant_content.append(block)

    messages.append({"role": "assistant", "content": assistant_content})

    # Tool results as a user message
    tool_result_blocks = []
    for tr in tool_results:
        tool_result_blocks.append({
            "type": "tool_result",
            "tool_use_id": tr["id"],
            "content": tr["result"],
        })
    messages.append({"role": "user", "content": tool_result_blocks})

    return messages


# ============================================================================
# Main ReAct Agent Loop
# ============================================================================

def run_agent(
    task: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    max_steps: int = 10,
    verbose: bool = True,
) -> str:
    """
    Run the ReAct agent to solve a task using iterative reasoning and tool use.

    The agent alternates between:
      1. Calling the LLM to reason about the task and decide on an action.
      2. Executing the chosen tool and feeding the result back to the LLM.

    This loop continues until the LLM provides a final answer or the
    maximum number of steps is reached.

    Args:
        task: The user's task or question to solve.
        provider: LLM provider — 'gemini', 'openai', or 'anthropic'.
        model: Specific model name (uses provider default if None).
        max_steps: Maximum number of reasoning-action steps allowed.
        verbose: Whether to print the reasoning trace to the console.

    Returns:
        The agent's final answer as a string.
    """
    if verbose:
        print(f"\n{BOLD}{MAGENTA}{'=' * 70}{RESET}")
        print(f"{BOLD}{MAGENTA}  ReAct Agent — {provider.upper()} ({model or 'default'}){RESET}")
        print(f"{BOLD}{MAGENTA}  Max Steps: {max_steps}{RESET}")
        print(f"{BOLD}{MAGENTA}{'=' * 70}{RESET}")
        print(f"\n{BOLD}📋 Task:{RESET} {task}\n")

    # Initialize conversation history with the user's task
    messages = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"{BOLD}{CYAN}--- Step {step}/{max_steps} ---{RESET}")

        # Call the LLM with the current conversation and available tools
        try:
            response = call_llm(messages, TOOL_DEFINITIONS, provider=provider, model=model)
        except Exception as e:
            error_msg = f"LLM API call failed: {e}"
            logger.error(error_msg)
            if verbose:
                print(f"{RED}❌ {error_msg}{RESET}")
            return f"Error: {error_msg}"

        # Print the agent's reasoning (thought)
        if response["text"] and verbose:
            print(f"{YELLOW}💭 Thought:{RESET}")
            for line in response["text"].strip().split("\n"):
                print(f"   {line}")
            print()

        # Decision point: final answer or tool use?
        if response["stop_reason"] == "end_turn":
            # The agent has finished — return the final answer
            final_answer = response["text"].strip()
            if verbose:
                print(f"{GREEN}✅ Final Answer:{RESET}")
                for line in final_answer.split("\n"):
                    print(f"   {line}")
                print(f"\n{DIM}Completed in {step} step(s).{RESET}\n")
            return final_answer

        if response["stop_reason"] == "tool_use":
            # Execute each tool call
            tool_results = []

            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                tool_id = tool_call.get("id", "")

                if verbose:
                    print(f"{MAGENTA}🔧 Action:{RESET} {tool_name}({json.dumps(tool_args)})")

                # Execute the tool
                result = execute_tool(tool_name, tool_args)

                if verbose:
                    print(f"{GREEN}📊 Observation:{RESET}")
                    # Truncate very long results for display
                    display_result = result if len(result) <= 500 else result[:500] + "... [truncated]"
                    for line in display_result.split("\n"):
                        print(f"   {line}")
                    print()

                tool_results.append({
                    "name": tool_name,
                    "id": tool_id,
                    "result": result,
                })

            # Append the assistant response and tool results to conversation history
            if provider == "gemini":
                messages = _append_assistant_and_tool_results_gemini(messages, response, tool_results)
            elif provider == "openai":
                messages = _append_assistant_and_tool_results_openai(messages, response, tool_results)
            elif provider == "anthropic":
                messages = _append_assistant_and_tool_results_anthropic(messages, response, tool_results)

    # Max steps reached without a final answer
    timeout_msg = (
        f"Max steps reached ({max_steps}) without completing the task. "
        "The agent was unable to arrive at a final answer within the allowed number of steps."
    )
    if verbose:
        print(f"\n{RED}⚠️  {timeout_msg}{RESET}\n")

    return timeout_msg

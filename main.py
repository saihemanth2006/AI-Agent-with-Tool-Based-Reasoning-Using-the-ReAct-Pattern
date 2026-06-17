#!/usr/bin/env python3
"""
ReAct Agent — Main Entry Point.

Provides both CLI and interactive modes for running the AI agent
with tool-based reasoning.

Usage:
    # Run with a single task from the command line
    python main.py --task "What is the weather in Tokyo?"

    # Interactive mode
    python main.py --interactive

    # Specify a provider and model
    python main.py --provider openai --model gpt-4o --task "Calculate 2**10"

    # Adjust the maximum number of steps
    python main.py --max-steps 15 --task "What is the weather in London?"
"""

import argparse
import logging
import sys
import os

# Ensure UTF-8 output on Windows to support Unicode characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from agent.react_agent import run_agent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ANSI colour codes
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    """Print a decorative startup banner."""
    banner = f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🤖  ReAct Agent — AI with Tool-Based Reasoning                 ║
║                                                                  ║
║   Reasoning + Acting for Multi-Step Problem Solving              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def run_interactive_mode(provider: str, model: str, max_steps: int):
    """
    Run the agent in interactive mode, allowing the user to submit
    multiple tasks in a REPL-style loop.

    Args:
        provider: The LLM provider to use.
        model: The specific model name.
        max_steps: Maximum steps per task.
    """
    print(f"\n{BOLD}Interactive Mode{RESET}")
    print(f"{DIM}Type your task and press Enter. Type 'quit' or 'exit' to stop.{RESET}")
    print(f"{DIM}Provider: {provider} | Model: {model or 'default'} | Max Steps: {max_steps}{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        try:
            result = run_agent(
                task=user_input,
                provider=provider,
                model=model,
                max_steps=max_steps,
                verbose=True,
            )
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            logger.exception("Agent execution failed")


def main():
    """Parse CLI arguments and run the agent."""
    parser = argparse.ArgumentParser(
        description="ReAct Agent — AI with Tool-Based Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --task "What is the weather in Tokyo?"
  python main.py --task "Find the weather in Paris and calculate 5 factorial"
  python main.py --provider openai --task "Search the web for Python 3.13 features"
  python main.py --interactive
  python main.py --interactive --provider anthropic --model claude-sonnet-4-20250514
        """,
    )

    parser.add_argument(
        "--task", "-t",
        type=str,
        help="The task for the agent to solve.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (REPL).",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "gemini"),
        choices=["gemini", "openai", "anthropic"],
        help="LLM provider to use (default: gemini).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get("LLM_MODEL"),
        help="Specific model name (uses provider default if not set).",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=int(os.environ.get("MAX_STEPS", "10")),
        help="Maximum number of agent steps (default: 10).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print the full reasoning trace (default: True).",
    )

    args = parser.parse_args()

    print_banner()

    if args.interactive:
        run_interactive_mode(args.provider, args.model, args.max_steps)
    elif args.task:
        result = run_agent(
            task=args.task,
            provider=args.provider,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        print(f"\n{BOLD}{'=' * 70}{RESET}")
        print(f"{BOLD}Final Result:{RESET}")
        print(result)
    else:
        parser.print_help()
        print(f"\n{YELLOW}Please provide a --task or use --interactive mode.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# 🤖 AI Agent with Tool-Based Reasoning — ReAct Pattern

An AI agent that solves multi-step problems by combining **Reasoning** and **Acting** (ReAct) with external tools. The agent iteratively thinks through a problem, calls the right tools to gather information or perform computations, observes the results, and continues until it arrives at a comprehensive answer.

## ✨ Features

- **ReAct Loop** — Iterative Reasoning + Acting pattern with full reasoning trace
- **5 Built-in Tools** — Weather lookup, web search, safe math evaluation, Python execution, and file reading
- **3 LLM Providers** — Google Gemini, OpenAI, and Anthropic (configurable)
- **Safety Mechanisms** — Configurable `max_steps` limit, sandboxed code execution, file access restrictions
- **Error Recovery** — Graceful handling of tool failures with informative error messages
- **Interactive & CLI Modes** — Single-task execution or REPL-style interactive mode
- **Full Logging** — Colored console output showing every thought, action, and observation
- **Dockerized** — Ready-to-run with Docker and Docker Compose

## 📁 Project Structure

```
.
├── agent/
│   ├── __init__.py          # Package init
│   ├── tools.py             # Tool implementations + JSON schemas + registry
│   └── react_agent.py       # ReAct loop + LLM provider integrations
├── tests/
│   ├── __init__.py
│   └── test_agent.py        # Comprehensive test suite
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image definition
├── docker-compose.yml       # Docker Compose configuration
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **At least one LLM API key** (Gemini recommended — free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/saihemanth2006/AI-Agent-with-Tool-Based-Reasoning-Using-the-ReAct-Pattern.git
cd AI-Agent-with-Tool-Based-Reasoning-Using-the-ReAct-Pattern
```

### 2. Set Up Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key(s)
# At minimum, set ONE of these:
#   GEMINI_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here
#   ANTHROPIC_API_KEY=your_key_here
```

### 4. Run the Agent

```bash
# Single task
python main.py --task "What is the weather in New York City, and what is 5 factorial?"

# Interactive mode
python main.py --interactive

# Use a specific provider
python main.py --provider openai --task "Calculate 2**10 + sqrt(144)"
```

## 🐳 Docker Setup

### Build and Run with Docker Compose

```bash
# Copy and configure your .env file first
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker compose build
docker compose run --rm react-agent

# Or run a single task
docker compose run --rm react-agent --task "What is the weather in London?"
```

### Build and Run with Docker Directly

```bash
docker build -t react-agent .
docker run -it --env-file .env react-agent --interactive
```

## 🔧 Available Tools

| Tool | Description | Implementation |
|------|-------------|----------------|
| `get_weather(city)` | Current weather conditions for any city | [wttr.in](https://wttr.in) API (no key required) |
| `search_web(query)` | Web search results and snippets | DuckDuckGo Instant Answer API (no key required) |
| `calculate(expression)` | Safe mathematical expression evaluation | `numexpr` library (no `eval()`) |
| `run_python(code)` | Execute Python code in a sandboxed subprocess | `subprocess` with 30s timeout |
| `read_file(path)` | Read local text files with access control | Path validation + 100KB size limit |

## 📋 Sample Prompts

### Single-Tool Tasks

```bash
python main.py --task "What is the current weather in Paris?"
python main.py --task "Calculate the value of 2**20 - 1"
python main.py --task "Run this Python code: print([i**2 for i in range(10)])"
```

### Multi-Tool Tasks (Required for Evaluation)

```bash
# Weather + Calculation
python main.py --task "What's the weather in New York City, and what is 5 factorial?"

# Weather comparison
python main.py --task "Find the weather in Paris and London, then calculate the temperature difference"

# Search + Python
python main.py --task "Search for the population of Japan and write Python code to calculate how many years it would take to double at 0.5% growth rate"
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --tb=short
```

## ⚙️ Configuration

All configuration is done through environment variables (or the `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LLM_PROVIDER` | `gemini` | Which LLM provider to use |
| `LLM_MODEL` | *(auto)* | Specific model name |
| `MAX_STEPS` | `10` | Max reasoning steps per task |
| `ALLOWED_READ_DIR` | `./` | Directory the `read_file` tool can access |

## 🔄 How the ReAct Loop Works

```
User Task
    │
    ▼
┌─────────────────────────────────┐
│  Step 1: Call LLM               │
│  ┌───────────────────────────┐  │
│  │ 💭 Thought: "I need to    │  │
│  │    get the weather first" │  │
│  │ 🔧 Action: get_weather()  │  │
│  └───────────────────────────┘  │
│  📊 Observation: "Sunny, 25°C"  │
│                                 │
│  Step 2: Call LLM               │
│  ┌───────────────────────────┐  │
│  │ 💭 Thought: "Now I need   │  │
│  │    to calculate..."       │  │
│  │ 🔧 Action: calculate()    │  │
│  └───────────────────────────┘  │
│  📊 Observation: "120"          │
│                                 │
│  Step 3: Call LLM               │
│  ┌───────────────────────────┐  │
│  │ 💭 Thought: "I have all   │  │
│  │    the info I need"       │  │
│  │ ✅ Final Answer            │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │
    ▼
Final Answer returned to user
```

## 📝 Example Output

```
══════════════════════════════════════════════════════════════════════
  🤖  ReAct Agent — GEMINI (default)
  Max Steps: 10
══════════════════════════════════════════════════════════════════════

📋 Task: What's the weather in New York City, and what is 5 factorial?

--- Step 1/10 ---
💭 Thought:
   I need to get the weather in New York City and calculate 5 factorial.
   Let me start with the weather.

🔧 Action: get_weather({"city": "New York City"})
📊 Observation:
   Weather in New York, United States of America:
     Condition: Partly cloudy
     Temperature: 24°C (75°F)
     Feels Like: 25°C
     Humidity: 65%
     Wind Speed: 15 km/h

--- Step 2/10 ---
💭 Thought:
   I have the weather data. Now I need to calculate 5 factorial (5!).

🔧 Action: calculate({"expression": "5 * 4 * 3 * 2 * 1"})
📊 Observation:
   5 * 4 * 3 * 2 * 1 = 120

--- Step 3/10 ---
✅ Final Answer:
   The weather in New York City is partly cloudy with a temperature
   of 24°C (75°F) and 65% humidity. 5 factorial (5!) equals 120.

Completed in 3 step(s).
```

## 🛡️ Safety Features

- **Max Steps Limit** — Prevents infinite loops and excessive API usage
- **Sandboxed Code Execution** — `run_python` runs in a subprocess with a 30-second timeout
- **Safe Math Evaluation** — Uses `numexpr` instead of dangerous `eval()`
- **File Access Control** — `read_file` validates paths to prevent directory traversal
- **File Size Limits** — Maximum 100KB file size for the `read_file` tool
- **Error Handling** — All tools return descriptive error messages instead of crashing

## 📄 License

This project is for educational purposes as part of a backend development assignment.

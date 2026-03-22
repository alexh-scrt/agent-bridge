# agent_bridge

> Your AI coding assistant, reachable from anywhere — just message it.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

agent_bridge is a self-hosted FastAPI server that connects **Telegram** and **Discord** to a local AI coding assistant. Send natural language instructions from your phone, and receive completed code or task results delivered straight back to your chat — no terminal required. It queues long-running AI jobs asynchronously, supports any OpenAI-compatible backend or local Ollama instance, and keeps a full SQLite task history you can query any time.

---

## Quick Start

**1. Install**

```bash
git clone https://github.com/your-org/agent_bridge.git
cd agent_bridge
pip install .
```

**2. Configure**

```bash
cp .env.example .env
# Edit .env with your tokens and AI backend URL
```

**3. Run**

```bash
uvicorn agent_bridge.app:create_app --factory --host 0.0.0.0 --port 8000
```

Or use the built-in entry point:

```bash
agent-bridge
```

The server is now live at `http://localhost:8000`. Register your Telegram/Discord webhook URLs to point at your public host and start sending prompts.

---

## Features

- **Unified webhook ingestion** — Accepts messages from both Telegram and Discord with per-user allowlist authentication, so only trusted users can trigger AI jobs.
- **Async task queue** — Long-running AI prompts are queued and executed concurrently up to a configurable limit, keeping the server responsive at all times.
- **Pluggable AI backend** — Works with any OpenAI-compatible API (e.g., OpenAI, Groq, LiteLLM) or a local Ollama instance, with streaming response support.
- **Automatic result delivery** — Completed code and text results are sent back to the originating chat channel, formatted as proper code blocks for each platform.
- **SQLite task history** — Every job is persisted with full status tracking (`pending` → `running` → `done` / `failed`). Query job status directly from chat with `/status`.

---

## Usage Examples

### Send a coding prompt from Telegram or Discord

Just message your bot in the allowed channel:

```
Write a Python function that parses a JSON file and returns a list of unique keys
```

The bot acknowledges the task immediately, then delivers the result when complete:

````
✅ Task abc123 complete

```python
def unique_keys(path: str) -> list[str]:
    import json
    with open(path) as f:
        data = json.load(f)
    return list({k for d in data for k in d})
```
````

### Check task status from chat

```
/status abc123
```

```
Task abc123 — running (started 12s ago)
```

### Query the REST API directly

```bash
# List recent tasks
curl http://localhost:8000/tasks

# Get a specific task
curl http://localhost:8000/tasks/abc123

# Manually trigger result delivery
curl -X POST http://localhost:8000/tasks/abc123/deliver
```

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

---

## Project Structure

```
agent_bridge/
├── agent_bridge/
│   ├── __init__.py        # Package init, version export
│   ├── app.py             # FastAPI application factory, lifespan hooks
│   ├── routes.py          # Webhook endpoints, task queries, callbacks
│   ├── queue.py           # Async task queue with concurrency control
│   ├── ai_client.py       # OpenAI-compatible / Ollama async HTTP client
│   ├── messenger.py       # Telegram & Discord result delivery abstraction
│   ├── db.py              # SQLite task store (aiosqlite)
│   ├── models.py          # Pydantic models for payloads, tasks, responses
│   └── config.py          # Settings loaded from environment variables
├── tests/
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_db.py
│   ├── test_ai_client.py
│   ├── test_queue.py
│   ├── test_messenger.py
│   └── test_routes.py
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Configuration

Copy `.env.example` to `.env` and set the values for your deployment. All settings are loaded from environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes* | — | Bot token from @BotFather |
| `DISCORD_BOT_TOKEN` | Yes* | — | Bot token from Discord Developer Portal |
| `ALLOWED_USER_IDS` | Yes | — | Comma-separated list of allowed Telegram/Discord user IDs |
| `AI_BACKEND_URL` | Yes | — | Base URL of your OpenAI-compatible API or Ollama (`http://localhost:11434`) |
| `AI_MODEL` | No | `gpt-4o` | Model name to pass to the AI backend |
| `AI_API_KEY` | No | — | API key (not required for local Ollama) |
| `MAX_CONCURRENT_TASKS` | No | `3` | Max number of AI jobs running simultaneously |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8000` | Server bind port |
| `LOG_LEVEL` | No | `info` | Logging verbosity (`debug`/`info`/`warning`/`error`) |
| `DATABASE_PATH` | No | `agent_bridge.db` | Path to the SQLite database file |

*At least one of `TELEGRAM_BOT_TOKEN` or `DISCORD_BOT_TOKEN` must be provided.

### Example `.env`

```dotenv
TELEGRAM_BOT_TOKEN=123456:ABC-your-token-here
DISCORD_BOT_TOKEN=your-discord-bot-token
ALLOWED_USER_IDS=111222333,444555666
AI_BACKEND_URL=http://localhost:11434
AI_MODEL=llama3
MAX_CONCURRENT_TASKS=2
LOG_LEVEL=info
```

---

## Running Tests

```bash
pip install '.[dev]'
pytest
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*

# agent_bridge

A self-hosted messaging bridge that connects **Telegram** and **Discord** to a local AI coding assistant. Send natural language instructions from your phone and receive completed code or task results asynchronously.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Setting Up Telegram](#setting-up-telegram)
- [Setting Up Discord](#setting-up-discord)
- [AI Backend Configuration](#ai-backend-configuration)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

`agent_bridge` is a FastAPI-based server that acts as a bridge between chat platforms (Telegram and Discord) and a local or remote AI coding assistant. It allows developers to:

1. **Send a coding prompt** from Telegram or Discord (e.g. "Write a binary search implementation in Python").
2. **Receive the AI-generated response** back in the same chat, asynchronously — even if the AI takes minutes to respond.
3. **Track task status** using built-in `/status` and `/tasks` commands.
4. **Optionally notify external systems** via webhook callbacks when tasks complete.

The bridge supports any **OpenAI-compatible API** (OpenAI, Azure OpenAI, Groq, Together AI, etc.) or a **local Ollama** instance.

---

## Key Features

| Feature | Details |
|---|---|
| **Unified webhook ingestion** | Accepts messages from both Telegram Bot API webhooks and Discord via a secure HTTP endpoint |
| **Per-user allowlist** | Restrict access to specific Telegram user IDs or Discord user IDs |
| **Async task queue** | Long-running AI tasks are queued and processed concurrently without blocking |
| **Configurable concurrency** | Limit simultaneous AI requests with `MAX_CONCURRENT_TASKS` |
| **Pluggable AI backends** | OpenAI-compatible API or local Ollama — switch with a single env var |
| **Streaming support** | Collects streamed responses from the AI and delivers the full result |
| **Code block formatting** | Responses are formatted with Markdown code fences for readability |
| **SQLite task history** | All tasks are persisted with full status tracking (pending/running/done/failed) |
| **Status commands** | `/status <id>` and `/tasks` commands work from both Telegram and Discord |
| **External webhook callbacks** | Optionally POST task results to an external URL on completion |
| **REST API** | Full CRUD API for task management and manual re-delivery |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         agent_bridge                            │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │ Telegram Bot │   │   FastAPI Routes  │   │ Discord Bot   │  │
│  │   Webhook    │──▶│                  │◀──│   Endpoint    │  │
│  └──────────────┘   │  /telegram/...   │   └───────────────┘  │
│                      │  /discord/...    │                       │
│                      │  /tasks/...      │                       │
│                      │  /health         │                       │
│                      └────────┬─────────┘                       │
│                               │                                 │
│                               ▼                                 │
│                      ┌─────────────────┐                        │
│                      │   Task Queue    │                        │
│                      │  (asyncio-based)│                        │
│                      │  concurrency    │                        │
│                      │  semaphore      │                        │
│                      └────────┬────────┘                        │
│                               │                                 │
│              ┌────────────────┼────────────────┐               │
│              ▼                ▼                 ▼               │
│     ┌────────────────┐ ┌──────────┐ ┌──────────────────┐      │
│     │   AI Client    │ │ Database │ │    Messenger      │      │
│     │  (httpx-based) │ │ (SQLite) │ │ Telegram/Discord  │      │
│     │  OpenAI/Ollama │ │aiosqlite │ │    delivery       │      │
│     └────────────────┘ └──────────┘ └──────────────────┘      │
│              │                                                  │
│              ▼                                                  │
│     ┌────────────────┐                                         │
│     │  AI Backend    │                                         │
│     │  (Ollama or    │                                         │
│     │  OpenAI API)   │                                         │
│     └────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

1. User sends a message on Telegram or Discord.
2. The chat platform calls the agent_bridge webhook endpoint.
3. The route handler validates the user (allowlist check) and creates a task record in SQLite.
4. The task is enqueued in the async task queue and the user receives an immediate acknowledgement.
5. A worker picks up the task, calls the AI backend, and saves the result.
6. The messenger delivers the result back to the originating chat.
7. Optionally, an external callback URL is POSTed with the task result.

---

## Requirements

- **Python 3.11+**
- A **Telegram Bot token** (from [@BotFather](https://t.me/BotFather)) — optional if only using Discord
- A **Discord Bot token** — optional if only using Telegram
- An **AI backend**: local [Ollama](https://ollama.com/) or any OpenAI-compatible API
- A publicly reachable HTTPS URL for the Telegram webhook (can use [ngrok](https://ngrok.com/) for local development)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourorg/agent_bridge.git
cd agent_bridge
```

### 2. Install dependencies

```bash
# Using pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings (see Configuration Reference below)
```

Minimum required configuration for Telegram + Ollama:

```dotenv
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
SECRET_TOKEN=your-random-secret-here
AI_BACKEND_TYPE=ollama
AI_BACKEND_URL=http://localhost:11434
AI_MODEL=llama3
```

### 4. Start your AI backend

```bash
# With Ollama:
ollama serve
ollama pull llama3

# Or point AI_BACKEND_URL at your OpenAI-compatible API
```

### 5. Run the server

```bash
# Via the CLI entry point:
agent-bridge

# Or via uvicorn directly:
uvicorn agent_bridge.app:app --host 0.0.0.0 --port 8000

# Or via Python:
python -m agent_bridge.app
```

### 6. Register the Telegram webhook

Telegram requires a public HTTPS URL. For local development, use ngrok:

```bash
ngrok http 8000
```

Then register the webhook with Telegram:

```bash
curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-ngrok-url.ngrok.io/telegram/webhook"}'
```

### 7. Test it!

Open Telegram, find your bot, and send:
```
Write a Python function that reverses a linked list.
```

You'll receive an acknowledgement immediately, and the AI response will arrive within seconds to minutes depending on your backend.

---

## Configuration Reference

All configuration is via environment variables or a `.env` file in the project root.

### Server

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server listen port |
| `LOG_LEVEL` | `info` | Logging verbosity: `debug`, `info`, `warning`, `error`, `critical` |
| `SECRET_TOKEN` | `changeme` | Shared secret for authenticating incoming webhook calls. **Change this in production!** |
| `DATABASE_URL` | `agent_bridge.db` | Path to the SQLite database file. Use `:memory:` for testing. |

### Telegram

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | *(none)* | Your Telegram Bot API token from @BotFather. Required to enable Telegram. |
| `TELEGRAM_ALLOWED_USERS` | *(empty = allow all)* | Comma-separated list of Telegram user IDs to allow. Leave empty to allow everyone. |

Example:
```dotenv
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_ALLOWED_USERS=111111,222222,333333
```

### Discord

| Variable | Default | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | *(none)* | Your Discord Bot token. Required to enable Discord. |
| `DISCORD_ALLOWED_USERS` | *(empty = allow all)* | Comma-separated list of Discord user snowflake IDs to allow. |
| `DISCORD_ALLOWED_GUILDS` | *(empty = allow all)* | Comma-separated list of Discord guild/server IDs to allow. |

Example:
```dotenv
DISCORD_BOT_TOKEN=your.discord.bot.token
DISCORD_ALLOWED_USERS=123456789012345678,987654321098765432
DISCORD_ALLOWED_GUILDS=111111111111111111
```

### AI Backend

| Variable | Default | Description |
|---|---|---|
| `AI_BACKEND_TYPE` | `ollama` | Backend type: `ollama` or `openai` |
| `AI_BACKEND_URL` | `http://localhost:11434` | Base URL of the AI backend |
| `AI_MODEL` | `llama3` | Model name to use for completions |
| `AI_API_KEY` | *(none)* | API key for OpenAI-compatible backends. Not required for Ollama. |
| `AI_SYSTEM_PROMPT` | *(coding assistant prompt)* | System prompt sent with every request |
| `AI_MAX_TOKENS` | `4096` | Maximum tokens in the AI response |
| `AI_TEMPERATURE` | `0.2` | Sampling temperature (0.0 = deterministic, 1.0 = creative) |
| `AI_TIMEOUT_SECONDS` | `120` | HTTP timeout for AI backend requests |

### Task Queue

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENT_TASKS` | `3` | Maximum number of AI tasks running simultaneously |
| `TASK_TIMEOUT_SECONDS` | `300` | Maximum wall-clock seconds a single task may run before being marked FAILED |

### Callbacks

| Variable | Default | Description |
|---|---|---|
| `CALLBACK_URL` | *(none)* | Optional URL to POST task completion payloads to |
| `CALLBACK_SECRET` | *(none)* | Optional secret sent in `X-Callback-Secret` header with callback POSTs |

---

## Setting Up Telegram

### 1. Create a bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts.
3. Copy the bot token (format: `123456789:ABCdefGHI...`).
4. Set `TELEGRAM_BOT_TOKEN` in your `.env`.

### 2. Get your user ID (optional, for allowlisting)

Message [@userinfobot](https://t.me/userinfobot) to get your Telegram user ID, then add it to `TELEGRAM_ALLOWED_USERS`.

### 3. Register the webhook

Telegram delivers messages to your server via HTTPS webhook. You need a publicly accessible URL.

**For production** (replace with your real domain):
```bash
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://yourdomain.com/telegram/webhook"}'
```

**For local development** with ngrok:
```bash
# Start ngrok
ngrok http 8000

# Register the ngrok URL
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://abcd1234.ngrok.io/telegram/webhook"}'
```

**Verify webhook registration:**
```bash
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

### 4. Available Telegram commands

| Command | Description |
|---|---|
| *(any text)* | Submits the text as an AI coding task |
| `/status <task_id>` | Check the status of a specific task by its short or full ID |
| `/tasks` | List your 5 most recent tasks |
| `/help` | Show the help message |
| `/start` | Show the help message (also sent when first starting the bot) |

---

## Setting Up Discord

The Discord integration uses a **push model**: your agent_bridge Discord bot component (or a simple Discord gateway bot) forwards messages to the `/discord/webhook` endpoint.

### 1. Create a Discord application and bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and give it a name.
3. Go to the **Bot** section and click **Add Bot**.
4. Under **Token**, click **Reset Token** and copy the token.
5. Set `DISCORD_BOT_TOKEN` in your `.env`.

### 2. Configure bot permissions

The bot needs the following permissions:
- **Send Messages** — to deliver AI results
- **Read Message History** — optional, for context
- **View Channels** — to see channels it's added to

In the **OAuth2 → URL Generator**, select:
- Scopes: `bot`
- Bot Permissions: `Send Messages`, `View Channels`

Copy the generated URL and open it to invite the bot to your server.

### 3. Configure message forwarding

Since Discord uses a gateway (WebSocket) rather than webhooks for incoming messages, you need a small adapter bot to forward messages to agent_bridge. A minimal example:

```python
import discord
import httpx

client = discord.Client(intents=discord.Intents.default())
AGENT_BRIDGE_URL = "http://localhost:8000"
SECRET_TOKEN = "your-secret-here"

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    payload = {
        "message": {
            "id": str(message.id),
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else None,
            "author": {
                "id": str(message.author.id),
                "username": message.author.name,
                "discriminator": message.author.discriminator,
                "bot": message.author.bot,
            },
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
        }
    }
    
    async with httpx.AsyncClient() as http:
        await http.post(
            f"{AGENT_BRIDGE_URL}/discord/webhook",
            json=payload,
            headers={"X-Secret-Token": SECRET_TOKEN},
        )

client.run("your-discord-bot-token")
```

### 4. Available Discord commands

| Command | Description |
|---|---|
| *(any message)* | Submits the message as an AI coding task |
| `/status <task_id>` | Check the status of a specific task |
| `/tasks` | List the 5 most recent tasks for this channel |
| `/help` | Show the help message |

---

## AI Backend Configuration

### Ollama (local, recommended for privacy)

```bash
# Install Ollama: https://ollama.com/
ollama serve

# Pull a model
ollama pull llama3
ollama pull codellama
ollama pull deepseek-coder
```

```dotenv
AI_BACKEND_TYPE=ollama
AI_BACKEND_URL=http://localhost:11434
AI_MODEL=llama3
AI_TEMPERATURE=0.2
AI_MAX_TOKENS=4096
```

**Recommended models for coding tasks:**
- `codellama` — Meta's code-focused model
- `deepseek-coder` — Excellent at code generation
- `llama3` — Good general-purpose model
- `mistral` — Fast and capable

### OpenAI

```dotenv
AI_BACKEND_TYPE=openai
AI_BACKEND_URL=https://api.openai.com
AI_MODEL=gpt-4o
AI_API_KEY=sk-your-openai-api-key
AI_TEMPERATURE=0.2
AI_MAX_TOKENS=4096
```

### Azure OpenAI

```dotenv
AI_BACKEND_TYPE=openai
AI_BACKEND_URL=https://your-resource.openai.azure.com
AI_MODEL=gpt-4o
AI_API_KEY=your-azure-api-key
```

### Groq (fast inference)

```dotenv
AI_BACKEND_TYPE=openai
AI_BACKEND_URL=https://api.groq.com/openai
AI_MODEL=llama3-70b-8192
AI_API_KEY=your-groq-api-key
```

### Together AI

```dotenv
AI_BACKEND_TYPE=openai
AI_BACKEND_URL=https://api.together.xyz
AI_MODEL=codellama/CodeLlama-34b-Instruct-hf
AI_API_KEY=your-together-api-key
```

### Custom system prompt

You can customise the system prompt to tune the AI's behaviour:

```dotenv
AI_SYSTEM_PROMPT=You are an expert Python developer. Always write clean, well-documented, PEP 8-compliant code with type hints. Include docstrings and error handling.
```

---

## API Reference

The full interactive API documentation is available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc) when the server is running.

### Authentication

All endpoints except `/health` and `/telegram/webhook` require the `X-Secret-Token` header:

```bash
curl -H "X-Secret-Token: your-secret" http://localhost:8000/tasks
```

> **Note:** `/telegram/webhook` is authenticated by Telegram itself and does not require the secret token header. The `/discord/webhook` endpoint does require the header.

### Endpoints

#### `GET /health`

Returns server health status and version.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "0.1.0"}
```

#### `POST /telegram/webhook`

Receives Telegram Bot API update payloads. Called automatically by Telegram's servers.

#### `POST /discord/webhook`

Receives Discord message payloads forwarded by your Discord bot adapter.

**Request body:**
```json
{
  "message": {
    "id": "1234567890",
    "channel_id": "9876543210",
    "guild_id": "1111111111",
    "author": {
      "id": "4444444444",
      "username": "developer",
      "discriminator": "0",
      "bot": false
    },
    "content": "Write a merge sort in Python",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### `GET /tasks`

List tasks with optional filtering and pagination.

```bash
# All tasks
curl -H "X-Secret-Token: secret" http://localhost:8000/tasks

# Filter by status
curl -H "X-Secret-Token: secret" "http://localhost:8000/tasks?status=done"

# Filter by platform
curl -H "X-Secret-Token: secret" "http://localhost:8000/tasks?platform=telegram"

# Pagination
curl -H "X-Secret-Token: secret" "http://localhost:8000/tasks?limit=10&offset=20"
```

**Query parameters:**
- `status` — Filter by: `pending`, `running`, `done`, `failed`
- `platform` — Filter by: `telegram`, `discord`
- `chat_id` — Filter by chat/channel ID
- `user_id` — Filter by user ID
- `limit` — Maximum results (1–200, default 50)
- `offset` — Pagination offset (default 0)

#### `GET /tasks/{task_id}`

Retrieve a single task by UUID.

```bash
curl -H "X-Secret-Token: secret" \
  http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000
```

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "platform": "telegram",
  "chat_id": "123456",
  "user_id": "789012",
  "prompt": "Write a binary search implementation",
  "status": "done",
  "result": "def binary_search(arr, target): ...",
  "error": null,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:45Z"
}
```

#### `DELETE /tasks/{task_id}`

Delete a task record.

```bash
curl -X DELETE -H "X-Secret-Token: secret" \
  http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000
```

Returns `204 No Content` on success, `404` if not found.

#### `POST /tasks/{task_id}/deliver`

Manually re-deliver a completed or failed task result to the originating chat.

Useful if the initial delivery failed (bot was offline, network error, etc.).

```bash
curl -X POST -H "X-Secret-Token: secret" \
  http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/deliver
```

- Returns `200` with `{"ok": true}` on success
- Returns `404` if task not found
- Returns `409` if task is still `pending` or `running`
- Returns `502` if the messenger fails to deliver

#### `POST /tasks/{task_id}/callback`

Manually trigger the external webhook callback for a completed task.

Requires `CALLBACK_URL` to be configured.

```bash
curl -X POST -H "X-Secret-Token: secret" \
  http://localhost:8000/tasks/550e8400-e29b-41d4-a716-446655440000/callback
```

- Returns `503` if `CALLBACK_URL` is not configured
- Returns `404` if task not found
- Returns `409` if task is not in a terminal state

### Webhook Callback Payload

When `CALLBACK_URL` is configured, agent_bridge POSTs the following JSON payload on task completion:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "platform": "telegram",
  "chat_id": "123456",
  "user_id": "789012",
  "status": "done",
  "result": "def binary_search(arr, target): ...",
  "error": null,
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:00:45Z"
}
```

If `CALLBACK_SECRET` is set, the request includes an `X-Callback-Secret` header for verification.

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_queue.py -v

# Run with coverage
pytest --cov=agent_bridge --cov-report=term-missing

# Run tests matching a pattern
pytest -k "test_telegram" -v
```

### Test structure

| File | Coverage |
|---|---|
| `tests/test_config.py` | Settings loading, validation, defaults |
| `tests/test_models.py` | Pydantic model validation |
| `tests/test_db.py` | SQLite CRUD operations |
| `tests/test_ai_client.py` | AI client HTTP communication (mocked) |
| `tests/test_queue.py` | Async task queue, concurrency, callbacks |
| `tests/test_messenger.py` | Message formatting and delivery |
| `tests/test_routes.py` | HTTP endpoint integration tests |

---

## Deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY agent_bridge/ ./agent_bridge/

EXPOSE 8000

CMD ["uvicorn", "agent_bridge.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  agent_bridge:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN}
      - SECRET_TOKEN=${SECRET_TOKEN}
      - AI_BACKEND_TYPE=ollama
      - AI_BACKEND_URL=http://ollama:11434
      - AI_MODEL=llama3
      - DATABASE_URL=/data/agent_bridge.db
      - LOG_LEVEL=info
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

Run:
```bash
docker-compose up -d

# Pull the AI model inside the ollama container
docker-compose exec ollama ollama pull llama3
```

### systemd (Linux)

Create `/etc/systemd/system/agent_bridge.service`:

```ini
[Unit]
Description=agent_bridge AI messaging bridge
After=network.target

[Service]
Type=exec
User=www-data
WorkingDirectory=/opt/agent_bridge
EnvironmentFile=/opt/agent_bridge/.env
ExecStart=/opt/agent_bridge/venv/bin/uvicorn agent_bridge.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable agent_bridge
sudo systemctl start agent_bridge
sudo systemctl status agent_bridge
```

### Nginx reverse proxy

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeout for long AI responses
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### Production checklist

- [ ] Set a strong `SECRET_TOKEN` (at least 32 random characters)
- [ ] Set `LOG_LEVEL=warning` or `LOG_LEVEL=error` to reduce log volume
- [ ] Use a persistent `DATABASE_URL` path (not `:memory:`)
- [ ] Configure `TELEGRAM_ALLOWED_USERS` and/or `DISCORD_ALLOWED_USERS` to restrict access
- [ ] Use HTTPS for all webhook URLs
- [ ] Set up log rotation for uvicorn logs
- [ ] Monitor disk space (SQLite database grows with task history)
- [ ] Consider periodic cleanup: `DELETE /tasks` for old `done`/`failed` tasks

---

## Error Handling

### User authentication errors

When a user not in the allowlist sends a message, agent_bridge:

1. Logs a warning with the user's ID
2. Sends a friendly "not authorised" message back to the user
3. Returns `{"ok": true, "ignored": true}` to the webhook caller
4. Does **not** create a task record

To add a user to the allowlist, update `TELEGRAM_ALLOWED_USERS` or `DISCORD_ALLOWED_USERS` and restart the server (or send a `SIGHUP` if hot-reload is enabled).

### AI backend failures

When the AI backend is unavailable or returns an error:

| Error Type | Behaviour |
|---|---|
| Connection refused | Task marked `failed` with `"AI backend error: Cannot connect..."` |
| HTTP 4xx/5xx | Task marked `failed` with the HTTP status code in the error |
| Timeout | Task marked `failed` with `"AI backend error: request timed out"` |
| Task timeout | Task marked `failed` with `"Task timed out after Xs"` |

In all failure cases:
- The task record in SQLite is updated to `failed` status with an `error` field
- The messenger attempts to deliver the error message back to the user
- Callbacks (messenger + external URL) are still fired
- The queue continues processing subsequent tasks normally

### Messenger delivery errors

If the messenger fails to deliver a result (e.g. the bot was kicked, network error):

- The error is logged
- The task status in the database is **not** changed (it remains `done` or `failed`)
- You can manually re-deliver using `POST /tasks/{id}/deliver`
- The queue continues processing normally (messenger errors are non-fatal)

### Task status transitions

```
pending ──▶ running ──▶ done
                 └──▶ failed
```

Only `done` and `failed` are terminal states. Tasks cannot transition backward.

### Handling long-running tasks

If a task exceeds `TASK_TIMEOUT_SECONDS`, it is forcibly cancelled and marked `failed`. Increase this value for models that take a long time:

```dotenv
TASK_TIMEOUT_SECONDS=600  # 10 minutes
```

### Rate limiting

There is currently no built-in rate limiting. For production use, consider adding nginx-based rate limiting or a FastAPI middleware. The allowlist feature provides basic access control.

---

## Troubleshooting

### Telegram webhook not receiving messages

```bash
# Check webhook info
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

Look for:
- `url` — should be your server URL
- `last_error_message` — describes any delivery errors
- `pending_update_count` — messages waiting to be delivered

Common fixes:
- Ensure your server URL is publicly accessible over HTTPS
- Check your server's TLS certificate is valid
- Verify the server is running: `curl https://yourdomain.com/health`

### Tasks stuck in `pending` status

This usually means the task queue workers are not running. Check:

```bash
# Server logs
journalctl -u agent_bridge -f

# Verify health
curl http://localhost:8000/health
```

If the server restarted while tasks were `running`, they will remain in `running` state. You can manually clean these up:

```bash
# List stuck running tasks
curl -H "X-Secret-Token: secret" "http://localhost:8000/tasks?status=running"
```

### AI backend connection errors

```bash
# Test Ollama directly
curl http://localhost:11434/api/chat \
  -d '{"model":"llama3","messages":[{"role":"user","content":"hello"}],"stream":false}'

# Test OpenAI
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer ${AI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}'
```

### Database errors

Check the database file permissions:

```bash
ls -la agent_bridge.db
# Should be readable/writable by the server process
chmod 644 agent_bridge.db
```

For corruption issues:
```bash
sqlite3 agent_bridge.db "PRAGMA integrity_check;"
```

### Debug mode

Set `LOG_LEVEL=debug` in your `.env` for detailed logging of all requests, AI payloads, and queue operations:

```dotenv
LOG_LEVEL=debug
```

> ⚠️ Debug mode logs may include prompt content. Do not use in production with sensitive data.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Check code style: `ruff check .`
6. Submit a pull request

### Development setup

```bash
git clone https://github.com/yourorg/agent_bridge.git
cd agent_bridge
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check .
ruff format --check .
```

### Project structure

```
agent_bridge/
├── __init__.py        # Package init, version
├── config.py          # Environment-based configuration (pydantic-settings)
├── models.py          # Pydantic models for payloads and records
├── db.py              # SQLite task store (aiosqlite)
├── ai_client.py       # Async AI backend client (httpx)
├── queue.py           # Async task queue with concurrency control
├── messenger.py       # Telegram/Discord message delivery
├── routes.py          # FastAPI route handlers
└── app.py             # Application factory and lifespan management

tests/
├── test_config.py
├── test_models.py
├── test_db.py
├── test_ai_client.py
├── test_queue.py
├── test_messenger.py
└── test_routes.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) — The web framework powering the API
- [python-telegram-bot](https://python-telegram-bot.org/) — Telegram Bot API client
- [discord.py](https://discordpy.readthedocs.io/) — Discord API client
- [Ollama](https://ollama.com/) — Local LLM runtime
- [httpx](https://www.python-httpx.org/) — Async HTTP client
- [aiosqlite](https://aiosqlite.omnilib.dev/) — Async SQLite wrapper
- [pydantic](https://docs.pydantic.dev/) — Data validation

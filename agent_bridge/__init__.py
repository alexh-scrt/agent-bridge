"""agent_bridge: A self-hosted messaging bridge connecting Telegram and Discord to a local AI coding assistant.

This package provides a FastAPI-based server that:
- Ingests messages from Telegram and Discord webhooks
- Queues and executes AI coding prompts asynchronously
- Delivers results back to the originating chat channel
- Maintains a SQLite-backed task history
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("agent_bridge")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__: list[str] = ["__version__"]

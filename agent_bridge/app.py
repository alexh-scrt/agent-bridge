"""FastAPI application factory for agent_bridge.

Provides the :func:`create_app` factory function that wires together all
application components:

- Settings loading
- Database initialisation and teardown
- Task queue start and graceful stop
- Messenger setup
- Router registration
- Lifespan management

Also exposes a :func:`main` entry point for running the server directly
via ``python -m agent_bridge.app`` or the ``agent-bridge`` CLI script.

Typical usage::

    from agent_bridge.app import create_app
    app = create_app()
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_bridge import __version__
from agent_bridge.config import Settings, get_settings
from agent_bridge.db import Database
from agent_bridge.messenger import Messenger
from agent_bridge.models import ErrorResponse
from agent_bridge.queue import TaskQueue
from agent_bridge.routes import router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan handler
# ---------------------------------------------------------------------------


def _build_lifespan(settings: Settings, testing: bool = False):
    """Construct an async lifespan context manager for the FastAPI application.

    Handles startup (database init, queue start) and shutdown (queue stop,
    database close) in the correct order.

    Args:
        settings: Application settings used to configure all components.
        testing: If ``True``, skips token verification on incoming requests
            and uses the provided settings without re-reading the environment.

    Returns:
        An async context manager suitable for ``FastAPI(lifespan=...)``,
        taking the app instance as its argument.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Async lifespan context manager for startup and shutdown."""
        # ---------------------------------------------------------------- #
        # STARTUP
        # ---------------------------------------------------------------- #
        logger.info("agent_bridge v%s starting up", __version__)

        # Initialise the database
        db = Database(settings.database_url)
        await db.initialize()
        logger.info("Database ready at '%s'", settings.database_url)

        # Initialise the messenger
        messenger = Messenger(settings)

        # Build the messenger callback for the queue
        async def messenger_callback(record: Any) -> None:
            try:
                await messenger.send(record)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Messenger failed to deliver result for task %s: %s",
                    record.id,
                    exc,
                )

        # Initialise and start the task queue
        queue = TaskQueue(
            settings=settings,
            db=db,
            messenger=messenger_callback,
        )
        await queue.start()
        logger.info(
            "Task queue started (workers=%d, max_concurrent=%d)",
            settings.max_concurrent_tasks,
            settings.max_concurrent_tasks,
        )

        # Store components in app state for access by route handlers
        app.state.settings = settings
        app.state.db = db
        app.state.queue = queue
        app.state.messenger = messenger
        app.state.testing = testing

        logger.info(
            "Telegram integration: %s",
            "enabled" if settings.telegram_enabled else "disabled",
        )
        logger.info(
            "Discord integration: %s",
            "enabled" if settings.discord_enabled else "disabled",
        )
        logger.info(
            "AI backend: %s at %s (model=%s)",
            settings.ai_backend_type,
            settings.ai_backend_url,
            settings.ai_model,
        )

        yield  # Application is running

        # ---------------------------------------------------------------- #
        # SHUTDOWN
        # ---------------------------------------------------------------- #
        logger.info("agent_bridge shutting down gracefully")

        await queue.stop(timeout=30.0)
        logger.info("Task queue stopped")

        await db.close()
        logger.info("Database connection closed")

        logger.info("agent_bridge shutdown complete")

    return lifespan


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    testing: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    This factory wires together all components and registers all routes. It
    is the primary entry point for both production use and testing.

    Args:
        settings: Optional pre-constructed :class:`~agent_bridge.config.Settings`
            instance. If ``None``, settings are loaded from the environment
            via :func:`~agent_bridge.config.get_settings`.
        testing: If ``True``, the application runs in testing mode, which
            disables secret-token verification on incoming requests and
            suppresses some warnings.

    Returns:
        A fully configured :class:`fastapi.FastAPI` application instance.

    Example::

        app = create_app()  # production
        app = create_app(settings=test_settings, testing=True)  # test
    """
    if settings is None:
        if testing:
            os.environ.setdefault("AGENT_BRIDGE_TESTING", "1")
        settings = get_settings()

    # ------------------------------------------------------------------
    # FastAPI app configuration
    # ------------------------------------------------------------------
    app = FastAPI(
        title="agent_bridge",
        description=(
            "A self-hosted messaging bridge that connects Telegram and Discord "
            "to a local AI coding assistant. Send natural language instructions "
            "from your phone and receive completed code asynchronously."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_build_lifespan(settings, testing=testing),
    )

    # ------------------------------------------------------------------
    # CORS middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production as appropriate
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Global exception handlers
    # ------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all exception handler that returns a structured error response."""
        logger.exception(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                detail="An unexpected error occurred. Please try again later.",
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        """Handle ValueError as a 400 Bad Request."""
        logger.warning(
            "ValueError on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="bad_request",
                detail=str(exc),
            ).model_dump(),
        )

    # ------------------------------------------------------------------
    # Router registration
    # ------------------------------------------------------------------
    app.include_router(router)

    logger.info("agent_bridge application created (testing=%s)", testing)
    return app


# ---------------------------------------------------------------------------
# Convenience: default app instance
# ---------------------------------------------------------------------------

# This module-level ``app`` instance is used by uvicorn when running:
#   uvicorn agent_bridge.app:app
#
# It is created lazily so that importing this module in tests does NOT
# immediately read environment variables or open database connections.
_app_instance: FastAPI | None = None


def _get_default_app() -> FastAPI:
    """Return (or lazily create) the default application instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


# Expose the app at module level for uvicorn
app = _get_default_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the agent_bridge server using uvicorn.

    This function is the entry point for the ``agent-bridge`` CLI script
    defined in ``pyproject.toml``. It reads host, port, and log level from
    the application settings.

    Example usage::

        agent-bridge
        # or:
        python -m agent_bridge.app
    """
    import uvicorn  # type: ignore[import]

    srv_settings = get_settings()

    logging.basicConfig(
        level=srv_settings.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Starting agent_bridge on %s:%d (log_level=%s)",
        srv_settings.host,
        srv_settings.port,
        srv_settings.log_level,
    )

    uvicorn.run(
        "agent_bridge.app:app",
        host=srv_settings.host,
        port=srv_settings.port,
        log_level=srv_settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()

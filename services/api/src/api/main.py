"""Точка входа FastAPI-приложения StockLens API."""

from fastapi import FastAPI

from api.core.lifespan import lifespan
from api.core.logging import configure_logging
from api.core.middleware import RequestIdMiddleware
from api.core.problem import install_exception_handlers
from api.core.settings import ApiSettings
from api.routers import bot, data, health, monitoring, portfolio


def create_app() -> FastAPI:
    """Создать и настроить экземпляр FastAPI.

    Используется как factory для uvicorn (--factory флаг).
    """
    settings = ApiSettings.model_validate({})
    configure_logging(pretty=settings.log_pretty)

    app = FastAPI(
        title="StockLens API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.state.settings = settings

    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(data.router)
    app.include_router(monitoring.router)
    app.include_router(portfolio.router)
    app.include_router(bot.router)

    return app

"""Точка входа FastAPI-приложения StockLens API."""

from fastapi import Depends, FastAPI

from api.core.auth.deps import require_auth
from api.core.auth.settings import AuthSettings
from api.core.lifespan import lifespan
from api.core.logging import configure_logging
from api.core.middleware import RequestIdMiddleware
from api.core.problem import install_exception_handlers
from api.core.settings import ApiSettings
from api.routers import auth as auth_router
from api.routers import bot, data, health, monitoring, portfolio, watchlist


def create_app() -> FastAPI:
    """Создать и настроить экземпляр FastAPI.

    Используется как factory для uvicorn (--factory флаг).
    """
    settings = ApiSettings.model_validate({})
    auth_settings = AuthSettings.model_validate({})
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
    app.state.auth_settings = auth_settings

    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth_router.router)

    _auth_dep = [Depends(require_auth)]
    app.include_router(data.router, dependencies=_auth_dep)
    app.include_router(monitoring.router, dependencies=_auth_dep)
    app.include_router(portfolio.router, dependencies=_auth_dep)
    app.include_router(watchlist.router, dependencies=_auth_dep)
    app.include_router(bot.router, dependencies=_auth_dep)

    return app

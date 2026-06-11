"""Жизненный цикл FastAPI-приложения: инициализация и завершение ресурсов.

Порядок старта: configure engine → wait_for_schema → init redis → yield → cleanup.
SchemaNotReadyError на старте завершает процесс с ненулевым кодом — Docker restart policy
перезапустит контейнер, пока миграции не применятся.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from api.core.exceptions import SchemaNotReadyError
from api.core.settings import ApiSettings

logger = structlog.get_logger(__name__)


async def _wait_for_schema(
    engine: AsyncEngine,
    attempts: int,
    interval: float,
) -> None:
    """Ожидать готовности схемы БД (наличие строки в alembic_version).

    Поднимает SchemaNotReadyError если схема не появилась за отведённые попытки.
    """
    for attempt in range(1, attempts + 1):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT version_num FROM alembic_version"))
            logger.info("schema_ready", attempt=attempt)
            return
        except Exception as exc:
            logger.warning(
                "schema_not_ready",
                attempt=attempt,
                max_attempts=attempts,
                reason=str(exc),
            )
            if attempt < attempts:
                await asyncio.sleep(interval)

    raise SchemaNotReadyError(attempts)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Инициализировать и освобождать ресурсы приложения.

    Хранит engine, session_factory и redis-клиент в app.state для доступа через DI.
    """
    settings: ApiSettings = app.state.settings

    engine = create_async_engine(
        str(settings.database_url_async),
        pool_pre_ping=True,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    await _wait_for_schema(
        engine,
        attempts=settings.schema_wait_attempts,
        interval=settings.schema_wait_interval_seconds,
    )

    # Стабы redis-py не параметризуют from_url по decode_responses — присваиваем через app.state.
    app.state.redis = Redis.from_url(str(settings.redis_url), decode_responses=True)
    app.state.session_factory = session_factory

    logger.info("api_started")
    yield

    await app.state.redis.aclose()
    await engine.dispose()
    logger.info("api_stopped")

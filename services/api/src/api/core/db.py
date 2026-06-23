"""Зависимости FastAPI для получения AsyncSession и Redis-клиента из app.state."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.cache import RedisClientProtocol
from api.core.settings import ApiSettings


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Открыть AsyncSession из session_factory, хранящегося в app.state."""
    async with request.app.state.session_factory() as session:
        yield session


def get_redis(request: Request) -> RedisClientProtocol:
    """Получить Redis-клиент из app.state."""
    client: RedisClientProtocol = request.app.state.redis
    return client


def get_api_settings(request: Request) -> ApiSettings:
    """Получить настройки сервиса из app.state."""
    settings: ApiSettings = request.app.state.settings
    return settings


SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[RedisClientProtocol, Depends(get_redis)]
SettingsDep = Annotated[ApiSettings, Depends(get_api_settings)]

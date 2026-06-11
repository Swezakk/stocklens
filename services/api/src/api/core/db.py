"""Зависимости FastAPI для получения AsyncSession и Redis-клиента из app.state."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.cache import RedisClientProtocol


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Открыть AsyncSession из session_factory, хранящегося в app.state."""
    async with request.app.state.session_factory() as session:
        yield session


def get_redis(request: Request) -> RedisClientProtocol:
    """Получить Redis-клиент из app.state."""
    client: RedisClientProtocol = request.app.state.redis
    return client


SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[RedisClientProtocol, Depends(get_redis)]

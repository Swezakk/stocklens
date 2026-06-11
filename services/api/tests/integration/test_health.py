"""Integration-тесты health-эндпоинтов."""

from collections.abc import AsyncGenerator

import pytest
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.integration


async def test_health_live_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_health_ready_returns_200_when_db_and_redis_up(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "ok"
    assert body["status"] in ("ready", "degraded")


class _BrokenRedis:
    """Redis-клиент, который падает на каждый вызов — имитирует недоступность."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("Redis down")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise ConnectionError("Redis down")

    async def ping(self) -> bool:
        raise ConnectionError("Redis down")

    async def aclose(self) -> None:
        pass


async def test_health_ready_returns_degraded_when_redis_down(
    db_session: AsyncSession,
    test_settings: ApiSettings,
) -> None:
    broken_redis: RedisClientProtocol = _BrokenRedis()

    app2 = create_app()
    app2.state.settings = test_settings

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_redis() -> RedisClientProtocol:
        return broken_redis

    app2.dependency_overrides[get_session] = override_session
    app2.dependency_overrides[get_redis] = override_redis

    async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["cache"] == "degraded"

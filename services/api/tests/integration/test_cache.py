"""Регрессия кэша свечей: данные реально сохраняются в Redis (round-trip).

Раньше репозиторий кэшировал ORM-объекты Candle, которые json не сериализует —
set_json молча проглатывал TypeError, и кэш никогда не заполнялся.
"""

import pytest
from api.core.cache import RedisClientProtocol
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.seed import seed_candle, seed_security

pytestmark = pytest.mark.integration


async def test_candles_are_cached_and_roundtrip(
    client: AsyncClient,
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
) -> None:
    """После первого запроса свечи лежат в кэше, второй запрос отдаёт те же данные."""
    security = await seed_security(db_session)
    await seed_candle(db_session, security_id=security.id)

    first = await client.get("/api/v1/data/candles", params={"ticker": "SBER"})
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["total"] == 1
    assert first_body["items"][0]["close"] == "283.000000"

    # Ключ строится из security_id и дефолтов пагинации (limit=50, offset=0).
    cache_key = f"candles:{security.id}:None:None:50:0"
    cached_raw = await real_redis_client.get(cache_key)
    assert cached_raw is not None, "Свечи не попали в кэш — сериализация round-trip сломана"

    second = await client.get("/api/v1/data/candles", params={"ticker": "SBER"})
    assert second.status_code == 200
    assert second.json() == first_body

"""Реализация CandleRepository с кэшированием через RedisCache.

Кэшируются и возвращаются DTO (CandleOut), а не ORM-объекты: ORM не сериализуется
в JSON, поэтому кэш-слой работает только с сериализуемым представлением.
"""

from datetime import date
from typing import cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Candle

from api.core.cache import RedisCache
from api.schemas.market import CandleOut


class SqlCandleRepository:
    """Читает свечи из PostgreSQL; результаты кэшируются в Redis как DTO."""

    def __init__(self, session: AsyncSession, cache: RedisCache, ttl: int) -> None:
        self._session = session
        self._cache = cache
        self._ttl = ttl

    async def list_candles(
        self,
        security_id: int,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CandleOut], int]:
        """Вернуть страницу свечей (DTO) и общее число. Результат кэшируется по ключу."""
        cache_key = f"candles:{security_id}:{date_from}:{date_to}:{limit}:{offset}"

        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            payload = cast(dict[str, object], cached)
            raw_items = cast(list[dict[str, object]], payload["items"])
            items = [CandleOut.model_validate(item) for item in raw_items]
            return items, cast(int, payload["total"])

        candles, total = await self._fetch_from_db(security_id, date_from, date_to, limit, offset)
        items = [CandleOut.model_validate(candle) for candle in candles]
        await self._cache.set_json(
            cache_key,
            {"items": [item.model_dump(mode="json") for item in items], "total": total},
            self._ttl,
        )
        return items, total

    async def _fetch_from_db(
        self,
        security_id: int,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Candle], int]:
        base_query = select(Candle).where(Candle.security_id == security_id)
        if date_from is not None:
            base_query = base_query.where(Candle.trade_date >= date_from)
        if date_to is not None:
            base_query = base_query.where(Candle.trade_date <= date_to)

        count_result = await self._session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total: int = count_result.scalar_one()

        rows_result = await self._session.execute(
            base_query.order_by(Candle.trade_date.desc()).limit(limit).offset(offset)
        )
        candles = list(rows_result.scalars().all())
        return candles, total

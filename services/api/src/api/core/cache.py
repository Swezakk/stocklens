"""Обёртка над Redis для кэширования JSON-значений.

При недоступности Redis логирует предупреждение и прозрачно обращается к фабрике —
кэш не является единственной точкой отказа для HTTP-запроса.
"""

import json
from collections.abc import Awaitable, Callable
from datetime import date
from decimal import Decimal
from typing import Protocol, TypeVar, cast

import structlog

T = TypeVar("T")

logger = structlog.get_logger(__name__)


class RedisClientProtocol(Protocol):
    """Минимальный интерфейс Redis-клиента для API-сервиса.

    decode_responses=True на стороне клиента гарантирует, что get() возвращает str | None.
    """

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...

    async def incr(self, key: str) -> int: ...

    async def expire(self, key: str, seconds: int) -> None: ...

    async def ping(self) -> bool: ...

    async def aclose(self) -> None: ...


def _default_encoder(obj: object) -> str:
    """Decimal и date сериализуются как строки — достаточно для read-only кэша."""
    if isinstance(obj, Decimal | date):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class RedisCache:
    """Кэш на базе Redis с безопасным падением (cache fallthrough on error)."""

    def __init__(self, client: RedisClientProtocol) -> None:
        self._client = client

    async def get_json(self, key: str) -> object | None:
        """Получить объект из кэша. Возвращает None при промахе или ошибке Redis."""
        try:
            raw: str | None = await self._client.get(key)
            if raw is None:
                return None
            return cast(object, json.loads(raw))
        except Exception:
            logger.warning("cache_unavailable", operation="get", key=key)
            return None

    async def set_json(self, key: str, value: object, ttl: int) -> None:
        """Сохранить объект в кэш с TTL. При ошибке Redis — silent fallthrough."""
        try:
            serialized = json.dumps(value, default=_default_encoder)
            await self._client.set(key, serialized, ex=ttl)
        except Exception:
            logger.warning("cache_unavailable", operation="set", key=key)

    async def get_or_set(
        self,
        key: str,
        ttl: int,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Вернуть значение из кэша или вычислить через factory и сохранить.

        При любой ошибке Redis вызывает factory напрямую (кэш — оптимизация, не SPoF).
        Данные из кэша приводятся к T через cast: граница I/O, caller несёт ответственность
        за совместимость T с тем, что было сохранено через set_json.
        """
        cached = await self.get_json(key)
        if cached is not None:
            return cast(T, cached)
        result = await factory()
        await self.set_json(key, result, ttl)
        return result

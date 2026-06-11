"""Unit-тесты для RedisCache — проверяют fallthrough при ошибке Redis."""

from api.core.cache import RedisCache, RedisClientProtocol


class _WorkingRedis:
    """Имитация Redis, которая работает нормально. Реализует RedisClientProtocol."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


class _BrokenRedis:
    """Имитация Redis, которая бросает исключение на каждый вызов. Реализует RedisClientProtocol."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("Redis недоступен")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise ConnectionError("Redis недоступен")

    async def ping(self) -> bool:
        raise ConnectionError("Redis недоступен")

    async def aclose(self) -> None:
        pass


def _working() -> RedisCache:
    client: RedisClientProtocol = _WorkingRedis()
    return RedisCache(client)


def _broken() -> RedisCache:
    client: RedisClientProtocol = _BrokenRedis()
    return RedisCache(client)


async def test_cache_get_or_set_stores_and_retrieves_value() -> None:
    cache = _working()
    call_count = 0

    async def factory() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {"value": 42}

    result1 = await cache.get_or_set("key1", ttl=60, factory=factory)
    result2 = await cache.get_or_set("key1", ttl=60, factory=factory)

    assert result1 == {"value": 42}
    assert result2 == {"value": 42}
    assert call_count == 1, "factory должна вызываться только при промахе кэша"


async def test_cache_falls_through_to_factory_on_redis_error() -> None:
    cache = _broken()
    call_count = 0

    async def factory() -> str:
        nonlocal call_count
        call_count += 1
        return "result"

    result = await cache.get_or_set("key2", ttl=60, factory=factory)

    assert result == "result"
    assert call_count == 1, "factory должна вызываться при ошибке Redis"


async def test_cache_get_json_returns_none_on_redis_error() -> None:
    result = await _broken().get_json("any_key")
    assert result is None


async def test_cache_set_json_silently_fails_on_redis_error() -> None:
    await _broken().set_json("key", {"data": 1}, ttl=60)

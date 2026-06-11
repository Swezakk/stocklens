"""Конфигурация API-сервиса через pydantic-settings.

Все параметры читаются из переменных окружения. Прямое обращение к os.environ запрещено.
"""

import functools

from pydantic import PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings


class ApiSettings(BaseSettings):
    """Параметры сервиса stocklens-api.

    Переменные окружения (без префикса):
    - DATABASE_URL_ASYNC  — asyncpg DSN (postgresql+asyncpg://...)
    - REDIS_URL           — Redis DSN (redis://...)
    - LOG_PRETTY          — True для ConsoleRenderer в разработке, False для JSON в проде
    - CACHE_TTL_CANDLES_SECONDS — TTL кэша свечей в Redis (сек.)
    - CACHE_TTL_DEFAULT_SECONDS — TTL кэша прочих ресурсов (сек.)
    - SCHEMA_WAIT_ATTEMPTS      — число попыток ожидания готовности схемы БД
    - SCHEMA_WAIT_INTERVAL_SECONDS — пауза между попытками (сек.)
    """

    database_url_async: PostgresDsn
    redis_url: RedisDsn
    log_pretty: bool = False
    cache_ttl_candles_seconds: int = 3600
    cache_ttl_default_seconds: int = 300
    schema_wait_attempts: int = 30
    schema_wait_interval_seconds: float = 2.0

    model_config = {
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",
    }


@functools.lru_cache(maxsize=1)
def get_settings() -> "ApiSettings":
    """Вернуть кэшированный экземпляр ApiSettings (читается из env один раз)."""
    return ApiSettings.model_validate({})

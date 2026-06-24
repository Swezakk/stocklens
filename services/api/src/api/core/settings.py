"""Конфигурация API-сервиса через pydantic-settings.

Все параметры читаются из переменных окружения. Прямое обращение к os.environ запрещено.
"""

import functools
from datetime import date

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
    - WATCHLIST_GRACE_SECONDS  — TTL «ожидания материализации» (сек.), после которого
                                  статус PENDING → NOT_FOUND если данных так и нет
    """

    database_url_async: PostgresDsn
    redis_url: RedisDsn
    log_pretty: bool = False
    cache_ttl_candles_seconds: int = 3600
    cache_ttl_default_seconds: int = 300
    schema_wait_attempts: int = 30
    schema_wait_interval_seconds: float = 2.0
    watchlist_grace_seconds: int = 3600

    # ML-serving (ml-spec §8.6). Загрузка моделей из реестра MLflow по алиасу.
    mlflow_tracking_uri: str = "http://mlflow:5000"
    ml_volatility_model: str = "stocklens-volatility"
    ml_trend_model: str = "stocklens-trend"
    ml_model_alias: str = "production"
    ml_required_for_ready: bool = True
    ml_load_attempts: int = 10
    ml_load_interval_seconds: float = 3.0
    # Окно истории для инференса = окно обучения (пост-2022, структурный разрыв; ml-spec D8).
    ml_train_start: date = date(2022, 4, 1)

    # Оценка режима волатильности (ml-spec §9).
    volatility_regime_quantile: float = 0.80
    volatility_regime_lookback: int = 252

    model_config = {
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",
    }


@functools.lru_cache(maxsize=1)
def get_settings() -> "ApiSettings":
    """Вернуть кэшированный экземпляр ApiSettings (читается из env один раз)."""
    return ApiSettings.model_validate({})

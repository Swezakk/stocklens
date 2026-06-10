"""Конфигурация пакета stocklens-core через переменные окружения.

env_file намеренно не задан — окружение инжектируется Docker Compose;
в тестах используется monkeypatch.setenv.
"""

from pydantic import PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Настройки подключений и параметры домена, читаемые из переменных окружения."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    database_url: PostgresDsn
    database_url_async: PostgresDsn
    redis_url: RedisDsn
    tickers_universe: str = "IMOEX"

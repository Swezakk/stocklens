"""Конфигурация ingestor через pydantic-settings.

Все параметры читаются из переменных окружения. Прямое обращение к os.environ запрещено.
"""

from pathlib import Path

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings


class IngestorSettings(BaseSettings):
    """Параметры сервиса-сборщика рыночных данных.

    Переменные окружения (префикс отсутствует):
    - DATABASE_URL — синхронный DSN PostgreSQL (postgresql+psycopg://...)
    - TICKERS_UNIVERSE — «IMOEX» (состав индекса) или запятая-список тикеров
    - LOG_PRETTY — True для ConsoleRenderer в разработке, False для JSON в проде
    - HEARTBEAT_PATH — путь к файлу-сигналу живости для healthcheck контейнера
    - SCHEMA_WAIT_ATTEMPTS — сколько раз ждать готовности схемы БД
    - SCHEMA_WAIT_INTERVAL_SECONDS — пауза между попытками ожидания схемы (сек.)
    """

    database_url: PostgresDsn
    tickers_universe: str = "IMOEX"
    log_pretty: bool = False
    heartbeat_path: Path = Path("/tmp/ingestor-heartbeat")
    schema_wait_attempts: int = 30
    schema_wait_interval_seconds: float = 5.0

    model_config = {
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",
    }

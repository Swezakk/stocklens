"""Ожидание готовности схемы БД перед стартом сборщиков.

depends_on в docker-compose гарантирует только старт контейнера миграций,
но не атомарность коммита схемы. Этот модуль добавляет явную проверку.
"""

import time
from collections.abc import Callable

import structlog
from sqlalchemy import Engine, text

from ingestor.exceptions import SchemaNotReadyError

log = structlog.get_logger(__name__)


def wait_for_schema(
    engine: Engine,
    attempts: int,
    interval: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Ожидать, пока alembic_version не появится в БД.

    Args:
        engine: Синхронный SQLAlchemy engine.
        attempts: Максимальное число попыток.
        interval: Пауза между попытками в секундах.
        sleep: Injectable функция паузы (тесты передают фейк без реального сна).

    Raises:
        SchemaNotReadyError: Если схема не стала готовой за все попытки.
    """
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT version_num FROM alembic_version"))
            log.info("schema_ready", attempt=attempt)
            return
        except Exception as exc:
            log.warning(
                "schema_not_ready",
                attempt=attempt,
                max_attempts=attempts,
                error=str(exc),
            )
            if attempt < attempts:
                sleep(interval)

    raise SchemaNotReadyError(
        f"Схема БД не готова после {attempts} попыток. Убедитесь, что миграции выполнены."
    )

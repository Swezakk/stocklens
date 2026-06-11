"""Точка входа ingestor: инициализация → ожидание схемы → backfill → планировщик."""

import sys

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ingestor.backfill import run_backfill
from ingestor.iss_client import MoexIssClient
from ingestor.logging_setup import configure_logging
from ingestor.scheduler import build_scheduler
from ingestor.schema_wait import wait_for_schema
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)


def main() -> None:
    """Запустить ingestor: настройка → БД → backfill → планировщик."""
    settings = IngestorSettings.model_validate({})
    configure_logging(settings.log_pretty)

    log.info("ingestor_starting", tickers_universe=settings.tickers_universe)

    engine = create_engine(str(settings.database_url), pool_pre_ping=True)
    session_factory: sessionmaker[Session] = sessionmaker(engine)

    try:
        wait_for_schema(
            engine,
            attempts=settings.schema_wait_attempts,
            interval=settings.schema_wait_interval_seconds,
        )
    except Exception:
        log.error("schema_wait_failed", exc_info=True)
        sys.exit(1)

    client = MoexIssClient()

    run_backfill(client, session_factory, settings)

    scheduler = build_scheduler(client, session_factory, settings)
    log.info("scheduler_starting")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("ingestor_shutdown")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()

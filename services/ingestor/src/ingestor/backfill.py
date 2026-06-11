"""Первоначальный backfill при старте ingestor.

При первом запуске тянет всю историю котировок и индекса.
При последующих — догоняет только пропуск от последней записи.
"""

import structlog
from sqlalchemy.orm import Session, sessionmaker

from ingestor.collectors.moex import run_all_collectors
from ingestor.iss_client import MoexIssClient
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)


def run_backfill(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Выполнить первоначальный или догоняющий backfill всех источников.

    Единый путь оркестрации с плановыми запусками — run_all_collectors:
    securities первым (FK candles → securities), сбой одного источника
    не прерывает остальные.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    log.info("backfill_started")
    run_all_collectors(client, session_factory, settings)
    log.info("backfill_finished")

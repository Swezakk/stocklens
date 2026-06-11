"""Первоначальный backfill при старте ingestor.

При первом запуске тянет всю историю котировок, индекса, курсов валют и ключевой ставки.
При последующих — догоняет только пропуск от последней записи.
Новости из RSS не восстанавливаются (нет архива) — только текущий фид при каждом запуске.
"""

import structlog
from sqlalchemy.orm import Session, sessionmaker

from ingestor.collectors.cbr import backfill_currency_rates, sync_key_rate
from ingestor.collectors.moex import run_all_collectors
from ingestor.iss_client import MoexIssClient
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)


def run_backfill(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> None:
    """Выполнить первоначальный или догоняющий backfill всех восстановимых источников.

    Порядок: MOEX-данные первыми (FK candles/dividends → securities),
    затем ЦБ. Сбой одного источника не прерывает остальные.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.
    """
    log.info("backfill_started")
    run_all_collectors(client, session_factory, settings)
    backfill_currency_rates(session_factory, settings)
    sync_key_rate(session_factory, settings)
    log.info("backfill_finished")

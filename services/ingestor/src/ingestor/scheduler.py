"""Планировщик задач ingestor на базе APScheduler 3.x (BlockingScheduler).

Расписание:
- 23:55 ежедневно — свечи и индекс (после закрытия торгов).
- 08:00 ежедневно — ценные бумаги, дивиденды, сплиты (корпоративные данные утром).
- каждые 60 секунд — heartbeat для healthcheck контейнера.

misfire_grace_time=300: если задача пропущена (перезапуск контейнера),
запускается с задержкой до 5 минут, иначе пропускается.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.orm import Session, sessionmaker

from ingestor import heartbeat
from ingestor.collectors.moex import (
    sync_candles,
    sync_dividends,
    sync_index,
    sync_securities,
    sync_splits,
)
from ingestor.iss_client import MoexIssClient
from ingestor.settings import IngestorSettings

_MISFIRE_GRACE_SECONDS = 300


def build_scheduler(
    client: MoexIssClient,
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
) -> BlockingScheduler:
    """Создать и настроить BlockingScheduler с заданиями ingestor.

    Args:
        client: Клиент MOEX ISS.
        session_factory: Фабрика синхронных сессий.
        settings: Конфигурация ingestor.

    Returns:
        Готовый к запуску планировщик (start() не вызван).
    """
    scheduler = BlockingScheduler(timezone="Europe/Moscow")

    scheduler.add_job(
        sync_candles,
        "cron",
        hour=23,
        minute=55,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="candles_daily",
    )

    scheduler.add_job(
        sync_index,
        "cron",
        hour=23,
        minute=55,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="index_daily",
    )

    scheduler.add_job(
        sync_securities,
        "cron",
        hour=8,
        minute=0,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="securities_daily",
    )

    scheduler.add_job(
        sync_dividends,
        "cron",
        hour=8,
        minute=0,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="dividends_daily",
    )

    scheduler.add_job(
        sync_splits,
        "cron",
        hour=8,
        minute=0,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="splits_daily",
    )

    scheduler.add_job(
        heartbeat.touch,
        "interval",
        seconds=60,
        kwargs={"path": settings.heartbeat_path},
        id="heartbeat",
    )

    return scheduler

"""Планировщик задач ingestor на базе APScheduler 3.x (BlockingScheduler).

Расписание:
- 10:00 ежедневно — утренний догон свечей и индекса (MOEX публикует дневную свечу
  предыдущего торгового дня утром следующего дня; без этого синка 23:55 отстаёт на сутки).
- 23:55 ежедневно — вечерний backstop свечей и индекса (на случай задержки публикации).
- 08:00 ежедневно — ценные бумаги, дивиденды, сплиты (корпоративные данные утром).
- 13:00 ежедневно — курсы валют и ключевая ставка ЦБ.
- каждые 30 минут — сбор новостей из RSS.
- каждые 60 секунд — heartbeat для healthcheck контейнера.

misfire_grace_time=300: если задача пропущена (перезапуск контейнера),
запускается с задержкой до 5 минут, иначе пропускается.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.orm import Session, sessionmaker

from ingestor import heartbeat
from ingestor.collectors.cbr import sync_currency_rates, sync_key_rate
from ingestor.collectors.moex import (
    sync_candles,
    sync_dividends,
    sync_index,
    sync_securities,
    sync_splits,
)
from ingestor.collectors.rss import sync_news
from ingestor.iss_client import MoexIssClient
from ingestor.sentiment import OnnxSentimentScorer
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
    scorer = OnnxSentimentScorer(
        model_dir=settings.sentiment_model_dir,
        model_id=settings.sentiment_model_id,
    )

    scheduler = BlockingScheduler(timezone="Europe/Moscow")

    # MOEX publishes the prior day's daily candle the next morning; 23:55 alone trails by a day.
    scheduler.add_job(
        sync_candles,
        "cron",
        hour=10,
        minute=0,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="candles_morning",
    )

    scheduler.add_job(
        sync_index,
        "cron",
        hour=10,
        minute=0,
        kwargs={"client": client, "session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="index_morning",
    )

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
        sync_currency_rates,
        "cron",
        hour=13,
        minute=0,
        kwargs={"session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="cbr_rates_daily",
    )

    scheduler.add_job(
        sync_key_rate,
        "cron",
        hour=13,
        minute=0,
        kwargs={"session_factory": session_factory, "settings": settings},
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="cbr_key_rate_daily",
    )

    scheduler.add_job(
        sync_news,
        "cron",
        minute="*/30",
        kwargs={
            "session_factory": session_factory,
            "settings": settings,
            "scorer": scorer,
        },
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        max_instances=1,
        id="rss_news_periodic",
    )

    scheduler.add_job(
        heartbeat.touch,
        "interval",
        seconds=60,
        kwargs={"path": settings.heartbeat_path},
        id="heartbeat",
    )

    return scheduler

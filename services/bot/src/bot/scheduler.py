"""Планировщик задач бота: sweep алертов + ежедневный дайджест (DESIGN §11).

Два задания:
- alert_sweep: каждые N минут → POST /bot/alerts/pending → format_alert + send_message_safe.
- digest: cron 08:30 МСК → claim_digest (once-per-day guard) → gather_digest → send.

AsyncIOScheduler.start() не блокирует — polling продолжает работать в том же event loop.
"""

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog
from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bot.api_client.client import ApiClient
from bot.api_client.errors import ApiError
from bot.formatting import format_alert
from bot.responses import digest_response
from bot.send import send_message_safe
from bot.settings import BotSettings

_log = structlog.get_logger()

_MOSCOW_TZ = ZoneInfo("Europe/Moscow")
_MISFIRE_GRACE_SECONDS = 300


def _now_moscow() -> datetime:
    """Текущее время в часовом поясе Москвы (инъектируется в тестах)."""
    return datetime.now(tz=_MOSCOW_TZ)


async def alert_sweep_job(*, bot: Bot, client: ApiClient) -> None:
    """Запросить сработавшие алерты и отправить каждый в соответствующий чат.

    Ошибка одного алерта (форматирование, отправка) не прерывает обработку остальных.
    """
    try:
        alerts = await client.get_pending_alerts()
    except ApiError as exc:
        _log.warning("alert_sweep_api_error", detail=exc.user_message)
        return
    except Exception:
        _log.exception("alert_sweep_unexpected_error")
        return

    for alert in alerts:
        try:
            html = format_alert(alert)
            await send_message_safe(bot, alert.chat_id, html)
        except Exception:
            _log.exception("alert_sweep_send_error", chat_id=alert.chat_id, ticker=alert.ticker)


async def digest_job(
    *,
    bot: Bot,
    client: ApiClient,
    digest_chat_id: int,
    clock: Callable[[], datetime] = _now_moscow,
) -> None:
    """Зарезервировать дайджест, собрать и отправить владельцу ровно один раз в день.

    claim_digest гарантирует однократность даже при рестарте бота или двойном срабатывании
    (coalesce=True в планировщике подавляет накопленные вызовы, но claim — страховка).
    """
    today = clock().date()
    try:
        claimed = await client.claim_digest(for_date=today)
    except ApiError as exc:
        _log.warning("digest_claim_api_error", detail=exc.user_message)
        return
    except Exception:
        _log.exception("digest_claim_unexpected_error")
        return

    if not claimed:
        _log.info("digest_already_sent", date=today.isoformat())
        return

    try:
        html = await digest_response(client, today)
        await send_message_safe(bot, digest_chat_id, html)
    except Exception:
        _log.exception("digest_gather_unexpected_error")


def build_scheduler(bot: Bot, client: ApiClient, settings: BotSettings) -> AsyncIOScheduler:
    """Создать AsyncIOScheduler с двумя заданиями бота.

    Возвращает планировщик без вызова start() — вызывающий код управляет жизненным циклом.
    """
    scheduler = AsyncIOScheduler(timezone=str(_MOSCOW_TZ))

    scheduler.add_job(
        alert_sweep_job,
        "interval",
        minutes=settings.alert_poll_interval_minutes,
        kwargs={"bot": bot, "client": client},
        id="alert_sweep",
        max_instances=1,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        coalesce=True,
    )

    scheduler.add_job(
        digest_job,
        "cron",
        hour=settings.digest_hour_msk,
        minute=settings.digest_minute_msk,
        kwargs={"bot": bot, "client": client, "digest_chat_id": settings.digest_chat_id},
        id="digest_daily",
        max_instances=1,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
        coalesce=True,
    )

    return scheduler

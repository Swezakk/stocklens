"""Эндпоинты управления Telegram-подписками на алерты и оценки алертов."""

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query
from starlette.responses import Response

from api.core.cache import AlertNxStore, RedisCache
from api.core.db import RedisDep, SessionDep
from api.repositories.alert import (
    SqlCloseRepository,
    SqlDividendAlertRepository,
    SqlNewsAlertRepository,
)
from api.repositories.bot import SqlBotSubscriptionRepository
from api.repositories.security import SqlSecurityRepository
from api.schemas.bot import DigestClaimOut, PendingAlertOut, SubscriptionIn, SubscriptionOut
from api.services.alert_evaluation import AlertEvaluationService
from api.services.bot import BotSubscriptionService

router = APIRouter(prefix="/api/v1/bot", tags=["bot"])

ChatIdDep = Annotated[int, Query(description="Telegram chat_id пользователя")]

_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def _today_moscow() -> date:
    """Вернуть текущую торговую дату по московскому часовому поясу (UTC+3)."""
    return datetime.now(tz=_MOSCOW_TZ).date()


def _subscription_service(session: SessionDep) -> BotSubscriptionService:
    """Собрать BotSubscriptionService из зависимостей запроса."""
    return BotSubscriptionService(
        repo=SqlBotSubscriptionRepository(session),
        security_repo=SqlSecurityRepository(session),
    )


def _alert_nx_store(redis: RedisDep) -> AlertNxStore:
    """Адаптировать RedisClientProtocol в AlertNxStore через RedisCache."""
    return RedisCache(redis)


def _evaluation_service(session: SessionDep, redis: RedisDep) -> AlertEvaluationService:
    """Собрать AlertEvaluationService из зависимостей запроса."""
    return AlertEvaluationService(
        bot_repo=SqlBotSubscriptionRepository(session),
        security_repo=SqlSecurityRepository(session),
        close_repo=SqlCloseRepository(session),
        news_repo=SqlNewsAlertRepository(session),
        dividend_repo=SqlDividendAlertRepository(session),
        redis=_alert_nx_store(redis),
        today=_today_moscow,
    )


@router.get(
    "/subscriptions",
    response_model=list[SubscriptionOut],
    summary="Список подписок по chat_id",
    description="Возвращает все активные подписки Telegram-пользователя.",
)
async def list_subscriptions(
    session: SessionDep,
    chat_id: ChatIdDep,
) -> list[SubscriptionOut]:
    """GET /bot/subscriptions?chat_id= — подписки пользователя."""
    return await _subscription_service(session).list_by_chat(chat_id)


@router.post(
    "/subscriptions",
    response_model=SubscriptionOut,
    status_code=201,
    summary="Создать подписку",
    description=(
        "Создать новую подписку на алерт. Все типы требуют 'ticker' (должен быть в БД). "
        "price_level дополнительно требует 'level' > 0. "
        "dividend_upcoming поддерживает 'lead_days' (1..30, default 3). "
        "422 если параметры невалидны или тикер неизвестен."
    ),
)
async def create_subscription(
    session: SessionDep,
    body: SubscriptionIn,
) -> SubscriptionOut:
    """POST /bot/subscriptions — создать подписку."""
    return await _subscription_service(session).create(body)


@router.delete(
    "/subscriptions/{sub_id}",
    status_code=204,
    summary="Удалить подписку",
    description="Удалить подписку по id. 404 если подписка не найдена.",
)
async def delete_subscription(session: SessionDep, sub_id: int) -> Response:
    """DELETE /bot/subscriptions/{sub_id} — удалить подписку, 204 при успехе."""
    await _subscription_service(session).delete(sub_id)
    return Response(status_code=204)


@router.post(
    "/alerts/pending",
    response_model=list[PendingAlertOut],
    summary="Список сработавших алертов",
    description=(
        "Оценить все активные подписки и вернуть список алертов, готовых к отправке. "
        "Использует Redis NX для дедупликации: один алерт не будет возвращён дважды "
        "за торговый день (price_level) или 3 дня (sentiment_spike). "
        "Вызывается ботом по расписанию."
    ),
)
async def get_pending_alerts(
    session: SessionDep,
    redis: RedisDep,
) -> list[PendingAlertOut]:
    """POST /bot/alerts/pending — список сработавших алертов."""
    return await _evaluation_service(session, redis).collect_pending()


@router.post(
    "/digest/claim",
    response_model=DigestClaimOut,
    summary="Зарезервировать дайджест на дату",
    description=(
        "Атомарно пометить дайджест за указанную дату как отправленный. "
        "Первый вызов за дату возвращает claimed=true; повторные — false. "
        "Гарантирует однократную отправку дайджеста при рестарте бота. "
        "При недоступности Redis — fail-open: returned claimed=true."
    ),
)
async def claim_digest(
    session: SessionDep,
    redis: RedisDep,
    for_date: Annotated[
        date | None,
        Query(description="Дата дайджеста в формате YYYY-MM-DD (по умолчанию — сегодня по МСК)"),
    ] = None,
) -> DigestClaimOut:
    """POST /bot/digest/claim?for_date= — зарезервировать дайджест."""
    target_date = for_date if for_date is not None else _today_moscow()
    claimed = await _evaluation_service(session, redis).digest_claim(target_date)
    return DigestClaimOut(claimed=claimed)

"""Эндпоинты управления Telegram-подписками на алерты."""

from typing import Annotated

from fastapi import APIRouter, Query
from starlette.responses import Response

from api.core.db import SessionDep
from api.repositories.bot import SqlBotSubscriptionRepository
from api.schemas.bot import SubscriptionIn, SubscriptionOut
from api.services.bot import BotSubscriptionService

router = APIRouter(prefix="/api/v1/bot", tags=["bot"])

ChatIdDep = Annotated[int, Query(description="Telegram chat_id пользователя")]


def _service(session: SessionDep) -> BotSubscriptionService:
    """Собрать BotSubscriptionService из зависимостей запроса."""
    return BotSubscriptionService(repo=SqlBotSubscriptionRepository(session))


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
    return await _service(session).list_by_chat(chat_id)


@router.post(
    "/subscriptions",
    response_model=SubscriptionOut,
    status_code=201,
    summary="Создать подписку",
    description=(
        "Создать новую подписку на алерт. Для price_level обязателен параметр 'level'. "
        "422 если параметры невалидны."
    ),
)
async def create_subscription(
    session: SessionDep,
    body: SubscriptionIn,
) -> SubscriptionOut:
    """POST /bot/subscriptions — создать подписку."""
    return await _service(session).create(body)


@router.delete(
    "/subscriptions/{sub_id}",
    status_code=204,
    summary="Удалить подписку",
    description="Удалить подписку по id. 404 если подписка не найдена.",
)
async def delete_subscription(session: SessionDep, sub_id: int) -> Response:
    """DELETE /bot/subscriptions/{sub_id} — удалить подписку, 204 при успехе."""
    await _service(session).delete(sub_id)
    return Response(status_code=204)

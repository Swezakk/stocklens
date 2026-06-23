"""Сборка текста ответов на команды (DESIGN §11).

Отделено от aiogram-хендлеров ради тестируемости: функции принимают API-клиент и примитивы
(chat_id, args), возвращают готовую HTML-строку — все ветки (успех / пустой ввод / ошибка)
покрываются юнит-тестами реальным клиентом + respx, без рантайма Telegram. Сетевой сбой API
становится русским сообщением (``ApiError.user_message``), а не traceback (правило ошибок).
"""

from datetime import date

import structlog

from bot import formatting
from bot.api_client.client import ApiClient
from bot.api_client.dto import SubscriptionIn
from bot.api_client.errors import ApiError
from bot.digest import gather_digest
from bot.subscriptions import ParseError, parse_subscribe, parse_unsubscribe

_log = structlog.get_logger()


def start_response() -> str:
    """Текст /start (статический список команд)."""
    return formatting.START_TEXT


async def portfolio_response(client: ApiClient) -> str:
    """Текст /portfolio: сводка портфеля или сообщение об ошибке API."""
    try:
        summary = await client.get_portfolio_summary()
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_portfolio(summary)


async def digest_response(client: ApiClient, today: date) -> str:
    """Текст /digest: дайджест или сообщение об ошибке API."""
    try:
        data = await gather_digest(client, today)
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_digest(data)


async def subscribe_response(client: ApiClient, chat_id: int, args: str | None) -> str:
    """Текст /subscribe: создать подписку, либо подсказка + текущие подписки, либо ошибка."""
    if not args:
        return await _subscribe_help(client, chat_id)
    parsed = parse_subscribe(args)
    if isinstance(parsed, ParseError):
        return parsed.message
    try:
        created = await client.create_subscription(
            SubscriptionIn(chat_id=chat_id, kind=parsed.kind, params=parsed.params)
        )
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_subscription_created(created)


async def unsubscribe_response(client: ApiClient, chat_id: int, args: str | None) -> str:
    """Текст /unsubscribe: удалить по id, либо список подписок с id, либо ошибка."""
    if not args:
        return await _subscriptions_list(client, chat_id)
    parsed = parse_unsubscribe(args)
    if isinstance(parsed, ParseError):
        return parsed.message
    try:
        await client.delete_subscription(parsed)
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_unsubscribed(parsed)


async def _subscribe_help(client: ApiClient, chat_id: int) -> str:
    """Подсказка /subscribe + текущие подписки (сбой списка не скрывает подсказку)."""
    try:
        subscriptions = await client.list_subscriptions(chat_id)
    except ApiError:
        return formatting.SUBSCRIBE_USAGE
    return formatting.SUBSCRIBE_USAGE + "\n\n" + formatting.format_subscriptions(subscriptions)


async def _subscriptions_list(client: ApiClient, chat_id: int) -> str:
    """Список подписок с id + подсказка /unsubscribe."""
    try:
        subscriptions = await client.list_subscriptions(chat_id)
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_subscriptions(subscriptions) + "\n\n" + formatting.UNSUBSCRIBE_USAGE


def _error_text(error: ApiError) -> str:
    """Залогировать сбой API и вернуть пользовательское русское сообщение."""
    _log.warning("api_error", detail=error.user_message)
    return error.user_message

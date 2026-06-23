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
    """Текст /start: приветствие с актуальным статусом алертов."""
    return formatting.START_TEXT


def help_response() -> str:
    """Текст /help: справка по командам, видам алертов и подпискам."""
    return formatting.HELP_TEXT


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


async def subscribe_response(client: ApiClient, chat_id: int, args: str) -> str:
    """Текст /subscribe с аргументами (текстовый путь): создать подписку или вернуть ошибку."""
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


async def unsubscribe_response(client: ApiClient, args: str) -> str:
    """Текст /unsubscribe с id (текстовый путь): удалить подписку или вернуть ошибку."""
    parsed = parse_unsubscribe(args)
    if isinstance(parsed, ParseError):
        return parsed.message
    try:
        await client.delete_subscription(parsed)
    except ApiError as exc:
        return _error_text(exc)
    return formatting.format_unsubscribed(parsed)


def _error_text(error: ApiError) -> str:
    """Залогировать сбой API и вернуть пользовательское русское сообщение."""
    _log.warning("api_error", detail=error.user_message)
    return error.user_message

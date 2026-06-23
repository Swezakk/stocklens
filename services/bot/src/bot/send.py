"""Безопасная отправка Telegram-сообщений от бота (DESIGN §11).

Единственный путь проактивной отправки (в отличие от message.answer в хендлерах).
Все ошибки aiogram поглощаются с логированием — падение отправки одного алерта
не должно прерывать цикл sweep или крашить планировщик.
"""

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter

_log = structlog.get_logger()


async def send_message_safe(bot: Bot, chat_id: int, html: str) -> bool:
    """Отправить HTML-сообщение в чат; при любой ошибке Telegram — залогировать и вернуть False.

    Только TelegramAPIError поглощается — unexpected ошибки Python пробрасываются,
    чтобы не скрывать баги реализации.
    """
    try:
        await bot.send_message(chat_id=chat_id, text=html)
        return True
    except TelegramRetryAfter as exc:
        _log.warning(
            "telegram_send_rate_limited",
            chat_id=chat_id,
            retry_after=exc.retry_after,
        )
        return False
    except TelegramAPIError as exc:
        _log.warning(
            "telegram_send_failed",
            chat_id=chat_id,
            error=str(exc),
        )
        return False

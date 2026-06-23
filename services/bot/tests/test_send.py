"""Tests for send_message_safe — must swallow all aiogram send errors and return False.

TDD: tests written before implementation; each test checks a distinct error branch.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from bot.send import send_message_safe

_CHAT_ID = 123456789
_HTML = "<b>Test</b>"


def _make_method_mock() -> Any:
    """Return a minimal mock satisfying TelegramAPIError(method, message) constructor."""
    return MagicMock()


def _make_bot(side_effect: Exception | None = None) -> Any:
    """Return a fake bot whose send_message either succeeds or raises side_effect."""
    bot = AsyncMock()
    if side_effect is not None:
        bot.send_message.side_effect = side_effect
    else:
        bot.send_message.return_value = MagicMock()
    return bot


async def test_send_message_safe_returns_true_on_success() -> None:
    bot = _make_bot()
    result = await send_message_safe(bot, _CHAT_ID, _HTML)
    assert result is True
    bot.send_message.assert_called_once_with(chat_id=_CHAT_ID, text=_HTML)


async def test_send_message_safe_swallows_forbidden_error_and_returns_false() -> None:
    error = TelegramForbiddenError(method=_make_method_mock(), message="bot was blocked")
    bot = _make_bot(side_effect=error)
    result = await send_message_safe(bot, _CHAT_ID, _HTML)
    assert result is False


async def test_send_message_safe_swallows_bad_request_and_returns_false() -> None:
    error = TelegramBadRequest(method=_make_method_mock(), message="chat not found")
    bot = _make_bot(side_effect=error)
    result = await send_message_safe(bot, _CHAT_ID, _HTML)
    assert result is False


async def test_send_message_safe_swallows_retry_after_and_returns_false() -> None:
    error = TelegramRetryAfter(
        method=_make_method_mock(), message="flood control exceeded", retry_after=30
    )
    bot = _make_bot(side_effect=error)
    result = await send_message_safe(bot, _CHAT_ID, _HTML)
    assert result is False


async def test_send_message_safe_swallows_base_telegram_api_error_and_returns_false() -> None:
    error = TelegramAPIError(method=_make_method_mock(), message="unknown api error")
    bot = _make_bot(side_effect=error)
    result = await send_message_safe(bot, _CHAT_ID, _HTML)
    assert result is False


async def test_send_message_safe_does_not_swallow_unexpected_errors() -> None:
    """Non-Telegram errors must propagate so they aren't silently lost."""
    bot = _make_bot(side_effect=RuntimeError("unexpected"))
    with pytest.raises(RuntimeError, match="unexpected"):
        await send_message_safe(bot, _CHAT_ID, _HTML)

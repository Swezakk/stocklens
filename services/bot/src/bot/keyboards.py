"""Чистые билдеры inline-клавиатур (aiogram 3.x InlineKeyboardBuilder).

Модуль не имеет зависимостей от aiogram-рантайма (Bot/Dispatcher) и от HTTP-клиента:
принимает чистые Python-данные, возвращает InlineKeyboardMarkup. Unit-тестируется
без рантайма Telegram.
"""

from collections.abc import Sequence

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from stocklens_core.enums import AlertKind

from bot.api_client.dto import SubscriptionOut
from bot.callbacks import (
    DeleteSubCb,
    MenuAction,
    MenuCb,
    WizCancelCb,
    WizKindCb,
    WizManualCb,
    WizTickerCb,
)
from bot.subscriptions import _SUBSCRIBABLE_KINDS

_MAX_TICKER_BUTTONS = 8
_MAIN_MENU_ROW_WIDTH = 2
_KIND_ROW_WIDTH = 1
_TICKER_ROW_WIDTH = 3

_KIND_LABELS: dict[AlertKind, str] = {
    AlertKind.PRICE_LEVEL: "📉 Уровень цены",
    AlertKind.SENTIMENT_SPIKE: "⚠️ Всплеск негатива",
    AlertKind.DIVIDEND_UPCOMING: "💰 Дивидендная отсечка",
}


def main_menu() -> InlineKeyboardMarkup:
    """Главное меню быстрых действий: Портфель, Дайджест, Подписки.

    Используется в /start и /help для удобного доступа к командам без ввода.
    """
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Портфель", callback_data=MenuCb(action=MenuAction.PORTFOLIO).pack())
    builder.button(text="📰 Дайджест", callback_data=MenuCb(action=MenuAction.DIGEST).pack())
    builder.button(text="🔔 Подписки", callback_data=MenuCb(action=MenuAction.SUBS).pack())
    builder.adjust(_MAIN_MENU_ROW_WIDTH)
    return builder.as_markup()


def subscriptions_kb(subs: Sequence[SubscriptionOut]) -> InlineKeyboardMarkup:
    """Список подписок: одна строка на подписку с кнопкой ❌, плюс кнопка ➕ Добавить.

    Позволяет удалять подписки тапом без ввода id. Кнопка ➕ запускает мастера.
    """
    builder = InlineKeyboardBuilder()
    for sub in subs:
        builder.button(
            text=f"❌ #{sub.id}",
            callback_data=DeleteSubCb(sub_id=sub.id).pack(),
        )
    builder.button(text="➕ Добавить", callback_data=MenuCb(action=MenuAction.SUBSCRIBE).pack())
    adjust = [1] * len(subs) + [1]
    builder.adjust(*adjust)
    return builder.as_markup()


def wizard_kind_kb() -> InlineKeyboardMarkup:
    """Шаг 1 мастера: выбор вида алерта из доступных для подписки видов."""
    builder = InlineKeyboardBuilder()
    for kind in _SUBSCRIBABLE_KINDS:
        label = _KIND_LABELS.get(kind, kind.value)
        builder.button(text=label, callback_data=WizKindCb(kind=kind).pack())
    builder.button(text="❌ Отмена", callback_data=WizCancelCb().pack())
    builder.adjust(_KIND_ROW_WIDTH)
    return builder.as_markup()


def wizard_ticker_kb(tickers: Sequence[str]) -> InlineKeyboardMarkup:
    """Шаг 2 мастера: выбор тикера из портфеля (до N) + ручной ввод + отмена."""
    builder = InlineKeyboardBuilder()
    for ticker in list(tickers)[:_MAX_TICKER_BUTTONS]:
        builder.button(
            text=ticker,
            callback_data=WizTickerCb(ticker=ticker).pack(),
        )
    builder.button(text="✏️ Ввести вручную", callback_data=WizManualCb().pack())
    builder.button(text="❌ Отмена", callback_data=WizCancelCb().pack())
    ticker_count = min(len(tickers), _MAX_TICKER_BUTTONS)
    if ticker_count > 0:
        full_rows, remainder = divmod(ticker_count, _TICKER_ROW_WIDTH)
        row_widths = [_TICKER_ROW_WIDTH] * full_rows
        if remainder:
            row_widths.append(remainder)
        row_widths.extend([1, 1])
    else:
        row_widths = [1, 1]
    builder.adjust(*row_widths)
    return builder.as_markup()

"""CallbackData-фабрики для inline-кнопок бота (aiogram 3.x).

Каждый класс описывает один тип колбэка: префикс гарантирует уникальность данных
при роутинге aiogram. Чистый слой: ни aiogram-объектов, ни httpx — только pack/unpack.
Unit-тестируется без рантайма Telegram (pack ↔ unpack round-trip).
"""

from enum import StrEnum

from aiogram.filters.callback_data import CallbackData
from stocklens_core.enums import AlertKind


class MenuAction(StrEnum):
    """Действия quick-action-кнопок главного меню (запрет хардкода строк в логике)."""

    PORTFOLIO = "portfolio"
    DIGEST = "digest"
    SUBS = "subs"
    SUBSCRIBE = "subscribe"


class MenuCb(CallbackData, prefix="menu"):
    """Quick-action из приветственного/help сообщения: действие пользователя."""

    action: MenuAction


class DeleteSubCb(CallbackData, prefix="delsub"):
    """Удалить подписку по id (кнопка ❌ в списке подписок)."""

    sub_id: int


class WizKindCb(CallbackData, prefix="wkind"):
    """Шаг мастера /subscribe: пользователь выбрал вид алерта."""

    kind: AlertKind


class WizTickerCb(CallbackData, prefix="wticker"):
    """Шаг мастера: пользователь выбрал тикер из пикера портфеля."""

    ticker: str


class WizManualCb(CallbackData, prefix="wmanual"):
    """Шаг мастера: пользователь нажал «Ввести вручную» — перейти к текстовому вводу."""


class WizCancelCb(CallbackData, prefix="wcancel"):
    """Отмена мастера /subscribe на любом шаге."""

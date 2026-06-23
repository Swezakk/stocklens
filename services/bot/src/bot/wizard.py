"""Чистая логика мастера /subscribe: валидация ввода и сборка SubscriptionIn.

Не зависит от aiogram-рантайма — принимает примитивы (kind, ticker, level),
возвращает SubscriptionIn или строку с RU-ошибкой. Unit-тестируется без Telegram.
"""

from dataclasses import dataclass

from stocklens_core.enums import AlertKind

from bot.api_client.dto import SubscriptionIn
from bot.subscriptions import _PARAM_LEVEL, _PARAM_TICKER, _SUBSCRIBABLE_KINDS

#: Максимальная длина тикера (MOEX: до 10 символов).
_MAX_TICKER_LEN = 10

_ERR_TICKER_EMPTY = "Тикер не может быть пустым."
_ERR_TICKER_TOO_LONG = f"Тикер не длиннее {_MAX_TICKER_LEN} символов."
_ERR_TICKER_NOT_ALPHA = "Тикер должен содержать только буквы и цифры."
_ERR_LEVEL_EMPTY = "Укажите числовой уровень цены, например: 250"
_ERR_LEVEL_NOT_NUMBER = "Уровень цены должен быть числом, например: 250"
_ERR_LEVEL_NOT_POSITIVE = "Уровень цены должен быть положительным числом."
_ERR_KIND_NOT_SUBSCRIBABLE = "Этот вид алерта недоступен для подписки."


@dataclass(frozen=True)
class WizardError:
    """Ошибка мастера с готовым RU-сообщением для пользователя."""

    message: str


def validate_ticker(raw: str) -> str | WizardError:
    """Нормализовать и проверить введённый тикер; вернуть UPPER-форму или ошибку."""
    ticker = raw.strip().upper()
    if not ticker:
        return WizardError(_ERR_TICKER_EMPTY)
    if len(ticker) > _MAX_TICKER_LEN:
        return WizardError(_ERR_TICKER_TOO_LONG)
    if not ticker.isalnum():
        return WizardError(_ERR_TICKER_NOT_ALPHA)
    return ticker


def validate_level(raw: str) -> float | WizardError:
    """Разобрать введённый уровень цены; вернуть float или ошибку."""
    stripped = raw.strip()
    if not stripped:
        return WizardError(_ERR_LEVEL_EMPTY)
    try:
        value = float(stripped)
    except ValueError:
        return WizardError(_ERR_LEVEL_NOT_NUMBER)
    if value <= 0:
        return WizardError(_ERR_LEVEL_NOT_POSITIVE)
    return value


def build_subscription(
    chat_id: int,
    kind: AlertKind,
    ticker: str,
    level: float | None,
) -> SubscriptionIn | WizardError:
    """Собрать SubscriptionIn из компонентов мастера после всей валидации.

    Для PRICE_LEVEL level обязателен — guard здесь как safety net; основной
    контроль обеспечен FSM-потоком (entering_level не пропустить без level).
    """
    if kind not in _SUBSCRIBABLE_KINDS:
        return WizardError(_ERR_KIND_NOT_SUBSCRIBABLE)
    params: dict[str, object] = {_PARAM_TICKER: ticker}
    if kind is AlertKind.PRICE_LEVEL:
        if level is None:
            return WizardError(_ERR_LEVEL_EMPTY)
        params[_PARAM_LEVEL] = level
    return SubscriptionIn(chat_id=chat_id, kind=kind, params=params)

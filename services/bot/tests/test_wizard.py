"""Тесты чистой логики мастера /subscribe: валидация тикера/уровня и сборка SubscriptionIn."""

from bot.api_client.dto import SubscriptionIn
from bot.wizard import WizardError, build_subscription, validate_level, validate_ticker
from stocklens_core.enums import AlertKind

_CHAT_ID = 42


def test_validate_ticker_normalizes_to_upper() -> None:
    result = validate_ticker("sber")
    assert result == "SBER"


def test_validate_ticker_strips_whitespace() -> None:
    result = validate_ticker("  GAZP  ")
    assert result == "GAZP"


def test_validate_ticker_empty_returns_error() -> None:
    result = validate_ticker("")
    assert isinstance(result, WizardError)
    assert result.message


def test_validate_ticker_whitespace_only_returns_error() -> None:
    result = validate_ticker("   ")
    assert isinstance(result, WizardError)


def test_validate_ticker_too_long_returns_error() -> None:
    result = validate_ticker("A" * 11)
    assert isinstance(result, WizardError)


def test_validate_ticker_non_alnum_returns_error() -> None:
    result = validate_ticker("SB-ER")
    assert isinstance(result, WizardError)


def test_validate_ticker_max_length_accepted() -> None:
    result = validate_ticker("A" * 10)
    assert result == "A" * 10


def test_validate_level_parses_integer_string() -> None:
    result = validate_level("250")
    assert result == 250.0


def test_validate_level_parses_float_string() -> None:
    result = validate_level("250.50")
    assert result == 250.50


def test_validate_level_empty_returns_error() -> None:
    result = validate_level("")
    assert isinstance(result, WizardError)


def test_validate_level_non_numeric_returns_error() -> None:
    result = validate_level("abc")
    assert isinstance(result, WizardError)


def test_validate_level_zero_returns_error() -> None:
    result = validate_level("0")
    assert isinstance(result, WizardError)


def test_validate_level_negative_returns_error() -> None:
    result = validate_level("-100")
    assert isinstance(result, WizardError)


def test_build_subscription_price_level_requires_level() -> None:
    result = build_subscription(_CHAT_ID, AlertKind.PRICE_LEVEL, "SBER", None)
    assert isinstance(result, WizardError)


def test_build_subscription_price_level_with_level_assembles_subscription_in() -> None:
    result = build_subscription(_CHAT_ID, AlertKind.PRICE_LEVEL, "SBER", 250.0)
    assert isinstance(result, SubscriptionIn)
    assert result.kind is AlertKind.PRICE_LEVEL
    assert result.params.get("ticker") == "SBER"
    assert result.params.get("level") == 250.0
    assert result.chat_id == _CHAT_ID


def test_build_subscription_sentiment_spike_does_not_require_level() -> None:
    result = build_subscription(_CHAT_ID, AlertKind.SENTIMENT_SPIKE, "GAZP", None)
    assert isinstance(result, SubscriptionIn)
    assert result.kind is AlertKind.SENTIMENT_SPIKE
    assert result.params.get("ticker") == "GAZP"
    assert "level" not in result.params


def test_build_subscription_dividend_upcoming_does_not_require_level() -> None:
    result = build_subscription(_CHAT_ID, AlertKind.DIVIDEND_UPCOMING, "LKOH", None)
    assert isinstance(result, SubscriptionIn)
    assert result.kind is AlertKind.DIVIDEND_UPCOMING


def test_build_subscription_volatility_regime_returns_error() -> None:
    result = build_subscription(_CHAT_ID, AlertKind.VOLATILITY_REGIME, "SBER", None)
    assert isinstance(result, WizardError)

"""Тесты разбора аргументов /subscribe и /unsubscribe (чистые функции, DESIGN §11)."""

from bot.subscriptions import ParsedSubscribe, ParseError, parse_subscribe, parse_unsubscribe
from stocklens_core.enums import AlertKind


def test_parse_price_level_extracts_ticker_and_level() -> None:
    result = parse_subscribe("price_level sber 250.5")
    assert isinstance(result, ParsedSubscribe)
    assert result.kind is AlertKind.PRICE_LEVEL
    assert result.params == {"ticker": "SBER", "level": 250.5}


def test_parse_price_level_without_level_is_error() -> None:
    result = parse_subscribe("price_level SBER")
    assert isinstance(result, ParseError)


def test_parse_price_level_non_numeric_level_is_error() -> None:
    result = parse_subscribe("price_level SBER abc")
    assert isinstance(result, ParseError)


def test_parse_sentiment_spike_with_ticker() -> None:
    result = parse_subscribe("sentiment_spike gazp")
    assert isinstance(result, ParsedSubscribe)
    assert result.kind is AlertKind.SENTIMENT_SPIKE
    assert result.params == {"ticker": "GAZP"}


def test_parse_dividend_upcoming_without_ticker_has_empty_params() -> None:
    result = parse_subscribe("dividend_upcoming")
    assert isinstance(result, ParsedSubscribe)
    assert result.kind is AlertKind.DIVIDEND_UPCOMING
    assert result.params == {}


def test_parse_unknown_kind_is_error() -> None:
    assert isinstance(parse_subscribe("moon_phase SBER"), ParseError)


def test_parse_volatility_regime_is_deferred_error() -> None:
    result = parse_subscribe("volatility_regime SBER")
    assert isinstance(result, ParseError)
    assert "ML" in result.message


def test_parse_unsubscribe_returns_int_id() -> None:
    assert parse_unsubscribe("3") == 3


def test_parse_unsubscribe_non_numeric_is_error() -> None:
    assert isinstance(parse_unsubscribe("abc"), ParseError)


def test_parse_unsubscribe_empty_is_error() -> None:
    assert isinstance(parse_unsubscribe(""), ParseError)

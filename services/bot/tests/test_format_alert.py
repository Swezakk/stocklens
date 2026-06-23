"""Tests for format_alert — one test per AlertKind, plus VOLATILITY_REGIME safe fallback.

TDD: verifies the HTML output contains expected data without prescribing exact wording.
"""

from datetime import date
from decimal import Decimal

from bot.api_client.dto import PendingAlert
from bot.formatting import format_alert
from stocklens_core.enums import AlertKind, Currency


def _price_level_alert() -> PendingAlert:
    return PendingAlert(
        chat_id=111,
        kind=AlertKind.PRICE_LEVEL,
        ticker="SBER",
        level=Decimal("250.00"),
        close=Decimal("251.50"),
    )


def _sentiment_spike_alert() -> PendingAlert:
    return PendingAlert(
        chat_id=111,
        kind=AlertKind.SENTIMENT_SPIKE,
        ticker="GAZP",
        article_id=42,
        article_title="Газпром снизил дивиденды",
        article_url="https://kommersant.ru/a/1",
        article_published_at=None,
    )


def _dividend_upcoming_alert() -> PendingAlert:
    return PendingAlert(
        chat_id=111,
        kind=AlertKind.DIVIDEND_UPCOMING,
        ticker="LKOH",
        ex_date=date(2026, 7, 1),
        dividend_value=Decimal("600.00"),
        dividend_currency=Currency.RUB,
    )


def _volatility_regime_alert() -> PendingAlert:
    return PendingAlert(
        chat_id=111,
        kind=AlertKind.VOLATILITY_REGIME,
        ticker="MOEX",
    )


def test_format_alert_price_level_contains_ticker_level_close() -> None:
    text = format_alert(_price_level_alert())
    assert "SBER" in text
    assert "250" in text
    assert "251" in text


def test_format_alert_sentiment_spike_contains_ticker_and_link() -> None:
    text = format_alert(_sentiment_spike_alert())
    assert "GAZP" in text
    assert "https://kommersant.ru/a/1" in text
    assert "Газпром снизил дивиденды" in text


def test_format_alert_dividend_upcoming_contains_ticker_date_and_value() -> None:
    text = format_alert(_dividend_upcoming_alert())
    assert "LKOH" in text
    assert "01.07.2026" in text
    assert "600" in text


def test_format_alert_dividend_upcoming_non_rub_uses_currency_symbol() -> None:
    """Дивиденд в USD рендерится символом $, не рублём (регрессия валюты)."""
    alert = PendingAlert(
        chat_id=111,
        kind=AlertKind.DIVIDEND_UPCOMING,
        ticker="GMKN",
        ex_date=date(2026, 7, 1),
        dividend_value=Decimal("5.00"),
        dividend_currency=Currency.USD,
    )
    text = format_alert(alert)
    assert "$" in text
    assert "₽" not in text


def test_format_alert_volatility_regime_does_not_raise() -> None:
    """VOLATILITY_REGIME has no detail fields — must not crash; returns safe fallback."""
    text = format_alert(_volatility_regime_alert())
    assert "MOEX" in text
    assert len(text) > 0

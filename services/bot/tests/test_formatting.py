"""Тесты форматирования сообщений бота (чистые функции, HTML, DESIGN §11).

Ассерты — по устойчивым подстрокам (ключевые суммы/знаки/тикеры/даты), а не по полному
тексту: точная вёрстка может меняться, а контракт «эти данные присутствуют» — нет.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

from bot.api_client.dto import (
    NewsOut,
    PortfolioSummaryOut,
    PositionOut,
    SentimentOut,
    SubscriptionOut,
)
from bot.digest_model import DigestData, UpcomingDividend
from stocklens_core.enums import AlertKind, Currency, SentimentLabel

from bot import formatting


def _position(ticker: str = "SBER", pnl: Decimal | None = Decimal("150.00")) -> PositionOut:
    has_price = pnl is not None
    return PositionOut(
        ticker=ticker,
        quantity=10,
        avg_price=Decimal("250.00"),
        opened_at=datetime(2026, 1, 10, 12, 0, tzinfo=UTC),
        current_price=Decimal("265.00") if has_price else None,
        current_value=Decimal("2650.00") if has_price else None,
        unrealized_pnl=pnl,
    )


def _summary(positions: list[PositionOut] | None = None) -> PortfolioSummaryOut:
    return PortfolioSummaryOut(
        positions=[_position()] if positions is None else positions,
        total_value=Decimal("2650.00"),
        total_cost=Decimal("2500.00"),
        total_unrealized_pnl=Decimal("150.00"),
        portfolio_return_pct=6.0,
        imoex_return_pct=-2.0,
        sharpe=1.1,
        max_drawdown=0.12,
        imoex_sharpe=0.4,
        imoex_max_drawdown=0.2,
        period_from=date(2026, 1, 1),
        period_to=date(2026, 6, 23),
    )


def _news(title: str = "РУСАЛ снизил прогноз", tickers: tuple[str, ...] = ("RUAL",)) -> NewsOut:
    return NewsOut(
        id=1,
        source="rss_kommersant",
        url="https://example.com/a",
        title=title,
        summary="...",
        published_at=datetime(2026, 6, 20, 15, 14, tzinfo=UTC),
        sentiment=SentimentOut(label=SentimentLabel.NEGATIVE, score=0.9, model_version="v1"),
        tickers=list(tickers),
    )


def test_format_portfolio_includes_pnl_return_and_ticker() -> None:
    text = formatting.format_portfolio(_summary())
    assert "Портфель" in text
    assert "+150.00" in text
    assert "+6.00%" in text
    assert "IMOEX" in text
    assert "SBER" in text


def test_format_portfolio_empty_positions_shows_placeholder() -> None:
    assert "пуст" in formatting.format_portfolio(_summary(positions=[])).lower()


def test_format_portfolio_position_without_price_shows_no_market_price() -> None:
    text = formatting.format_portfolio(_summary(positions=[_position(pnl=None)]))
    assert "нет рыночной цены" in text


def test_format_subscriptions_empty_shows_hint() -> None:
    assert "нет активных подписок" in formatting.format_subscriptions([]).lower()


def test_format_subscriptions_lists_id_label_and_params() -> None:
    sub = SubscriptionOut(
        id=5, chat_id=7, kind=AlertKind.PRICE_LEVEL, params={"ticker": "SBER", "level": 250}
    )
    text = formatting.format_subscriptions([sub])
    assert "5" in text
    assert "уровень цены" in text
    assert "SBER" in text


def test_format_digest_combines_portfolio_dividends_and_news() -> None:
    data = DigestData(
        summary=_summary(),
        dividends=[
            UpcomingDividend(
                ticker="SBER",
                ex_date=date(2026, 6, 25),
                value=Decimal("33.30"),
                currency=Currency.RUB,
            )
        ],
        negative_news=[_news()],
    )
    text = formatting.format_digest(data)
    assert "Портфель" in text
    assert "25.06.2026" in text
    assert "РУСАЛ" in text


def test_format_digest_dividend_non_rub_uses_currency_symbol() -> None:
    """Дивиденд в USD в дайджесте — символ $, не рубль (регрессия валюты)."""
    data = DigestData(
        summary=_summary(),
        dividends=[
            UpcomingDividend(
                ticker="GMKN",
                ex_date=date(2026, 6, 25),
                value=Decimal("5.00"),
                currency=Currency.USD,
            )
        ],
        negative_news=[],
    )
    text = formatting.format_digest(data)
    assert "5.00 $" in text
    assert "5.00 ₽" not in text


def test_format_digest_empty_sections_show_placeholders() -> None:
    text = formatting.format_digest(DigestData(summary=_summary(), dividends=[], negative_news=[]))
    assert "отсечек нет" in text
    assert "новостей нет" in text


def test_format_subscription_created_confirms_label_and_id() -> None:
    sub = SubscriptionOut(
        id=9, chat_id=7, kind=AlertKind.SENTIMENT_SPIKE, params={"ticker": "GAZP"}
    )
    text = formatting.format_subscription_created(sub)
    assert "создана" in text
    assert "9" in text


def test_format_unsubscribed_confirms_id() -> None:
    assert "3" in formatting.format_unsubscribed(3)


def test_start_text_describes_alerts_as_active() -> None:
    assert "алерты подключаются" not in formatting.START_TEXT.lower()
    assert "работают" in formatting.START_TEXT or "sweep" in formatting.START_TEXT


def test_start_text_lists_main_commands() -> None:
    assert "/portfolio" in formatting.START_TEXT
    assert "/digest" in formatting.START_TEXT
    assert "/subscribe" in formatting.START_TEXT
    assert "/help" in formatting.START_TEXT


def test_help_text_describes_three_alert_types() -> None:
    help_lower = formatting.HELP_TEXT.lower()
    assert "уровень цены" in help_lower
    assert "всплеск негатива" in help_lower
    assert "дивиденд" in help_lower


def test_help_text_mentions_schedule() -> None:
    assert "08:30" in formatting.HELP_TEXT or "30 минут" in formatting.HELP_TEXT


def test_format_subscription_created_has_checkmark() -> None:
    sub = SubscriptionOut(
        id=9, chat_id=7, kind=AlertKind.SENTIMENT_SPIKE, params={"ticker": "GAZP"}
    )
    text = formatting.format_subscription_created(sub)
    assert "✅" in text


def test_format_unsubscribed_has_checkmark() -> None:
    assert "✅" in formatting.format_unsubscribed(3)


def test_format_subscriptions_header_has_bell_emoji() -> None:
    sub = SubscriptionOut(id=1, chat_id=7, kind=AlertKind.PRICE_LEVEL, params={"ticker": "SBER"})
    text = formatting.format_subscriptions([sub])
    assert "🔔" in text

"""Smoke-тесты DTO: зеркало JSON-контракта API парсится без потери точности."""

from decimal import Decimal
from typing import Any

from dashboard.api_client.dto import (
    CandleOut,
    CollectorRunOut,
    MoversOut,
    NewsOut,
    OptimizationStrategy,
    Page,
    PortfolioSummaryOut,
    SecurityOut,
)
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel


def test_page_parses_generic_items(page_payload: dict[str, Any]) -> None:
    page = Page[SecurityOut].model_validate(page_payload)
    assert page.total == 1
    assert page.items[0].ticker == "SBER"


def test_candle_preserves_decimal_precision(candle_payload: dict[str, Any]) -> None:
    candle = CandleOut.model_validate(candle_payload)
    assert candle.close == Decimal("310.75")
    assert candle.value == Decimal("478123456.50")
    assert candle.volume == 1542000


def test_movers_parses_change_pct(movers_payload: dict[str, Any]) -> None:
    movers = MoversOut.model_validate(movers_payload)
    assert movers.gainers[0].change_pct == 6.91
    assert movers.losers[0].change_pct == -4.41


def test_news_parses_sentiment_enum(news_payload: dict[str, Any]) -> None:
    news = NewsOut.model_validate(news_payload)
    assert news.sentiment is not None
    assert news.sentiment.label is SentimentLabel.POSITIVE
    assert news.tickers == ["SBER"]


def test_collector_run_parses_status_enum(collector_run_payload: dict[str, Any]) -> None:
    run = CollectorRunOut.model_validate(collector_run_payload)
    assert run.status is CollectorRunStatus.SUCCESS
    assert run.error_message is None


def test_portfolio_summary_parses(portfolio_summary_payload: dict[str, Any]) -> None:
    summary = PortfolioSummaryOut.model_validate(portfolio_summary_payload)
    assert summary.total_value == Decimal("3107.50")
    position = summary.positions[0]
    assert position.unrealized_pnl == Decimal("307.50")
    assert isinstance(position.current_value, Decimal)


def test_optimization_strategy_mirrors_api_values() -> None:
    assert OptimizationStrategy.MAX_SHARPE.value == "max_sharpe"
    assert OptimizationStrategy.MIN_VOLATILITY.value == "min_volatility"
    assert {s.value for s in OptimizationStrategy} == {
        "max_sharpe",
        "min_volatility",
        "target_return",
        "target_risk",
        "max_utility",
    }


def test_currency_enum_imported_from_core() -> None:
    assert Currency.RUB.value == "RUB"

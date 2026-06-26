"""Smoke-тесты DTO: зеркало JSON-контракта API парсится без потери точности."""

from decimal import Decimal
from typing import Any

from dashboard.api_client.dto import (
    CandleOut,
    CollectorRunOut,
    MoversOut,
    NewsOut,
    OptimizationStrategy,
    OptimizeResult,
    Page,
    PortfolioSummaryOut,
    SecurityOut,
    VolatilityForecastHistoryOut,
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


def _optimize_payload(**overrides: Any) -> dict[str, Any]:
    """Образец JSON ответа /portfolio/optimize (зеркало OptimizeResult)."""
    payload: dict[str, Any] = {
        "strategy": "max_sharpe",
        "requested_strategy": "max_sharpe",
        "weights": {"SBER": 1.0},
        "expected_return": 0.12,
        "volatility": 0.2,
        "sharpe": 0.6,
        "frontier": [{"volatility": 0.2, "expected_return": 0.12}],
        "equal_weight_sharpe": 0.5,
        "imoex_sharpe": 0.4,
    }
    payload.update(overrides)
    return payload


def test_optimize_result_round_trips_requested_strategy_and_fallback() -> None:
    """Авто-фолбэк: эффективная стратегия min-vol при запрошенной max-Sharpe + причина."""
    result = OptimizeResult.model_validate(
        _optimize_payload(
            strategy="min_volatility",
            requested_strategy="max_sharpe",
            fallback_reason="Максимизация Шарпа невозможна: применена минимизация риска.",
        )
    )
    assert result.strategy is OptimizationStrategy.MIN_VOLATILITY
    assert result.requested_strategy is OptimizationStrategy.MAX_SHARPE
    assert result.fallback_reason == "Максимизация Шарпа невозможна: применена минимизация риска."


def test_optimize_result_defaults_fallback_reason_to_none() -> None:
    """Без авто-фолбэка: API не присылает fallback_reason — поле по умолчанию None."""
    result = OptimizeResult.model_validate(_optimize_payload())
    assert result.fallback_reason is None
    assert result.requested_strategy is OptimizationStrategy.MAX_SHARPE


def test_currency_enum_imported_from_core() -> None:
    assert Currency.RUB.value == "RUB"


def test_volatility_forecast_history_parses_without_live_fields() -> None:
    """Ответ API без live_metrics/live_sample_size (старая версия) парсится без ошибок."""
    payload = {
        "ticker": "SBER",
        "model": "garch",
        "model_version": "v1.0",
        "metrics_vs_baseline": {"qlike": 0.512, "qlike_baseline": 0.731, "rmse": 0.042},
        "points": [],
    }
    history = VolatilityForecastHistoryOut.model_validate(payload)
    assert history.live_metrics is None
    assert history.live_sample_size == 0


def test_volatility_forecast_history_parses_with_live_fields() -> None:
    """Ответ API с live_metrics/live_sample_size парсится корректно."""
    payload = {
        "ticker": "SBER",
        "model": "garch",
        "model_version": "v1.0",
        "metrics_vs_baseline": {"qlike": 0.512, "qlike_baseline": 0.731, "rmse": 0.042},
        "points": [],
        "live_metrics": {"qlike": 0.534, "qlike_baseline": 0.748, "rmse": 0.044},
        "live_sample_size": 15,
    }
    history = VolatilityForecastHistoryOut.model_validate(payload)
    assert history.live_metrics is not None
    assert history.live_metrics.qlike == 0.534
    assert history.live_sample_size == 15

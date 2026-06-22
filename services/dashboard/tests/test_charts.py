"""Smoke-тесты Plotly-билдеров графиков (DESIGN.md §5, §10).

Билдеры — чистые относительно Streamlit (возвращают go.Figure), поэтому проверяются без
поднятия UI: каждая фигура обязана сериализоваться (`fig.to_json()` не падает) и нести
ожидаемые трейсы/цвета из токенов theme. Лэйаут как таковой не тестируется (DESIGN §10),
тестируется состав данных и семантика цвета. Нормировка rebase-to-100 — pure-хелпер —
покрыта отдельно (база/пусто/неположительная база).
"""

from datetime import date
from decimal import Decimal

import pytest
from dashboard.api_client.dto import CandleOut, DividendOut, FrontierPoint
from dashboard.components.charts import (
    _rebase_to_100,
    build_candlestick_chart,
    build_comparison_chart,
    build_efficient_frontier_chart,
    build_index_line_chart,
    build_portfolio_vs_imoex_chart,
    build_sentiment_trend_chart,
    build_word_frequency_chart,
)
from dashboard.components.transforms import SentimentDayPoint
from stocklens_core.enums import Currency, SentimentLabel

from dashboard import theme


def _candle(trade_date: date, close: str) -> CandleOut:
    """Собрать CandleOut с заданными датой и close (прочие OHLCV — производные)."""
    close_dec = Decimal(close)
    return CandleOut(
        id=1,
        security_id=1,
        trade_date=trade_date,
        open=close_dec,
        high=close_dec,
        low=close_dec,
        close=close_dec,
        volume=1000,
        value=Decimal("1000000.00"),
        is_weekend_session=False,
    )


def test_candlestick_chart_serializes_with_volume_and_dividend_markers() -> None:
    candles = [_candle(date(2026, 6, 18), "100.00"), _candle(date(2026, 6, 19), "110.00")]
    dividends = [
        DividendOut(
            id=1,
            security_id=1,
            ex_date=date(2026, 6, 19),
            value=Decimal("12.50"),
            currency=Currency.RUB,
        )
    ]
    fig = build_candlestick_chart(candles, dividends)
    payload = fig.to_json()
    assert "candlestick" in payload
    assert "y2" in payload  # субплот объёма
    assert fig.data[0].increasing.line.color == theme.UP
    assert fig.data[0].decreasing.line.color == theme.DOWN
    # Дивидендная отсечка нанесена shape+annotation (не add_vline), цвет — WARNING.
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].line.color == theme.WARNING
    assert len(fig.layout.annotations) == 1


def test_index_line_chart_full_mode_serializes_with_accent_line() -> None:
    points = [(date(2026, 6, 18), 2800.0), (date(2026, 6, 19), 2850.0)]
    fig = build_index_line_chart(points)
    assert fig.to_json()
    assert len(fig.data) == 1
    assert fig.data[0].line.color == theme.ACCENT
    assert fig.layout.xaxis.visible is None  # оси видимы в полном режиме


def test_index_line_chart_sparkline_mode_hides_axes() -> None:
    points = [(date(2026, 6, 18), 2800.0), (date(2026, 6, 19), 2850.0)]
    fig = build_index_line_chart(points, sparkline=True)
    assert fig.to_json()
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False
    assert fig.layout.showlegend is False


def test_comparison_chart_rebases_each_series_to_100() -> None:
    series = {
        "SBER": [_candle(date(2026, 6, 18), "200.00"), _candle(date(2026, 6, 19), "220.00")],
        "GAZP": [_candle(date(2026, 6, 18), "120.00"), _candle(date(2026, 6, 19), "114.00")],
    }
    fig = build_comparison_chart(series)
    assert fig.to_json()
    assert len(fig.data) == 2
    # Первая точка каждой серии нормирована к 100; цвета — categorical, не семантические.
    assert fig.data[0].y[0] == 100.0
    assert fig.data[0].y[1] == pytest.approx(110.0)  # 220/200 × 100
    assert fig.data[1].y[1] == pytest.approx(95.0)  # 114/120 × 100
    assert fig.data[0].line.color == theme.CHART_CATEGORICAL[0]
    assert fig.data[1].line.color == theme.CHART_CATEGORICAL[1]


def test_comparison_chart_skips_empty_series() -> None:
    series: dict[str, list[CandleOut]] = {
        "SBER": [_candle(date(2026, 6, 18), "200.00")],
        "EMPTY": [],
    }
    fig = build_comparison_chart(series)
    assert fig.to_json()
    assert len(fig.data) == 1
    assert fig.data[0].name == "SBER"


def test_rebase_to_100_normalizes_from_first_point() -> None:
    assert _rebase_to_100([200.0, 220.0, 180.0]) == pytest.approx([100.0, 110.0, 90.0])


def test_rebase_to_100_empty_series_returns_empty() -> None:
    assert _rebase_to_100([]) == []


def test_rebase_to_100_non_positive_base_returns_empty() -> None:
    assert _rebase_to_100([0.0, 5.0]) == []


def test_portfolio_vs_imoex_chart_serializes_with_accent_and_muted() -> None:
    dates = [date(2026, 6, 18), date(2026, 6, 19)]
    fig = build_portfolio_vs_imoex_chart(dates, [100.0, 110.0], [100.0, 104.0])
    assert fig.to_json()
    assert fig.data[0].line.color == theme.ACCENT
    assert fig.data[1].line.color == theme.MUTED_TEXT


def test_efficient_frontier_chart_serializes_with_markers() -> None:
    frontier = [
        FrontierPoint(volatility=0.10, expected_return=0.08),
        FrontierPoint(volatility=0.20, expected_return=0.15),
    ]
    fig = build_efficient_frontier_chart(
        frontier,
        selected=(0.15, 0.12),
        equal_weight=(0.18, 0.11),
        imoex=(0.22, 0.10),
    )
    assert fig.to_json()
    # Кривая фронтира + три маркера-бенчмарка.
    assert len(fig.data) == 4
    assert fig.data[0].line.color == theme.ACCENT


def test_sentiment_trend_chart_colors_markers_by_sign() -> None:
    series = [
        SentimentDayPoint(
            day=date(2026, 6, 18), mean_score=0.80, counts={SentimentLabel.POSITIVE: 2}
        ),
        SentimentDayPoint(
            day=date(2026, 6, 19), mean_score=0.30, counts={SentimentLabel.NEGATIVE: 1}
        ),
    ]
    fig = build_sentiment_trend_chart(series)
    assert fig.to_json()
    marker_colors = tuple(fig.data[0].marker.color)
    assert marker_colors == (theme.UP, theme.DOWN)


def test_word_frequency_chart_serializes_with_accent_bars() -> None:
    fig = build_word_frequency_chart([("сбербанк", 5), ("прибыль", 3)])
    assert fig.to_json()
    assert fig.data[0].marker.color == theme.ACCENT
    assert fig.data[0].orientation == "h"

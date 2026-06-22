"""Plotly-билдеры графиков дашборда (DESIGN.md §5, §2.3).

Каждый билдер собирает go.Figure с ЯВНЫМИ цветами трейсов из theme (источник истины
графиков) и применяет theme.apply_dark_template. Графики рендерятся через
render_chart → st.plotly_chart(fig, theme=None): тему контролирует наш шаблон, а не
Streamlit, без двойного применения (config.toml chart* к Plotly-пути не применяются).

Билдеры чистые относительно Streamlit (возвращают Figure) — рендерит только
render_chart. Семантика цвета: candlestick up/down и дивиденды — семантические токены;
портфель — акцент-тил, бенчмарк IMOEX — приглушённый mutedText (бенчмарк отступает);
мульти-серии — categorical-палитра; частотные слова — монохром-тил.
"""

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from dashboard.api_client.dto import CandleOut, DividendOut, FrontierPoint
from dashboard.components.transforms import DeltaDirection, SentimentDayPoint

#: Высота субплота объёма относительно цены (candlestick доминирует, объём — подложка).
_VOLUME_ROW_HEIGHT = 0.22

#: Цвета точки динамики тона по знаку среднего балла (diverging-логика, DESIGN §5).
_SENTIMENT_TREND_COLORS: dict[DeltaDirection, str] = {
    DeltaDirection.UP: theme.UP,
    DeltaDirection.DOWN: theme.DOWN,
    DeltaDirection.FLAT: theme.FLAT,
}

#: Центр шкалы тональности (score 0..1): выше — позитив, ниже — негатив.
_SENTIMENT_NEUTRAL_MIDPOINT = 0.5


def build_candlestick_chart(
    candles: Sequence[CandleOut],
    dividends: Sequence[DividendOut] = (),
) -> go.Figure:
    """Свечной график цены с субплотом объёма и дивидендными отсечками (DESIGN §5).

    up `#34D399` / down `#F87171` (семантические токены); объём — приглушённый тил;
    ex-дивидендные даты — вертикальные аннотации.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=[candle.trade_date for candle in candles],
            open=[float(candle.open) for candle in candles],
            high=[float(candle.high) for candle in candles],
            low=[float(candle.low) for candle in candles],
            close=[float(candle.close) for candle in candles],
            name="Цена",
            increasing_line_color=theme.UP,
            decreasing_line_color=theme.DOWN,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[candle.trade_date for candle in candles],
            y=[candle.volume for candle in candles],
            name="Объём",
            marker_color=theme.ACCENT_STRONG,
            opacity=0.4,
            yaxis="y2",
        )
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis={"domain": [_VOLUME_ROW_HEIGHT, 1.0], "title": "Цена"},
        yaxis2={"domain": [0.0, _VOLUME_ROW_HEIGHT - 0.02], "title": "Объём"},
    )
    _add_dividend_markers(fig, dividends)
    return theme.apply_dark_template(fig)


def _add_dividend_markers(fig: go.Figure, dividends: Sequence[DividendOut]) -> None:
    """Нанести вертикальные линии-аннотации ex-дивидендных дат на свечной график.

    Используются явные add_shape + add_annotation, а не add_vline: последний усредняет
    x-координаты для позиции аннотации и падает на date-оси (date не суммируется).
    Дата передаётся ISO-строкой — нативный формат date-оси Plotly.
    """
    for dividend in dividends:
        ex_date_iso = dividend.ex_date.isoformat()
        fig.add_shape(
            type="line",
            x0=ex_date_iso,
            x1=ex_date_iso,
            yref="paper",
            y0=0,
            y1=1,
            line={"color": theme.WARNING, "width": 1, "dash": "dot"},
        )
        fig.add_annotation(
            x=ex_date_iso,
            yref="paper",
            y=1,
            text="div",
            showarrow=False,
            font={"color": theme.WARNING, "size": 11},
            yshift=8,
        )


def build_portfolio_vs_imoex_chart(
    dates: Sequence[object],
    portfolio: Sequence[float],
    imoex: Sequence[float],
) -> go.Figure:
    """Линия/площадь: портфель (акцент-тил) vs IMOEX (приглушённый бенчмарк) (DESIGN §5).

    Портфель — основная серия акцентом; IMOEX-бенчмарк отступает приглушённым цветом.
    Применяется к equity-кривой бэктеста и динамике P&L.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(portfolio),
            name="Портфель",
            mode="lines",
            line={"color": theme.ACCENT, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(imoex),
            name="IMOEX",
            mode="lines",
            line={"color": theme.MUTED_TEXT, "width": 1.5, "dash": "dash"},
        )
    )
    return theme.apply_dark_template(fig)


def build_efficient_frontier_chart(
    frontier: Sequence[FrontierPoint],
    selected: tuple[float, float] | None = None,
    equal_weight: tuple[float, float] | None = None,
    imoex: tuple[float, float] | None = None,
) -> go.Figure:
    """Scatter эффективной границы Марковица с маркерами стратегий (DESIGN §5).

    Кривая фронтира (тил) + выбранная стратегия (акцент) + equal-weight + IMOEX как
    точки-бенчмарки. Координаты маркеров — (volatility, expected_return).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[point.volatility for point in frontier],
            y=[point.expected_return for point in frontier],
            name="Эффективная граница",
            mode="lines",
            line={"color": theme.ACCENT, "width": 2},
        )
    )
    _add_frontier_marker(fig, selected, "Стратегия", theme.LINK, "star")
    _add_frontier_marker(fig, equal_weight, "Равные веса", theme.CHART_CATEGORICAL[1], "circle")
    _add_frontier_marker(fig, imoex, "IMOEX", theme.MUTED_TEXT, "diamond")
    fig.update_layout(
        xaxis_title="Риск (волатильность)",
        yaxis_title="Ожидаемая доходность",
    )
    return theme.apply_dark_template(fig)


def _add_frontier_marker(
    fig: go.Figure,
    point: tuple[float, float] | None,
    name: str,
    color: str,
    symbol: str,
) -> None:
    """Нанести точку-бенчмарк (volatility, expected_return) на график фронтира."""
    if point is None:
        return
    volatility, expected_return = point
    fig.add_trace(
        go.Scatter(
            x=[volatility],
            y=[expected_return],
            name=name,
            mode="markers",
            marker={"color": color, "size": 12, "symbol": symbol},
        )
    )


def build_sentiment_trend_chart(series: Sequence[SentimentDayPoint]) -> go.Figure:
    """Линия динамики среднего тона по дням; цвет точки — по знаку (DESIGN §5).

    Линия-связка приглушённая; маркеры окрашены diverging-логикой (позитив/негатив/
    нейтраль) — знак среднего балла центрируется на 0.5 (диапазон score 0..1).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[point.day for point in series],
            y=[point.mean_score for point in series],
            name="Средний тон",
            mode="lines+markers",
            line={"color": theme.MUTED_TEXT, "width": 1.5},
            marker={
                "color": [_sentiment_marker_color(point.mean_score) for point in series],
                "size": 9,
            },
        )
    )
    fig.update_layout(yaxis_title="Средний балл тональности")
    return theme.apply_dark_template(fig)


def _sentiment_marker_color(mean_score: float) -> str:
    """Цвет маркера динамики тона по знаку среднего балла (центр шкалы score = 0.5)."""
    if mean_score > _SENTIMENT_NEUTRAL_MIDPOINT:
        return _SENTIMENT_TREND_COLORS[DeltaDirection.UP]
    if mean_score < _SENTIMENT_NEUTRAL_MIDPOINT:
        return _SENTIMENT_TREND_COLORS[DeltaDirection.DOWN]
    return _SENTIMENT_TREND_COLORS[DeltaDirection.FLAT]


def build_word_frequency_chart(frequencies: Sequence[tuple[str, int]]) -> go.Figure:
    """Горизонтальный бар топ-частотных слов; монохром-тил без облака (DESIGN §5).

    Бары упорядочены по возрастанию частоты сверху вниз (самое частое — сверху).
    """
    ordered = list(reversed(frequencies))
    fig = go.Figure(
        go.Bar(
            x=[count for _, count in ordered],
            y=[word for word, _ in ordered],
            orientation="h",
            marker_color=theme.ACCENT,
        )
    )
    fig.update_layout(xaxis_title="Частота")
    return theme.apply_dark_template(fig)


def render_chart(fig: go.Figure) -> None:
    """Отрендерить фигуру с нашим шаблоном (DESIGN §2.3): theme=None, контейнерная ширина.

    theme=None отключает палитру Streamlit — цвета берутся из apply_dark_template/токенов,
    без двойного применения темы.
    """
    st.plotly_chart(fig, theme=None, use_container_width=True)

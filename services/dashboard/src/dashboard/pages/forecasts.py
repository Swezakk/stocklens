"""Страница «Прогнозы» дашборда (DESIGN.md §10.4, ml-spec §10).

Волатильность: форвардный сигнал на 5 дн (вверху) + график «прогноз vs факт»
(реализованная за последующие 5 дней) + плашка метрик QLIKE модели vs baseline +
версия модели; выбор тикера. Тренд (P↑ + SHAP) — отложен до появления trend-модели
(показана честная строка, не интерактивная заглушка).

`render()` — тонкая оркестрация: нетривиальные преобразования вынесены в чистые
типизированные хелперы (`_ticker_options`, `_clamp_lookback`, `_format_metric`,
`_build_live_block`, `_build_forward_block`), покрытые unit-тестами. Каждый сетевой
вызов обрабатывает три ветки (успех / ошибка сервера / сеть недоступна) через
`components.feedback`.

Источник «факта» — реализованная волатильность `sqrt(rv_target)` из API (тот же показатель,
что таргетирует модель): дашборд её не считает (инвариант №3). График — track record
выпущенных прогнозов, не walk-forward кривая; число QLIKE — оффлайн-оценка из реестра.

Метрики разделены на два явных блока:
- «Бэктест (офлайн)» — QLIKE walk-forward на сотнях исторических точек из реестра моделей.
- «Боевые прогнозы (live)» — QLIKE по реальным созревшим прогнозам (n ≥ порога API).
Сравнивать корректно только модель vs baseline внутри каждого блока, не live vs офлайн.
"""

from collections.abc import Sequence
from typing import NamedTuple

import streamlit as st

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import (
    SecurityOut,
    VolatilityForecastHistoryOut,
    VolatilityMetricsOut,
    VolatilityRegimeOut,
)
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import fetch_all_securities, fetch_volatility_forecast_history
from dashboard.auth import get_api_client
from dashboard.components import filters
from dashboard.components.charts import build_forecast_vs_actual_chart, render_chart
from dashboard.components.feedback import render_empty, render_error
from dashboard.components.kpi import stat_cell
from dashboard.components.layout import card

#: Заголовок страницы и подписи секций (RU-копи — пользовательские строки).
_PAGE_TITLE = "Прогнозы"
_VOL_SECTION = "Волатильность: прогноз vs факт"
_TREND_SECTION = "Тренд"

#: Прочерк для недоступного значения (модель не загружена / нет версии).
_DASH = "—"

#: Границы окна истории, совпадают с Query(ge=5, le=365) эндпоинта /predict/volatility/history.
_MIN_LOOKBACK = 5
_MAX_LOOKBACK = 365

#: RU-копи пустых результатов и поясняющих строк.
_EMPTY_SECURITIES = "Список бумаг пуст: данные ещё не собраны."
_EMPTY_HISTORY = "Нет данных о волатильности по бумаге {ticker} за выбранный период."
_CHART_NOTE = (
    "Факт vs наши выпущенные прогнозы (не walk-forward кривая). Реализованная "
    "волатильность известна с задержкой 5 торговых дней — правый край ряда «исход ожидается»."
)
_BACKTEST_SECTION_LABEL = "Бэктест (офлайн)"
_LIVE_SECTION_LABEL = "Боевые прогнозы (live)"
_BACKTEST_NOTE = "QLIKE — walk-forward оценка на исторических данных (сотни точек); меньше — лучше."
_LIVE_NOTE_TEMPLATE = "QLIKE по {n} реальным созревшим прогнозам; меньше — лучше."
_LIVE_ACCUMULATING_TEMPLATE = "Боевая метрика накапливается: созрело {n} пар."
_METRICS_CAPTION = (
    "Офлайн = бэктест walk-forward на сотнях точек. "
    "Live = реальные созревшие прогнозы (n штук). "
    "Сравнивать стоит модель vs baseline внутри каждого блока, не live-число с офлайн-числом."
)
_NO_MODEL_NOTE = (
    "Модель волатильности не загружена: прогнозы и метрики недоступны, "
    "показана только реализованная волатильность."
)
_TREND_NOTE = "Прогноз тренда (вероятность роста и SHAP-объяснение) — с появлением trend-модели."

_FORWARD_VOL_LABEL = "Прогноз волатильности (5 дн)"
_FORWARD_REGIME_LABEL = "Режим"
_FORWARD_DATE_LABEL = "По данным на"
_FORWARD_REGIME_NORMAL = "Нормальный"
_FORWARD_REGIME_ELEVATED = "Повышенный"
_FORWARD_ELEVATED_WARNING = "повышенная волатильность"
_FORWARD_UNAVAILABLE = "Текущий прогноз недоступен (модель не загружена или мало истории)."
_FORWARD_CAPTION_TEMPLATE = (
    "Действующий сигнал: ожидаемая волатильность на ближайшие 5 торговых дней. "
    "«Повышенный» = выше {quantile_pct:.0f}-го перцентиля за {lookback} дн."
)

_VOL_PERCENT = 100.0

_DATE_FORMAT = "%d.%m.%Y"


class _ForwardBlock(NamedTuple):
    """Данные для отрисовки форвардного блока волатильности (5 дн).

    ``is_available=False`` когда forward=None (модель не загружена / мало истории) —
    остальные поля в этом случае не используются. Хелпер ``_build_forward_block`` гарантирует
    валидный объект в обоих состояниях, поэтому ``_render_forward`` только ветвится по флагу.
    """

    is_available: bool
    vol_pct: str
    regime_label: str
    is_elevated: bool
    date_label: str
    caption: str


def _build_forward_block(forward: VolatilityRegimeOut | None) -> _ForwardBlock:
    """Построить данные форвардного блока из ``VolatilityRegimeOut`` или ``None``.

    Чистая функция без зависимостей от Streamlit — покрывается unit-тестами.
    """
    if forward is None:
        return _ForwardBlock(
            is_available=False,
            vol_pct="",
            regime_label="",
            is_elevated=False,
            date_label="",
            caption="",
        )
    regime_label = _FORWARD_REGIME_ELEVATED if forward.is_elevated else _FORWARD_REGIME_NORMAL
    return _ForwardBlock(
        is_available=True,
        vol_pct=f"{forward.volatility * _VOL_PERCENT:.1f}%",
        regime_label=regime_label,
        is_elevated=forward.is_elevated,
        date_label=forward.predicted_for.strftime(_DATE_FORMAT),
        caption=_FORWARD_CAPTION_TEMPLATE.format(
            quantile_pct=forward.quantile * _VOL_PERCENT,
            lookback=forward.lookback,
        ),
    )


class _LiveBlock(NamedTuple):
    """Данные для отрисовки блока «Боевые прогнозы (live)».

    ``model_qlike`` / ``baseline_qlike`` — отформатированные значения или ``None``,
    когда метрик нет (режим накопления). ``annotation`` — пояснение с числом N.
    ``is_accumulating`` — ``True`` когда live_metrics отсутствуют (пар недостаточно).
    """

    model_qlike: str | None
    baseline_qlike: str | None
    annotation: str
    is_accumulating: bool


def _build_live_block(
    live_metrics: VolatilityMetricsOut | None,
    live_sample_size: int,
) -> _LiveBlock:
    """Построить данные live-блока из live_metrics и числа созревших пар.

    Порог достаточности пар определяет API, не дашборд: показываем N, не «N из порога».
    """
    if live_metrics is not None:
        return _LiveBlock(
            model_qlike=_format_metric(live_metrics.qlike),
            baseline_qlike=_format_metric(live_metrics.qlike_baseline),
            annotation=_LIVE_NOTE_TEMPLATE.format(n=live_sample_size),
            is_accumulating=False,
        )
    return _LiveBlock(
        model_qlike=None,
        baseline_qlike=None,
        annotation=_LIVE_ACCUMULATING_TEMPLATE.format(n=live_sample_size),
        is_accumulating=True,
    )


def _ticker_options(securities: Sequence[SecurityOut]) -> list[str]:
    """Список тикеров для селектора, упорядоченный по алфавиту (детерминизм UI)."""
    return sorted(security.ticker for security in securities)


def _clamp_lookback(period_days: int) -> int:
    """Привести период (дней) к допустимому окну эндпоинта `[5, 365]` (защита от HTTP 422)."""
    return max(_MIN_LOOKBACK, min(period_days, _MAX_LOOKBACK))


def _format_metric(value: float) -> str:
    """Отформатировать метрику QLIKE/RMSE: три знака после запятой (компактно, без шума)."""
    return f"{value:.3f}"


def render() -> None:
    """Отрисовать страницу «Прогнозы»: волатильность (прогноз vs факт) + заметка про тренд."""
    st.title(_PAGE_TITLE)
    client = get_api_client()

    securities = _load_securities(client)
    if securities is None:
        return
    tickers = _ticker_options(securities)
    if not tickers:
        render_empty(_EMPTY_SECURITIES)
        return

    with st.sidebar:
        ticker = filters.select_ticker(tickers, key="forecasts_ticker")
        period_days = filters.select_period(key="forecasts_period")
    if ticker is None:
        render_empty(_EMPTY_SECURITIES)
        return

    lookback = _clamp_lookback(period_days)
    with card("forecasts-volatility"):
        _render_volatility(client, ticker, lookback)
    with card("forecasts-trend"):
        _render_trend_note()


def _load_securities(client: ApiClient) -> list[SecurityOut] | None:
    """Загрузить список бумаг с обработкой трёх веток; None при ошибке (сообщение показано)."""
    try:
        return fetch_all_securities(client, is_active=True)
    except ApiError as exc:
        render_error(exc.user_message)
        return None


def _render_forward(forward: VolatilityRegimeOut | None) -> None:
    """Отрисовать форвардный блок (действующий сигнал) в верхней части секции волатильности."""
    block = _build_forward_block(forward)
    if not block.is_available:
        st.caption(_FORWARD_UNAVAILABLE)
        return
    col_vol, col_regime, col_date = st.columns(3)
    with col_vol:
        stat_cell(_FORWARD_VOL_LABEL, block.vol_pct)
    with col_regime:
        stat_cell(_FORWARD_REGIME_LABEL, block.regime_label)
    with col_date:
        stat_cell(_FORWARD_DATE_LABEL, block.date_label)
    if block.is_elevated:
        st.warning(_FORWARD_ELEVATED_WARNING)
    st.caption(block.caption)


def _render_volatility(client: ApiClient, ticker: str, lookback: int) -> None:
    """Отрисовать форвардный сигнал, плашку метрик и график «прогноз vs факт» волатильности."""
    st.subheader(_VOL_SECTION)
    try:
        history = fetch_volatility_forecast_history(client, ticker=ticker, lookback=lookback)
    except ApiError as exc:
        render_error(exc.user_message)
        return

    _render_forward(history.forward)
    _render_metrics(history)
    if not history.points:
        render_empty(_EMPTY_HISTORY.format(ticker=ticker))
        return

    render_chart(build_forecast_vs_actual_chart(history.points, forward=history.forward))
    st.caption(_CHART_NOTE)


def _render_metrics(history: VolatilityForecastHistoryOut) -> None:
    """Два явных блока: «Бэктест (офлайн)» и «Боевые прогнозы (live)».

    Офлайн-блок — QLIKE walk-forward из реестра + версия модели.
    Live-блок — QLIKE по созревшим прогнозам или сообщение о накоплении.
    Сравнение корректно только внутри блока (модель vs baseline), не между блоками.
    """
    offline_metrics = history.metrics_vs_baseline
    if offline_metrics is None:
        st.warning(_NO_MODEL_NOTE)
        return

    st.markdown(f"**{_BACKTEST_SECTION_LABEL}**")
    backtest_model_col, backtest_baseline_col, version_col = st.columns(3)
    with backtest_model_col:
        stat_cell("QLIKE (модель)", _format_metric(offline_metrics.qlike))
    with backtest_baseline_col:
        stat_cell("QLIKE (baseline)", _format_metric(offline_metrics.qlike_baseline))
    with version_col:
        stat_cell("Версия модели", history.model_version or _DASH)
    st.caption(_BACKTEST_NOTE)

    st.markdown(f"**{_LIVE_SECTION_LABEL}**")
    live_block = _build_live_block(history.live_metrics, history.live_sample_size)
    if live_block.is_accumulating:
        st.info(live_block.annotation)
    else:
        live_model_col, live_baseline_col = st.columns(2)
        with live_model_col:
            stat_cell("QLIKE (модель)", live_block.model_qlike or _DASH)
        with live_baseline_col:
            stat_cell("QLIKE (baseline)", live_block.baseline_qlike or _DASH)
        st.caption(live_block.annotation)

    st.caption(_METRICS_CAPTION)


def _render_trend_note() -> None:
    """Честная строка про отложенный прогноз тренда (без интерактивной заглушки)."""
    st.subheader(_TREND_SECTION)
    st.caption(_TREND_NOTE)

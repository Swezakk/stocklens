"""Страница «Прогнозы» дашборда (DESIGN.md §10.4, ml-spec §10).

Волатильность: график «прогноз vs факт» (реализованная за последующие 5 дней) + плашка
метрик QLIKE модели vs baseline + версия модели; выбор тикера. Тренд (P↑ + SHAP) —
отложен до появления trend-модели (показана честная строка, не интерактивная заглушка).

`render()` — тонкая оркестрация: нетривиальные преобразования вынесены в чистые
типизированные хелперы (`_ticker_options`, `_clamp_lookback`, `_format_metric`), покрытые
unit-тестами. Каждый сетевой вызов обрабатывает три ветки (успех / ошибка сервера / сеть
недоступна) через `components.feedback`.

Источник «факта» — реализованная волатильность `sqrt(rv_target)` из API (тот же показатель,
что таргетирует модель): дашборд её не считает (инвариант №3). График — track record
выпущенных прогнозов, не walk-forward кривая; число QLIKE — оффлайн-оценка из реестра.
"""

from collections.abc import Sequence

import streamlit as st

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import SecurityOut, VolatilityForecastHistoryOut
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
_QLIKE_NOTE = "QLIKE — оффлайн-оценка модели против naive baseline (меньше — лучше)."
_NO_MODEL_NOTE = (
    "Модель волатильности не загружена: прогнозы и метрики недоступны, "
    "показана только реализованная волатильность."
)
_TREND_NOTE = "Прогноз тренда (вероятность роста и SHAP-объяснение) — с появлением trend-модели."


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


def _render_volatility(client: ApiClient, ticker: str, lookback: int) -> None:
    """Отрисовать плашку метрик и график «прогноз vs факт» волатильности (три ветки вызова)."""
    st.subheader(_VOL_SECTION)
    try:
        history = fetch_volatility_forecast_history(client, ticker=ticker, lookback=lookback)
    except ApiError as exc:
        render_error(exc.user_message)
        return

    _render_metrics(history)
    if not history.points:
        render_empty(_EMPTY_HISTORY.format(ticker=ticker))
        return

    render_chart(build_forecast_vs_actual_chart(history.points))
    st.caption(_CHART_NOTE)


def _render_metrics(history: VolatilityForecastHistoryOut) -> None:
    """Плашка QLIKE модели/baseline и версии модели; degrade-заметка при незагруженной модели."""
    metrics = history.metrics_vs_baseline
    model_col, baseline_col, version_col = st.columns(3)
    with model_col:
        stat_cell("QLIKE (модель)", _format_metric(metrics.qlike) if metrics else _DASH)
    with baseline_col:
        stat_cell("QLIKE (baseline)", _format_metric(metrics.qlike_baseline) if metrics else _DASH)
    with version_col:
        stat_cell("Версия модели", history.model_version or _DASH)

    st.caption(_NO_MODEL_NOTE if metrics is None else _QLIKE_NOTE)


def _render_trend_note() -> None:
    """Честная строка про отложенный прогноз тренда (без интерактивной заглушки)."""
    st.subheader(_TREND_SECTION)
    st.caption(_TREND_NOTE)

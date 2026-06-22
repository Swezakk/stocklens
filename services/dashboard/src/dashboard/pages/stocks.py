"""Страница «Акции» дашборда (DESIGN.md §10.2).

Sidebar-фильтры: тикер (из `/data/securities`) и период. Главный вид — свечной график
цены с субплотом объёма и дивидендными отсечками (`/data/candles` + `/data/dividends`).
Доп-режим — сравнение нескольких бумаг нормированными close-сериями (rebase-to-100,
categorical-палитра).

`render()` — тонкая оркестрация: всё нетривиальное преобразование данных вынесено в
чистые типизированные хелперы (`_period_bounds`, `_ticker_options`,
`_sort_candles_ascending`, `_latest_close_delta`, `_dividends_in_window`), покрытые
unit-тестами. Каждый сетевой вызов обрабатывает три ветки (успех / ошибка сервера /
сеть недоступна) через `components.feedback`; пустой результат — `render_empty` с RU-копи.

Свечи `/data/candles` приходят по `trade_date DESC` (новейшая первая): хелперы это
учитывают — для оси графика серия сортируется по возрастанию, а «последний close»
читается с начала исходного DESC-списка.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import CandleOut, DividendOut, SecurityPage
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import (
    fetch_candles,
    fetch_dividends,
    fetch_securities,
    get_client,
)
from dashboard.auth import (
    build_on_unauthorized,
    build_token_provider,
    get_token_manager,
)
from dashboard.components import filters
from dashboard.components.charts import (
    build_candlestick_chart,
    build_comparison_chart,
    render_chart,
)
from dashboard.components.feedback import render_empty, render_error
from dashboard.components.kpi import delta_badge_from_values, stat_cell

#: Заголовок страницы (RU-копи — пользовательская строка).
_PAGE_TITLE = "Акции"

#: Подписи режимов: одиночный свечной vs сравнение нескольких бумаг (DESIGN §10.2).
_TAB_SINGLE = "Свечи"
_TAB_COMPARE = "Сравнение"

#: Метка KPI последней цены закрытия.
_LATEST_CLOSE_LABEL = "Последний close"

#: Подписи виджетов и секций (RU-копи).
_COMPARE_TICKERS_LABEL = "Бумаги для сравнения"
_COMPARE_KEY = "stocks_compare_tickers"
_PRICE_SECTION = "Цена и объём"
_COMPARE_SECTION = "Сравнение динамики"

#: RU-копи пустых результатов трёх вызовов (запрос прошёл, но данных под фильтр нет).
_EMPTY_SECURITIES = "Список бумаг пуст: данные ещё не собраны."
_EMPTY_CANDLES = "Нет свечей по бумаге {ticker} за выбранный период."
_EMPTY_COMPARE_PICK = "Выберите хотя бы две бумаги, чтобы сравнить динамику."
_EMPTY_COMPARE_DATA = "Нет данных по выбранным бумагам за период для сравнения."

#: Минимум бумаг для осмысленного сравнения динамики.
_MIN_COMPARE_TICKERS = 2

#: Минимум свечей для расчёта дельты дня (текущий close против предыдущего).
_MIN_CANDLES_FOR_DELTA = 2

#: Предел свечей за период: годовой максимум периодов (365) + запас на сессии выходного дня.
_CANDLE_LIMIT = 400

#: Потолок выборки бумаг: список MOEX-бумаг дашборда заведомо умещается на одной странице.
_SECURITIES_LIMIT = 500

#: Часовой пояс отсчёта периода: окно свечей считается по московскому календарю (CLAUDE.md).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def _period_bounds(period_days: int, reference: date) -> tuple[str, str]:
    """ISO-границы периода `[reference - period_days, reference]` для фильтра свечей.

    `reference` инъектируется (не читается из часов внутри) — функция чистая и
    unit-тестируема без подмены времени. Возвращает ISO-строки `(date_from, date_to)`
    в формате, который ждёт API (`date | None`).
    """
    date_from = reference - timedelta(days=period_days)
    return date_from.isoformat(), reference.isoformat()


def _ticker_options(page: SecurityPage) -> list[str]:
    """Список тикеров для селектора, упорядоченный по алфавиту (детерминизм UI).

    Берёт тикеры из страницы бумаг как есть (фильтр активности — на стороне fetch);
    сортировка делает порядок опций стабильным независимо от порядка ответа API.
    """
    return sorted(security.ticker for security in page.items)


def _sort_candles_ascending(candles: Sequence[CandleOut]) -> list[CandleOut]:
    """Отсортировать свечи по возрастанию `trade_date` для хронологической оси графика.

    `/data/candles` отдаёт `trade_date DESC` (новейшая первая); свечной билдер ожидает
    хронологический порядок слева направо. Вход не мутируется (возвращается новый список).
    """
    return sorted(candles, key=lambda candle: candle.trade_date)


def _latest_close_delta(candles: Sequence[CandleOut]) -> tuple[float, float] | None:
    """Пара `(последний close, предыдущий close)` для KPI-дельты дня; None, если данных мало.

    Свечи приходят `DESC` (новейшая — индекс 0), поэтому последний close — `candles[0]`,
    предыдущий — `candles[1]`. Decimal приводится к float: kpi-хелперы float-only.
    Меньше двух свечей → None (дельту не от чего считать).
    """
    if len(candles) < _MIN_CANDLES_FOR_DELTA:
        return None
    return float(candles[0].close), float(candles[1].close)


def _dividends_in_window(
    dividends: Sequence[DividendOut],
    date_from: str,
    date_to: str,
) -> list[DividendOut]:
    """Оставить дивидендные отсечки, чьи ex-даты попадают в окно графика `[date_from, date_to]`.

    `/data/dividends` не фильтруется по дате (вся история бумаги), а отсечки рисуются только
    в пределах видимого периода свечей — иначе аннотации повиснут вне области графика.
    Границы — ISO-строки периода; сравнение по `date` (ISO-строки сравнимы лексикографически,
    но сопоставляем типобезопасно через `date.fromisoformat`).
    """
    window_from = date.fromisoformat(date_from)
    window_to = date.fromisoformat(date_to)
    return [dividend for dividend in dividends if window_from <= dividend.ex_date <= window_to]


def _resolve_client() -> ApiClient:
    """Получить singleton ApiClient, привязанный к auth-менеджеру сессии (DESIGN §6, §7).

    Импурно (session_state + cache_resource), поэтому живёт в orchestration, а не в
    чистых хелперах. Провайдер токена и хук 401 — из auth.py.
    """
    manager = get_token_manager()
    return get_client(
        build_token_provider(manager),
        build_on_unauthorized(manager),
    )


def render() -> None:
    """Отрисовать страницу «Акции»: фильтры → свечи/объём/дивиденды + режим сравнения."""
    st.title(_PAGE_TITLE)
    client = _resolve_client()

    securities = _load_securities(client)
    if securities is None:
        return
    tickers = _ticker_options(securities)
    if not tickers:
        render_empty(_EMPTY_SECURITIES)
        return

    with st.sidebar:
        ticker = filters.select_ticker(tickers, key="stocks_ticker")
        period_days = filters.select_period(key="stocks_period")
    if ticker is None:
        render_empty(_EMPTY_SECURITIES)
        return

    date_from, date_to = _period_bounds(period_days, datetime.now(tz=_MOSCOW_TZ).date())

    single_tab, compare_tab = st.tabs([_TAB_SINGLE, _TAB_COMPARE])
    with single_tab:
        _render_single(client, ticker, date_from, date_to)
    with compare_tab:
        _render_comparison(client, tickers, date_from, date_to)


def _load_securities(client: ApiClient) -> SecurityPage | None:
    """Загрузить список бумаг с обработкой трёх веток; None при ошибке (сообщение показано)."""
    try:
        return fetch_securities(client, is_active=True, limit=_SECURITIES_LIMIT)
    except ApiError as exc:
        render_error(exc.user_message)
        return None


def _render_single(client: ApiClient, ticker: str, date_from: str, date_to: str) -> None:
    """Отрисовать свечной график с объёмом, дивидендными отсечками и KPI последнего close."""
    st.subheader(_PRICE_SECTION)
    try:
        candle_page = fetch_candles(
            client,
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
            limit=_CANDLE_LIMIT,
        )
        dividend_page = fetch_dividends(client, ticker=ticker, limit=_CANDLE_LIMIT)
    except ApiError as exc:
        render_error(exc.user_message)
        return

    candles = candle_page.items
    if not candles:
        render_empty(_EMPTY_CANDLES.format(ticker=ticker))
        return

    _render_latest_close(candles)
    dividends = _dividends_in_window(dividend_page.items, date_from, date_to)
    fig = build_candlestick_chart(_sort_candles_ascending(candles), dividends)
    render_chart(fig)


def _render_latest_close(candles: Sequence[CandleOut]) -> None:
    """Показать KPI-ячейку последнего close с дельтой ко вчерашнему дню (если есть пара свечей)."""
    pair = _latest_close_delta(candles)
    if pair is None:
        stat_cell(_LATEST_CLOSE_LABEL, f"{float(candles[0].close):.2f}")
        return
    latest, previous = pair
    badge = delta_badge_from_values(current=latest, previous=previous)
    stat_cell(_LATEST_CLOSE_LABEL, f"{latest:.2f}", delta=badge)


def _render_comparison(
    client: ApiClient,
    tickers: Sequence[str],
    date_from: str,
    date_to: str,
) -> None:
    """Отрисовать сравнение нормированных close-серий нескольких бумаг (rebase-to-100)."""
    st.subheader(_COMPARE_SECTION)
    selected = st.multiselect(
        _COMPARE_TICKERS_LABEL,
        options=list(tickers),
        key=_COMPARE_KEY,
    )
    if len(selected) < _MIN_COMPARE_TICKERS:
        render_empty(_EMPTY_COMPARE_PICK)
        return

    series_by_ticker = _load_comparison_series(client, selected, date_from, date_to)
    if series_by_ticker is None:
        return
    if not any(series for series in series_by_ticker.values()):
        render_empty(_EMPTY_COMPARE_DATA)
        return

    fig = build_comparison_chart(series_by_ticker)
    render_chart(fig)


def _load_comparison_series(
    client: ApiClient,
    selected: Sequence[str],
    date_from: str,
    date_to: str,
) -> dict[str, list[CandleOut]] | None:
    """Загрузить close-серии выбранных бумаг (хронологически) для сравнения; None при ошибке.

    Серии сортируются по возрастанию даты — `build_comparison_chart` рисует слева направо.
    """
    series_by_ticker: dict[str, list[CandleOut]] = {}
    try:
        for ticker in selected:
            page = fetch_candles(
                client,
                ticker=ticker,
                date_from=date_from,
                date_to=date_to,
                limit=_CANDLE_LIMIT,
            )
            series_by_ticker[ticker] = _sort_candles_ascending(page.items)
    except ApiError as exc:
        render_error(exc.user_message)
        return None
    return series_by_ticker

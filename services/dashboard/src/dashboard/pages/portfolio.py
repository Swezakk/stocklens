"""Страница «Портфель» дашборда (DESIGN.md §10.5).

Состав (DESIGN §10.5):
- таблица позиций с текущей оценкой и нереализованным P&L + суммарный P&L бейджем;
- форма добавления и удаление позиции — **единственный write-путь дашборда** (DESIGN §5):
  POST /portfolio/positions и DELETE /portfolio/positions/{ticker} через ApiClient;
- equity-кривая бэктеста «портфель vs IMOEX» + риск-метрики (Sharpe, max drawdown)
  карточками StatCell против бенчмарка;
- эффективная граница Марковица (efficient frontier) с маркерами стратегий.

Все сетевые вызовы проходят три ветки (успех / ошибка сервера / сеть недоступна) через
components.feedback — пустых экранов без объяснения нет (DESIGN §10). Денежные поля DTO —
Decimal: приводятся к float только на границе KPI-хелперов (они float-only). Метки времени
API — UTC, отображаются по Europe/Moscow.

render() — тонкая оркестрация: всё нетривиальное форматирование вынесено в чистые
типизированные хелперы (`_position_row`, `_format_pnl`, …) и покрыто unit-тестами; сам
UI-layout не тестируется (DESIGN §10).
"""

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import (
    BacktestResultOut,
    EquityPointOut,
    OptimizeResult,
    PortfolioSummaryOut,
    PositionIn,
    PositionOut,
)
from dashboard.api_client.errors import ApiError, ApiServerError
from dashboard.api_client.fetch import (
    fetch_backtest,
    fetch_optimize,
    fetch_portfolio_summary,
)
from dashboard.auth import get_api_client
from dashboard.components import charts, feedback
from dashboard.components.kpi import delta_badge_from_values, render_delta_badge, stat_cell
from dashboard.components.transforms import DELTA_GLYPHS, DeltaDirection

#: Заголовок страницы (RU-копи — пользовательская строка).
_PAGE_TITLE = "Портфель"

#: Часовой пояс отображения меток времени (DESIGN: отображение — Europe/Moscow).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")

#: Типографский минус U+2212 для отрицательных сумм (DESIGN §2.2), не ASCII «-».
_MINUS_SIGN = "−"

#: Заглушка для отсутствующей рыночной оценки (нет свежей котировки по бумаге).
_NO_DATA_PLACEHOLDER = "—"

#: Точность отображения денежных сумм (рубли с копейками).
_MONEY_QUANTUM = Decimal("0.01")

#: Глубина истории для сводки/оптимизации (год — дефолт period_days API, DESIGN §9).
_SUMMARY_PERIOD_DAYS = 365

#: Глубина бэктеста equity-кривой в месяцах (дефолт months_back API).
_BACKTEST_MONTHS_BACK = 12

#: HTTP 422: API так сигналит «портфель пуст / истории/тикеров недостаточно» (роутер
#: portfolio.py) — для пользователя это пустое состояние первого запуска, не сбой сервиса.
_HTTP_UNPROCESSABLE_ENTITY = 422

#: RU-заголовки колонок таблицы позиций (они же ключи строки _PositionRow).
_COL_TICKER = "Тикер"
_COL_QUANTITY = "Количество"
_COL_AVG_PRICE = "Средняя цена"
_COL_CURRENT_PRICE = "Текущая цена"
_COL_CURRENT_VALUE = "Стоимость"
_COL_PNL = "P&L"

#: Порядок колонок таблицы позиций (стабилен и при пустом наборе строк).
_POSITION_COLUMNS = (
    _COL_TICKER,
    _COL_QUANTITY,
    _COL_AVG_PRICE,
    _COL_CURRENT_PRICE,
    _COL_CURRENT_VALUE,
    _COL_PNL,
)

#: RU-копи пустых состояний и сообщений формы.
_EMPTY_POSITIONS = "Портфель пуст: добавьте первую позицию через форму ниже."
_EMPTY_EQUITY = "Недостаточно истории котировок для построения кривой капитала."
_EMPTY_FRONTIER = "Недостаточно данных для построения эффективной границы (нужно ≥ 2 бумаги)."
_POSITION_CREATED = "Позиция по тикеру {ticker} сохранена."
_POSITION_DELETED = "Позиция по тикеру {ticker} удалена."
_DELETE_NEEDS_TICKER = "Не удалось удалить позицию: тикер не выбран."

#: Ключи виджетов формы (разводят виджеты Streamlit по ключу, без хардкода в логике).
_KEY_NEW_TICKER = "portfolio_new_ticker"
_KEY_NEW_QUANTITY = "portfolio_new_quantity"
_KEY_NEW_PRICE = "portfolio_new_price"
_KEY_NEW_DATE = "portfolio_new_date"
_KEY_DELETE_TICKER = "portfolio_delete_ticker"


#: Тип строки таблицы позиций: RU-заголовок колонки → отформатированное значение.
_PositionRow = dict[str, str | int]


def _format_money(value: Decimal | None) -> str:
    """Форматировать денежную сумму (рубли, 2 знака); None → заглушка «—».

    Отрицательные суммы несут типографский минус U+2212 (DESIGN §2.2), не ASCII «-».
    """
    if value is None:
        return _NO_DATA_PLACEHOLDER
    quantized = value.quantize(_MONEY_QUANTUM)
    if quantized < 0:
        return f"{_MINUS_SIGN}{abs(quantized):.2f}"
    return f"{quantized:.2f}"


def _pnl_direction(value: Decimal) -> DeltaDirection:
    """Классифицировать знак P&L: > 0 рост, < 0 падение, = 0 без изменения (DESIGN §2.2)."""
    if value > 0:
        return DeltaDirection.UP
    if value < 0:
        return DeltaDirection.DOWN
    return DeltaDirection.FLAT


def _format_pnl(value: Decimal | None) -> str:
    """Форматировать нереализованный P&L со знаком (DESIGN §2.2); None → заглушка «—».

    Знак — явный канал-дубль помимо цвета (a11y): «+» для роста, U+2212 для падения,
    без знака для нуля.
    """
    if value is None:
        return _NO_DATA_PLACEHOLDER
    quantized = value.quantize(_MONEY_QUANTUM)
    direction = _pnl_direction(quantized)
    if direction is DeltaDirection.UP:
        return f"+{quantized:.2f}"
    if direction is DeltaDirection.DOWN:
        return f"{_MINUS_SIGN}{abs(quantized):.2f}"
    return f"{quantized:.2f}"


def _vs_imoex_delta(
    portfolio_return_pct: float,
    imoex_return_pct: float,
) -> tuple[DeltaDirection, str]:
    """Разница доходности портфеля и IMOEX в процентных пунктах (DESIGN §2.2, §10.5).

    Бенчмарк IMOEX — это уже проценты, поэтому сравнение ведётся в процентных ПУНКТАХ
    (portfolio − imoex), не в относительном изменении: «портфель обгоняет индекс на N п.п.».
    Знак и глиф — каналы-дубли помимо цвета (a11y).
    """
    gap = portfolio_return_pct - imoex_return_pct
    if gap > 0:
        return DeltaDirection.UP, f"{DELTA_GLYPHS[DeltaDirection.UP]} +{gap:.2f} п.п."
    if gap < 0:
        return (
            DeltaDirection.DOWN,
            f"{DELTA_GLYPHS[DeltaDirection.DOWN]} {_MINUS_SIGN}{abs(gap):.2f} п.п.",
        )
    return DeltaDirection.FLAT, f"{DELTA_GLYPHS[DeltaDirection.FLAT]} 0.00 п.п."


def _position_row(position: PositionOut) -> _PositionRow:
    """Собрать строку таблицы позиций из PositionOut (чистое форматирование, DESIGN §5).

    Денежные поля Decimal форматируются строками с типографским минусом; отсутствующая
    рыночная оценка (нет свежей котировки) показывается заглушкой «—», не нулём.
    """
    return {
        _COL_TICKER: position.ticker,
        _COL_QUANTITY: position.quantity,
        _COL_AVG_PRICE: _format_money(position.avg_price),
        _COL_CURRENT_PRICE: _format_money(position.current_price),
        _COL_CURRENT_VALUE: _format_money(position.current_value),
        _COL_PNL: _format_pnl(position.unrealized_pnl),
    }


def _positions_dataframe(positions: Sequence[PositionOut]) -> pd.DataFrame:
    """Собрать DataFrame таблицы позиций с RU-заголовками колонок (DESIGN §5)."""
    rows = [_position_row(position) for position in positions]
    return pd.DataFrame(rows, columns=_POSITION_COLUMNS)


def _equity_series(
    curve: Sequence[EquityPointOut],
) -> tuple[list[date], list[float], list[float]]:
    """Разложить equity-кривую бэктеста в параллельные серии для charts-билдера.

    Возвращает `(даты, портфель, IMOEX)` — ровно форма, которую ждёт
    build_portfolio_vs_imoex_chart (DTO-агностичный билдер). Пустая кривая → три
    пустых списка (страница показывает render_empty, не строит график).
    """
    dates = [point.date for point in curve]
    portfolio = [point.portfolio for point in curve]
    imoex = [point.imoex for point in curve]
    return dates, portfolio, imoex


def _ticker_options(positions: Sequence[PositionOut]) -> list[str]:
    """Тикеры текущих позиций для селектора удаления (отсортированы детерминированно)."""
    return sorted(position.ticker for position in positions)


def _is_empty_state_error(error: ApiError) -> bool:
    """True, если ошибка API — это 422 «данных недостаточно» (пустое состояние, не сбой).

    Роутер портфеля отвечает 422 при пустом портфеле / нехватке истории / < 2 тикеров —
    для владельца на первом запуске это пустое состояние, которое показывается через
    render_empty, а не красным баннером ошибки сервера (DESIGN §10: внятные пустые экраны).
    """
    return isinstance(error, ApiServerError) and error.status == _HTTP_UNPROCESSABLE_ENTITY


def _render_load_failure(error: ApiError, empty_message: str) -> None:
    """Отрисовать ветку неуспеха загрузки: 422 → пустое состояние, иначе → ошибка сервиса."""
    if _is_empty_state_error(error):
        feedback.render_empty(empty_message)
    else:
        feedback.render_error(error.user_message)


def render() -> None:
    """Отрисовать страницу «Портфель» (DESIGN §10.5) — тонкая оркестрация."""
    st.title(_PAGE_TITLE)
    client = get_api_client()

    summary = _load_summary(client)
    _render_positions_section(client, summary)
    _render_equity_section(client)
    _render_frontier_section(client)


def _load_summary(client: ApiClient) -> PortfolioSummaryOut | None:
    """Загрузить сводку портфеля (три ветки сетевого вызова, DESIGN §10).

    422 «портфель пуст / истории недостаточно» рендерится пустым состоянием (первый запуск
    владельца), прочие ошибки — баннером сервиса; None означает «таблицы/P&L нет».
    """
    try:
        return fetch_portfolio_summary(client, period_days=_SUMMARY_PERIOD_DAYS)
    except ApiError as exc:
        _render_load_failure(exc, _EMPTY_POSITIONS)
        return None


def _render_positions_section(
    client: ApiClient,
    summary: PortfolioSummaryOut | None,
) -> None:
    """Секция позиций: таблица + суммарный P&L + write-форма и удаление (DESIGN §10.5)."""
    st.subheader("Позиции и P&L", anchor=False)
    positions = summary.positions if summary is not None else []
    if summary is not None and positions:
        _render_positions_table(positions)
        _render_total_pnl(summary)
    elif summary is not None:
        feedback.render_empty(_EMPTY_POSITIONS)

    _render_position_form(client)
    _render_delete_position(client, positions)


def _render_positions_table(positions: Sequence[PositionOut]) -> None:
    """Отрисовать таблицу позиций (st.dataframe, RU-заголовки, табличные цифры).

    Денежные колонки — TextColumn: значения уже отформатированы строками (типографский
    минус и заглушка «—»), а NumberColumn-формат не поддерживает оба этих случая.
    """
    st.dataframe(
        _positions_dataframe(positions),
        hide_index=True,
        use_container_width=True,
        column_config={
            _COL_TICKER: st.column_config.TextColumn(_COL_TICKER),
            _COL_QUANTITY: st.column_config.NumberColumn(_COL_QUANTITY, format="%d"),
            _COL_AVG_PRICE: st.column_config.TextColumn(_COL_AVG_PRICE),
            _COL_CURRENT_PRICE: st.column_config.TextColumn(_COL_CURRENT_PRICE),
            _COL_CURRENT_VALUE: st.column_config.TextColumn(_COL_CURRENT_VALUE),
            _COL_PNL: st.column_config.TextColumn(_COL_PNL),
        },
    )


def _render_total_pnl(summary: PortfolioSummaryOut) -> None:
    """Суммарная оценка портфеля + нереализованный P&L бейджем (DESIGN §5, §10.5)."""
    pnl_direction = _pnl_direction(summary.total_unrealized_pnl)
    pnl_text = f"{_format_pnl(summary.total_unrealized_pnl)} ₽"
    pnl_badge = render_delta_badge(pnl_direction, pnl_text)
    vs_imoex_direction, vs_imoex_text = _vs_imoex_delta(
        summary.portfolio_return_pct,
        summary.imoex_return_pct,
    )
    value_col, return_col = st.columns(2)
    with value_col:
        stat_cell(
            "Стоимость портфеля",
            f"{_format_money(summary.total_value)} ₽",
            delta=pnl_badge,
        )
    with return_col:
        stat_cell(
            "Доходность портфеля",
            f"{summary.portfolio_return_pct:.2f}%",
            delta=render_delta_badge(vs_imoex_direction, vs_imoex_text),
        )


def _render_position_form(client: ApiClient) -> None:
    """Форма добавления/обновления позиции — write-путь дашборда (DESIGN §5).

    На сабмит: валидация формы → PositionIn → client.create_position. Успех чистит кэш
    сводки и делает rerun; ошибки (валидация / сеть / сервер) — RU-копи через feedback.
    """
    with st.expander("Добавить или обновить позицию", icon=":material/add:"):
        with st.form("portfolio_position_form", clear_on_submit=True):
            ticker = st.text_input("Тикер", key=_KEY_NEW_TICKER)
            quantity = st.number_input(
                "Количество",
                min_value=1,
                step=1,
                value=1,
                key=_KEY_NEW_QUANTITY,
            )
            avg_price = st.number_input(
                "Средняя цена, ₽",
                min_value=0.01,
                step=0.01,
                value=0.01,
                format="%.2f",
                key=_KEY_NEW_PRICE,
            )
            opened_on = st.date_input("Дата открытия", key=_KEY_NEW_DATE)
            submitted = st.form_submit_button("Сохранить позицию", type="primary")
        if submitted:
            _submit_position(client, ticker, int(quantity), avg_price, opened_on)


def _submit_position(
    client: ApiClient,
    ticker: str,
    quantity: int,
    avg_price: float,
    opened_on: date,
) -> None:
    """Собрать PositionIn, отправить POST и обработать три ветки результата (DESIGN §7)."""
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        feedback.render_error("Не удалось сохранить позицию: укажите тикер.")
        return
    try:
        position = PositionIn(
            ticker=normalized_ticker,
            quantity=quantity,
            avg_price=Decimal(str(avg_price)),
            opened_at=datetime.combine(opened_on, datetime.min.time(), tzinfo=_MOSCOW_TZ),
        )
    except ValueError as exc:
        feedback.render_error(f"Не удалось сохранить позицию: {exc}")
        return

    try:
        client.create_position(position.model_dump(mode="json"))
    except ApiError as exc:
        feedback.render_error(exc.user_message)
        return

    _invalidate_portfolio_caches()
    st.success(_POSITION_CREATED.format(ticker=normalized_ticker), icon=":material/check:")
    st.rerun()


def _render_delete_position(client: ApiClient, positions: Sequence[PositionOut]) -> None:
    """Удаление позиции по тикеру (write-путь): селектор + кнопка, три ветки (DESIGN §5)."""
    options = _ticker_options(positions)
    if not options:
        return
    with st.expander("Удалить позицию", icon=":material/delete:"):
        ticker = st.selectbox("Тикер позиции", options=options, key=_KEY_DELETE_TICKER)
        if st.button("Удалить позицию", type="secondary"):
            _submit_delete(client, ticker)


def _submit_delete(client: ApiClient, ticker: str | None) -> None:
    """Отправить DELETE позиции и обработать три ветки результата (DESIGN §7)."""
    if not ticker:
        feedback.render_error(_DELETE_NEEDS_TICKER)
        return
    try:
        client.delete_position(ticker)
    except ApiError as exc:
        feedback.render_error(exc.user_message)
        return

    _invalidate_portfolio_caches()
    st.success(_POSITION_DELETED.format(ticker=ticker), icon=":material/check:")
    st.rerun()


def _invalidate_portfolio_caches() -> None:
    """Сбросить кэш fetch портфеля после write (DESIGN §8: ручной точечный .clear()).

    Чистятся обёртки, чьи данные меняет write-путь: сводка, бэктест и оптимизация зависят
    от состава позиций, поэтому после create/delete их кэш невалиден.
    """
    fetch_portfolio_summary.clear()
    fetch_backtest.clear()
    fetch_optimize.clear()


def _render_equity_section(client: ApiClient) -> None:
    """Equity-кривая бэктеста портфель vs IMOEX + риск-метрики карточками (DESIGN §10.5)."""
    st.subheader("Динамика портфеля vs IMOEX", anchor=False)
    backtest = _load_backtest(client)
    if backtest is None:
        return
    if not backtest.equity_curve:
        feedback.render_empty(_EMPTY_EQUITY)
        return

    dates, portfolio, imoex = _equity_series(backtest.equity_curve)
    charts.render_chart(charts.build_portfolio_vs_imoex_chart(dates, portfolio, imoex))
    _render_risk_metrics(backtest)


def _load_backtest(client: ApiClient) -> BacktestResultOut | None:
    """Загрузить бэктест equity-кривой (три ветки сетевого вызова, DESIGN §10).

    422 «портфель пуст / данных недостаточно» — пустое состояние, не сбой сервиса.
    """
    try:
        return fetch_backtest(client, months_back=_BACKTEST_MONTHS_BACK)
    except ApiError as exc:
        _render_load_failure(exc, _EMPTY_EQUITY)
        return None


def _render_risk_metrics(backtest: BacktestResultOut) -> None:
    """Риск-метрики бэктеста (Sharpe, max drawdown) карточками vs IMOEX (DESIGN §10.5)."""
    sharpe_col, drawdown_col = st.columns(2)
    with sharpe_col:
        stat_cell(
            "Sharpe портфеля",
            f"{backtest.portfolio_sharpe:.2f}",
            delta=delta_badge_from_values(
                current=backtest.portfolio_sharpe,
                previous=backtest.imoex_sharpe,
            ),
        )
    with drawdown_col:
        stat_cell(
            "Макс. просадка портфеля",
            f"{backtest.portfolio_max_drawdown * 100:.2f}%",
            delta=delta_badge_from_values(
                current=backtest.portfolio_max_drawdown,
                previous=backtest.imoex_max_drawdown,
            ),
        )


def _render_frontier_section(client: ApiClient) -> None:
    """Эффективная граница Марковица с маркерами стратегий (DESIGN §10.5)."""
    st.subheader("Эффективная граница", anchor=False)
    result = _load_optimize(client)
    if result is None:
        return
    if not result.frontier:
        feedback.render_empty(_EMPTY_FRONTIER)
        return

    chart = charts.build_efficient_frontier_chart(
        frontier=result.frontier,
        selected=(result.volatility, result.expected_return),
    )
    charts.render_chart(chart)


def _load_optimize(client: ApiClient) -> OptimizeResult | None:
    """Запросить оптимизацию текущих позиций (три ветки сетевого вызова, DESIGN §10).

    Оптимизация кэшируется по нормализованным параметрам (period_days, стратегия max-Sharpe;
    tickers=None — текущие позиции); дорогой солвер Марковица не пересчитывается на каждый
    rerun. 422 «нужно ≥ 2 бумаги» — валидная ошибка сервера; ``cache_data`` не кэширует
    исключения, поэтому пустое состояние перерешается заново при следующем rerun.
    """
    try:
        return fetch_optimize(client, period_days=_SUMMARY_PERIOD_DAYS)
    except ApiError as exc:
        _render_load_failure(exc, _EMPTY_FRONTIER)
        return None

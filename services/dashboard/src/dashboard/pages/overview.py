"""Страница «Обзор» дашборда (DESIGN.md §10.1).

Состав (DESIGN §10.1):
- Верх — KPI-полоса StatCell: IMOEX + дельта дня, USD/EUR/CNY, ключевая ставка ЦБ,
  каждая со спарклайном (build_index_line_chart(sparkline=True)).
- Центр — две компактные таблицы муверов (рост / падение) с DeltaBadge на строку.
- Низ — линия IMOEX за выбранный период (filters.select_period).
- Футер — «Данные обновлены: <время МСК>» из последнего успешного collector_run.

render() — тонкая оркестрация: всё нетривиальное преобразование данных вынесено в
чистые типизированные хелперы (``_latest_update_time``, ``_index_points``,
``_currency_kpi``, ``_mover_badge``, …) и покрыто unit-тестами. Каждый сетевой вызов
проходит три ветки feedback (успех / ошибка сервера / сеть недоступна) — пустых
экранов без объяснения нет (DESIGN §5, §10).
"""

import html
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import streamlit as st
from stocklens_core.enums import CollectorRunStatus, Currency

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import (
    CollectorRunOut,
    CurrencyRateOut,
    IndexValueOut,
    KeyRateOut,
    MoverOut,
)
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import (
    fetch_collector_runs,
    fetch_currency_rates,
    fetch_index,
    fetch_index_window,
    fetch_key_rate,
    fetch_movers,
)
from dashboard.auth import get_api_client
from dashboard.components import filters
from dashboard.components.charts import build_index_line_chart, render_chart
from dashboard.components.feedback import render_empty, render_error
from dashboard.components.kpi import delta_badge_from_values, render_delta_badge, stat_cell
from dashboard.components.layout import card
from dashboard.components.transforms import DELTA_GLYPHS, DeltaDirection

#: Заголовок страницы (RU-копи — пользовательская строка).
_PAGE_TITLE = "Обзор"

#: Часовой пояс отображения времени обновления (CLAUDE.md: отображение — Europe/Moscow).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")

#: Код биржевого индекса по умолчанию (KPI и нижняя линия — IMOEX).
_INDEX_CODE = "IMOEX"

#: Лимит выборки индекса для KPI: последняя точка против предыдущей (дельта дня).
_INDEX_KPI_LIMIT = 2

#: Лимит выборки индекса для спарклайна KPI: полная недавняя серия формирует силуэт линии.
#: Дельта дня по-прежнему берётся по последним двум точкам; при прежнем limit=2 спарклайн
#: вырождался в прямую диагональ из двух точек (регресс — чинится бо́льшим окном).
_INDEX_SPARKLINE_LIMIT = 30

#: Число лидеров в каждой таблице муверов (рост / падение).
_MOVERS_LIMIT = 5

#: Лимит выборки последних запусков сборщиков при поиске свежего SUCCESS-времени.
_RUNS_SCAN_LIMIT = 50

#: Валюты KPI-полосы в порядке отображения (рубль — база, не показывается).
_KPI_CURRENCIES: tuple[Currency, ...] = (Currency.USD, Currency.EUR, Currency.CNY)

#: Подписи KPI-ячеек (RU-копи — пользовательские строки).
_LABEL_INDEX = "IMOEX"
_LABEL_KEY_RATE = "Ключевая ставка"
_KPI_CURRENCY_SUFFIX = "/RUB"

#: Формат времени обновления в подписи футера (день и время по календарю Москвы).
_UPDATE_TIME_FORMAT = "%d.%m.%Y %H:%M"

#: RU-копи разделов и пустых состояний страницы.
_SECTION_KPI_EMPTY = "Рыночные показатели пока недоступны."
_SECTION_GAINERS = "Лидеры роста"
_SECTION_LOSERS = "Лидеры падения"
_SECTION_MOVERS = "Движение дня"
_SECTION_INDEX = "Динамика IMOEX"
_MOVERS_EMPTY = "Нет данных о лидерах роста и падения за торговый день."
_INDEX_EMPTY = "Нет значений индекса IMOEX за выбранный период."
_UPDATE_PREFIX = "Данные обновлены:"
_UPDATE_UNKNOWN = "Данные обновлены: время последнего успешного сбора недоступно."

#: Спецификаторы дробной точности значений KPI (Fira Code tnum выровняет разряды).
_INDEX_VALUE_PRECISION = 2
_CURRENCY_VALUE_PRECISION = 4
_KEY_RATE_VALUE_PRECISION = 2


def _period_bounds(period_days: int, reference: date) -> tuple[str, str]:
    """ISO-границы окна ``[reference - period_days, reference]`` для запроса линии индекса.

    Линия IMOEX за выбранный период задаётся окном дат, а не record-count: ``period_days``
    — календарные дни выбранного периода, а ``limit`` API считал бы записи (≈250 торговых
    дней в году) и при 365 упёрся бы в HTTP 422. ``reference`` инъектируется — функция чистая.
    """
    date_from = reference - timedelta(days=period_days)
    return date_from.isoformat(), reference.isoformat()


def _index_points(values: Sequence[IndexValueOut]) -> list[tuple[date, float]]:
    """Привести значения индекса к ``(дата, close)``-парам по возрастанию даты.

    ``build_index_line_chart`` принимает ``(date, float)``-пары; ``/data/index`` отдаёт
    DTO с Decimal-close и торговой датой. Серия сортируется по дате независимо от
    порядка ответа API (KPI берёт последнюю точку, спарклайн — форму линии).
    """
    ordered = sorted(values, key=lambda value: value.trade_date)
    return [(value.trade_date, float(value.close)) for value in ordered]


def _latest_two_closes(values: Sequence[IndexValueOut]) -> tuple[float, float] | None:
    """Вернуть ``(последний_close, предыдущий_close)`` индекса или None.

    Дельта дня KPI считается как последний close против предыдущего; при менее чем двух
    точках дельта неопределима (None) — ячейка покажется без бейджа.
    """
    points = _index_points(values)
    if len(points) < _INDEX_KPI_LIMIT:
        return None
    return points[-1][1], points[-2][1]


def _latest_currency_rate(rates: Sequence[CurrencyRateOut]) -> Decimal | None:
    """Вернуть самый свежий курс валюты по дате котировки или None при пустой выборке."""
    if not rates:
        return None
    return max(rates, key=lambda rate: rate.rate_date).rate


def _latest_key_rate(rates: Sequence[KeyRateOut]) -> Decimal | None:
    """Вернуть самую свежую ключевую ставку по дате или None при пустой выборке."""
    if not rates:
        return None
    return max(rates, key=lambda rate: rate.rate_date).rate


def _mover_direction(change_pct: float) -> DeltaDirection:
    """Классифицировать направление движения бумаги по знаку ``change_pct`` (API-значение)."""
    if change_pct > 0:
        return DeltaDirection.UP
    if change_pct < 0:
        return DeltaDirection.DOWN
    return DeltaDirection.FLAT


def _mover_badge(change_pct: float) -> str:
    """Собрать HTML DeltaBadge строки мувера из ``change_pct`` API (три канала a11y).

    Знак и направление берутся из авторитетного ``change_pct`` API (не пересчёт из
    close/prev_close), чтобы отображаемый процент совпадал с источником истины. Знак
    отрицательной дельты — типографский минус U+2212 (как в format_delta).
    """
    direction = _mover_direction(change_pct)
    glyph = DELTA_GLYPHS[direction]
    if direction is DeltaDirection.FLAT:
        text = f"{glyph} 0.00%"
    else:
        sign = "+" if direction is DeltaDirection.UP else "−"
        text = f"{glyph} {sign}{abs(change_pct):.2f}%"
    return render_delta_badge(direction=direction, text=text)


def _mover_close_text(mover: MoverOut) -> str:
    """Цена закрытия дня мувера, форматированная под табличные цифры числовой колонки."""
    return f"{float(mover.close):.{_INDEX_VALUE_PRECISION}f}"


def _latest_update_time(runs: Sequence[CollectorRunOut]) -> datetime | None:
    """Вернуть московское время последнего успешного завершённого запуска сборщика.

    Из журнала ``/monitoring/runs`` берутся запуски со статусом SUCCESS и непустым
    ``finished_at``; среди них выбирается максимальное время завершения и переводится
    в Europe/Moscow для подписи футера (DESIGN §10.1). Нет подходящих запусков → None.
    """
    finished = [
        run.finished_at
        for run in runs
        if run.status is CollectorRunStatus.SUCCESS and run.finished_at is not None
    ]
    if not finished:
        return None
    return max(finished).astimezone(_MOSCOW_TZ)


def _format_update_footer(moment: datetime | None) -> str:
    """Собрать строку футера обновления; None → честная подпись о недоступности времени."""
    if moment is None:
        return _UPDATE_UNKNOWN
    return f"{_UPDATE_PREFIX} {moment.strftime(_UPDATE_TIME_FORMAT)} МСК"


def _render_index_kpi(client: ApiClient) -> None:
    """KPI-ячейка IMOEX: последнее значение + дельта дня + спарклайн (DESIGN §10.1)."""
    try:
        page = fetch_index(client, index_code=_INDEX_CODE, limit=_INDEX_SPARKLINE_LIMIT)
    except ApiError as exc:
        render_error(exc.user_message)
        return
    if not page.items:
        render_empty(_SECTION_KPI_EMPTY)
        return

    points = _index_points(page.items)
    last_value = points[-1][1]
    closes = _latest_two_closes(page.items)
    badge = (
        None if closes is None else delta_badge_from_values(current=closes[0], previous=closes[1])
    )
    value = f"{last_value:,.{_INDEX_VALUE_PRECISION}f}".replace(",", " ")
    stat_cell(label=_LABEL_INDEX, value=value, delta=badge)
    render_chart(build_index_line_chart(points, sparkline=True))


def _render_currency_kpi(client: ApiClient, currency: Currency) -> None:
    """KPI-ячейка курса валюты к рублю: свежий курс + спарклайн (DESIGN §10.1)."""
    try:
        page = fetch_currency_rates(client, currency=currency)
    except ApiError as exc:
        render_error(exc.user_message)
        return
    if not page.items:
        render_empty(_SECTION_KPI_EMPTY)
        return

    latest = _latest_currency_rate(page.items)
    label = f"{currency.value}{_KPI_CURRENCY_SUFFIX}"
    value = "—" if latest is None else f"{float(latest):.{_CURRENCY_VALUE_PRECISION}f}"
    stat_cell(label=label, value=value)
    points = sorted(((rate.rate_date, float(rate.rate)) for rate in page.items), key=lambda p: p[0])
    if points:
        render_chart(build_index_line_chart(points, sparkline=True))


def _render_key_rate_kpi(client: ApiClient) -> None:
    """KPI-ячейка ключевой ставки ЦБ РФ: свежее значение + спарклайн (DESIGN §10.1)."""
    try:
        page = fetch_key_rate(client)
    except ApiError as exc:
        render_error(exc.user_message)
        return
    if not page.items:
        render_empty(_SECTION_KPI_EMPTY)
        return

    latest = _latest_key_rate(page.items)
    value = "—" if latest is None else f"{float(latest):.{_KEY_RATE_VALUE_PRECISION}f}%"
    stat_cell(label=_LABEL_KEY_RATE, value=value)
    points = sorted(((rate.rate_date, float(rate.rate)) for rate in page.items), key=lambda p: p[0])
    if points:
        render_chart(build_index_line_chart(points, sparkline=True))


def _render_kpi_strip(client: ApiClient) -> None:
    """KPI-полоса: IMOEX, USD/EUR/CNY к рублю, ключевая ставка (DESIGN §10.1).

    Каждая ячейка изолирована собственной обработкой трёх веток: сбой одного источника
    не валит остальные KPI (как partial-семантика сборщиков, CLAUDE.md «Правила данных»).
    """
    columns = st.columns(2 + len(_KPI_CURRENCIES))
    with columns[0]:
        _render_index_kpi(client)
    for offset, currency in enumerate(_KPI_CURRENCIES, start=1):
        with columns[offset]:
            _render_currency_kpi(client, currency)
    with columns[1 + len(_KPI_CURRENCIES)]:
        _render_key_rate_kpi(client)


def _mover_row_markdown(mover: MoverOut) -> str:
    """Собрать плотную строку-грид мувера: тикер · имя · close · DeltaBadge (DESIGN §10.1).

    Грид-таблица (CSS .mover-row) с выровненными числовыми колонками вместо прежней flex-строки
    «подпись слева, бейдж улетел вправо»: в полуширотной колонке space-between давал асимметрию
    и провал (DESIGN §4 — плотность, без пустот). Тикер/имя из MOEX экранируются html.escape
    (как в kpi.py), бейдж — наш доверенный HTML.
    """
    ticker = html.escape(mover.ticker)
    name = html.escape(mover.name)
    close = html.escape(_mover_close_text(mover))
    badge = _mover_badge(mover.change_pct)
    return (
        '<div class="mover-row">'
        f'<span class="mover-row__ticker">{ticker}</span>'
        f'<span class="mover-row__name">{name}</span>'
        f'<span class="mover-row__close">{close}</span>'
        f'<span class="mover-row__badge">{badge}</span>'
        "</div>"
    )


def _render_mover_table(title: str, movers: Sequence[MoverOut]) -> None:
    """Отрисовать компактную таблицу муверов (рост или падение) с overline-меткой колонки.

    Метка — overline (11px caps), а не ``st.subheader`` (h3): иначе она одного размера с
    секцией «Движение дня» (тоже h3) и иерархия читается плоско (DESIGN §3).
    """
    st.markdown(f'<span class="sl-overline">{html.escape(title)}</span>', unsafe_allow_html=True)
    if not movers:
        render_empty(_MOVERS_EMPTY)
        return
    for mover in movers:
        st.markdown(_mover_row_markdown(mover), unsafe_allow_html=True)


def _render_movers(client: ApiClient) -> None:
    """Две таблицы муверов бок о бок: лидеры роста и падения дня (DESIGN §10.1)."""
    st.subheader(_SECTION_MOVERS)
    try:
        movers = fetch_movers(client, limit=_MOVERS_LIMIT)
    except ApiError as exc:
        render_error(exc.user_message)
        return
    if not movers.gainers and not movers.losers:
        render_empty(_MOVERS_EMPTY)
        return

    gainers_col, losers_col = st.columns(2, gap="small")
    with gainers_col:
        _render_mover_table(_SECTION_GAINERS, movers.gainers)
    with losers_col:
        _render_mover_table(_SECTION_LOSERS, movers.losers)


def _render_index_chart(client: ApiClient) -> None:
    """Линия IMOEX за выбранный пользователем период (filters.select_period, DESIGN §10.1)."""
    st.subheader(_SECTION_INDEX)
    period_days = filters.select_period(key="overview_period")
    date_from, date_to = _period_bounds(period_days, datetime.now(tz=_MOSCOW_TZ).date())
    try:
        values = fetch_index_window(
            client,
            index_code=_INDEX_CODE,
            date_from=date_from,
            date_to=date_to,
        )
    except ApiError as exc:
        render_error(exc.user_message)
        return
    if not values:
        render_empty(_INDEX_EMPTY)
        return
    render_chart(build_index_line_chart(_index_points(values)))


def _render_header(client: ApiClient) -> None:
    """Шапка Обзора: заголовок слева + время обновления справа на одной строке (DESIGN §10.1).

    Заголовок-H1 один занимал бы всю ширину строки; время последнего сбора, выровненное по
    правому краю, даёт шапке смысл и использует ширину вместо висящего внизу футера.
    """
    title_col, update_col = st.columns([2, 1], vertical_alignment="bottom")
    with title_col:
        st.title(_PAGE_TITLE)
    with update_col:
        _render_update_indicator(client)


def _render_update_indicator(client: ApiClient) -> None:
    """Индикатор времени последнего успешного сбора, выровненный по правому краю шапки.

    Свежесть — второстепенный индикатор шапки: при сбое его источника он опускается (страница
    строится без него), а не рисует красный баннер в заголовке. Остальные ветки данных
    страницы показывают свой feedback сами.
    """
    try:
        page = fetch_collector_runs(
            client,
            status=CollectorRunStatus.SUCCESS,
            limit=_RUNS_SCAN_LIMIT,
        )
    except ApiError:
        return
    text = _format_update_footer(_latest_update_time(page.items))
    st.markdown(f'<div class="overview-updated">{html.escape(text)}</div>', unsafe_allow_html=True)


def render() -> None:
    """Отрисовать страницу «Обзор»: шапка → KPI-полоса → муверы → линия IMOEX (DESIGN §10.1).

    Каждая секция — карточка на inset-границе (st.container(border=True), CSS превращает её в
    Linear instrument-panel): группировка границей+тенью вместо разделителей st.divider.
    """
    client = get_api_client()
    _render_header(client)
    with card("overview-kpi"):
        _render_kpi_strip(client)
    with card("overview-movers"):
        _render_movers(client)
    with card("overview-index"):
        _render_index_chart(client)

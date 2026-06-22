"""Рыночный контекст в сайдбаре дашборда (DESIGN.md §4).

Сайдбар нёс только навигацию и бренд — много места без контекста. Компактная сводка
рынка (IMOEX + дельта дня, ключевая ставка ЦБ) даёт постоянный рыночный пульс на любой
странице, оправдывая место сайдбара.

Сводка — второстепенный адорнмент шапки навигации: при сбое её источников блок опускается
(страница строится без него), а не рисует красный баннер в сайдбаре. Значения берутся из
тех же кэшируемых fetch-обёрток, что и страница «Обзор» (повторный rerun отдаёт кэш).
"""

import html
from collections.abc import Sequence

import streamlit as st

from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import IndexValueOut, KeyRateOut
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import fetch_index, fetch_key_rate
from dashboard.components.kpi import delta_badge_from_values

#: Код индекса рыночной сводки и размер выборки (последняя точка против предыдущей — дельта дня).
_INDEX_CODE = "IMOEX"
_INDEX_LIMIT = 2

#: RU-копи блока сводки (пользовательские строки).
_SECTION = "Рынок сейчас"
_LABEL_INDEX = "IMOEX"
_LABEL_RATE = "Ставка ЦБ"

#: Точность отображения значений сводки.
_INDEX_PRECISION = 2
_RATE_PRECISION = 2


def _latest_two_closes(values: Sequence[IndexValueOut]) -> tuple[float, float] | None:
    """Вернуть ``(последний_close, предыдущий_close)`` индекса по дате или None при < 2 точках."""
    if len(values) < _INDEX_LIMIT:
        return None
    ordered = sorted(values, key=lambda value: value.trade_date)
    return float(ordered[-1].close), float(ordered[-2].close)


def _latest_close(values: Sequence[IndexValueOut]) -> float | None:
    """Вернуть самый свежий close индекса по дате или None при пустой выборке."""
    if not values:
        return None
    return float(max(values, key=lambda value: value.trade_date).close)


def _latest_rate(rates: Sequence[KeyRateOut]) -> float | None:
    """Вернуть самую свежую ключевую ставку по дате или None при пустой выборке."""
    if not rates:
        return None
    return float(max(rates, key=lambda rate: rate.rate_date).rate)


def _format_index(value: float) -> str:
    """Отформатировать значение индекса с разрядным разделителем-пробелом (как KPI Обзора)."""
    return f"{value:,.{_INDEX_PRECISION}f}".replace(",", " ")


def _context_html(index_value: str, delta_badge: str, rate_value: str) -> str:
    """Собрать HTML компактного блока рыночной сводки (метка слева, значение справа).

    ``delta_badge`` — доверенный HTML из ``delta_badge_from_values`` (его текст уже экранирован);
    числовые значения экранируются html.escape, как в kpi.py.
    """
    return (
        '<div class="sl-market-context">'
        f'<span class="sl-market-context__head">{html.escape(_SECTION)}</span>'
        '<div class="sl-market-context__row">'
        f'<span class="sl-market-context__label">{html.escape(_LABEL_INDEX)}</span>'
        f'<span class="sl-market-context__value">{html.escape(index_value)} {delta_badge}</span>'
        "</div>"
        '<div class="sl-market-context__row">'
        f'<span class="sl-market-context__label">{html.escape(_LABEL_RATE)}</span>'
        f'<span class="sl-market-context__value">{html.escape(rate_value)}</span>'
        "</div>"
        "</div>"
    )


def render_market_context(client: ApiClient) -> None:
    """Отрисовать рыночную сводку в сайдбаре: IMOEX + дельта дня, ключевая ставка (DESIGN §4).

    Тонкая оркестрация: фетчит индекс и ставку через кэшируемые обёртки; при сбое любого
    источника или пустой выборке блок не строится (второстепенный адорнмент, не сбой страницы).
    """
    try:
        index_page = fetch_index(client, index_code=_INDEX_CODE, limit=_INDEX_LIMIT)
        rate_page = fetch_key_rate(client)
    except ApiError:
        return

    last_close = _latest_close(index_page.items)
    rate = _latest_rate(rate_page.items)
    if last_close is None or rate is None:
        return

    closes = _latest_two_closes(index_page.items)
    delta_badge = (
        "" if closes is None else delta_badge_from_values(current=closes[0], previous=closes[1])
    )
    html_block = _context_html(
        index_value=_format_index(last_close),
        delta_badge=delta_badge,
        rate_value=f"{rate:.{_RATE_PRECISION}f}%",
    )
    with st.sidebar:
        st.markdown(html_block, unsafe_allow_html=True)

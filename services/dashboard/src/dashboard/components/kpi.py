"""KPI-ячейка и бейдж дельты дашборда (DESIGN.md §5).

StatCell — компактная KPI-ячейка (метка + значение Fira Code tnum + опциональный
DeltaBadge), НЕ запрещённый hero-metric шаблон (один гигантский градиентный номер).
DeltaBadge несёт три канала-дубля (цвет + глиф + знак) для a11y: цвет никогда не
единственный индикатор.

Разметка инжектится через st.markdown(unsafe_allow_html=True) с CSS-классами из
assets/dashboard.css. Все интерполируемые значения экранируются html.escape: имена
бумаг MOEX могут содержать `&`/кавычки и сломать разметку.
"""

import html

import streamlit as st

from dashboard.components.transforms import DeltaDirection, format_delta


def render_delta_badge(direction: DeltaDirection, text: str) -> str:
    """Собрать HTML бейджа дельты (три канала: цвет-класс + глиф + знак уже в `text`).

    Возвращает строку (не рендерит) — StatCell встраивает её в свою разметку.
    `text` приходит из format_delta и уже содержит глиф со знаком; экранируется.
    """
    return f'<span class="delta-badge delta-badge--{direction.value}">{html.escape(text)}</span>'


def delta_badge_from_values(current: float, previous: float) -> str:
    """Удобный билдер бейджа из пары значений (считает дельту через format_delta)."""
    text, direction = format_delta(current=current, previous=previous)
    return render_delta_badge(direction=direction, text=text)


def stat_cell(label: str, value: str, *, delta: str | None = None) -> None:
    """Отрендерить компактную KPI-ячейку (DESIGN §5), не hero-metric.

    `label` — подпись (caption); `value` — отформатированное значение (Fira Code tnum);
    `delta` — готовый HTML бейджа из render_delta_badge / delta_badge_from_values, либо
    None (тогда бейдж не показывается). Текстовые поля экранируются; `delta` — наш
    доверенный HTML, не пользовательский ввод.
    """
    badge_html = f'<div class="statcell__delta">{delta}</div>' if delta else ""
    st.markdown(
        '<div class="statcell">'
        f'<span class="statcell__label">{html.escape(label)}</span>'
        f'<span class="statcell__value">{html.escape(value)}</span>'
        f"{badge_html}"
        "</div>",
        unsafe_allow_html=True,
    )

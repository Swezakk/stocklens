"""Селекторы фильтров дашборда: тикер / период / тональность (DESIGN.md §4, §10).

Функции возвращают выбранные значения и не имеют побочных эффектов помимо отрисовки
самого виджета (никакого фетча/записи). Тональность выбирается значениями SentimentLabel
из stocklens-core — без хардкода строковых литералов (инвариант №4). Подписи русские.
"""

from collections.abc import Sequence

import streamlit as st
from stocklens_core.enums import SentimentLabel

from dashboard.components.sentiment import SENTIMENT_LABELS_RU

#: Предустановленные периоды графиков (подпись → число дней истории).
PERIOD_OPTIONS: dict[str, int] = {
    "1 месяц": 30,
    "3 месяца": 90,
    "6 месяцев": 180,
    "1 год": 365,
}


def select_ticker(tickers: Sequence[str], *, key: str = "ticker") -> str | None:
    """Селектор одного тикера из доступных бумаг; None при пустом списке.

    `key` разводит несколько селекторов на одной странице (виджеты Streamlit по ключу).
    """
    if not tickers:
        return None
    return st.selectbox("Тикер", options=list(tickers), key=key)


def select_period(*, default_label: str = "3 месяца", key: str = "period") -> int:
    """Селектор периода истории; возвращает число дней (DESIGN §10)."""
    labels = list(PERIOD_OPTIONS)
    index = labels.index(default_label) if default_label in PERIOD_OPTIONS else 0
    chosen_label = st.selectbox("Период", options=labels, index=index, key=key)
    return PERIOD_OPTIONS[chosen_label]


def select_sentiments(*, key: str = "sentiment") -> list[SentimentLabel]:
    """Мультиселектор меток тональности новостей; пустой список = без фильтра.

    Возвращает значения SentimentLabel (контракт API), а не строки. Подписи русские
    через format_func; внутреннее значение виджета — сам enum.
    """
    selected = st.multiselect(
        "Тональность",
        options=list(SentimentLabel),
        format_func=lambda label: SENTIMENT_LABELS_RU[label],
        key=key,
    )
    return list(selected)

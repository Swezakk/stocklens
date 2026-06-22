"""Чип тональности новости (DESIGN.md §2.2, §5, §10).

SentimentChip — пилюля с русской текстовой меткой и цветом из theme.SENTIMENT_COLORS.
Текстовая метка — обязательный второй канал (a11y): тональность не передаётся одним
лишь цветом. Цвет уже зашит в CSS-класс sentiment-chip--{label} (assets/dashboard.css);
theme.SENTIMENT_COLORS — источник истины этих хексов и используется для inline-fallback.

Метки — на русском (правило: пользовательские строки русские). Маппинг по SentimentLabel
из stocklens-core, без хардкода строковых литералов тональности (инвариант №4).
"""

import html

import streamlit as st
from stocklens_core.enums import SentimentLabel

from dashboard import theme

#: Русские подписи меток тональности (второй a11y-канал помимо цвета).
#: Единый источник для чипа и фильтра новостей — без дублирования копи.
SENTIMENT_LABELS_RU: dict[SentimentLabel, str] = {
    SentimentLabel.POSITIVE: "Позитив",
    SentimentLabel.NEUTRAL: "Нейтрально",
    SentimentLabel.NEGATIVE: "Негатив",
}


def render_sentiment_chip(label: SentimentLabel) -> str:
    """Собрать HTML чипа тональности (цвет из theme + русская текстовая метка).

    Возвращает строку (не рендерит) — лента новостей встраивает чип в строку статьи.
    Цвет дублируется inline из theme.SENTIMENT_COLORS поверх CSS-класса: чип читаем,
    даже если строка статьи рендерится вне основного CSS-контекста.
    """
    color = theme.SENTIMENT_COLORS[label]
    text = html.escape(SENTIMENT_LABELS_RU[label])
    return (
        f'<span class="sentiment-chip sentiment-chip--{label.value}" '
        f'style="color: {color}">{text}</span>'
    )


def sentiment_chip(label: SentimentLabel) -> None:
    """Отрендерить чип тональности новости (DESIGN §5)."""
    st.markdown(render_sentiment_chip(label), unsafe_allow_html=True)

"""Карточка-секция Linear instrument-panel (DESIGN.md §5).

``st.container(border=True)`` даёт лишь тонкий нативный бордер; Linear-приём — глубина
charcoal-заливкой + 1px-границей + мягкой тенью на onyx-холсте. Чтобы навесить это
стабильным CSS (без хрупких emotion-классов), контейнеру задаётся ``key`` с общим
префиксом ``sl-card-``: Streamlit рендерит его как класс ``st-key-sl-card-<key>``, по
которому CSS (assets/dashboard.css) и стилизует все карточки одним правилом.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

#: Общий префикс ключа карточки → стабильный класс ``st-key-sl-card-<key>`` (CSS-хук).
_CARD_KEY_PREFIX = "sl-card-"


@contextmanager
def card(key: str) -> Iterator[None]:
    """Контекст-менеджер карточки-секции Linear: bordered-контейнер со стабильным CSS-хуком.

    ``key`` обязан быть уникальным в пределах страницы (требование Streamlit к ключам
    виджетов); префикс ``sl-card-`` добавляется автоматически, поэтому в CSS все карточки
    ловятся селектором по этому префиксу, а не по хрупкому хеш-классу.
    """
    with st.container(border=True, key=f"{_CARD_KEY_PREFIX}{key}"):
        yield

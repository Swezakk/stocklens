"""Фидбэк-компоненты сетевого вызова: ошибка / пусто / загрузка (DESIGN.md §5, §10).

Три ветки каждого сетевого вызова страницы (успех / ошибка сервера / сеть недоступна)
проходят через эти функции — пустых экранов без объяснения нет. Тексты для пользователя
русские. Принимают готовую строку (контракт §5: render_error(msg)), а не доменную
ошибку, — страница достаёт `.user_message` из ApiError и передаёт сюда. Это держит
feedback вне зависимости от api_client.errors.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

#: Дефолтное сообщение пустого результата, если страница не уточнила контекст.
_DEFAULT_EMPTY_MESSAGE = "Нет данных для отображения по выбранным фильтрам."

#: Дефолтная подпись индикатора загрузки.
_DEFAULT_LOADING_MESSAGE = "Загрузка данных…"


def render_error(message: str) -> None:
    """Показать ошибку сетевого вызова (DESIGN §5).

    `message` — готовая русская строка из `ApiError.user_message` (сущность + причина).
    """
    st.error(message, icon=":material/error:")


def render_empty(message: str = _DEFAULT_EMPTY_MESSAGE) -> None:
    """Показать сообщение о пустом результате (успех без данных).

    Отличается от ошибки: запрос прошёл, но данных под фильтр нет — это не сбой.
    """
    st.info(message, icon=":material/inbox:")


def render_info(message: str) -> None:
    """Показать информационный баннер (нейтральная нотификация, не ошибка и не пустота).

    `message` — готовая русская строка (например, причина авто-фолбэка стратегии из API).
    Иконка отличает баннер от пустого состояния (inbox) и ошибки (error).
    """
    st.info(message, icon=":material/info:")


@contextmanager
def render_loading(message: str = _DEFAULT_LOADING_MESSAGE) -> Iterator[None]:
    """Контекст-менеджер индикатора загрузки вокруг блока сетевого вызова."""
    with st.spinner(message):
        yield

"""Страница «Мониторинг» дашборда (DESIGN.md §10.6, §12).

Три блока над журналом запусков сборщиков (``/monitoring/runs``):

1. **Статусные плитки источников** — последний запуск каждого источника со статусом
   ``CollectorRunStatus`` (success / partial / failed). Статус несёт три канала-дубля
   (цвет + иконка ``:material/`` + русский текст) — цвет никогда не единственный
   индикатор (DESIGN §12).
2. **Журнал** — таблица всех запусков (источник, время МСК, статус, +записей, ошибка)
   с русскими заголовками.
3. **Последние ошибки** — запуски со статусом ``FAILED`` либо непустым ``error_message``,
   самые свежие сверху.

Орхестрация ``render`` тонкая: всё формирование данных вынесено в чистые типизированные
хелперы (``_latest_run_per_source``, ``_recent_errors``, ``_status_visual``,
``_journal_row``, ``_format_msk``), покрытые unit-тестами; сам layout не тестируется
(DESIGN §5). Единственный сетевой вызов раскладывается по трём веткам через ``feedback``
(успех / ошибка сервера / сеть недоступна) — пустых экранов без объяснения нет.

Время в БД и в ответах API — UTC; отображение — Europe/Moscow (CLAUDE.md «Правила данных»).
"""

from datetime import datetime
from typing import NamedTuple
from zoneinfo import ZoneInfo

import streamlit as st
from stocklens_core.enums import CollectorRunStatus

from dashboard.api_client.dto import CollectorRunOut
from dashboard.api_client.errors import ApiError
from dashboard.api_client.fetch import fetch_collector_runs
from dashboard.auth import get_api_client
from dashboard.components.feedback import render_empty, render_error
from dashboard.components.layout import card
from dashboard.theme import STATUS_BADGE_COLORS, STATUS_BADGE_ICONS, BadgeColor

#: Заголовок страницы (RU-копи — пользовательская строка).
_PAGE_TITLE = "Мониторинг"

#: Подзаголовки блоков страницы (RU-копи).
_TILES_HEADER = "Источники"
_JOURNAL_HEADER = "Журнал запусков"
_ERRORS_HEADER = "Последние ошибки"

#: Пустой результат: журнал запусков ещё не наполнен (успех без данных).
_EMPTY_RUNS_MESSAGE = "Журнал запусков сборщиков пуст: данные ещё не собирались."

#: Сообщение блока ошибок, когда сбоев нет (успех без данных — это норма, не ошибка).
_NO_ERRORS_MESSAGE = "Ошибок сбора нет: все источники отработали штатно."

#: Глубина журнала: один экран последних запусков (потолок limit API = 200, DESIGN §9).
_JOURNAL_LIMIT = 100

#: Число колонок-плиток статусов в ряд (компактная плотная раскладка, DESIGN §4).
_TILE_COLUMNS = 3

#: Часовой пояс отображения времени запусков (UTC в БД → МСК в UI).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")

#: Формат отображения метки времени запуска в МСК (дата + время до минут).
_MSK_TIME_FORMAT = "%d.%m.%Y %H:%M"

#: Заглушка отсутствующего значения в таблице (незавершённый запуск, пустая ошибка).
_EMPTY_CELL = "—"

#: Русские заголовки колонок журнала (ключи строки = заголовки, без column_config-переименования).
_COL_SOURCE = "Источник"
_COL_STARTED = "Начало (МСК)"
_COL_FINISHED = "Завершение (МСК)"
_COL_STATUS = "Статус"
_COL_RECORDS = "Записей добавлено"
_COL_ERROR = "Ошибка"


class _StatusVisual(NamedTuple):
    """Тройной a11y-канал статуса запуска: русский текст + ``:material/``-иконка + цвет.

    Цвет — не единственный индикатор (DESIGN §12): текст и иконка дублируют его для
    пользователей, не различающих цвета. ``accent`` — именованный цвет ``st.badge``
    из ``theme.STATUS_BADGE_COLORS``, несущий тот же сигнал, что иконка и текст.
    """

    label: str
    icon: str
    accent: BadgeColor


#: Русский текст статуса (user-facing копи живёт на странице, не в theme).
_STATUS_LABELS: dict[CollectorRunStatus, str] = {
    CollectorRunStatus.SUCCESS: "Успешно",
    CollectorRunStatus.PARTIAL: "Частично",
    CollectorRunStatus.FAILED: "Сбой",
}


#: Визуальное представление каждого статуса: текст со страницы + цвет/иконка из theme.
_STATUS_VISUALS: dict[CollectorRunStatus, _StatusVisual] = {
    status: _StatusVisual(
        label=_STATUS_LABELS[status],
        icon=STATUS_BADGE_ICONS[status],
        accent=STATUS_BADGE_COLORS[status],
    )
    for status in CollectorRunStatus
}


def _status_visual(status: CollectorRunStatus) -> _StatusVisual:
    """Вернуть тройной a11y-канал (текст + иконка + цвет) для статуса запуска.

    Карта покрывает все значения ``CollectorRunStatus``; отсутствие ключа означало бы
    рассинхрон с доменным enum и должно падать явно, а не молча подставлять дефолт.
    """
    return _STATUS_VISUALS[status]


def _to_moscow(moment: datetime) -> datetime:
    """Перевести UTC-метку времени API в Europe/Moscow для отображения (CLAUDE.md)."""
    return moment.astimezone(_MOSCOW_TZ)


def _format_msk(moment: datetime | None) -> str:
    """Отформатировать метку времени запуска в МСК; ``None`` (незавершённый) → прочерк."""
    if moment is None:
        return _EMPTY_CELL
    return _to_moscow(moment).strftime(_MSK_TIME_FORMAT)


def _latest_run_per_source(runs: list[CollectorRunOut]) -> list[CollectorRunOut]:
    """Последний запуск каждого источника, отсортированный по имени источника.

    Группировка по ``source``; в группе берётся запуск с максимальным ``started_at``
    (самый свежий). Порядок результата детерминирован — по алфавиту источника, чтобы
    раскладка плиток не «прыгала» между rerun. Пустой вход → пустой список.
    """
    latest_by_source: dict[str, CollectorRunOut] = {}
    for run in runs:
        current = latest_by_source.get(run.source)
        if current is None or run.started_at > current.started_at:
            latest_by_source[run.source] = run
    return [latest_by_source[source] for source in sorted(latest_by_source)]


def _recent_errors(runs: list[CollectorRunOut]) -> list[CollectorRunOut]:
    """Запуски-сбои (статус ``FAILED`` или непустой ``error_message``), свежие сверху.

    Отбор по двум каналам: явный статус ``FAILED`` И/ИЛИ заполненное ``error_message``
    (``partial`` тоже может нести сообщение об упавшем источнике). Сортировка по убыванию
    ``started_at`` — последний сбой первым. Пустой вход или отсутствие сбоев → пустой список.
    """
    failed = [
        run
        for run in runs
        if run.status is CollectorRunStatus.FAILED
        or bool(run.error_message and run.error_message.strip())
    ]
    return sorted(failed, key=lambda run: run.started_at, reverse=True)


def _journal_row(run: CollectorRunOut) -> dict[str, str | int]:
    """Собрать одну строку журнала с русскими заголовками и временем в МСК.

    Ключи словаря — готовые русские заголовки колонок (Streamlit делает их именами
    столбцов), поэтому ``column_config``-переименование не нужно. Метки времени — в МСК;
    незавершённое завершение и пустая ошибка показываются прочерком.
    """
    return {
        _COL_SOURCE: run.source,
        _COL_STARTED: _format_msk(run.started_at),
        _COL_FINISHED: _format_msk(run.finished_at),
        _COL_STATUS: _status_visual(run.status).label,
        _COL_RECORDS: run.records_added,
        _COL_ERROR: run.error_message if run.error_message is not None else _EMPTY_CELL,
    }


def _load_runs() -> list[CollectorRunOut]:
    """Загрузить журнал запусков через API (единственный сетевой вызов страницы).

    Клиент собирается из связанных провайдеров auth (токен + хук 401). Исключение
    ``ApiError`` всплывает в ``render``, где раскладывается по веткам ``feedback`` —
    здесь не глушится.
    """
    page = fetch_collector_runs(get_api_client(), limit=_JOURNAL_LIMIT)
    return page.items


def _render_status_tiles(runs: list[CollectorRunOut]) -> None:
    """Отрисовать плитки последних статусов по источникам (три канала: цвет+иконка+текст)."""
    st.subheader(_TILES_HEADER)
    latest = _latest_run_per_source(runs)
    columns = st.columns(_TILE_COLUMNS)
    for index, run in enumerate(latest):
        visual = _status_visual(run.status)
        with columns[index % _TILE_COLUMNS], card(f"mon-tile-{run.source}"):
            st.markdown(f"**{run.source}**")
            st.badge(visual.label, icon=visual.icon, color=visual.accent)
            st.caption(f"Обновлено: {_format_msk(run.started_at)}")


def _render_journal(runs: list[CollectorRunOut]) -> None:
    """Отрисовать таблицу журнала запусков с русскими заголовками и временем в МСК."""
    st.subheader(_JOURNAL_HEADER)
    st.dataframe(
        [_journal_row(run) for run in runs],
        hide_index=True,
        use_container_width=True,
    )


def _render_recent_errors(runs: list[CollectorRunOut]) -> None:
    """Отрисовать блок последних ошибок сбора; при отсутствии сбоев — нейтральная заметка."""
    st.subheader(_ERRORS_HEADER)
    errors = _recent_errors(runs)
    if not errors:
        render_empty(_NO_ERRORS_MESSAGE)
        return
    st.dataframe(
        [_journal_row(run) for run in errors],
        hide_index=True,
        use_container_width=True,
    )


def render() -> None:
    """Отрисовать страницу «Мониторинг»: плитки источников, журнал, последние ошибки.

    Тонкая орхестрация: один сетевой вызов (``_load_runs``) с раскладкой по трём веткам
    ``feedback``; всё формирование данных — в чистых хелперах. Пустой журнал → нейтральная
    заметка ``render_empty`` (успех без данных, а не сбой).
    """
    st.title(_PAGE_TITLE)

    try:
        runs = _load_runs()
    except ApiError as exc:
        render_error(exc.user_message)
        return

    if not runs:
        render_empty(_EMPTY_RUNS_MESSAGE)
        return

    # Источники не оборачиваем в карточку: сами плитки — уже карточки (вложенность запрещена).
    _render_status_tiles(runs)
    with card("mon-journal"):
        _render_journal(runs)
    with card("mon-errors"):
        _render_recent_errors(runs)

"""Тесты чистых хелперов страницы «Мониторинг» (DESIGN.md §10.6, §12).

Покрывают формирование данных, вынесенное из тонкого ``render`` (layout не тестируется,
DESIGN §5):

- ``_latest_run_per_source`` — группировка по источнику, выбор самого свежего по
  ``started_at``, детерминированный порядок, пустой вход;
- ``_recent_errors`` — отбор сбоев (``FAILED`` или непустой ``error_message``), порядок
  «свежие сверху», отсутствие сбоев;
- ``_status_visual`` — тройной a11y-канал (текст + ``:material/``-иконка + цвет) для всех
  статусов;
- ``_format_msk`` — отображение UTC-метки в Europe/Moscow и прочерк для ``None``;
- ``_journal_row`` — русские заголовки колонок, время в МСК, прочерки для пустых ячеек.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from dashboard.api_client.dto import CollectorRunOut
from dashboard.pages.monitoring import (
    _EMPTY_CELL,
    _format_msk,
    _journal_row,
    _latest_run_per_source,
    _recent_errors,
    _status_visual,
)
from stocklens_core.enums import CollectorRunStatus

_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


def _run(
    *,
    run_id: int,
    source: str,
    started_at: datetime,
    status: CollectorRunStatus = CollectorRunStatus.SUCCESS,
    finished_at: datetime | None = None,
    records_added: int = 0,
    error_message: str | None = None,
) -> CollectorRunOut:
    """Собрать CollectorRunOut с заданными источником, временем старта и статусом."""
    return CollectorRunOut(
        id=run_id,
        source=source,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        records_added=records_added,
        error_message=error_message,
    )


def test_latest_run_per_source_empty_returns_empty() -> None:
    assert _latest_run_per_source([]) == []


def test_latest_run_per_source_picks_newest_started_at_within_source() -> None:
    older = _run(
        run_id=1,
        source="moex",
        started_at=datetime(2026, 6, 20, 10, 0, tzinfo=UTC),
    )
    newer = _run(
        run_id=2,
        source="moex",
        started_at=datetime(2026, 6, 22, 10, 0, tzinfo=UTC),
    )

    result = _latest_run_per_source([older, newer])

    assert [run.id for run in result] == [newer.id]


def test_latest_run_per_source_groups_and_sorts_by_source_name() -> None:
    moex = _run(run_id=1, source="moex", started_at=datetime(2026, 6, 22, 9, 0, tzinfo=UTC))
    rss = _run(run_id=2, source="rss", started_at=datetime(2026, 6, 22, 9, 5, tzinfo=UTC))
    cbr = _run(run_id=3, source="cbr", started_at=datetime(2026, 6, 22, 9, 10, tzinfo=UTC))

    result = _latest_run_per_source([moex, rss, cbr])

    assert [run.source for run in result] == ["cbr", "moex", "rss"]


def test_recent_errors_empty_returns_empty() -> None:
    assert _recent_errors([]) == []


def test_recent_errors_selects_failed_and_error_message_runs() -> None:
    ok = _run(run_id=1, source="moex", started_at=datetime(2026, 6, 22, 9, 0, tzinfo=UTC))
    failed = _run(
        run_id=2,
        source="rss",
        started_at=datetime(2026, 6, 22, 9, 5, tzinfo=UTC),
        status=CollectorRunStatus.FAILED,
    )
    partial_with_message = _run(
        run_id=3,
        source="cbr",
        started_at=datetime(2026, 6, 22, 9, 10, tzinfo=UTC),
        status=CollectorRunStatus.PARTIAL,
        error_message="Источник недоступен",
    )

    result = _recent_errors([ok, failed, partial_with_message])

    assert {run.id for run in result} == {failed.id, partial_with_message.id}


def test_recent_errors_orders_newest_started_at_first() -> None:
    early = _run(
        run_id=1,
        source="rss",
        started_at=datetime(2026, 6, 22, 9, 0, tzinfo=UTC),
        status=CollectorRunStatus.FAILED,
    )
    late = _run(
        run_id=2,
        source="moex",
        started_at=datetime(2026, 6, 22, 11, 0, tzinfo=UTC),
        status=CollectorRunStatus.FAILED,
    )

    result = _recent_errors([early, late])

    assert [run.id for run in result] == [late.id, early.id]


def test_status_visual_covers_all_statuses_with_three_channels() -> None:
    for status in CollectorRunStatus:
        visual = _status_visual(status)
        assert visual.label
        assert visual.icon.startswith(":material/")
        assert visual.accent in {"green", "orange", "red"}


def test_format_msk_converts_utc_to_moscow_display() -> None:
    moment = datetime(2026, 6, 22, 7, 30, tzinfo=UTC)

    formatted = _format_msk(moment)

    expected = moment.astimezone(_MOSCOW_TZ).strftime("%d.%m.%Y %H:%M")
    assert formatted == expected
    assert formatted == "22.06.2026 10:30"


def test_format_msk_none_returns_dash() -> None:
    assert _format_msk(None) == _EMPTY_CELL


def test_journal_row_has_russian_headers_and_msk_time() -> None:
    run = _run(
        run_id=1,
        source="moex",
        started_at=datetime(2026, 6, 22, 7, 0, tzinfo=UTC),
        finished_at=datetime(2026, 6, 22, 7, 5, tzinfo=UTC),
        status=CollectorRunStatus.SUCCESS,
        records_added=42,
    )

    row = _journal_row(run)

    assert row["Источник"] == "moex"
    assert row["Начало (МСК)"] == "22.06.2026 10:00"
    assert row["Завершение (МСК)"] == "22.06.2026 10:05"
    assert row["Статус"] == "Успешно"
    assert row["Записей добавлено"] == 42
    assert row["Ошибка"] == _EMPTY_CELL


def test_journal_row_unfinished_run_shows_dash_for_finished_at() -> None:
    run = _run(
        run_id=1,
        source="rss",
        started_at=datetime(2026, 6, 22, 7, 0, tzinfo=UTC),
        finished_at=None,
        status=CollectorRunStatus.PARTIAL,
        error_message="RSS-канал не ответил",
    )

    row = _journal_row(run)

    assert row["Завершение (МСК)"] == _EMPTY_CELL
    assert row["Статус"] == "Частично"
    assert row["Ошибка"] == "RSS-канал не ответил"

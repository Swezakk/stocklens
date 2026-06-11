"""Контекстный менеджер для журналирования запусков сборщиков в collector_runs.

Гарантия: запись в БД всегда фиксируется, даже при исключении внутри блока.
Исключение не пробрасывается наружу — один источник не должен останавливать остальные.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

import structlog
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import CollectorRunStatus
from stocklens_core.models.operations import CollectorRun

log = structlog.get_logger(__name__)


class RunHandle:
    """Дескриптор текущего запуска: накапливает счётчики и статус.

    Используется внутри collector_run() для передачи состояния из тела блока
    в финализирующий код контекстного менеджера.
    """

    def __init__(self) -> None:
        self._records: int = 0
        self._partial_reason: str | None = None
        self.failed: bool = False
        self.error_message: str | None = None

    def add_records(self, count: int) -> None:
        """Добавить count к счётчику записей текущего запуска."""
        self._records += count

    def mark_partial(self, reason: str) -> None:
        """Отметить запуск как частичный (PARTIAL) с причиной."""
        self._partial_reason = reason

    @property
    def records_added(self) -> int:
        return self._records

    @property
    def status(self) -> CollectorRunStatus:
        if self.failed:
            return CollectorRunStatus.FAILED
        if self._partial_reason is not None:
            return CollectorRunStatus.PARTIAL
        return CollectorRunStatus.SUCCESS


@contextmanager
def collector_run(
    session_factory: sessionmaker[Session],
    source: str,
) -> Generator[RunHandle, None, None]:
    """Контекстный менеджер: создаёт запись CollectorRun, фиксирует результат.

    Исключение внутри блока логируется и записывается в error_message,
    но наружу не пробрасывается — следующий источник продолжит работу.

    Args:
        session_factory: Фабрика синхронных SQLAlchemy-сессий.
        source: Имя источника (например «moex_candles»).

    Yields:
        RunHandle для накопления счётчиков и пометки PARTIAL.
    """
    handle = RunHandle()
    run_id: int | None = None

    with session_factory() as session:
        run = CollectorRun(
            source=source,
            started_at=datetime.now(UTC),
            status=CollectorRunStatus.FAILED,
            records_added=0,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    log.info("collector_run_started", source=source, run_id=run_id)

    try:
        yield handle
    except Exception as exc:
        handle.failed = True
        handle.error_message = str(exc)
        log.error(
            "collector_run_failed",
            source=source,
            run_id=run_id,
            error=str(exc),
            exc_info=True,
        )
    finally:
        with session_factory() as session:
            run_record = session.get(CollectorRun, run_id)
            if run_record is not None:
                run_record.finished_at = datetime.now(UTC)
                run_record.status = handle.status
                run_record.records_added = handle.records_added
                run_record.error_message = handle.error_message
                session.commit()

        log.info(
            "collector_run_finished",
            source=source,
            run_id=run_id,
            status=handle.status,
            records_added=handle.records_added,
        )

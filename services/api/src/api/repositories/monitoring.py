"""Реализация MonitoringRepository на AsyncSession (SQLAlchemy 2.0)."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import CollectorRunStatus
from stocklens_core.models.operations import CollectorRun


class SqlMonitoringRepository:
    """Читает журнал запусков сборщиков из PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_runs(
        self,
        source: str | None,
        status: CollectorRunStatus | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CollectorRun], int]:
        """Вернуть страницу запусков (сортировка: started_at desc) и общее число."""
        base_query = select(CollectorRun)
        if source is not None:
            base_query = base_query.where(CollectorRun.source == source)
        if status is not None:
            base_query = base_query.where(CollectorRun.status == status)

        count_result = await self._session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total: int = count_result.scalar_one()

        rows_result = await self._session.execute(
            base_query.order_by(CollectorRun.started_at.desc()).limit(limit).offset(offset)
        )
        runs = list(rows_result.scalars().all())
        return runs, total

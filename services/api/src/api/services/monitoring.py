"""Сервис чтения журнала запусков сборщиков."""

from stocklens_core.enums import CollectorRunStatus

from api.repositories.protocols import MonitoringRepository
from api.schemas.common import Page
from api.schemas.monitoring import CollectorRunOut


class MonitoringService:
    """Оркестрирует MonitoringRepository, маппит ORM → DTO."""

    def __init__(self, repo: MonitoringRepository) -> None:
        self._repo = repo

    async def list_runs(
        self,
        source: str | None,
        status: CollectorRunStatus | None,
        limit: int,
        offset: int,
    ) -> Page[CollectorRunOut]:
        """Вернуть страницу запусков сборщиков."""
        runs, total = await self._repo.list_runs(
            source=source,
            status=status,
            limit=limit,
            offset=offset,
        )
        items = [CollectorRunOut.model_validate(r) for r in runs]
        return Page(items=items, total=total, limit=limit, offset=offset)

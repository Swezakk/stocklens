"""Эндпоинты мониторинга: журнал запусков сборщиков."""

from typing import Annotated

from fastapi import APIRouter, Query
from stocklens_core.enums import CollectorRunStatus

from api.core.db import SessionDep
from api.core.pagination import PageDep
from api.repositories.monitoring import SqlMonitoringRepository
from api.schemas.common import Page
from api.schemas.monitoring import CollectorRunOut
from api.services.monitoring import MonitoringService

router = APIRouter(prefix="/api/v1", tags=["monitoring"])

SourceDep = Annotated[str | None, Query(description="Фильтр по источнику данных")]
StatusDep = Annotated[CollectorRunStatus | None, Query(description="Фильтр по статусу запуска")]


@router.get(
    "/monitoring/runs",
    response_model=Page[CollectorRunOut],
    summary="Журнал запусков сборщиков",
    description="Возвращает запуски сборщиков данных в порядке убывания времени старта.",
)
async def list_runs(
    session: SessionDep,
    page: PageDep,
    source: SourceDep = None,
    status: StatusDep = None,
) -> Page[CollectorRunOut]:
    """GET /monitoring/runs — журнал запусков сборщиков, сортировка started_at desc."""
    repo = SqlMonitoringRepository(session)
    service = MonitoringService(repo)
    return await service.list_runs(
        source=source,
        status=status,
        limit=page.limit,
        offset=page.offset,
    )

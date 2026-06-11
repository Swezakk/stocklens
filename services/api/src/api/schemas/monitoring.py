"""DTO для мониторинга: запуски сборщиков."""

from datetime import datetime

from pydantic import BaseModel
from stocklens_core.enums import CollectorRunStatus


class CollectorRunOut(BaseModel):
    """DTO запуска сборщика данных."""

    model_config = {"from_attributes": True}

    id: int
    source: str
    started_at: datetime
    finished_at: datetime | None
    status: CollectorRunStatus
    records_added: int
    error_message: str | None

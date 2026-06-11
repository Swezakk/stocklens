"""DTO для операций со списком наблюдения."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, field_validator


class WatchlistStatus(StrEnum):
    """Статус материализации бумаги из вотчлиста.

    READY    — бумага существует в securities И имеет хотя бы одну свечу.
    PENDING  — добавлена недавно (в пределах grace-периода), ingestor ещё не успел.
    NOT_FOUND — grace-период истёк, данных так и нет (MOEX 404 или задержка > ожидаемой).
    """

    READY = "ready"
    PENDING = "pending"
    NOT_FOUND = "not_found"


class WatchlistItemIn(BaseModel):
    """Входные данные для добавления тикера в список наблюдения."""

    ticker: str

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        """Тикер приводится к верхнему регистру без пробелов."""
        return v.strip().upper()


class WatchlistItemOut(BaseModel):
    """Выходные данные элемента вотчлиста с производным статусом."""

    model_config = {"from_attributes": True}

    ticker: str
    added_at: datetime
    status: WatchlistStatus
    has_data: bool

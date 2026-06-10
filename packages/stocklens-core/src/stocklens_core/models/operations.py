"""ORM-модель журнала запусков сборщиков данных."""

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from stocklens_core.enums import CollectorRunStatus
from stocklens_core.models.base import Base, str_enum_type


class CollectorRun(Base):
    """Запись о запуске сборщика: источник, временные метки, статус и счётчик записей."""

    __tablename__ = "collector_runs"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    source: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    status: Mapped[CollectorRunStatus] = mapped_column(
        str_enum_type(CollectorRunStatus), nullable=False
    )
    records_added: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

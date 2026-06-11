"""ORM-модель ML-прогнозов."""

from datetime import date, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from stocklens_core.enums import PredictionKind
from stocklens_core.models.base import Base, str_enum_type


class Prediction(Base):
    """ML-прогноз волатильности или тренда для инструмента."""

    __tablename__ = "predictions"
    __table_args__ = (
        # Явное короткое имя: имя по naming_convention превышает лимит
        # идентификатора PostgreSQL (63 символа) и было бы молча обрезано.
        sa.UniqueConstraint(
            "security_id",
            "predicted_for",
            "horizon_days",
            "kind",
            "model_version",
            name="uq_predictions_natural_key",
        ),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    security_id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.ForeignKey("securities.id"), nullable=False
    )
    predicted_for: Mapped[date] = mapped_column(sa.Date, nullable=False)
    horizon_days: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    kind: Mapped[PredictionKind] = mapped_column(str_enum_type(PredictionKind), nullable=False)
    value: Mapped[float] = mapped_column(sa.Float, nullable=False)
    model_version: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

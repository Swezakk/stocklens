"""SQL-репозиторий ML-прогнозов (ml-spec §8.4, D2).

API — единственный write-путь в ``predictions``. Запись — идемпотентный upsert по
натуральному ключу (constraint ``uq_predictions_natural_key``); ``get_value`` — read-through
кэш (повторный инференс в тот же день той же версией модели не делает работу).
"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import PredictionKind
from stocklens_core.models.prediction import Prediction

_NATURAL_KEY_CONSTRAINT = "uq_predictions_natural_key"


class SqlPredictionRepository:
    """Чтение/запись прогнозов в таблицу ``predictions`` через AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_value(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        model_version: str,
    ) -> float | None:
        """Вернуть сохранённое значение прогноза по натуральному ключу или None."""
        result = await self._session.execute(
            select(Prediction.value).where(
                Prediction.security_id == security_id,
                Prediction.predicted_for == predicted_for,
                Prediction.horizon_days == horizon_days,
                Prediction.kind == kind,
                Prediction.model_version == model_version,
            )
        )
        value = result.scalar_one_or_none()
        return None if value is None else float(value)

    async def upsert(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        value: float,
        model_version: str,
    ) -> None:
        """Создать/обновить прогноз по натуральному ключу. Коммитит транзакцию."""
        stmt = (
            pg_insert(Prediction)
            .values(
                security_id=security_id,
                predicted_for=predicted_for,
                horizon_days=horizon_days,
                kind=kind,
                value=value,
                model_version=model_version,
            )
            .on_conflict_do_update(
                constraint=_NATURAL_KEY_CONSTRAINT,
                set_={"value": value},
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()

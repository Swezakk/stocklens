"""Интеграционные тесты SqlPredictionRepository против PostgreSQL (ml-spec §8.4, §11.2).

Проверяет идемпотентность upsert по натуральному ключу и read-through (get_value).
Контейнер module-scoped, репозиторий коммитит — тесты используют разные тикеры/ключи.
"""

from datetime import date

import pytest
from api.repositories.prediction import SqlPredictionRepository
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import PredictionKind
from stocklens_core.models.prediction import Prediction

from tests.integration.seed import seed_security

pytestmark = pytest.mark.integration

_PREDICTED_FOR = date(2024, 6, 20)
_HORIZON = 5
_VERSION = "3"


async def _count(session: AsyncSession, security_id: int, version: str) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(Prediction)
        .where(
            Prediction.security_id == security_id,
            Prediction.predicted_for == _PREDICTED_FOR,
            Prediction.horizon_days == _HORIZON,
            Prediction.kind == PredictionKind.VOLATILITY,
            Prediction.model_version == version,
        )
    )
    return int(result.scalar_one())


async def test_predict_repo_upsert_then_get_value_roundtrip(db_session: AsyncSession) -> None:
    security = await seed_security(db_session, ticker="VOLA")
    repo = SqlPredictionRepository(db_session)

    await repo.upsert(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, 0.0312, _VERSION
    )
    value = await repo.get_value(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, _VERSION
    )

    assert value == pytest.approx(0.0312)


async def test_predict_repo_upsert_is_idempotent_on_natural_key(db_session: AsyncSession) -> None:
    security = await seed_security(db_session, ticker="VOLB")
    repo = SqlPredictionRepository(db_session)

    await repo.upsert(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, 0.030, _VERSION
    )
    await repo.upsert(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, 0.041, _VERSION
    )

    assert await _count(db_session, security.id, _VERSION) == 1
    value = await repo.get_value(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, _VERSION
    )
    assert value == pytest.approx(0.041)  # повторный upsert обновил значение, не создал дубль


async def test_predict_repo_get_value_returns_none_for_absent(db_session: AsyncSession) -> None:
    security = await seed_security(db_session, ticker="VOLC")
    repo = SqlPredictionRepository(db_session)

    value = await repo.get_value(
        security.id, _PREDICTED_FOR, _HORIZON, PredictionKind.VOLATILITY, "999"
    )

    assert value is None

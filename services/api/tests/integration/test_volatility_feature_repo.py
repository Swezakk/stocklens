"""Интеграционные тесты SqlVolatilityFeatureRepository против PostgreSQL (ml-spec §8.5).

Проверяет, что свечи/дивиденды/сплиты читаются в DataFrame'ы нужной формы и сортировки —
ровно в виде, который ожидает сборщик фич stocklens_ml.
"""

from datetime import date

import pytest
from api.repositories.volatility_features import SqlVolatilityFeatureRepository
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Split

from tests.integration.seed import seed_candles_range, seed_dividend, seed_security

pytestmark = pytest.mark.integration


async def test_feature_repo_loads_candles_dividends_splits(db_session: AsyncSession) -> None:
    security = await seed_security(db_session, ticker="FEAT")
    dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)]
    await seed_candles_range(db_session, security.id, dates)
    await seed_dividend(db_session, security.id, ex_date=date(2024, 1, 11))
    db_session.add(Split(security_id=security.id, split_date=date(2024, 1, 12), before=1, after=10))
    await db_session.flush()

    repo = SqlVolatilityFeatureRepository(db_session)
    candles = await repo.load_candles(security.id)
    dividends = await repo.load_dividends(security.id)
    splits = await repo.load_splits(security.id)

    assert list(candles.columns) == [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_weekend_session",
    ]
    assert len(candles) == 3
    assert list(candles["trade_date"]) == dates  # отсортировано по возрастанию даты
    assert candles["high"].iloc[0] > candles["low"].iloc[0]

    assert len(dividends) == 1
    assert dividends["value"].iloc[0] == pytest.approx(33.0)

    assert len(splits) == 1
    assert int(splits["before"].iloc[0]) == 1
    assert int(splits["after"].iloc[0]) == 10


async def test_feature_repo_returns_empty_frames_for_unknown_security(
    db_session: AsyncSession,
) -> None:
    repo = SqlVolatilityFeatureRepository(db_session)

    candles = await repo.load_candles(999_999)

    assert candles.empty
    assert list(candles.columns) == [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_weekend_session",
    ]

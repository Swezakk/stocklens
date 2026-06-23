"""Integration-тесты маршрута POST /predict/volatility (ml-spec §8.3, §11.2).

Полная вертикаль через HTTP: router → service → репозитории (реальный PG) + стаб-модель
(app.state.ml из conftest). Плюс readiness 503 при обязательной, но незагруженной модели.
"""

import math
from collections.abc import AsyncGenerator
from datetime import date

import pandas as pd
import pytest
from api.core.auth.settings import AuthSettings
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from api.ml.bundle import ModelBundle
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import STUB_VARIANCE
from tests.integration.seed import seed_candles_range, seed_security

pytestmark = pytest.mark.integration


def _business_dates(n: int, start: str = "2023-06-01") -> list[date]:
    dates: list[date] = pd.bdate_range(start=start, periods=n).date.tolist()
    return dates


async def test_predict_volatility_returns_prediction(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="PRED")
    await seed_candles_range(db_session, security.id, _business_dates(130))
    await db_session.commit()

    response = await client.post("/api/v1/predict/volatility", json={"ticker": "PRED"})

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "PRED"
    assert body["volatility"] == pytest.approx(math.sqrt(STUB_VARIANCE))
    assert body["model"] == "garch"
    assert body["model_version"] == "test-1"
    assert body["horizon_days"] == 5
    assert body["metrics_vs_baseline"]["qlike"] == pytest.approx(0.844)


async def test_predict_volatility_returns_404_for_unknown_ticker(client: AsyncClient) -> None:
    response = await client.post("/api/v1/predict/volatility", json={"ticker": "NOPE"})

    assert response.status_code == 404


async def test_predict_volatility_returns_422_for_insufficient_history(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="THIN")
    await seed_candles_range(db_session, security.id, _business_dates(30))
    await db_session.commit()

    response = await client.post("/api/v1/predict/volatility", json={"ticker": "THIN"})

    assert response.status_code == 422


async def test_health_ready_returns_503_when_models_required_and_unavailable(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    required_settings = ApiSettings.model_validate({"ml_required_for_ready": True})
    app = create_app()
    app.state.settings = required_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = ModelBundle(volatility=None)

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_redis() -> RedisClientProtocol:
        return real_redis_client

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = override_redis

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["models"] == "unavailable"

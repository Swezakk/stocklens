"""Integration-тесты GET /api/v1/predict/volatility/history (ml-spec §10).

Полная вертикаль через HTTP: router → service → репозитории (реальный PG) + стаб-модель.
Проверяет: 200 с realized и forecast точками, 404 для неизвестного тикера.
"""

from collections.abc import AsyncGenerator
from datetime import date

import pandas as pd
import pytest
from api.core.auth.deps import require_auth
from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import PredictionKind
from stocklens_core.models.prediction import Prediction

from tests.integration.conftest import stub_bundle
from tests.integration.seed import seed_candles_range, seed_security

pytestmark = pytest.mark.integration


def _business_dates(n: int, start: str = "2023-01-02") -> list[date]:
    return list(pd.bdate_range(start=start, periods=n).date)


async def _seed_prediction(
    session: AsyncSession,
    security_id: int,
    predicted_for: date,
    value: float,
    model_version: str = "test-1",
) -> None:
    row = Prediction(
        security_id=security_id,
        predicted_for=predicted_for,
        horizon_days=5,
        kind=PredictionKind.VOLATILITY,
        value=value,
        model_version=model_version,
    )
    session.add(row)
    await session.flush()


async def test_forecast_history_returns_realized_and_forecast_points(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    security = await seed_security(db_session, ticker="HIST")
    dates = _business_dates(180)
    await seed_candles_range(db_session, security.id, dates)

    # Засеваем прогноз на дату внутри окна lookback=120 (последние 120 торговых дат)
    # Берём дату из середины окна, чтобы realized тоже там был (rv_target не NaN)
    forecast_date = dates[-60]
    await _seed_prediction(db_session, security.id, forecast_date, value=0.04)
    await db_session.commit()

    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = stub_bundle()

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_redis() -> RedisClientProtocol:
        return real_redis_client

    async def override_require_auth() -> Principal:
        return Principal(sub="testowner", scopes=[], claims={})

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = override_redis
    app.dependency_overrides[require_auth] = override_require_auth

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/api/v1/predict/volatility/history",
            params={"ticker": "HIST", "lookback": 120},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "HIST"
    assert body["model_version"] == "test-1"
    assert body["model"] == "garch"
    assert body["metrics_vs_baseline"] is not None
    assert body["metrics_vs_baseline"]["qlike"] == pytest.approx(0.844)

    points = body["points"]
    assert len(points) > 0

    realized_points = [p for p in points if p["realized"] is not None]
    assert len(realized_points) > 0, "Ожидались точки с realized в окне lookback=120"

    forecast_str = forecast_date.isoformat()
    forecast_points = [p for p in points if p["date"] == forecast_str]
    assert len(forecast_points) == 1
    assert forecast_points[0]["forecast"] == pytest.approx(0.04)


async def test_forecast_history_returns_422_for_invalid_lookback(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = stub_bundle()

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_redis() -> RedisClientProtocol:
        return real_redis_client

    async def override_require_auth() -> Principal:
        return Principal(sub="testowner", scopes=[], claims={})

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = override_redis
    app.dependency_overrides[require_auth] = override_require_auth

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/api/v1/predict/volatility/history",
            params={"ticker": "ANY", "lookback": 1},
        )

    assert response.status_code == 422


async def test_forecast_history_returns_404_for_unknown_ticker(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = stub_bundle()

    async def override_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_redis() -> RedisClientProtocol:
        return real_redis_client

    async def override_require_auth() -> Principal:
        return Principal(sub="testowner", scopes=[], claims={})

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = override_redis
    app.dependency_overrides[require_auth] = override_require_auth

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get(
            "/api/v1/predict/volatility/history",
            params={"ticker": "NOHIST"},
        )

    assert response.status_code == 404

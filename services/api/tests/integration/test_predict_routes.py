"""Integration-тесты маршрута POST /predict/volatility (ml-spec §8.3, §11.2).

Полная вертикаль через HTTP: router → service → репозитории (реальный PG) + стаб-модель
(app.state.ml из conftest). Плюс readiness 503 при обязательной, но незагруженной модели.
"""

import math
from collections.abc import AsyncGenerator
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from api.core.auth.deps import require_auth
from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from api.ml.bundle import ModelBundle
from api.ml.trend import TREND_FEATURE_COLUMNS
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import PredictionKind, TrendDirection
from stocklens_core.models.market import Candle
from stocklens_core.models.prediction import Prediction

from tests.integration.conftest import (
    STUB_PROB_UP,
    STUB_TREND_BASE_VALUE,
    STUB_TREND_HORIZON_DAYS,
    STUB_TREND_MODEL_VERSION,
    STUB_VARIANCE,
    stub_bundle,
)
from tests.integration.seed import seed_candles_range, seed_security

pytestmark = pytest.mark.integration


def _business_dates(n: int, start: str = "2023-06-01") -> list[date]:
    dates: list[date] = pd.bdate_range(start=start, periods=n).date.tolist()
    return dates


async def _seed_trend_candles(session: AsyncSession, security_id: int, dates: list[date]) -> None:
    """Засеять свечи с варьирующимся объёмом — volume_zscore требует ненулевого std окна.

    Общий seed_candles_range держит объём константой (1_000_000), из-за чего rolling-std
    объёма = 0 и фича volume_zscore вырождается в NaN. Здесь объём детерминированно колеблется.
    """
    price = Decimal("280.00")
    for i, trade_date in enumerate(dates):
        price = price + Decimal(str(i % 5 - 2))
        volume = 1_000_000 + (i % 7) * 50_000
        session.add(
            Candle(
                security_id=security_id,
                trade_date=trade_date,
                open=price - Decimal("1.00"),
                high=price + Decimal("2.00"),
                low=price - Decimal("2.00"),
                close=price,
                volume=volume,
                value=price * Decimal(str(volume)),
                is_weekend_session=False,
            )
        )
    await session.flush()


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


async def test_predict_trend_returns_prediction(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="TRND")
    await _seed_trend_candles(db_session, security.id, _business_dates(130))
    await db_session.commit()

    response = await client.post("/api/v1/predict/trend", json={"ticker": "TRND"})

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "TRND"
    assert body["prob_up"] == pytest.approx(STUB_PROB_UP)
    assert body["direction"] == TrendDirection.UP.value
    assert body["horizon_days"] == STUB_TREND_HORIZON_DAYS
    assert body["model_version"] == STUB_TREND_MODEL_VERSION
    assert body["base_value"] == pytest.approx(STUB_TREND_BASE_VALUE)
    assert len(body["shap"]) == len(TREND_FEATURE_COLUMNS)
    assert all({"feature", "value"} == set(item) for item in body["shap"])

    stored = (
        await db_session.execute(
            select(Prediction.value).where(
                Prediction.security_id == security.id,
                Prediction.kind == PredictionKind.TREND,
                Prediction.model_version == STUB_TREND_MODEL_VERSION,
            )
        )
    ).scalar_one()
    assert float(stored) == pytest.approx(STUB_PROB_UP)


async def test_predict_trend_returns_404_for_unknown_ticker(client: AsyncClient) -> None:
    response = await client.post("/api/v1/predict/trend", json={"ticker": "NONE"})

    assert response.status_code == 404


async def test_predict_trend_returns_422_for_insufficient_history(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="TFLAT")
    # Константный объём seed_candles_range → rolling-std=0 → volume_zscore NaN на as-of строке
    # → срабатывает history-гард predict_trend (InsufficientHistoryError → 422).
    await seed_candles_range(db_session, security.id, _business_dates(130))
    await db_session.commit()

    response = await client.post("/api/v1/predict/trend", json={"ticker": "TFLAT"})

    assert response.status_code == 422


async def test_predict_trend_returns_422_for_invalid_body(client: AsyncClient) -> None:
    response = await client.post("/api/v1/predict/trend", json={"ticker": ""})

    assert response.status_code == 422


async def test_predict_trend_returns_503_when_trend_model_unavailable(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    bundle = stub_bundle()
    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = ModelBundle(volatility=bundle.volatility, trend=None)

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
        response = await ac.post("/api/v1/predict/trend", json={"ticker": "ANY"})

    assert response.status_code == 503

"""Integration-тест алерта volatility_regime через poll-эндпоинт (ml-spec §9, тикет 7d3e9b21).

Полная вертикаль через HTTP: POST /bot/alerts/pending → AlertEvaluationService →
assess_volatility_regime (PredictionService) → репозитории (реальный PG). Модель — стаб с
заведомо высокой волатильностью (is_elevated гарантирован), остальное настоящее.
"""

from collections.abc import AsyncGenerator

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from api.core.auth.deps import require_auth
from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import AlertKind

from tests.integration.seed import seed_bot_subscription, seed_candles_range, seed_security

pytestmark = pytest.mark.integration


class _HighVolPredictor:
    """Стаб: дисперсия 1.0 → волатильность 1.0 (заведомо выше любого реализованного квантиля)."""

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.array([1.0], dtype=np.float64)


def _elevated_bundle() -> ModelBundle:
    return ModelBundle(
        volatility=LoadedVolatilityModel(
            predictor=_HighVolPredictor(),
            model_version="test-1",
            method="garch",
            metrics={"qlike": 0.84, "qlike_baseline": 2.2, "rmse": 0.0025},
            horizon_days=5,
        )
    )


async def test_pending_alerts_returns_volatility_regime_when_elevated(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    security = await seed_security(db_session, ticker="VLTR")
    dates = pd.bdate_range(start="2023-01-01", periods=180).date.tolist()
    await seed_candles_range(db_session, security.id, dates)
    await seed_bot_subscription(
        db_session, chat_id=42, kind=AlertKind.VOLATILITY_REGIME, params={"ticker": "VLTR"}
    )
    await db_session.commit()

    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = _elevated_bundle()

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
        response = await ac.post("/api/v1/bot/alerts/pending")

    assert response.status_code == 200
    alerts = response.json()
    volatility_alerts = [a for a in alerts if a["kind"] == AlertKind.VOLATILITY_REGIME.value]
    assert len(volatility_alerts) == 1
    alert = volatility_alerts[0]
    assert alert["ticker"] == "VLTR"
    assert alert["volatility"] == pytest.approx(1.0)
    assert alert["threshold"] < 1.0  # реализованный квантиль ниже стаб-прогноза

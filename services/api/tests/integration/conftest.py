"""Fixtures для integration-тестов API.

Паттерн: module-scoped PostgreSQL + Redis контейнеры → применяем миграции один раз
на модуль → каждый тест получает свежий engine и сессию (function scope).
Function-scope engine необходим: module-scope engine привязан к event loop первого теста,
а pytest-asyncio создаёт новый loop на каждую функцию.
"""

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from alembic import command
from alembic.config import Config
from api.core.auth.deps import require_auth
from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.cache import RedisClientProtocol
from api.core.db import get_redis, get_session
from api.core.settings import ApiSettings
from api.main import create_app
from api.ml.bundle import (
    LoadedTrendModel,
    LoadedVolatilityModel,
    ModelBundle,
    TrendShapResult,
)
from api.ml.trend import TREND_FEATURE_COLUMNS
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[4]

_TEST_SIGNING_KEY = "test-secret-for-integration-tests-only"
_TEST_OWNER_USERNAME = "testowner"
_TEST_OWNER_CREDENTIAL = "test-owner-credential-integration"
_TEST_PRINCIPAL = Principal(sub=_TEST_OWNER_USERNAME, scopes=[], claims={})

# create_app() жадно создаёт AuthSettings из env (owner_password обязателен) —
# задаём детерминированные AUTH_* до того, как фикстуры вызовут create_app().
os.environ["AUTH_SECRET"] = _TEST_SIGNING_KEY
os.environ["AUTH_OWNER_USERNAME"] = _TEST_OWNER_USERNAME
os.environ["AUTH_OWNER_PASSWORD"] = _TEST_OWNER_CREDENTIAL


#: Дисперсия стаб-модели (доли²) → волатильность sqrt = 0.03.
STUB_VARIANCE = 0.0009

#: P(up) стаб-модели тренда ≥ 0.5 → direction = TrendDirection.UP.
STUB_PROB_UP = 0.73

#: Базовое значение SHAP (логит) стаб-модели тренда.
STUB_TREND_BASE_VALUE = 0.12

#: Версия стаб-модели тренда.
STUB_TREND_MODEL_VERSION = "test-trend-1"

#: Горизонт стаб-модели тренда (дни).
STUB_TREND_HORIZON_DAYS = 5


class _StubVolatilityPredictor:
    """Стаб модели волатильности: возвращает фиксированную дисперсию (без MLflow/arch)."""

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.array([STUB_VARIANCE], dtype=np.float64)


class _StubTrendPredictor:
    """Стаб модели тренда: фиксированный P(up) + детерминированный SHAP (без MLflow/catboost).

    ``predict_proba`` отдаёт ``STUB_PROB_UP`` на каждую строку; ``shap`` — единичные вклады
    по числу фич TREND_FEATURE_COLUMNS и фиксированное базовое значение. SHAP-вклады — 2D
    формы (1, n_features), зеркало нативного CatBoost (срез без столбца базового значения).
    """

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.full(len(x), STUB_PROB_UP, dtype=np.float64)

    def shap(self, x: pd.DataFrame) -> TrendShapResult:
        contribs = np.full((1, len(TREND_FEATURE_COLUMNS)), 0.05, dtype=np.float64)
        return TrendShapResult(
            contribs=contribs,
            base_value=STUB_TREND_BASE_VALUE,
            feature_names=list(TREND_FEATURE_COLUMNS),
        )


def stub_bundle() -> ModelBundle:
    """Bundle со стаб-моделями волатильности и тренда — имитация прод-состояния (§8.1)."""
    return ModelBundle(
        volatility=LoadedVolatilityModel(
            predictor=_StubVolatilityPredictor(),
            model_version="test-1",
            method="garch",
            metrics={"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025},
            horizon_days=5,
        ),
        trend=LoadedTrendModel(
            predictor=_StubTrendPredictor(),
            model_version=STUB_TREND_MODEL_VERSION,
            horizon_days=STUB_TREND_HORIZON_DAYS,
        ),
    )


def _alembic_config(sync_url: str) -> Config:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", sync_url)
    return config


def _async_url(pg_container: PostgresContainer) -> str:
    sync_url: str = str(pg_container.get_connection_url())
    return sync_url.replace("postgresql+psycopg://", "postgresql+asyncpg://")


@pytest.fixture(scope="module")
def pg_container() -> PostgresContainer:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
        command.upgrade(_alembic_config(container.get_connection_url()), "head")
        yield container


@pytest.fixture(scope="module")
def redis_container() -> RedisContainer:
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest_asyncio.fixture
async def async_engine(pg_container: PostgresContainer) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(_async_url(pg_container), pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(async_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def real_redis_client(redis_container: RedisContainer) -> RedisClientProtocol:
    port = redis_container.get_exposed_port(6379)
    # cast на I/O-границе: decode_responses=True даёт str-возврат, Protocol статически не выводится.
    return cast(
        RedisClientProtocol,
        aioredis.from_url(f"redis://localhost:{port}/0", decode_responses=True),
    )


@pytest.fixture
def test_settings(pg_container: PostgresContainer, redis_container: RedisContainer) -> ApiSettings:
    redis_port = redis_container.get_exposed_port(6379)
    os.environ["DATABASE_URL_ASYNC"] = _async_url(pg_container)
    os.environ["REDIS_URL"] = f"redis://localhost:{redis_port}/0"
    os.environ["LOG_PRETTY"] = "true"
    # Большинство integration-тестов не поднимают lifespan (нет загрузки моделей) — ML не
    # обязателен для readiness; ML-тесты переопределяют это и выставляют app.state.ml сами.
    os.environ["ML_REQUIRED_FOR_READY"] = "false"
    return ApiSettings.model_validate({})


@pytest.fixture
def test_auth_settings() -> AuthSettings:
    """AuthSettings для integration-тестов с известным секретом."""
    return AuthSettings.model_validate(
        {
            "secret": _TEST_SIGNING_KEY,
            "owner_username": _TEST_OWNER_USERNAME,
            "owner_password": _TEST_OWNER_CREDENTIAL,
            "issuer": "https://stocklens.test",
            "audience": "stocklens-api-test",
        }
    )


@pytest_asyncio.fixture
async def client(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP-клиент с переопределённым require_auth — все существующие тесты не требуют токена."""
    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = stub_bundle()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_redis() -> RedisClientProtocol:
        return real_redis_client

    async def override_require_auth() -> Principal:
        return _TEST_PRINCIPAL

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[require_auth] = override_require_auth

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def noauth_client(
    db_session: AsyncSession,
    real_redis_client: RedisClientProtocol,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP-клиент БЕЗ override require_auth — для тестирования реальной аутентификации."""
    app = create_app()
    app.state.settings = test_settings
    app.state.auth_settings = test_auth_settings
    app.state.ml = stub_bundle()

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_redis() -> RedisClientProtocol:
        return real_redis_client

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

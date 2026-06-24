"""Маршруты ML-прогнозов (ml-spec §8.3, §10). Под /api/v1, response_model обязателен."""

from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.db import SessionDep, SettingsDep
from api.core.settings import ApiSettings
from api.ml.bundle import ModelBundle
from api.ml.deps import MlBundleDep
from api.repositories.prediction import SqlPredictionRepository
from api.repositories.security import SqlSecurityRepository
from api.repositories.volatility_features import SqlVolatilityFeatureRepository
from api.schemas.predict import (
    VolatilityForecastHistoryOut,
    VolatilityPredictionIn,
    VolatilityPredictionOut,
)
from api.services.prediction import PredictionService

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])

_DEFAULT_FORECAST_LOOKBACK = 90


def _service(
    session: AsyncSession, bundle: ModelBundle, settings: ApiSettings
) -> PredictionService:
    return PredictionService(
        security_repo=SqlSecurityRepository(session),
        feature_repo=SqlVolatilityFeatureRepository(session),
        prediction_repo=SqlPredictionRepository(session),
        bundle=bundle,
        settings=settings,
    )


@router.get(
    "/volatility/history",
    response_model=VolatilityForecastHistoryOut,
    summary="История прогнозов волатильности",
    description=(
        "График «прогноз vs реализованная волатильность» за последние lookback торговых дат "
        "(ml-spec §10). Реализованная волатильность — sqrt(rv_target) из feature-pipeline, "
        "тот же показатель, который таргетирует модель."
    ),
)
async def get_volatility_forecast_history(
    ticker: Annotated[str, Query(min_length=1, description="Тикер бумаги")],
    session: SessionDep,
    bundle: MlBundleDep,
    settings: SettingsDep,
    lookback: Annotated[int, Query(ge=5, le=365)] = _DEFAULT_FORECAST_LOOKBACK,
) -> VolatilityForecastHistoryOut:
    """История прогнозов волатильности с реализованными значениями для дашборда."""
    return await _service(session, bundle, settings).forecast_history(ticker, lookback)


@router.post(
    "/volatility",
    response_model=VolatilityPredictionOut,
    summary="Прогноз 5-дневной волатильности",
    description="Возвращает прогноз волатильности тикера на горизонт 5 дней (sqrt дисперсии).",
)
async def predict_volatility(
    payload: VolatilityPredictionIn,
    session: SessionDep,
    bundle: MlBundleDep,
    settings: SettingsDep,
) -> VolatilityPredictionOut:
    """Прогноз волатильности: тикер → 5-дневная волатильность + метрики vs baseline."""
    return await _service(session, bundle, settings).predict_volatility(payload.ticker)

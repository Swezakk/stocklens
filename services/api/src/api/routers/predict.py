"""Маршруты ML-прогнозов (ml-spec §8.3). Под /api/v1, response_model обязателен."""

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.db import SessionDep, SettingsDep
from api.core.settings import ApiSettings
from api.ml.bundle import ModelBundle
from api.ml.deps import MlBundleDep
from api.repositories.prediction import SqlPredictionRepository
from api.repositories.security import SqlSecurityRepository
from api.repositories.volatility_features import SqlVolatilityFeatureRepository
from api.schemas.predict import VolatilityPredictionIn, VolatilityPredictionOut
from api.services.prediction import PredictionService

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])


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

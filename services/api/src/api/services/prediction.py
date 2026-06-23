"""Сервис ML-прогнозов: волатильность (ml-spec §8.3, §8.4).

Оркестрирует: резолв тикера → загрузка истории → сборка фич (единый код с обучением) →
инференс волатильности в threadpool (CPU-bound рефит GARCH не блокирует event-loop) →
идемпотентная запись в predictions (D2). Read-through кэш: если прогноз на (бумага, дата,
горизонт, версия) уже записан — возвращаем его без рефита.
"""

import math

import pandas as pd
from starlette.concurrency import run_in_threadpool
from stocklens_core.enums import PredictionKind

from api.core.exceptions import (
    InsufficientHistoryError,
    ModelNotLoadedError,
    SecurityNotFoundError,
)
from api.core.settings import ApiSettings
from api.ml.bundle import ModelBundle
from api.ml.features import MIN_VOLATILITY_HISTORY, SERVING_FEATURES, build_serving_frame
from api.repositories.protocols import (
    PredictionRepository,
    SecurityRepository,
    VolatilityFeatureRepository,
)
from api.schemas.predict import VolatilityMetrics, VolatilityPredictionOut


class PredictionService:
    """Инференс ML-прогнозов поверх загруженных моделей и рыночной истории."""

    def __init__(
        self,
        *,
        security_repo: SecurityRepository,
        feature_repo: VolatilityFeatureRepository,
        prediction_repo: PredictionRepository,
        bundle: ModelBundle,
        settings: ApiSettings,
    ) -> None:
        self._security_repo = security_repo
        self._feature_repo = feature_repo
        self._prediction_repo = prediction_repo
        self._bundle = bundle
        self._settings = settings

    async def predict_volatility(self, ticker: str) -> VolatilityPredictionOut:
        """Прогноз 5-дневной волатильности по тикеру (sqrt прогноза дисперсии)."""
        model = self._bundle.volatility
        if model is None:
            raise ModelNotLoadedError(self._settings.ml_volatility_model)

        security = await self._security_repo.get_by_ticker(ticker)
        if security is None:
            raise SecurityNotFoundError(ticker)

        frame = build_serving_frame(
            await self._feature_repo.load_candles(security.id),
            await self._feature_repo.load_dividends(security.id),
            await self._feature_repo.load_splits(security.id),
            train_start=self._settings.ml_train_start,
            horizon=model.horizon_days,
        )
        valid = int(frame["r"].notna().sum())
        if valid < MIN_VOLATILITY_HISTORY:
            raise InsufficientHistoryError(ticker, valid, MIN_VOLATILITY_HISTORY)

        predicted_for = pd.Timestamp(frame.iloc[-1]["trade_date"]).date()
        cached = await self._prediction_repo.get_value(
            security.id,
            predicted_for,
            model.horizon_days,
            PredictionKind.VOLATILITY,
            model.model_version,
        )
        if cached is not None:
            volatility = cached
        else:
            variance = await run_in_threadpool(model.predictor.forecast, frame[SERVING_FEATURES])
            volatility = math.sqrt(float(variance[0]))
            await self._prediction_repo.upsert(
                security.id,
                predicted_for,
                model.horizon_days,
                PredictionKind.VOLATILITY,
                volatility,
                model.model_version,
            )

        return VolatilityPredictionOut(
            ticker=security.ticker,
            predicted_for=predicted_for,
            horizon_days=model.horizon_days,
            volatility=volatility,
            model=model.method,
            model_version=model.model_version,
            metrics_vs_baseline=VolatilityMetrics(
                qlike=model.metrics["qlike"],
                qlike_baseline=model.metrics["qlike_baseline"],
                rmse=model.metrics["rmse"],
            ),
        )

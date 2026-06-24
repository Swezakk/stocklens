"""Сервис ML-прогнозов: волатильность (ml-spec §8.3, §8.4, §9).

Оркестрирует: резолв тикера → загрузка истории → сборка фич (единый код с обучением) →
инференс волатильности в threadpool (CPU-bound рефит GARCH не блокирует event-loop) →
идемпотентная запись в predictions (D2). Read-through кэш: если прогноз на (бумага, дата,
горизонт, версия) уже записан — возвращаем его без рефита.
"""

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from starlette.concurrency import run_in_threadpool
from stocklens_core.enums import PredictionKind
from stocklens_core.models.market import Security

from api.core.exceptions import (
    InsufficientHistoryError,
    ModelNotLoadedError,
    SecurityNotFoundError,
)
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
from api.ml.features import MIN_VOLATILITY_HISTORY, SERVING_FEATURES, build_serving_frame
from api.repositories.protocols import (
    PredictionRepository,
    SecurityRepository,
    VolatilityFeatureRepository,
)
from api.schemas.predict import VolatilityMetrics, VolatilityPredictionOut, VolatilityRegime

#: Минимум ненулевых значений rv_target для устойчивой оценки режима волатильности.
#: Ниже 60 распределение слишком бедно для осмысленного 0.80-квантиля.
MIN_REGIME_OBSERVATIONS = 60


@dataclass(frozen=True)
class _Forecast:
    """Результат внутреннего пайплайна инференса волатильности."""

    security: Security
    frame: pd.DataFrame
    predicted_for: date
    volatility: float
    model: LoadedVolatilityModel


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
        result = await self._forecast(ticker)
        return VolatilityPredictionOut(
            ticker=result.security.ticker,
            predicted_for=result.predicted_for,
            horizon_days=result.model.horizon_days,
            volatility=result.volatility,
            model=result.model.method,
            model_version=result.model.model_version,
            metrics_vs_baseline=VolatilityMetrics(
                qlike=result.model.metrics["qlike"],
                qlike_baseline=result.model.metrics["qlike_baseline"],
                rmse=result.model.metrics["rmse"],
            ),
        )

    async def assess_volatility_regime(
        self, ticker: str, quantile: float, lookback: int
    ) -> VolatilityRegime:
        """Оценить режим волатильности: прогноз vs исторический квантиль (ml-spec §9).

        Raises:
            ModelNotLoadedError: модель волатильности не загружена из реестра.
            SecurityNotFoundError: тикер не найден в БД.
            InsufficientHistoryError: менее MIN_VOLATILITY_HISTORY валидных r
                или менее MIN_REGIME_OBSERVATIONS ненулевых rv_target.
        """
        result = await self._forecast(ticker)

        realized = np.sqrt(result.frame["rv_target"].dropna().to_numpy(dtype=float))
        if len(realized) < MIN_REGIME_OBSERVATIONS:
            raise InsufficientHistoryError(ticker, len(realized), MIN_REGIME_OBSERVATIONS)

        trailing = realized[-lookback:]
        threshold = float(np.quantile(trailing, quantile))
        return VolatilityRegime(
            ticker=result.security.ticker,
            predicted_for=result.predicted_for,
            volatility=result.volatility,
            threshold=threshold,
            is_elevated=result.volatility > threshold,
            quantile=quantile,
            lookback=lookback,
        )

    async def _forecast(self, ticker: str) -> _Forecast:
        """Общий пайплайн: резолв тикера → фичи → read-through кэш/инференс → уpsert.

        Raises:
            ModelNotLoadedError: если bundle.volatility is None.
            SecurityNotFoundError: если тикер не найден в БД.
            InsufficientHistoryError: если валидных r < MIN_VOLATILITY_HISTORY.
        """
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

        return _Forecast(
            security=security,
            frame=frame,
            predicted_for=predicted_for,
            volatility=volatility,
            model=model,
        )

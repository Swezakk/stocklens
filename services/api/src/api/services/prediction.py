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
import structlog
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
from api.schemas.predict import (
    ForecastRefreshSummary,
    VolatilityForecastHistoryOut,
    VolatilityForecastPoint,
    VolatilityMetrics,
    VolatilityPredictionOut,
    VolatilityRegime,
)

#: Минимум ненулевых значений rv_target для устойчивой оценки режима волатильности.
#: Ниже 60 распределение слишком бедно для осмысленного 0.80-квантиля.
MIN_REGIME_OBSERVATIONS = 60

#: Размер страницы при переборе активных бумаг. Цикл пагинирует до упора — это не потолок.
_REFRESH_PAGE_SIZE = 500

logger = structlog.get_logger(__name__)

#: Горизонт прогноза по умолчанию (дни) — используется когда модель не загружена,
#: чтобы build_serving_frame мог рассчитать rv_target (forward-window).
_DEFAULT_HORIZON_DAYS = 5


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

    async def refresh_active_forecasts(self) -> ForecastRefreshSummary:
        """Сгенерировать прогнозы волатильности для всех активных бумаг.

        Короткий выход: если модель не загружена, возвращает нулевой итог немедленно —
        ни один тикер не обрабатывается и predict_volatility не вызывается.

        Изоляция ошибок: SecurityNotFoundError и InsufficientHistoryError считаются
        ожидаемыми (skipped); любое другое исключение логируется и считается failed,
        но цикл продолжается до конца.
        """
        if self._bundle.volatility is None:
            return ForecastRefreshSummary(generated=0, skipped=0, failed=0, total=0)

        generated = 0
        skipped = 0
        failed = 0
        offset = 0

        while True:
            page, total = await self._security_repo.list_securities(
                is_active=True,
                limit=_REFRESH_PAGE_SIZE,
                offset=offset,
            )
            for security in page:
                try:
                    await self.predict_volatility(security.ticker)
                    generated += 1
                except (SecurityNotFoundError, InsufficientHistoryError):
                    skipped += 1
                except Exception:
                    failed += 1
                    logger.exception("forecast_refresh_ticker_failed", ticker=security.ticker)

            offset += len(page)
            if offset >= total:
                break

        logger.info(
            "forecast_refresh_complete",
            generated=generated,
            skipped=skipped,
            failed=failed,
            total=generated + skipped + failed,
        )
        return ForecastRefreshSummary(
            generated=generated,
            skipped=skipped,
            failed=failed,
            total=generated + skipped + failed,
        )

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

    async def forecast_history(self, ticker: str, lookback: int) -> VolatilityForecastHistoryOut:
        """История прогнозов волатильности vs реализованных значений (ml-spec §10).

        Читает сохранённые прогнозы из ``predictions`` и объединяет их с реализованной
        волатильностью (``sqrt(rv_target)``) из feature-фрейма — тот же показатель,
        который таргетирует модель. Не поднимает ошибок при отсутствии модели или
        нехватке истории: возвращает пустые points.
        """
        normalized = ticker.strip().upper()
        security = await self._security_repo.get_by_ticker(normalized)
        if security is None:
            raise SecurityNotFoundError(normalized)

        model = self._bundle.volatility
        horizon = model.horizon_days if model is not None else _DEFAULT_HORIZON_DAYS

        frame = build_serving_frame(
            await self._feature_repo.load_candles(security.id),
            await self._feature_repo.load_dividends(security.id),
            await self._feature_repo.load_splits(security.id),
            train_start=self._settings.ml_train_start,
            horizon=horizon,
        )

        model_name, model_version, metrics = _extract_model_meta(model)

        if frame.empty:
            return VolatilityForecastHistoryOut(
                ticker=security.ticker,
                model=model_name,
                model_version=model_version,
                metrics_vs_baseline=metrics,
                points=[],
            )

        all_dates: list[date] = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
        window_dates = all_dates[-lookback:]
        date_from = window_dates[0]
        date_to = window_dates[-1]

        realized_by_date = _build_realized_map(frame, all_dates)

        forecasts_by_date: dict[date, float]
        if model is not None and model_version is not None:
            forecasts_by_date = await self._prediction_repo.list_values(
                security.id, PredictionKind.VOLATILITY, model_version, date_from, date_to
            )
        else:
            forecasts_by_date = {}

        points = _join_forecast_points(window_dates, forecasts_by_date, realized_by_date)

        return VolatilityForecastHistoryOut(
            ticker=security.ticker,
            model=model_name,
            model_version=model_version,
            metrics_vs_baseline=metrics,
            points=points,
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


def _extract_model_meta(
    model: LoadedVolatilityModel | None,
) -> tuple[str | None, str | None, VolatilityMetrics | None]:
    """Извлечь имя, версию и метрики из загруженной модели (или None при отсутствии)."""
    if model is None:
        return None, None, None
    return (
        model.method,
        model.model_version,
        VolatilityMetrics(
            qlike=model.metrics["qlike"],
            qlike_baseline=model.metrics["qlike_baseline"],
            rmse=model.metrics["rmse"],
        ),
    )


def _build_realized_map(frame: pd.DataFrame, all_dates: list[date]) -> dict[date, float]:
    """Построить {trade_date: sqrt(rv_target)} для строк с ненулевым rv_target."""
    realized: dict[date, float] = {}
    rv_col = frame["rv_target"].to_numpy(dtype=float)
    for i, d in enumerate(all_dates):
        val = rv_col[i]
        if not np.isnan(val):
            realized[d] = float(np.sqrt(val))
    return realized


def _join_forecast_points(
    window_dates: list[date],
    forecasts_by_date: dict[date, float],
    realized_by_date: dict[date, float],
) -> list[VolatilityForecastPoint]:
    """Объединить прогнозы и реализованные значения в список точек, исключая пустые."""
    all_dates_in_window = sorted(set(window_dates) | set(forecasts_by_date))
    points: list[VolatilityForecastPoint] = []
    for d in all_dates_in_window:
        forecast = forecasts_by_date.get(d)
        realized = realized_by_date.get(d)
        if forecast is not None or realized is not None:
            points.append(VolatilityForecastPoint(date=d, forecast=forecast, realized=realized))
    return points

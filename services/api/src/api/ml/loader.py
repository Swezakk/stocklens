"""Загрузка ML-моделей из реестра MLflow в app.state (ml-spec §8.1).

Грузит модель волатильности по алиасу (``models:/<name>@<alias>``) через Models-from-Code —
исполняется самодостаточный артефакт (нужен ``arch`` в образе API) — и модель тренда
(нативный CatBoost-артефакт → P(up), а не метка pyfunc-обёртки; тикет 3455b248). Реестр на
старте может быть недоступен → ретраи для волатильности; после исчерпания возвращается
bundle с volatility=None: процесс остаётся живым (liveness), readiness репортит модель
недоступной (§8.2) — НЕ падаем. Загрузка моделей независима (инвариант «ошибка одного
источника не валит остальные»): промах тренда логируется и оставляет trend=None, не затрагивая
волатильность.
"""

import time

import structlog
from stocklens_ml.config import HORIZON_DAYS

import mlflow
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedTrendModel, LoadedVolatilityModel, ModelBundle
from api.ml.trend import CatBoostTrendPredictor
from mlflow import MlflowClient

logger = structlog.get_logger(__name__)


def _load_volatility(settings: ApiSettings) -> LoadedVolatilityModel:
    """Загрузить модель волатильности по алиасу + версию реестра (одна попытка)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.ml_volatility_model}@{settings.ml_model_alias}"
    loaded = mlflow.pyfunc.load_model(uri)
    model = loaded.unwrap_python_model()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.ml_volatility_model, settings.ml_model_alias
    ).version
    return LoadedVolatilityModel(
        predictor=model,
        model_version=str(version),
        method=str(model.method),
        metrics=dict(model.metrics),
        horizon_days=int(model.horizon),
    )


def _load_trend(settings: ApiSettings) -> LoadedTrendModel:
    """Загрузить нативный CatBoost-артефакт тренда по алиасу + версию реестра (одна попытка)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.ml_trend_model}@{settings.ml_model_alias}"
    model = mlflow.catboost.load_model(uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.ml_trend_model, settings.ml_model_alias
    ).version
    return LoadedTrendModel(
        predictor=CatBoostTrendPredictor(model),
        model_version=str(version),
        horizon_days=HORIZON_DAYS,
    )


def _load_volatility_with_retries(settings: ApiSettings) -> LoadedVolatilityModel | None:
    """Загрузить волатильность с ретраями; на исчерпание попыток — None (degraded readiness)."""
    for attempt in range(1, settings.ml_load_attempts + 1):
        try:
            volatility = _load_volatility(settings)
            logger.info(
                "ml_model_loaded",
                model=settings.ml_volatility_model,
                version=volatility.model_version,
                method=volatility.method,
            )
            return volatility
        except Exception as exc:
            logger.warning(
                "ml_model_load_failed",
                attempt=attempt,
                max_attempts=settings.ml_load_attempts,
                model=settings.ml_volatility_model,
                reason=str(exc),
            )
            if attempt < settings.ml_load_attempts:
                time.sleep(settings.ml_load_interval_seconds)

    logger.error("ml_model_unavailable", model=settings.ml_volatility_model)
    return None


def _try_load_trend(settings: ApiSettings) -> LoadedTrendModel | None:
    """Загрузить тренд одной попыткой; промах логируется и даёт None (не валит волатильность)."""
    try:
        trend = _load_trend(settings)
        logger.info(
            "ml_model_loaded",
            model=settings.ml_trend_model,
            version=trend.model_version,
        )
        return trend
    except Exception as exc:
        logger.warning(
            "ml_model_load_failed",
            model=settings.ml_trend_model,
            reason=str(exc),
        )
        return None


def load_bundle(settings: ApiSettings) -> ModelBundle:
    """Загрузить модели в bundle независимо друг от друга; промах любой не валит остальные.

    Синхронная (mlflow sync) — вызывается из lifespan через threadpool, чтобы не блокировать
    event-loop на старте. Волатильность грузится с ретраями (обязательная для readiness),
    тренд — одной попыткой в собственном try/except (опционален, не блокирует готовность).
    """
    volatility = _load_volatility_with_retries(settings)
    trend = _try_load_trend(settings)
    return ModelBundle(volatility=volatility, trend=trend)

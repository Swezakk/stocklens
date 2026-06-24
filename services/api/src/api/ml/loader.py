"""Загрузка ML-моделей из реестра MLflow в app.state (ml-spec §8.1).

Грузит модель волатильности по алиасу (``models:/<name>@<alias>``) через Models-from-Code —
исполняется самодостаточный артефакт (нужен ``arch`` в образе API). Реестр на старте может
быть недоступен → ретраи; после исчерпания возвращается пустой bundle (volatility=None):
процесс остаётся живым (liveness), readiness репортит модель недоступной (§8.2) — НЕ падаем.
Тренд отложен (модель не зарегистрирована) и здесь не грузится.
"""

import time

import structlog

import mlflow
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
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


def load_bundle(settings: ApiSettings) -> ModelBundle:
    """Загрузить модели в bundle с ретраями; на исчерпание — пустой bundle (degraded readiness).

    Синхронная (mlflow sync) — вызывается из lifespan через threadpool, чтобы не блокировать
    event-loop на старте.
    """
    for attempt in range(1, settings.ml_load_attempts + 1):
        try:
            volatility = _load_volatility(settings)
            logger.info(
                "ml_model_loaded",
                model=settings.ml_volatility_model,
                version=volatility.model_version,
                method=volatility.method,
            )
            return ModelBundle(volatility=volatility)
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
    return ModelBundle(volatility=None)

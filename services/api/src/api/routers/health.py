"""Эндпоинты проверки работоспособности сервиса."""

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from api.core.db import RedisDep, SessionDep, SettingsDep
from api.ml.deps import MlBundleDep

router = APIRouter(prefix="/api/v1", tags=["health"])


class LiveResponse(BaseModel):
    """Тело ответа /health/live."""

    status: Literal["alive"]


class ReadyResponse(BaseModel):
    """Тело ответа /health/ready."""

    status: Literal["ready", "degraded"]
    database: Literal["ok", "unavailable"]
    cache: Literal["ok", "degraded"]
    models: Literal["ok", "unavailable"]


@router.get(
    "/health/live",
    response_model=LiveResponse,
    summary="Liveness probe",
    description="Возвращает 200 если процесс запущен.",
)
async def health_live() -> LiveResponse:
    """Liveness: процесс жив."""
    return LiveResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Проверяет доступность БД и Redis. 503 если БД недоступна.",
)
async def health_ready(
    session: SessionDep,
    redis: RedisDep,
    bundle: MlBundleDep,
    settings: SettingsDep,
) -> JSONResponse:
    """Readiness: проверить БД, Redis и ML-модели.

    503 если БД недоступна или модели не загружены при ML_REQUIRED_FOR_READY=true (§8.2);
    200 с status='degraded' при недоступном Redis или информативно недоступных моделях.
    """
    db_status: Literal["ok", "unavailable"] = "ok"
    cache_status: Literal["ok", "degraded"] = "ok"

    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    try:
        await redis.ping()
    except Exception:
        cache_status = "degraded"

    models_status: Literal["ok", "unavailable"] = "ok" if bundle.ready() else "unavailable"

    overall_status: Literal["ready", "degraded"] = (
        "ready" if cache_status == "ok" and models_status == "ok" else "degraded"
    )

    models_block_ready = models_status == "unavailable" and settings.ml_required_for_ready
    http_status = 503 if db_status == "unavailable" or models_block_ready else 200

    body = ReadyResponse(
        status=overall_status,
        database=db_status,
        cache=cache_status,
        models=models_status,
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())

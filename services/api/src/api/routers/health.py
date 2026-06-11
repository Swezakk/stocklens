"""Эндпоинты проверки работоспособности сервиса."""

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from api.core.db import RedisDep, SessionDep

router = APIRouter(prefix="/api/v1", tags=["health"])


class LiveResponse(BaseModel):
    """Тело ответа /health/live."""

    status: Literal["alive"]


class ReadyResponse(BaseModel):
    """Тело ответа /health/ready."""

    status: Literal["ready", "degraded"]
    database: Literal["ok", "unavailable"]
    cache: Literal["ok", "degraded"]


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
) -> JSONResponse:
    """Readiness: проверить БД и Redis.

    503 если БД недоступна; 200 с status='degraded' если Redis недоступен.
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

    overall_status: Literal["ready", "degraded"] = (
        "degraded" if cache_status == "degraded" else "ready"
    )

    http_status = 503 if db_status == "unavailable" else 200

    body = ReadyResponse(
        status=overall_status,
        database=db_status,
        cache=cache_status,
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())

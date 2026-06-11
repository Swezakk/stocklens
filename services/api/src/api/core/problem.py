"""Обработчики исключений для RFC 9457 Problem Details.

Все 4xx/5xx ответы возвращают application/problem+json с полями:
type, title, status, detail, instance.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.core.exceptions import ApiError

_PROBLEM_MEDIA_TYPE = "application/problem+json"

logger = structlog.get_logger(__name__)


def _problem_response(
    request: Request,
    status: int,
    problem_type: str,
    title: str,
    detail: str,
) -> JSONResponse:
    """Сформировать JSONResponse в формате Problem Details (RFC 9457)."""
    return JSONResponse(
        status_code=status,
        media_type=_PROBLEM_MEDIA_TYPE,
        content={
            "type": problem_type,
            "title": title,
            "status": status,
            "detail": detail,
            "instance": str(request.url.path),
        },
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Зарегистрировать обработчики исключений на экземпляре FastAPI."""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _problem_response(
            request,
            status=exc.status,
            problem_type=exc.problem_type,
            title=exc.title,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        detail = "; ".join(
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return _problem_response(
            request,
            status=422,
            problem_type="https://stocklens.local/problems/validation-error",
            title="Ошибка валидации запроса",
            detail=detail,
        )

    @app.exception_handler(Exception)
    async def handle_generic_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            exc_type=type(exc).__name__,
            exc_str=str(exc),
        )
        return _problem_response(
            request,
            status=500,
            problem_type="https://stocklens.local/problems/internal-error",
            title="Внутренняя ошибка сервиса",
            detail="Внутренняя ошибка сервиса. Попробуйте повторить запрос позже.",
        )

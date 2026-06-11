"""ASGI-middleware для request-id и структурированного логирования запросов."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Назначить request_id каждому запросу, привязать к structlog-контексту."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        structlog.contextvars.clear_contextvars()

        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 1)

        response.headers[_REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_completed",
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response

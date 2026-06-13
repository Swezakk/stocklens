"""Эндпоинт выдачи JWT-токенов (OAuth2 Password Flow, только режим local)."""

from enum import StrEnum
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api.core.auth.deps import _get_auth_settings, check_login_rate_limit
from api.core.auth.issuer import issue_token
from api.core.db import RedisDep
from api.core.exceptions import UnauthorizedError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class OAuthGrantType(StrEnum):
    """OAuth2-тип выданного credentials-объекта."""

    BEARER = "bearer"


class TokenOut(BaseModel):
    """OAuth2-совместимый ответ с токеном доступа."""

    access_token: str
    token_type: str
    expires_in: int


def _client_ip_from_request(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


@router.post(
    "/token",
    response_model=TokenOut,
    summary="Получить токен доступа",
    description=(
        "OAuth2 Password Flow. Принимает username/password владельца, "
        "возвращает JWT-токен. Только в режиме mode=local."
    ),
)
async def login(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    redis: RedisDep,
) -> TokenOut:
    """POST /auth/token — выдать JWT при корректных учётных данных владельца."""
    settings = _get_auth_settings(request)
    client_ip = _client_ip_from_request(request)

    await check_login_rate_limit(request, redis, settings)

    outcome = "success"
    try:
        jwt_value = issue_token(form.username, form.password, settings)
    except UnauthorizedError:
        outcome = "failure"
        raise
    finally:
        logger.info(
            "auth_login",
            outcome=outcome,
            username=form.username,
            client_ip=client_ip,
        )

    return TokenOut(
        access_token=jwt_value,
        token_type=OAuthGrantType.BEARER,
        expires_in=settings.token_ttl_seconds,
    )

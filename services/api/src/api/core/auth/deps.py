"""FastAPI-зависимости для аутентификации и защиты эндпоинтов."""

from typing import Annotated

import structlog
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer

from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.auth.validator import decode_token
from api.core.cache import RedisClientProtocol
from api.core.exceptions import RateLimitError, UnauthorizedError

logger = structlog.get_logger(__name__)

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")

_RATE_LIMIT_KEY_PREFIX = "auth:rate:"
_RATE_LIMIT_GLOBAL_KEY = "auth:rate:global"


def _get_auth_settings(request: Request) -> AuthSettings:
    settings: AuthSettings = request.app.state.auth_settings
    return settings


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def check_login_rate_limit(
    request: Request,
    redis: RedisClientProtocol,
    settings: AuthSettings,
) -> None:
    """Проверить и инкрементировать счётчики rate-limit для /auth/token.

    Использует два Redis-ключа: по IP клиента и глобальный (защита от распределённого брутфорса).
    Поднимает RateLimitError (429) если лимит превышен.
    """
    ip = _client_ip(request)
    ip_key = f"{_RATE_LIMIT_KEY_PREFIX}{ip}"
    window = settings.login_rate_limit_window_seconds
    max_attempts = settings.login_rate_limit_max

    # INCR атомарен (без гонки read-modify-write); TTL окна ставится при первом инкременте.
    for key in (ip_key, _RATE_LIMIT_GLOBAL_KEY):
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window)
        if count > max_attempts:
            logger.warning(
                "auth_rate_limited",
                client_ip=ip,
                key=key,
                count=count,
                max_attempts=max_attempts,
            )
            raise RateLimitError(
                detail=f"Слишком много попыток входа. Повторите через {window} секунд.",
                retry_after_seconds=window,
            )


async def require_auth(
    token: Annotated[str, Depends(_oauth2_scheme)],
    request: Request,
) -> Principal:
    """Декодировать Bearer-токен и вернуть Principal.

    Регистрируется как зависимость на защищённых роутерах.
    Поднимает UnauthorizedError → HTTP 401 + WWW-Authenticate: Bearer.
    """
    settings = _get_auth_settings(request)
    try:
        principal = await decode_token(token, settings)
    except UnauthorizedError:
        raise
    except Exception as exc:
        raise UnauthorizedError("Недействительный токен") from exc

    return principal


PrincipalDep = Annotated[Principal, Depends(require_auth)]

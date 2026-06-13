"""Выдача локальных JWT-токенов для единственного владельца сервиса."""

import hmac
from datetime import UTC, datetime, timedelta

import jwt

from api.core.auth.settings import AuthSettings
from api.core.exceptions import UnauthorizedError

_ALGORITHM_LOCAL = "HS256"


def issue_token(username: str, password: str, settings: AuthSettings) -> str:
    """Выдать HS256-токен при совпадении учётных данных владельца.

    hmac.compare_digest используется для обоих полей — защита от timing-атак.
    Поднимает UnauthorizedError при неверных данных или неактивном режиме.
    """
    if settings.mode != "local":
        raise UnauthorizedError("Локальная выдача токенов отключена")

    if not settings.secret:
        raise UnauthorizedError("Локальный режим не настроен: AUTH_SECRET не задан")

    username_match = hmac.compare_digest(username.encode(), settings.owner_username.encode())
    password_match = hmac.compare_digest(password.encode(), settings.owner_password.encode())

    if not (username_match and password_match):
        raise UnauthorizedError("Неверные учётные данные")

    now = datetime.now(tz=UTC)
    exp = now + timedelta(seconds=settings.token_ttl_seconds)

    payload: dict[str, object] = {
        "sub": username,
        "scope": "",
        "iss": settings.issuer,
        "aud": settings.audience,
        "iat": now,
        "exp": exp,
    }

    return jwt.encode(payload, settings.secret, algorithm=_ALGORITHM_LOCAL)

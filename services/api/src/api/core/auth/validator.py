"""Валидация JWT-токенов для режимов local (HS256) и oidc (RS256).

Алгоритм ВСЕГДА берётся из конфига, никогда из заголовка токена.
Смешивать HS256 и RS256 в одном списке algorithms ЗАПРЕЩЕНО:
публичный RS256-ключ, переданный как HMAC-секрет, даёт успешную верификацию
HS256-токена — это известная атака на библиотеки JWT.
"""

import functools

import anyio
import jwt
from jwt import PyJWKClient

from api.core.auth.principal import Principal
from api.core.auth.settings import AuthSettings
from api.core.exceptions import UnauthorizedError

_ALGORITHMS_LOCAL = ["HS256"]
_ALGORITHMS_OIDC = ["RS256"]


@functools.lru_cache(maxsize=1)
def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Вернуть кэшированный PyJWKClient (создаётся один раз на процесс)."""
    return PyJWKClient(jwks_url)


def _build_principal(payload: dict[str, object]) -> Principal:
    """Построить Principal из декодированного payload."""
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise UnauthorizedError("Недействительный токен: отсутствует sub")

    scope_raw = payload.get("scope", "")
    scopes: list[str] = scope_raw.split() if isinstance(scope_raw, str) and scope_raw else []

    return Principal(sub=sub, scopes=scopes, claims=payload)


async def decode_token(token: str, settings: AuthSettings) -> Principal:
    """Декодировать и верифицировать JWT, вернуть Principal.

    Поднимает UnauthorizedError при любой проблеме с токеном.
    Алгоритм пиннирован из конфига — заголовок токена не используется.
    """
    try:
        if settings.mode == "local":
            return _decode_local(token, settings)
        return await _decode_oidc(token, settings)
    except UnauthorizedError:
        raise
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError(f"Недействительный токен: {exc}") from exc
    except Exception as exc:
        raise UnauthorizedError("Недействительный токен") from exc


def _decode_local(token: str, settings: AuthSettings) -> Principal:
    """Верификация HS256-токена локального issuer."""
    if not settings.secret:
        raise UnauthorizedError("Локальный режим не настроен: AUTH_SECRET не задан")

    payload: dict[str, object] = jwt.decode(
        token,
        settings.secret,
        algorithms=_ALGORITHMS_LOCAL,
        audience=settings.audience,
        issuer=settings.issuer,
        leeway=settings.leeway_seconds,
        options={"require": ["sub", "exp", "iat", "iss", "aud"]},
    )
    return _build_principal(payload)


async def _decode_oidc(token: str, settings: AuthSettings) -> Principal:
    """Верификация RS256-токена от OIDC-провайдера через JWKS."""
    if not settings.jwks_url:
        raise UnauthorizedError("OIDC-режим не настроен: AUTH_JWKS_URL не задан")

    client = _get_jwks_client(settings.jwks_url)

    signing_key = await anyio.to_thread.run_sync(lambda: client.get_signing_key_from_jwt(token))

    payload: dict[str, object] = jwt.decode(
        token,
        signing_key.key,
        algorithms=_ALGORITHMS_OIDC,
        audience=settings.audience,
        issuer=settings.issuer,
        leeway=settings.leeway_seconds,
        options={"require": ["sub", "exp", "iat", "iss", "aud"]},
    )
    return _build_principal(payload)

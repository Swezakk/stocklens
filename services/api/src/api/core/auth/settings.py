"""Настройки подсистемы аутентификации API StockLens.

Переменные окружения (с префиксом AUTH_):
- AUTH_MODE              — "local" или "oidc" (default: "local")
- AUTH_ISSUER            — iss claim в токенах
- AUTH_AUDIENCE          — aud claim в токенах
- AUTH_SECRET            — секрет HS256 (только для mode=local)
- AUTH_JWKS_URL          — URL JWKS-endpoint (только для mode=oidc)
- AUTH_OWNER_USERNAME    — имя владельца (только для mode=local)
- AUTH_OWNER_PASSWORD    — пароль владельца (только для mode=local)
- AUTH_TOKEN_TTL_SECONDS — срок жизни токена в секундах (default: 3600)
- AUTH_LEEWAY_SECONDS    — допуск при проверке exp/nbf (default: 60)
- AUTH_LOGIN_RATE_LIMIT_MAX             — макс. попыток входа в окне (default: 5)
- AUTH_LOGIN_RATE_LIMIT_WINDOW_SECONDS  — ширина окна в секундах (default: 60)
"""

import functools
from typing import Literal

from pydantic_settings import BaseSettings


class AuthSettings(BaseSettings):
    """Параметры аутентификации сервиса stocklens-api."""

    mode: Literal["local", "oidc"] = "local"
    issuer: str = "https://stocklens.local"
    audience: str = "stocklens-api"

    secret: str | None = None
    jwks_url: str | None = None

    owner_username: str = "admin"
    owner_password: str

    token_ttl_seconds: int = 3600
    leeway_seconds: int = 60

    login_rate_limit_max: int = 5
    login_rate_limit_window_seconds: int = 60

    model_config = {
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "AUTH_",
    }


@functools.lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    """Вернуть кэшированный экземпляр AuthSettings (читается из env один раз)."""
    return AuthSettings.model_validate({})

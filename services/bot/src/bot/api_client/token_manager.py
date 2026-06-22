"""Жизненный цикл JWT-токена бота к StockLens API (DESIGN.md §7, §11).

Бот — долгоживущий backend-процесс: токен кэшируется в памяти менеджера (не в сессии),
проактивно перевыпускается до истечения и форс-обновляется при реактивном 401. Single-flight
на ``asyncio.Lock`` защищает лимит логина API (5/60s) от параллельных mint.

Учётные данные владельца берутся из настроек (env) — бот логинится сам, без участия
пользователя. Часы инъектируются (``Callable[[], float]``) — проверка истечения unit-тестируема
без реального ожидания (DESIGN §8).
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import time

import httpx
from pydantic import BaseModel

from bot.api_client.errors import ApiUnavailableError, AuthError

#: Путь выдачи токена под префиксом версии (form-encoded OAuth2 Password Flow).
_LOGIN_PATH = "/auth/token"

#: HTTP 401 — неверные учётные данные владельца; ≥400 — прочий сбой выдачи токена.
_HTTP_UNAUTHORIZED = 401
_HTTP_BAD_REQUEST = 400


class TokenResponse(BaseModel):
    """Зеркало ответа POST /auth/token (TokenOut API): токен и время жизни."""

    access_token: str
    token_type: str
    expires_in: int


@dataclass(frozen=True)
class AuthConfig:
    """Неизменяемая конфигурация аутентификации бота (группирует параметры менеджера)."""

    api_base_url: str
    api_prefix: str
    auth_username: str
    auth_password: str
    token_refresh_margin_seconds: int
    request_timeout_seconds: float


class TokenManager:
    """Async-управление JWT: кэш в памяти, проактивный refresh, single-flight, форс-401.

    Параметры:
    - ``config`` — хост/префикс API, учётные данные владельца, margin, таймаут;
    - ``clock`` — источник epoch-времени (по умолчанию ``time.time``); инъекция делает
      проверку истечения unit-тестируемой.
    """

    def __init__(self, config: AuthConfig, clock: Callable[[], float] = time) -> None:
        self._config = config
        self._clock = clock
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._expiry: float = 0.0

    async def get_token(self) -> str:
        """Вернуть валидный Bearer-токен, перевыпустив при отсутствии или близости к истечению.

        Single-flight: после захвата лока повторно проверяем свежесть — если параллельный
        вызов уже выпустил токен, переиспользуем его, а не минтим повторно.
        """
        if self._token is not None and not self._is_expiring():
            return self._token
        async with self._lock:
            if self._token is not None and not self._is_expiring():
                return self._token
            return await self._mint()

    async def force_refresh(self) -> str:
        """Немедленно перевыпустить токен (хук ``ApiClient.on_unauthorized`` при реактивном 401).

        Single-flight: если параллельный обработчик 401 уже сменил токен, переиспользуем новый,
        не минтя снова (защита лимита логина API).
        """
        stale = self._token
        async with self._lock:
            if self._token is not None and self._token != stale:
                return self._token
            return await self._mint()

    async def _mint(self) -> str:
        """Запросить новый токен у API и сохранить с временем истечения (вызывается под локом).

        Раскладка: 401 → ``AuthError``; сетевая ошибка/таймаут или прочий ≥400 →
        ``ApiUnavailableError`` (бот не падает traceback'ом, а логирует и сообщает в чат).
        """
        url = f"{self._config.api_base_url}{self._config.api_prefix}{_LOGIN_PATH}"
        form = {"username": self._config.auth_username, "password": self._config.auth_password}
        try:
            async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
                response = await client.post(url, data=form)
        except httpx.RequestError as exc:
            raise ApiUnavailableError() from exc

        if response.status_code == _HTTP_UNAUTHORIZED:
            raise AuthError()
        if response.status_code >= _HTTP_BAD_REQUEST:
            raise ApiUnavailableError()

        token = TokenResponse.model_validate(response.json())
        self._token = token.access_token
        self._expiry = self._clock() + token.expires_in
        return token.access_token

    def _is_expiring(self) -> bool:
        """True, если до истечения осталось меньше margin (нужен проактивный refresh)."""
        return self._clock() > self._expiry - self._config.token_refresh_margin_seconds

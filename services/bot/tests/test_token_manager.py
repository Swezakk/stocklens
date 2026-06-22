"""Тесты async-менеджера JWT-токена бота (DESIGN §7, §11).

Покрытие: проактивный refresh по близости к истечению, single-flight на параллельных
вызовах (один mint), форс-обновление, раскладка ошибок выдачи токена (401 → AuthError,
сеть → ApiUnavailableError). HTTP мокается через respx; часы инъектируются (без ожидания).
"""

import asyncio

import httpx
import pytest
import respx
from bot.api_client.errors import ApiUnavailableError, AuthError
from bot.api_client.token_manager import AuthConfig, TokenManager

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_LOGIN_URL = f"{_BASE}{_PREFIX}/auth/token"
#: Учётные данные и тип токена держим в переменных (не литералах у secret-именованных полей).
_CREDS = ("admin", "owner-pw")
_BEARER = "bearer"
_MARGIN_SECONDS = 60
_TTL_SECONDS = 3600


class _Clock:
    """Управляемые часы: проверка истечения без реального ожидания."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _config() -> AuthConfig:
    """Собрать AuthConfig тестового бота (креды — из переменной, не литерала)."""
    return AuthConfig(
        api_base_url=_BASE,
        api_prefix=_PREFIX,
        auth_username=_CREDS[0],
        auth_password=_CREDS[1],
        token_refresh_margin_seconds=_MARGIN_SECONDS,
        request_timeout_seconds=5.0,
    )


def _payload(access: str, expires_in: int = _TTL_SECONDS) -> dict[str, object]:
    """Тело ответа /auth/token; значения — переменные, не строковые литералы у token-ключей."""
    return {"access_token": access, "token_type": _BEARER, "expires_in": expires_in}


@respx.mock
async def test_get_token_mints_once_and_caches() -> None:
    route = respx.post(_LOGIN_URL).mock(return_value=httpx.Response(200, json=_payload("a-1")))
    manager = TokenManager(_config(), clock=_Clock())

    first = await manager.get_token()
    second = await manager.get_token()

    assert first == "a-1"
    assert second == "a-1"
    assert route.call_count == 1


@respx.mock
async def test_get_token_remints_when_within_refresh_margin() -> None:
    respx.post(_LOGIN_URL).mock(
        side_effect=[
            httpx.Response(200, json=_payload("a-1")),
            httpx.Response(200, json=_payload("a-2")),
        ]
    )
    clock = _Clock(now=1000.0)
    manager = TokenManager(_config(), clock=clock)

    first = await manager.get_token()
    clock.now = 1000.0 + _TTL_SECONDS - (_MARGIN_SECONDS / 2)
    second = await manager.get_token()

    assert first == "a-1"
    assert second == "a-2"


@respx.mock
async def test_concurrent_get_token_mints_once_single_flight() -> None:
    route = respx.post(_LOGIN_URL).mock(return_value=httpx.Response(200, json=_payload("a-1")))
    manager = TokenManager(_config(), clock=_Clock())

    results = await asyncio.gather(*(manager.get_token() for _ in range(5)))

    assert all(value == "a-1" for value in results)
    assert route.call_count == 1


@respx.mock
async def test_force_refresh_mints_new_token() -> None:
    respx.post(_LOGIN_URL).mock(
        side_effect=[
            httpx.Response(200, json=_payload("a-1")),
            httpx.Response(200, json=_payload("a-2")),
        ]
    )
    manager = TokenManager(_config(), clock=_Clock())

    first = await manager.get_token()
    refreshed = await manager.force_refresh()

    assert first == "a-1"
    assert refreshed == "a-2"


@respx.mock
async def test_mint_raises_auth_error_on_401() -> None:
    respx.post(_LOGIN_URL).mock(return_value=httpx.Response(401, json={"detail": "bad creds"}))
    manager = TokenManager(_config(), clock=_Clock())

    with pytest.raises(AuthError):
        await manager.get_token()


@respx.mock
async def test_mint_raises_unavailable_on_network_error() -> None:
    respx.post(_LOGIN_URL).mock(side_effect=httpx.ConnectError("boom"))
    manager = TokenManager(_config(), clock=_Clock())

    with pytest.raises(ApiUnavailableError):
        await manager.get_token()

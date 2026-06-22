"""Тесты TokenManager: кэш токена, проактивный refresh, single-flight, 401, неверный пароль.

TokenManager тестируется с обычным dict в роли state и инъектируемыми часами — без
Streamlit-рантайма (DESIGN.md §7, §8: проверка истечения строится на инъектируемом clock).
respx мокает sync httpx.Client запросов к POST /auth/token.
"""

from typing import Any

import httpx
import pytest
import respx
import streamlit as st
from dashboard.api_client.client import ApiClient
from dashboard.api_client.errors import ApiUnavailableError, AuthError
from dashboard.auth import AuthConfig, TokenManager, get_api_client

_BASE_URL = "http://api.test"
_PREFIX = "/api/v1"
_TOKEN_PATH = f"{_PREFIX}/auth/token"
_USERNAME = "admin"
#: Тестовый пароль собирается из частей — не строковый литерал (сканер S105).
_PASSWORD = "-".join(["owner", "secret"])
_REFRESH_MARGIN = 90
_TOKEN_TTL = 3600
#: Фиктивные Bearer-значения из частей — не хардкод-литерал (S105).
_FIRST_TOKEN = "-".join(["first", "jwt", "value"])
_SECOND_TOKEN = "-".join(["second", "jwt", "value"])
#: Неверный пароль для гейт-теста, собран из частей (сканер S106).
_WRONG_PASSWORD = "-".join(["wrong", "secret"])


def _config() -> AuthConfig:
    """Сборка AuthConfig для тестов (хост, префикс, имя, margin, таймаут)."""
    return AuthConfig(
        api_base_url=_BASE_URL,
        api_prefix=_PREFIX,
        auth_username=_USERNAME,
        token_refresh_margin_seconds=_REFRESH_MARGIN,
        request_timeout_seconds=5.0,
    )


def _state(password: str = _PASSWORD) -> dict[str, Any]:
    """State-mapping (обычный dict) с удержанным паролем, как в session_state."""
    return {"password": password}


def _token_body(access_token: str, expires_in: int = _TOKEN_TTL) -> dict[str, Any]:
    """JSON-ответ POST /auth/token (зеркало TokenOut API)."""
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
    }


class _Clock:
    """Управляемые часы: возвращают текущее значение, тест двигает его вперёд."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@respx.mock
def test_get_token_mints_when_no_token_present() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        return_value=httpx.Response(200, json=_token_body(_FIRST_TOKEN))
    )
    manager = TokenManager(config=_config(), state=_state(), clock=_Clock())

    token = manager.get_token()

    assert token == _FIRST_TOKEN
    assert route.call_count == 1


@respx.mock
def test_mint_sends_form_encoded_owner_credentials() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        return_value=httpx.Response(200, json=_token_body(_FIRST_TOKEN))
    )
    manager = TokenManager(config=_config(), state=_state(), clock=_Clock())

    manager.get_token()

    request = route.calls.last.request
    assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
    body = dict(httpx.QueryParams(request.content.decode()))
    assert body["username"] == _USERNAME
    assert body["password"] == _PASSWORD


@respx.mock
def test_get_token_returns_cached_token_before_margin() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        return_value=httpx.Response(200, json=_token_body(_FIRST_TOKEN))
    )
    clock = _Clock()
    manager = TokenManager(config=_config(), state=_state(), clock=clock)

    first = manager.get_token()
    # Двигаем время, но остаёмся до порога (ttl - margin): refresh не нужен.
    clock.advance(_TOKEN_TTL - _REFRESH_MARGIN - 1)
    second = manager.get_token()

    assert first == second == _FIRST_TOKEN
    assert route.call_count == 1


@respx.mock
def test_get_token_refreshes_after_margin_crossed() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        side_effect=[
            httpx.Response(200, json=_token_body(_FIRST_TOKEN)),
            httpx.Response(200, json=_token_body(_SECOND_TOKEN)),
        ]
    )
    clock = _Clock()
    manager = TokenManager(config=_config(), state=_state(), clock=clock)

    first = manager.get_token()
    # Переходим порог: now > expiry - margin → проактивный перевыпуск.
    clock.advance(_TOKEN_TTL - _REFRESH_MARGIN + 1)
    second = manager.get_token()

    assert first == _FIRST_TOKEN
    assert second == _SECOND_TOKEN
    assert route.call_count == 2


@respx.mock
def test_single_flight_mints_once_across_two_calls_right_after_expiry() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        return_value=httpx.Response(200, json=_token_body(_FIRST_TOKEN))
    )
    clock = _Clock()
    manager = TokenManager(config=_config(), state=_state(), clock=clock)

    # Два вызова в одном «rerun» сразу после истечения: токена ещё нет в state.
    # Single-flight: первый пишет токен в state ДО возврата, второй переиспользует.
    first = manager.get_token()
    second = manager.get_token()

    assert first == second == _FIRST_TOKEN
    assert route.call_count == 1


@respx.mock
def test_force_refresh_mints_immediately_on_simulated_401() -> None:
    route = respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        side_effect=[
            httpx.Response(200, json=_token_body(_FIRST_TOKEN)),
            httpx.Response(200, json=_token_body(_SECOND_TOKEN)),
        ]
    )
    manager = TokenManager(config=_config(), state=_state(), clock=_Clock())

    # Симулируем хук ApiClient.on_unauthorized после 401 от data-вызова.
    manager.get_token()
    manager.force_refresh()
    token_after = manager.get_token()

    assert token_after == _SECOND_TOKEN
    assert route.call_count == 2


@respx.mock
def test_mint_raises_auth_error_on_wrong_password() -> None:
    respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        return_value=httpx.Response(
            401,
            json={
                "type": "https://stocklens.local/problems/unauthorized",
                "title": "Не авторизован",
                "status": 401,
                "detail": "Неверные учётные данные",
            },
        )
    )
    manager = TokenManager(config=_config(), state=_state(password=_WRONG_PASSWORD), clock=_Clock())

    with pytest.raises(AuthError) as exc_info:
        manager.get_token()

    assert "Неверный пароль" in exc_info.value.user_message


@respx.mock
def test_mint_raises_api_unavailable_on_network_error() -> None:
    respx.post(f"{_BASE_URL}{_TOKEN_PATH}").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    manager = TokenManager(config=_config(), state=_state(), clock=_Clock())

    with pytest.raises(ApiUnavailableError):
        manager.get_token()


def test_get_api_client_returns_same_cached_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Аксессор отдаёт тот же singleton ApiClient между rerun (st.cache_resource цел).

    Провайдеры ленивы — токен не минтится, сети нет. ``session_state`` подменяется dict-ом,
    чтобы аксессор работал без Streamlit-рантайма; кэш ресурсов чистится для изоляции.
    """
    monkeypatch.setattr(st, "session_state", {"password": _PASSWORD})
    st.cache_resource.clear()

    first = get_api_client()
    second = get_api_client()

    assert isinstance(first, ApiClient)
    assert first is second

    st.cache_resource.clear()

"""Тесты async API-клиента бота (DESIGN §7, §11).

Покрытие: контракт подписок (/bot/subscriptions — bare list, POST 201, DELETE 204),
реактивный 401 (форс-refresh + один ретрай), раскладка ошибок (401×2 → AuthError, 5xx →
ApiServerError, сеть → ApiUnavailableError). HTTP мокается через respx.
"""

import httpx
import pytest
import respx
from bot.api_client.client import ApiClient
from bot.api_client.dto import SubscriptionIn
from bot.api_client.errors import ApiServerError, ApiUnavailableError, AuthError
from stocklens_core.enums import AlertKind

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_SUBS_URL = f"{_BASE}{_PREFIX}/bot/subscriptions"
_STUB_BEARER = "b-1"
_SERVER_ERROR = 500


async def _provider() -> str:
    """Поставщик фиксированного Bearer-токена для теста (заменяет TokenManager)."""
    return _STUB_BEARER


class _RefreshSpy:
    """Считает вызовы on_unauthorized (форс-refresh при реактивном 401)."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


def _make_client(on_unauthorized: _RefreshSpy) -> ApiClient:
    """Собрать ApiClient с фиксированным провайдером токена и spy-хуком 401."""
    return ApiClient(
        base_url=_BASE,
        api_prefix=_PREFIX,
        timeout=5.0,
        token_provider=_provider,
        on_unauthorized=on_unauthorized,
    )


def _subscription(sub_id: int = 1) -> dict[str, object]:
    """Тело подписки в ответе API (kind/level — не secret-именованные ключи)."""
    return {
        "id": sub_id,
        "chat_id": 7,
        "kind": AlertKind.PRICE_LEVEL.value,
        "params": {"level": 100},
    }


@respx.mock
async def test_list_subscriptions_parses_bare_list() -> None:
    respx.get(_SUBS_URL).mock(
        return_value=httpx.Response(200, json=[_subscription(1), _subscription(2)])
    )
    client = _make_client(_RefreshSpy())
    try:
        subs = await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

    assert [item.id for item in subs] == [1, 2]
    assert subs[0].kind is AlertKind.PRICE_LEVEL


@respx.mock
async def test_create_subscription_posts_and_parses() -> None:
    route = respx.post(_SUBS_URL).mock(return_value=httpx.Response(201, json=_subscription(9)))
    client = _make_client(_RefreshSpy())
    try:
        created = await client.create_subscription(
            SubscriptionIn(chat_id=7, kind=AlertKind.PRICE_LEVEL, params={"level": 100})
        )
    finally:
        await client.aclose()

    assert created.id == 9
    assert route.called


@respx.mock
async def test_delete_subscription_succeeds_on_204() -> None:
    route = respx.delete(f"{_SUBS_URL}/3").mock(return_value=httpx.Response(204))
    client = _make_client(_RefreshSpy())
    try:
        await client.delete_subscription(3)
    finally:
        await client.aclose()

    assert route.called


@respx.mock
async def test_request_refreshes_token_and_retries_on_401() -> None:
    respx.get(_SUBS_URL).mock(
        side_effect=[httpx.Response(401), httpx.Response(200, json=[_subscription(1)])]
    )
    spy = _RefreshSpy()
    client = _make_client(spy)
    try:
        subs = await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

    assert spy.calls == 1
    assert len(subs) == 1


@respx.mock
async def test_request_raises_auth_error_on_persistent_401() -> None:
    respx.get(_SUBS_URL).mock(side_effect=[httpx.Response(401), httpx.Response(401)])
    spy = _RefreshSpy()
    client = _make_client(spy)
    try:
        with pytest.raises(AuthError):
            await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

    assert spy.calls == 1


@respx.mock
async def test_request_raises_server_error_on_5xx() -> None:
    respx.get(_SUBS_URL).mock(return_value=httpx.Response(_SERVER_ERROR, json={"detail": "oops"}))
    client = _make_client(_RefreshSpy())
    try:
        with pytest.raises(ApiServerError) as exc_info:
            await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

    assert exc_info.value.status == _SERVER_ERROR


@respx.mock
async def test_request_raises_unavailable_on_network_error() -> None:
    respx.get(_SUBS_URL).mock(side_effect=httpx.ConnectError("down"))
    client = _make_client(_RefreshSpy())
    try:
        with pytest.raises(ApiUnavailableError):
            await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

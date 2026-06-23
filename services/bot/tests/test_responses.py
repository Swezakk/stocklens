"""Тесты сборки ответов на команды (DESIGN §11).

Через реальный API-клиент + respx — покрывают ветки: успех, пустой ввод (подсказка+список),
ошибка разбора, сбой API (ApiError → русское сообщение). Без рантайма Telegram.
"""

from datetime import date

import httpx
import respx
from bot.api_client.client import ApiClient
from bot.responses import (
    digest_response,
    portfolio_response,
    start_response,
    subscribe_response,
    unsubscribe_response,
)
from stocklens_core.enums import AlertKind

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_STUB_BEARER = "b-1"
_CHAT_ID = 7
_TODAY = date(2026, 6, 23)
_SERVER_ERROR = 500


async def _provider() -> str:
    return _STUB_BEARER


async def _noop() -> None:
    return None


def _make_client() -> ApiClient:
    return ApiClient(
        base_url=_BASE,
        api_prefix=_PREFIX,
        timeout=5.0,
        token_provider=_provider,
        on_unauthorized=_noop,
    )


def _summary_json() -> dict[str, object]:
    return {
        "positions": [],
        "total_value": "0.00",
        "total_cost": "0.00",
        "total_unrealized_pnl": "0.00",
        "portfolio_return_pct": 0.0,
        "imoex_return_pct": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "imoex_sharpe": 0.0,
        "imoex_max_drawdown": 0.0,
        "period_from": "2026-01-01",
        "period_to": "2026-06-23",
    }


def _subscription_json(sub_id: int) -> dict[str, object]:
    return {
        "id": sub_id,
        "chat_id": _CHAT_ID,
        "kind": AlertKind.PRICE_LEVEL.value,
        "params": {"ticker": "SBER", "level": 250},
    }


def test_start_response_lists_commands() -> None:
    text = start_response()
    assert "/portfolio" in text
    assert "/subscribe" in text


@respx.mock
async def test_portfolio_response_success_renders_portfolio() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_summary_json())
    )
    client = _make_client()
    try:
        text = await portfolio_response(client)
    finally:
        await client.aclose()

    assert "Портфель" in text


@respx.mock
async def test_portfolio_response_server_error_returns_user_message() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(_SERVER_ERROR, json={"detail": "oops"})
    )
    client = _make_client()
    try:
        text = await portfolio_response(client)
    finally:
        await client.aclose()

    assert "Сервис данных" in text


@respx.mock
async def test_digest_response_network_error_returns_unavailable_message() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(side_effect=httpx.ConnectError("down"))
    client = _make_client()
    try:
        text = await digest_response(client, _TODAY)
    finally:
        await client.aclose()

    assert "недоступен" in text


@respx.mock
async def test_subscribe_response_no_args_shows_usage_and_subscriptions() -> None:
    respx.get(f"{_BASE}{_PREFIX}/bot/subscriptions").mock(
        return_value=httpx.Response(200, json=[_subscription_json(1)])
    )
    client = _make_client()
    try:
        text = await subscribe_response(client, _CHAT_ID, None)
    finally:
        await client.aclose()

    assert "price_level" in text
    assert "Ваши подписки" in text


@respx.mock
async def test_subscribe_response_creates_subscription() -> None:
    respx.post(f"{_BASE}{_PREFIX}/bot/subscriptions").mock(
        return_value=httpx.Response(201, json=_subscription_json(9))
    )
    client = _make_client()
    try:
        text = await subscribe_response(client, _CHAT_ID, "price_level SBER 250")
    finally:
        await client.aclose()

    assert "создана" in text


async def test_subscribe_response_parse_error_returns_message() -> None:
    client = _make_client()
    try:
        text = await subscribe_response(client, _CHAT_ID, "price_level SBER")
    finally:
        await client.aclose()

    assert "уровень" in text.lower()


@respx.mock
async def test_unsubscribe_response_deletes_by_id() -> None:
    respx.delete(f"{_BASE}{_PREFIX}/bot/subscriptions/3").mock(return_value=httpx.Response(204))
    client = _make_client()
    try:
        text = await unsubscribe_response(client, _CHAT_ID, "3")
    finally:
        await client.aclose()

    assert "удалена" in text


@respx.mock
async def test_unsubscribe_response_no_args_lists_subscriptions() -> None:
    respx.get(f"{_BASE}{_PREFIX}/bot/subscriptions").mock(
        return_value=httpx.Response(200, json=[_subscription_json(5)])
    )
    client = _make_client()
    try:
        text = await unsubscribe_response(client, _CHAT_ID, None)
    finally:
        await client.aclose()

    assert "Ваши подписки" in text
    assert "/unsubscribe" in text

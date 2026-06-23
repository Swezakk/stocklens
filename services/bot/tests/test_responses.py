"""Тесты сборки ответов на команды (DESIGN §11).

Через реальный API-клиент + respx — покрывают ветки: успех, ошибка API (ApiError → русское
сообщение), ошибка разбора аргументов. Текстовый путь /subscribe и /unsubscribe (с аргументами)
живёт в responses.py рядом с FSM-мастером в handlers.py. Без рантайма Telegram.
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

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_STUB_BEARER = "b-1"
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
async def test_subscribe_response_text_path_creates_subscription() -> None:
    respx.post(f"{_BASE}{_PREFIX}/bot/subscriptions").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": 7,
                "chat_id": 100,
                "kind": "price_level",
                "params": {"ticker": "SBER", "level": 250},
            },
        )
    )
    client = _make_client()
    try:
        text = await subscribe_response(client, chat_id=100, args="price_level SBER 250")
    finally:
        await client.aclose()

    assert "создана" in text


async def test_subscribe_response_parse_error_returns_hint() -> None:
    client = _make_client()
    try:
        text = await subscribe_response(client, chat_id=100, args="bogus_kind")
    finally:
        await client.aclose()

    assert "неизвестн" in text.lower()


@respx.mock
async def test_unsubscribe_response_text_path_deletes() -> None:
    respx.delete(f"{_BASE}{_PREFIX}/bot/subscriptions/5").mock(return_value=httpx.Response(204))
    client = _make_client()
    try:
        text = await unsubscribe_response(client, args="5")
    finally:
        await client.aclose()

    assert "удалена" in text


async def test_unsubscribe_response_non_numeric_returns_hint() -> None:
    client = _make_client()
    try:
        text = await unsubscribe_response(client, args="abc")
    finally:
        await client.aclose()

    assert "id" in text.lower()

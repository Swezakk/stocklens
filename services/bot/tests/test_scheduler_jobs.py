"""Tests for scheduler job bodies as plain async functions.

Tests call alert_sweep_job and digest_job directly (no APScheduler firing).
Uses respx mocks for API calls and a fake bot to assert send behaviour.
Uses injected clock for the digest's MSK-today decision.
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import httpx
import respx
from bot.api_client.client import ApiClient
from bot.scheduler import alert_sweep_job, digest_job, forecast_refresh_job

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_STUB_BEARER = "t-1"
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")


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


def _make_bot(send_ok: bool = True) -> Any:
    bot = AsyncMock()
    if send_ok:
        bot.send_message.return_value = MagicMock()
    else:
        bot.send_message.side_effect = RuntimeError("blocked")
    return bot


def _pending_alerts_json() -> list[dict[str, object]]:
    return [
        {
            "chat_id": 777,
            "kind": "price_level",
            "ticker": "SBER",
            "level": "250.00",
            "close": "251.50",
            "article_id": None,
            "article_title": None,
            "article_url": None,
            "article_published_at": None,
            "ex_date": None,
            "dividend_value": None,
            "dividend_currency": None,
        }
    ]


def _portfolio_summary_json() -> dict[str, object]:
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


def _index_page_json() -> dict[str, object]:
    return {
        "items": [
            {"trade_date": "2026-06-22", "close": "3200.50"},
            {"trade_date": "2026-06-21", "close": "3150.00"},
        ],
        "total": 2,
        "limit": 2,
        "offset": 0,
    }


def _news_page_empty() -> dict[str, object]:
    return {"items": [], "total": 0, "limit": 30, "offset": 0}


def _claim_response(claimed: bool) -> dict[str, object]:
    return {"claimed": claimed}


@respx.mock
async def test_alert_sweep_job_sends_pending_alerts() -> None:
    respx.post(f"{_BASE}{_PREFIX}/bot/alerts/pending").mock(
        return_value=httpx.Response(200, json=_pending_alerts_json())
    )
    bot = _make_bot(send_ok=True)
    client = _make_client()
    try:
        await alert_sweep_job(bot=bot, client=client)
    finally:
        await client.aclose()

    bot.send_message.assert_called_once()
    call_kwargs = bot.send_message.call_args.kwargs
    assert call_kwargs["chat_id"] == 777


@respx.mock
async def test_alert_sweep_job_handles_empty_pending_list() -> None:
    respx.post(f"{_BASE}{_PREFIX}/bot/alerts/pending").mock(
        return_value=httpx.Response(200, json=[])
    )
    bot = _make_bot()
    client = _make_client()
    try:
        await alert_sweep_job(bot=bot, client=client)
    finally:
        await client.aclose()

    bot.send_message.assert_not_called()


@respx.mock
async def test_alert_sweep_job_swallows_api_errors() -> None:
    """A network error must not propagate — job must not crash the scheduler."""
    respx.post(f"{_BASE}{_PREFIX}/bot/alerts/pending").mock(
        side_effect=httpx.ConnectError("refused")
    )
    bot = _make_bot()
    client = _make_client()
    try:
        await alert_sweep_job(bot=bot, client=client)
    finally:
        await client.aclose()
    # No exception raised — job handles failure gracefully


@respx.mock
async def test_digest_job_sends_digest_when_claim_succeeds() -> None:
    fixed_clock = datetime(2026, 6, 23, 8, 30, tzinfo=_MOSCOW_TZ)
    digest_chat_id = 999

    respx.post(f"{_BASE}{_PREFIX}/bot/digest/claim").mock(
        return_value=httpx.Response(200, json=_claim_response(True))
    )
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_portfolio_summary_json())
    )
    respx.get(f"{_BASE}{_PREFIX}/data/index").mock(
        return_value=httpx.Response(200, json=_index_page_json())
    )
    respx.get(f"{_BASE}{_PREFIX}/data/news").mock(
        return_value=httpx.Response(200, json=_news_page_empty())
    )

    bot = _make_bot()
    client = _make_client()

    try:
        await digest_job(
            bot=bot,
            client=client,
            digest_chat_id=digest_chat_id,
            clock=lambda: fixed_clock,
        )
    finally:
        await client.aclose()

    bot.send_message.assert_called_once()
    call_kwargs = bot.send_message.call_args.kwargs
    assert call_kwargs["chat_id"] == digest_chat_id


@respx.mock
async def test_digest_job_skips_send_when_claim_returns_false() -> None:
    """If claim returns False (already sent today), digest must NOT be sent again."""
    fixed_clock = datetime(2026, 6, 23, 8, 30, tzinfo=_MOSCOW_TZ)

    respx.post(f"{_BASE}{_PREFIX}/bot/digest/claim").mock(
        return_value=httpx.Response(200, json=_claim_response(False))
    )

    bot = _make_bot()
    client = _make_client()
    try:
        await digest_job(
            bot=bot,
            client=client,
            digest_chat_id=999,
            clock=lambda: fixed_clock,
        )
    finally:
        await client.aclose()

    bot.send_message.assert_not_called()


@respx.mock
async def test_digest_job_swallows_api_errors() -> None:
    """A network error during claim must not propagate."""
    fixed_clock = datetime(2026, 6, 23, 8, 30, tzinfo=_MOSCOW_TZ)

    respx.post(f"{_BASE}{_PREFIX}/bot/digest/claim").mock(side_effect=httpx.ConnectError("refused"))

    bot = _make_bot()
    client = _make_client()
    try:
        await digest_job(
            bot=bot,
            client=client,
            digest_chat_id=999,
            clock=lambda: fixed_clock,
        )
    finally:
        await client.aclose()
    # No exception raised


@respx.mock
async def test_forecast_refresh_job_triggers_endpoint() -> None:
    route = respx.post(f"{_BASE}{_PREFIX}/bot/forecasts/refresh").mock(
        return_value=httpx.Response(202, json={"accepted": True, "reason": None})
    )
    client = _make_client()
    try:
        await forecast_refresh_job(client=client)
    finally:
        await client.aclose()

    assert route.called


@respx.mock
async def test_forecast_refresh_job_swallows_api_errors() -> None:
    """A network error must not propagate — job must not crash the scheduler."""
    respx.post(f"{_BASE}{_PREFIX}/bot/forecasts/refresh").mock(
        side_effect=httpx.ConnectError("refused")
    )
    client = _make_client()
    try:
        await forecast_refresh_job(client=client)
    finally:
        await client.aclose()
    # No exception raised

"""Tests for IMOEX index gathering in digest (spec §357 requirement).

Verifies get_index client method via respx and the ImoexClose DTO propagation into DigestData.
"""

from datetime import date
from decimal import Decimal

import httpx
import respx
from bot.api_client.client import ApiClient
from bot.api_client.dto import IndexValue
from bot.digest import gather_digest

_BASE = "http://testapi"
_PREFIX = "/api/v1"
_STUB_BEARER = "b-1"
_TODAY = date(2026, 6, 23)


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


def _summary_json_empty() -> dict[str, object]:
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


def _index_page_json(items: list[dict[str, object]]) -> dict[str, object]:
    return {"items": items, "total": len(items), "limit": 2, "offset": 0}


def _news_page_empty() -> dict[str, object]:
    return {"items": [], "total": 0, "limit": 30, "offset": 0}


async def test_get_index_returns_index_value_list() -> None:
    client = _make_client()
    with respx.mock:
        respx.get(f"{_BASE}{_PREFIX}/data/index").mock(
            return_value=httpx.Response(
                200,
                json=_index_page_json(
                    [
                        {"trade_date": "2026-06-22", "close": "3200.50"},
                        {"trade_date": "2026-06-21", "close": "3150.00"},
                    ]
                ),
            )
        )
        try:
            values = await client.get_index(index_code="IMOEX", limit=2)
        finally:
            await client.aclose()

    assert len(values) == 2
    assert isinstance(values[0], IndexValue)
    assert values[0].trade_date == date(2026, 6, 22)
    assert values[0].close == Decimal("3200.50")


async def test_get_index_handles_empty_response() -> None:
    client = _make_client()
    with respx.mock:
        respx.get(f"{_BASE}{_PREFIX}/data/index").mock(
            return_value=httpx.Response(200, json=_index_page_json([]))
        )
        try:
            values = await client.get_index(index_code="IMOEX", limit=2)
        finally:
            await client.aclose()

    assert values == []


@respx.mock
async def test_gather_digest_includes_imoex_close() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_summary_json_empty())
    )
    respx.get(f"{_BASE}{_PREFIX}/data/index").mock(
        return_value=httpx.Response(
            200,
            json=_index_page_json(
                [
                    {"trade_date": "2026-06-22", "close": "3200.50"},
                    {"trade_date": "2026-06-21", "close": "3150.00"},
                ]
            ),
        )
    )
    respx.get(f"{_BASE}{_PREFIX}/data/news").mock(
        return_value=httpx.Response(200, json=_news_page_empty())
    )

    client = _make_client()
    try:
        data = await gather_digest(client, _TODAY)
    finally:
        await client.aclose()

    assert data.imoex_yesterday is not None
    assert data.imoex_yesterday.close == Decimal("3200.50")
    assert data.imoex_prior is not None
    assert data.imoex_prior.close == Decimal("3150.00")


@respx.mock
async def test_gather_digest_handles_missing_imoex_gracefully() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_summary_json_empty())
    )
    respx.get(f"{_BASE}{_PREFIX}/data/index").mock(
        return_value=httpx.Response(200, json=_index_page_json([]))
    )
    respx.get(f"{_BASE}{_PREFIX}/data/news").mock(
        return_value=httpx.Response(200, json=_news_page_empty())
    )

    client = _make_client()
    try:
        data = await gather_digest(client, _TODAY)
    finally:
        await client.aclose()

    assert data.imoex_yesterday is None
    assert data.imoex_prior is None

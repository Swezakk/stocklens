"""Тесты сбора дайджеста (DESIGN §11).

Чистые отборы (окно отсечек, пересечение тикеров с портфелем) — юнит; gather_digest —
через реальный API-клиент + respx (включая короткое замыкание на пустом портфеле).
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import respx
from bot.api_client.client import ApiClient
from bot.api_client.dto import DividendOut, NewsOut, SentimentOut
from bot.digest import gather_digest, select_portfolio_news, select_upcoming
from stocklens_core.enums import Currency, SentimentLabel

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


def _div(ex_date: date) -> DividendOut:
    return DividendOut(
        id=1, security_id=1, ex_date=ex_date, value=Decimal("33.30"), currency=Currency.RUB
    )


def _news_dto(published_at: datetime, tickers: list[str]) -> NewsOut:
    return NewsOut(
        id=1,
        source="rss",
        url="https://example.com/a",
        title="t",
        summary=None,
        published_at=published_at,
        sentiment=SentimentOut(label=SentimentLabel.NEGATIVE, score=0.9, model_version="v1"),
        tickers=tickers,
    )


def test_select_upcoming_filters_window_and_sorts() -> None:
    dividends = [
        _div(date(2026, 6, 20)),
        _div(date(2026, 6, 28)),
        _div(date(2026, 6, 25)),
        _div(date(2026, 8, 1)),
    ]
    result = select_upcoming(dividends, "SBER", _TODAY, 7)
    assert [item.ex_date for item in result] == [date(2026, 6, 25), date(2026, 6, 28)]
    assert all(item.ticker == "SBER" for item in result)


def test_select_portfolio_news_intersects_and_sorts_desc() -> None:
    news = [
        _news_dto(datetime(2026, 6, 18, 10, 0, tzinfo=UTC), ["GAZP"]),
        _news_dto(datetime(2026, 6, 20, 10, 0, tzinfo=UTC), ["SBER"]),
        _news_dto(datetime(2026, 6, 22, 10, 0, tzinfo=UTC), ["SBER", "LKOH"]),
    ]
    result = select_portfolio_news(news, frozenset({"SBER"}))
    assert [item.published_at for item in result] == [
        datetime(2026, 6, 22, 10, 0, tzinfo=UTC),
        datetime(2026, 6, 20, 10, 0, tzinfo=UTC),
    ]


def _position_json(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "quantity": 10,
        "avg_price": "250.00",
        "opened_at": "2026-01-10T12:00:00+00:00",
        "current_price": "265.00",
        "current_value": "2650.00",
        "unrealized_pnl": "150.00",
    }


def _summary_json(tickers: list[str]) -> dict[str, object]:
    return {
        "positions": [_position_json(ticker) for ticker in tickers],
        "total_value": "2650.00",
        "total_cost": "2500.00",
        "total_unrealized_pnl": "150.00",
        "portfolio_return_pct": 6.0,
        "imoex_return_pct": -2.0,
        "sharpe": 1.1,
        "max_drawdown": 0.12,
        "imoex_sharpe": 0.4,
        "imoex_max_drawdown": 0.2,
        "period_from": "2026-01-01",
        "period_to": "2026-06-23",
    }


def _dividend_page_json(ex_date: str) -> dict[str, object]:
    item = {"id": 1, "security_id": 1, "ex_date": ex_date, "value": "33.30", "currency": "RUB"}
    return {"items": [item], "total": 1, "limit": 100, "offset": 0}


def _news_page_json(tickers: list[str]) -> dict[str, object]:
    item = {
        "id": 1,
        "source": "rss",
        "url": "https://example.com/a",
        "title": "Заголовок",
        "summary": None,
        "published_at": datetime(2026, 6, 20, 15, 14, tzinfo=UTC).isoformat(),
        "sentiment": {"label": "negative", "score": 0.9, "model_version": "v1"},
        "tickers": tickers,
    }
    return {"items": [item], "total": 1, "limit": 30, "offset": 0}


@respx.mock
async def test_gather_digest_assembles_sections() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_summary_json(["SBER"]))
    )
    respx.get(f"{_BASE}{_PREFIX}/data/dividends").mock(
        return_value=httpx.Response(200, json=_dividend_page_json("2026-06-25"))
    )
    respx.get(f"{_BASE}{_PREFIX}/data/news").mock(
        return_value=httpx.Response(200, json=_news_page_json(["SBER"]))
    )
    client = _make_client()
    try:
        data = await gather_digest(client, _TODAY)
    finally:
        await client.aclose()

    assert [item.ticker for item in data.dividends] == ["SBER"]
    assert len(data.negative_news) == 1


@respx.mock
async def test_gather_digest_empty_portfolio_skips_dividends_and_news() -> None:
    respx.get(f"{_BASE}{_PREFIX}/portfolio/summary").mock(
        return_value=httpx.Response(200, json=_summary_json([]))
    )
    dividends_route = respx.get(f"{_BASE}{_PREFIX}/data/dividends").mock(
        return_value=httpx.Response(200, json=_dividend_page_json("2026-06-25"))
    )
    news_route = respx.get(f"{_BASE}{_PREFIX}/data/news").mock(
        return_value=httpx.Response(200, json=_news_page_json([]))
    )
    client = _make_client()
    try:
        data = await gather_digest(client, _TODAY)
    finally:
        await client.aclose()

    assert list(data.dividends) == []
    assert list(data.negative_news) == []
    assert not dividends_route.called
    assert not news_route.called

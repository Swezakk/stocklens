"""Integration-тесты data-эндпоинтов против реального PostgreSQL (testcontainers)."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.seed import (
    seed_candle,
    seed_dividend,
    seed_news_with_sentiment,
    seed_security,
)

pytestmark = pytest.mark.integration


async def test_list_securities_returns_empty_page_initially(client: AsyncClient) -> None:
    response = await client.get("/api/v1/data/securities")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "total" in body


async def test_list_securities_returns_seeded_security(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await seed_security(db_session, ticker="SBER_TEST_SEC")
    await db_session.commit()

    response = await client.get("/api/v1/data/securities")
    assert response.status_code == 200
    tickers = [s["ticker"] for s in response.json()["items"]]
    assert "SBER_TEST_SEC" in tickers


async def test_list_securities_pagination(client: AsyncClient, db_session: AsyncSession) -> None:
    for i in range(3):
        await seed_security(db_session, ticker=f"PAGE_TEST_{i}")
    await db_session.commit()

    response = await client.get("/api/v1/data/securities?limit=2&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) <= 2
    assert body["limit"] == 2


async def test_list_candles_returns_data_for_seeded_ticker(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="CANDLE_OK")
    await seed_candle(db_session, security_id=security.id)
    await db_session.commit()

    response = await client.get("/api/v1/data/candles?ticker=CANDLE_OK")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["items"][0]["security_id"] == security.id


async def test_list_candles_returns_404_for_unknown_ticker(client: AsyncClient) -> None:
    response = await client.get("/api/v1/data/candles?ticker=NONEXISTENT_XYZ")
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "https://stocklens.local/problems/security-not-found"
    assert "NONEXISTENT_XYZ" in body["detail"]


async def test_list_candles_problem_details_shape(client: AsyncClient) -> None:
    response = await client.get("/api/v1/data/candles?ticker=BADTICKER")
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert {"type", "title", "status", "detail", "instance"} <= body.keys()
    assert body["status"] == 404
    assert body["instance"] == "/api/v1/data/candles"


async def test_list_news_returns_seeded_article_with_sentiment(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="NEWS_TICKER")
    await seed_news_with_sentiment(db_session, security_id=security.id, ticker="NEWS_TICKER")
    await db_session.commit()

    response = await client.get("/api/v1/data/news")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    article = body["items"][0]
    assert "sentiment" in article
    assert "tickers" in article


async def test_list_news_filter_by_unknown_ticker_returns_empty(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/data/news?ticker=NOPE_TICKER")
    assert response.status_code == 200
    assert response.json()["total"] == 0


async def test_list_dividends_returns_seeded_dividend(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    security = await seed_security(db_session, ticker="DIV_TICKER")
    await seed_dividend(db_session, security_id=security.id)
    await db_session.commit()

    response = await client.get("/api/v1/data/dividends?ticker=DIV_TICKER")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["items"][0]["security_id"] == security.id


async def test_list_dividends_unknown_ticker_returns_empty(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/data/dividends?ticker=NODIV_TICKER")
    assert response.status_code == 200
    assert response.json()["total"] == 0

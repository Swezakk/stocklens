"""Интеграционные тесты маршрутов списка наблюдения.

testcontainers PostgreSQL + Redis, миграции применяются в pg_container fixture.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.seed import seed_candle, seed_security, seed_watchlist_item

pytestmark = pytest.mark.integration


async def test_post_watchlist_returns_201_and_pending_status(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /watchlist: новый тикер → 201, статус pending (бумага ещё не материализована)."""
    resp = await client.post("/api/v1/watchlist", json={"ticker": "gazp"})
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["ticker"] == "GAZP"
    assert data["status"] == "pending"
    assert data["has_data"] is False


async def test_post_watchlist_duplicate_returns_409(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /watchlist: дубль тикера → 409 Problem+JSON с RU-сообщением."""
    await seed_watchlist_item(db_session, "VTBR")
    await db_session.commit()

    resp = await client.post("/api/v1/watchlist", json={"ticker": "VTBR"})
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["status"] == 409
    assert "VTBR" in body["detail"]


async def test_get_watchlist_shows_ready_when_security_and_candle_exist(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /watchlist: бумага + свеча → статус ready, has_data=True."""
    sec = await seed_security(db_session, ticker="SBER_WL")
    await seed_candle(db_session, sec.id)
    await seed_watchlist_item(db_session, "SBER_WL")
    await db_session.commit()

    resp = await client.get("/api/v1/watchlist")
    assert resp.status_code == 200, resp.text
    items = resp.json()
    sber_items = [i for i in items if i["ticker"] == "SBER_WL"]
    assert len(sber_items) == 1
    assert sber_items[0]["status"] == "ready"
    assert sber_items[0]["has_data"] is True


async def test_delete_watchlist_item_returns_204(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE /watchlist/{ticker} → 204 при успехе."""
    await seed_watchlist_item(db_session, "LKOH_WL")
    await db_session.commit()

    resp = await client.delete("/api/v1/watchlist/LKOH_WL")
    assert resp.status_code == 204


async def test_delete_watchlist_item_twice_returns_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE /watchlist/{ticker} повторно → 404 Problem+JSON."""
    await seed_watchlist_item(db_session, "AFLT_WL")
    await db_session.commit()

    await client.delete("/api/v1/watchlist/AFLT_WL")
    resp2 = await client.delete("/api/v1/watchlist/AFLT_WL")
    assert resp2.status_code == 404
    body = resp2.json()
    assert "AFLT_WL" in body["detail"]


async def test_delete_watchlist_item_not_found_returns_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE /watchlist/{ticker} несуществующего → 404."""
    resp = await client.delete("/api/v1/watchlist/NONEXISTENT_WL")
    assert resp.status_code == 404

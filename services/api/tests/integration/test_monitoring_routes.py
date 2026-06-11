"""Integration-тесты monitoring-эндпоинтов."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import CollectorRunStatus

from tests.integration.seed import seed_collector_run

pytestmark = pytest.mark.integration


async def test_list_runs_returns_seeded_run(client: AsyncClient, db_session: AsyncSession) -> None:
    await seed_collector_run(db_session, source="moex_candles")
    await db_session.commit()

    response = await client.get("/api/v1/monitoring/runs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    sources = [r["source"] for r in body["items"]]
    assert "moex_candles" in sources


async def test_list_runs_filter_by_source(client: AsyncClient, db_session: AsyncSession) -> None:
    await seed_collector_run(db_session, source="rss_news")
    await db_session.commit()

    response = await client.get("/api/v1/monitoring/runs?source=rss_news")
    assert response.status_code == 200
    body = response.json()
    for run in body["items"]:
        assert run["source"] == "rss_news"


async def test_list_runs_filter_by_status(client: AsyncClient, db_session: AsyncSession) -> None:
    await seed_collector_run(db_session, source="cbr_rates", status=CollectorRunStatus.FAILED)
    await db_session.commit()

    response = await client.get(f"/api/v1/monitoring/runs?status={CollectorRunStatus.FAILED}")
    assert response.status_code == 200
    body = response.json()
    for run in body["items"]:
        assert run["status"] == CollectorRunStatus.FAILED


async def test_list_runs_ordered_by_started_at_desc(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await seed_collector_run(db_session, source="source_a")
    await seed_collector_run(db_session, source="source_b")
    await db_session.commit()

    response = await client.get("/api/v1/monitoring/runs?limit=200")
    assert response.status_code == 200
    items = response.json()["items"]
    if len(items) >= 2:
        started_ats = [item["started_at"] for item in items]
        assert started_ats == sorted(started_ats, reverse=True)

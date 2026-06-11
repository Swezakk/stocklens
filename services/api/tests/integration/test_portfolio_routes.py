"""Интеграционные тесты маршрутов портфеля и подписок.

testcontainers PostgreSQL + Redis, миграции применяются в pg_container fixture.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.seed import (
    seed_candles_range,
    seed_index_values_range,
    seed_key_rate,
    seed_portfolio_position,
    seed_security,
)

pytestmark = pytest.mark.integration


def _recent_weekdays(n: int, offset_days: int = 0) -> list[date]:
    """Вернуть n последних рабочих дней с опциональным смещением назад."""
    result: list[date] = []
    base = datetime.now(tz=UTC).date() - timedelta(days=offset_days)
    d = base
    while len(result) < n:
        if d.weekday() < 5:
            result.append(d)
        d -= timedelta(days=1)
    return sorted(result)


_SUMMARY_DATES = _recent_weekdays(14, offset_days=25)
_OPTIMIZE_DATES = _recent_weekdays(15, offset_days=60)
_OPTIMIZE_DATES_B = _recent_weekdays(15, offset_days=120)


async def test_upsert_position_then_list_returns_it(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /portfolio/positions → GET /portfolio/positions: позиция видна в списке."""
    await seed_security(db_session, ticker="SBER_T1")
    await db_session.commit()

    resp = await client.post(
        "/api/v1/portfolio/positions",
        json={
            "ticker": "SBER_T1",
            "quantity": 50,
            "avg_price": "280.00",
            "opened_at": "2024-01-01T00:00:00+00:00",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ticker"] == "SBER_T1"
    assert data["quantity"] == 50

    list_resp = await client.get("/api/v1/portfolio/positions")
    assert list_resp.status_code == 200
    tickers = [p["ticker"] for p in list_resp.json()]
    assert "SBER_T1" in tickers


async def test_upsert_position_twice_same_ticker_updates_row(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST дважды один тикер → upsert: одна строка, обновлённые данные."""
    await seed_security(db_session, ticker="LKOH_T1")
    await db_session.commit()

    payload = {
        "ticker": "LKOH_T1",
        "quantity": 10,
        "avg_price": "7000.00",
        "opened_at": "2024-01-01T00:00:00+00:00",
    }
    resp1 = await client.post("/api/v1/portfolio/positions", json=payload)
    assert resp1.status_code == 200

    payload2 = {**payload, "quantity": 20, "avg_price": "7200.00"}
    resp2 = await client.post("/api/v1/portfolio/positions", json=payload2)
    assert resp2.status_code == 200
    assert resp2.json()["quantity"] == 20

    list_resp = await client.get("/api/v1/portfolio/positions")
    lkoh_positions = [p for p in list_resp.json() if p["ticker"] == "LKOH_T1"]
    assert len(lkoh_positions) == 1
    assert lkoh_positions[0]["avg_price"] == "7200.000000"


async def test_delete_position_returns_204(client: AsyncClient, db_session: AsyncSession) -> None:
    """DELETE /portfolio/positions/{ticker} → 204 при успехе."""
    sec = await seed_security(db_session, ticker="SBERP_T1")
    await seed_portfolio_position(db_session, security_id=sec.id, quantity=10)
    await db_session.commit()

    resp = await client.delete("/api/v1/portfolio/positions/SBERP_T1")
    assert resp.status_code == 204

    resp2 = await client.delete("/api/v1/portfolio/positions/SBERP_T1")
    assert resp2.status_code == 404


async def test_delete_position_returns_404_for_unknown_ticker(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """DELETE неизвестный тикер → 404 Problem+JSON с русским сообщением."""
    resp = await client.delete("/api/v1/portfolio/positions/UNKNOWN_XYZ")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert "UNKNOWN_XYZ" in body["detail"]


async def test_upsert_unknown_ticker_returns_404_problem_json(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST с неизвестным тикером → 404 Problem+JSON с русским сообщением."""
    resp = await client.post(
        "/api/v1/portfolio/positions",
        json={
            "ticker": "NO_SUCH_TICKER",
            "quantity": 1,
            "avg_price": "100.00",
            "opened_at": "2024-01-01T00:00:00+00:00",
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == 404
    assert "NO_SUCH_TICKER" in body["detail"]


async def test_portfolio_summary_returns_finite_metrics(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /portfolio/summary: возвращает Шарп и MDD для портфеля с историей."""
    sec = await seed_security(db_session, ticker="GAZP_T1")
    await seed_portfolio_position(db_session, security_id=sec.id, quantity=100)
    await seed_candles_range(db_session, sec.id, _SUMMARY_DATES)
    await seed_index_values_range(db_session, _SUMMARY_DATES)
    await seed_key_rate(db_session, rate_date=_SUMMARY_DATES[0])
    await db_session.commit()

    resp = await client.get("/api/v1/portfolio/summary", params={"period_days": 60})
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert isinstance(data["sharpe"], float)
    assert isinstance(data["max_drawdown"], float)
    assert isinstance(data["imoex_sharpe"], float)
    assert data["max_drawdown"] <= 0.0
    assert data["period_from"] is not None


async def test_portfolio_summary_period_too_short_does_not_crash(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /portfolio/summary с минимальным period_days не возвращает 500.

    БД содержит данные из предыдущих тестов: некоторые попадают в период, другие нет.
    Позиции без данных в периоде исключаются из расчёта (не ломают пересечение дат).
    """
    resp = await client.get(
        "/api/v1/portfolio/summary",
        params={"period_days": 2},
    )
    assert resp.status_code in (200, 422)


async def test_portfolio_optimize_default_strategy_returns_new_shape(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /portfolio/optimize (strategy=max_sharpe по умолчанию): новая форма ответа."""
    sec1 = await seed_security(db_session, ticker="OPT_A")
    sec2 = await seed_security(db_session, ticker="OPT_B")
    await seed_candles_range(db_session, sec1.id, _OPTIMIZE_DATES, Decimal("100.00"))
    await seed_candles_range(db_session, sec2.id, _OPTIMIZE_DATES, Decimal("200.00"))
    await seed_index_values_range(db_session, _OPTIMIZE_DATES)
    await seed_key_rate(db_session, rate_date=_OPTIMIZE_DATES[0])
    await db_session.commit()

    resp = await client.post(
        "/api/v1/portfolio/optimize",
        json={"tickers": ["OPT_A", "OPT_B"], "period_days": 120},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["strategy"] == "max_sharpe"
    weights_sum = sum(data["weights"].values())
    assert abs(weights_sum - 1.0) < 0.01, f"Сумма весов = {weights_sum}"
    assert isinstance(data["expected_return"], float)
    assert isinstance(data["volatility"], float)
    assert isinstance(data["sharpe"], float)
    assert isinstance(data["frontier"], list)
    assert "equal_weight_sharpe" in data
    assert "imoex_sharpe" in data


async def test_portfolio_optimize_min_volatility_strategy(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /portfolio/optimize strategy=min_volatility: веса суммируются ≈ 1."""
    sec1 = await seed_security(db_session, ticker="OPT_C")
    sec2 = await seed_security(db_session, ticker="OPT_D")
    await seed_candles_range(db_session, sec1.id, _OPTIMIZE_DATES_B, Decimal("150.00"))
    await seed_candles_range(db_session, sec2.id, _OPTIMIZE_DATES_B, Decimal("250.00"))
    await seed_index_values_range(db_session, _OPTIMIZE_DATES_B)
    await seed_key_rate(db_session, rate_date=_OPTIMIZE_DATES_B[0])
    await db_session.commit()

    resp = await client.post(
        "/api/v1/portfolio/optimize",
        json={"tickers": ["OPT_C", "OPT_D"], "period_days": 180, "strategy": "min_volatility"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["strategy"] == "min_volatility"
    weights_sum = sum(data["weights"].values())
    assert abs(weights_sum - 1.0) < 0.01, f"Сумма весов min_vol = {weights_sum}"


async def test_portfolio_optimize_single_ticker_returns_422(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /portfolio/optimize с одним тикером → 422."""
    resp = await client.post(
        "/api/v1/portfolio/optimize",
        json={"tickers": ["SBER"], "period_days": 365},
    )
    assert resp.status_code == 422


async def test_bot_create_list_delete_subscription(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST → GET → DELETE полный цикл подписки."""
    resp_create = await client.post(
        "/api/v1/bot/subscriptions",
        json={"chat_id": 999, "kind": "sentiment_spike", "params": {}},
    )
    assert resp_create.status_code == 201
    sub_id = resp_create.json()["id"]

    resp_list = await client.get("/api/v1/bot/subscriptions", params={"chat_id": 999})
    assert resp_list.status_code == 200
    ids = [s["id"] for s in resp_list.json()]
    assert sub_id in ids

    resp_del = await client.delete(f"/api/v1/bot/subscriptions/{sub_id}")
    assert resp_del.status_code == 204

    resp_del2 = await client.delete(f"/api/v1/bot/subscriptions/{sub_id}")
    assert resp_del2.status_code == 404


async def test_bot_create_price_level_without_level_returns_422(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST price_level без параметра 'level' → 422 Problem+JSON с русским сообщением."""
    resp = await client.post(
        "/api/v1/bot/subscriptions",
        json={"chat_id": 111, "kind": "price_level", "params": {}},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "level" in body["detail"]

"""Интеграционные тесты эндпоинтов рыночных данных и бэктеста.

Покрывает: /data/index, /data/currency-rates, /data/key-rate, /data/movers,
/portfolio/backtest.
"""

from datetime import date
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import Currency
from stocklens_core.models.portfolio import PortfolioPosition

from tests.integration.seed import (
    seed_candles_range,
    seed_currency_rate,
    seed_index_value,
    seed_index_values_range,
    seed_key_rate,
    seed_portfolio_position,
    seed_security,
)

pytestmark = pytest.mark.integration

_DATES_5 = [date(2024, 3, d) for d in range(1, 6)]


class TestListIndex:
    async def test_list_index_happy_path(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + корректный Page[IndexValueOut] для существующих записей IMOEX."""
        await seed_index_value(db_session, trade_date=date(2024, 2, 15), close=Decimal("3200.00"))
        await seed_index_value(db_session, trade_date=date(2024, 2, 14), close=Decimal("3190.00"))
        await db_session.commit()

        resp = await client.get("/api/v1/data/index", params={"index_code": "IMOEX"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 2
        assert "items" in body
        assert all(k in body["items"][0] for k in ("trade_date", "close"))

    async def test_list_index_empty_returns_empty_page(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + пустой Page если записей нет."""
        resp = await client.get(
            "/api/v1/data/index",
            params={"index_code": "NONEXISTENT_IDX"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    async def test_list_index_date_filter(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Фильтр date_from/date_to корректно сужает выборку."""
        await seed_index_values_range(db_session, dates=_DATES_5, index_code="IMOEX2")
        await db_session.commit()

        resp = await client.get(
            "/api/v1/data/index",
            params={
                "index_code": "IMOEX2",
                "date_from": "2024-03-02",
                "date_to": "2024-03-03",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2

    async def test_list_index_invalid_pagination(self, client: AsyncClient) -> None:
        """422 при limit=0 (невалидная пагинация)."""
        resp = await client.get("/api/v1/data/index", params={"limit": 0})
        assert resp.status_code == 422


class TestListCurrencyRates:
    async def test_list_currency_rates_happy_path(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + корректный Page[CurrencyRateOut]."""
        await seed_currency_rate(db_session, currency=Currency.USD, rate_date=date(2024, 4, 1))
        await seed_currency_rate(
            db_session, currency=Currency.EUR, rate_date=date(2024, 4, 1), rate=Decimal("97.00")
        )
        await db_session.commit()

        resp = await client.get("/api/v1/data/currency-rates")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 2
        assert all(k in body["items"][0] for k in ("currency", "rate_date", "rate"))

    async def test_list_currency_rates_filter_by_currency(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Фильтр currency=USD возвращает только USD."""
        await seed_currency_rate(
            db_session, currency=Currency.USD, rate_date=date(2024, 5, 1)
        )
        await seed_currency_rate(
            db_session, currency=Currency.EUR, rate_date=date(2024, 5, 1), rate=Decimal("97.00")
        )
        await db_session.commit()

        resp = await client.get("/api/v1/data/currency-rates", params={"currency": "USD"})

        assert resp.status_code == 200
        body = resp.json()
        assert all(item["currency"] == "USD" for item in body["items"])

    async def test_list_currency_rates_invalid_currency(self, client: AsyncClient) -> None:
        """422 при невалидном значении currency."""
        resp = await client.get("/api/v1/data/currency-rates", params={"currency": "XYZ"})
        assert resp.status_code == 422

    async def test_list_currency_rates_empty(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + пустой список если нет данных за будущую дату."""
        resp = await client.get(
            "/api/v1/data/currency-rates",
            params={"date_from": "2099-01-01"},
        )

        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestListKeyRate:
    async def test_list_key_rate_happy_path(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + корректный Page[KeyRateOut]."""
        await seed_key_rate(db_session, rate_date=date(2024, 6, 1), rate=Decimal("16.00"))
        await seed_key_rate(db_session, rate_date=date(2024, 7, 1), rate=Decimal("15.00"))
        await db_session.commit()

        resp = await client.get("/api/v1/data/key-rate")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 2
        assert all(k in body["items"][0] for k in ("rate_date", "rate"))

    async def test_list_key_rate_date_filter(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Фильтр date_from возвращает только записи не ранее даты."""
        await seed_key_rate(db_session, rate_date=date(2025, 1, 1), rate=Decimal("21.00"))
        await seed_key_rate(db_session, rate_date=date(2025, 2, 1), rate=Decimal("21.00"))
        await db_session.commit()

        resp = await client.get("/api/v1/data/key-rate", params={"date_from": "2025-02-01"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["rate"] == "21.00"

    async def test_list_key_rate_empty(self, client: AsyncClient) -> None:
        """200 + пустой список если нет данных за будущую дату."""
        resp = await client.get("/api/v1/data/key-rate", params={"date_from": "2099-01-01"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_list_key_rate_invalid_pagination(self, client: AsyncClient) -> None:
        """422 при невалидной пагинации."""
        resp = await client.get("/api/v1/data/key-rate", params={"offset": -1})
        assert resp.status_code == 422


class TestGetMovers:
    async def test_get_movers_happy_path(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + структура MoversOut с gainers/losers."""
        sec = await seed_security(db_session, ticker="MVRS_SBER")
        await seed_candles_range(
            db_session,
            security_id=sec.id,
            dates=_DATES_5,
            base_price=Decimal("280.00"),
        )
        await db_session.commit()

        resp = await client.get("/api/v1/data/movers", params={"limit": 5})

        assert resp.status_code == 200
        body = resp.json()
        assert "gainers" in body
        assert "losers" in body

    async def test_get_movers_no_securities_returns_empty(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + пустые gainers/losers если нет бумаг с >=2 регулярными свечами."""
        resp = await client.get("/api/v1/data/movers")

        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["gainers"], list)
        assert isinstance(body["losers"], list)

    async def test_get_movers_invalid_limit_zero(self, client: AsyncClient) -> None:
        """422 при limit=0 (не входит в диапазон 1–50)."""
        resp = await client.get("/api/v1/data/movers", params={"limit": 0})
        assert resp.status_code == 422

    async def test_get_movers_invalid_limit_too_large(self, client: AsyncClient) -> None:
        """422 при limit=51 (превышает максимум 50)."""
        resp = await client.get("/api/v1/data/movers", params={"limit": 51})
        assert resp.status_code == 422


class TestPortfolioBacktest:
    async def test_backtest_empty_portfolio_returns_422(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """422 при пустом портфеле (тест сам гарантирует пустоту — не зависит от порядка)."""
        await db_session.execute(delete(PortfolioPosition))
        await db_session.commit()

        resp = await client.get("/api/v1/portfolio/backtest", params={"months_back": 3})

        assert resp.status_code == 422
        body = resp.json()
        assert "портфель" in body["detail"].lower() or "позиц" in body["detail"].lower()

    async def test_backtest_happy_path(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """200 + корректная структура BacktestResultOut при наличии позиций и котировок."""
        sec = await seed_security(db_session, ticker="BKTS_SBER")
        await seed_portfolio_position(db_session, security_id=sec.id)
        await seed_candles_range(
            db_session,
            security_id=sec.id,
            dates=_DATES_5,
            base_price=Decimal("280.00"),
        )
        await seed_index_values_range(db_session, dates=_DATES_5, index_code="IMOEX")
        await seed_key_rate(db_session, rate_date=date(2024, 8, 1))
        await db_session.commit()

        resp = await client.get("/api/v1/portfolio/backtest", params={"months_back": 120})

        assert resp.status_code == 200
        body = resp.json()
        assert body["months_back"] == 120
        assert "period_from" in body
        assert "period_to" in body
        assert "portfolio_return_pct" in body
        assert "imoex_return_pct" in body
        assert "equity_curve" in body
        assert len(body["equity_curve"]) >= 2

    async def test_backtest_invalid_months_back_zero(self, client: AsyncClient) -> None:
        """422 при months_back=0."""
        resp = await client.get("/api/v1/portfolio/backtest", params={"months_back": 0})
        assert resp.status_code == 422

    async def test_backtest_invalid_months_back_too_large(self, client: AsyncClient) -> None:
        """422 при months_back=121 (превышает максимум 120)."""
        resp = await client.get("/api/v1/portfolio/backtest", params={"months_back": 121})
        assert resp.status_code == 422

"""Реализация MarketDataRepository на AsyncSession (SQLAlchemy 2.0).

Читает индексы, курсы валют, ключевые ставки и вычисляет муверы дня.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import Currency
from stocklens_core.models.market import Candle, CurrencyRate, IndexValue, KeyRate, Security

from api.schemas.market import CurrencyRateOut, IndexValueOut, KeyRateOut, MoverOut

_MIN_CANDLES_FOR_MOVER = 2


class SqlMarketDataRepository:
    """Читает справочные рыночные данные из PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def index_series_page(
        self,
        index_code: str,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[IndexValueOut], int]:
        """Вернуть страницу значений индекса (desc by trade_date) и общее число записей."""
        filters = [IndexValue.index_code == index_code]
        if date_from is not None:
            filters.append(IndexValue.trade_date >= date_from)
        if date_to is not None:
            filters.append(IndexValue.trade_date <= date_to)

        total_result = await self._session.execute(
            select(func.count()).select_from(IndexValue).where(*filters)
        )
        total: int = total_result.scalar_one()

        rows = await self._session.execute(
            select(IndexValue.trade_date, IndexValue.close)
            .where(*filters)
            .order_by(IndexValue.trade_date.desc())
            .limit(limit)
            .offset(offset)
        )
        items = [IndexValueOut(trade_date=row.trade_date, close=row.close) for row in rows]
        return items, total

    async def currency_rates_page(
        self,
        currency: Currency | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CurrencyRateOut], int]:
        """Вернуть страницу курсов валют (desc by rate_date) и общее число записей."""
        filters = []
        if currency is not None:
            filters.append(CurrencyRate.currency == currency)
        if date_from is not None:
            filters.append(CurrencyRate.rate_date >= date_from)
        if date_to is not None:
            filters.append(CurrencyRate.rate_date <= date_to)

        count_query = select(func.count()).select_from(CurrencyRate)
        if filters:
            count_query = count_query.where(*filters)
        total_result = await self._session.execute(count_query)
        total = total_result.scalar_one()

        data_query = (
            select(CurrencyRate.currency, CurrencyRate.rate_date, CurrencyRate.rate)
            .order_by(CurrencyRate.rate_date.desc())
        )
        if filters:
            data_query = data_query.where(*filters)
        rows = await self._session.execute(data_query.limit(limit).offset(offset))
        items = [
            CurrencyRateOut(currency=row.currency, rate_date=row.rate_date, rate=row.rate)
            for row in rows
        ]
        return items, total

    async def key_rates_page(
        self,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[KeyRateOut], int]:
        """Вернуть страницу ключевых ставок (desc by rate_date) и общее число записей."""
        filters = []
        if date_from is not None:
            filters.append(KeyRate.rate_date >= date_from)
        if date_to is not None:
            filters.append(KeyRate.rate_date <= date_to)

        count_query = select(func.count()).select_from(KeyRate)
        if filters:
            count_query = count_query.where(*filters)
        total_result = await self._session.execute(count_query)
        total = total_result.scalar_one()

        data_query = select(KeyRate.rate_date, KeyRate.rate).order_by(KeyRate.rate_date.desc())
        if filters:
            data_query = data_query.where(*filters)
        rows = await self._session.execute(data_query.limit(limit).offset(offset))
        items = [KeyRateOut(rate_date=row.rate_date, rate=row.rate) for row in rows]
        return items, total

    async def active_securities_latest_closes(
        self,
        limit_per_security: int = 2,
    ) -> list[MoverOut]:
        """Вернуть MoverOut для каждой активной бумаги с >=2 регулярными свечами.

        Использует оконную функцию ROW_NUMBER() для получения
        последних двух свечей на бумагу без N+1 запросов.
        """
        row_num = (
            func.row_number()
            .over(
                partition_by=Candle.security_id,
                order_by=Candle.trade_date.desc(),
            )
            .label("rn")
        )
        subq = (
            select(
                Candle.security_id,
                Candle.trade_date,
                Candle.close,
                row_num,
            )
            .join(Security, Security.id == Candle.security_id)
            .where(
                Security.is_active.is_(True),
                Candle.is_weekend_session.is_(False),
            )
            .subquery()
        )

        rows = await self._session.execute(
            select(
                subq.c.security_id,
                subq.c.trade_date,
                subq.c.close,
                subq.c.rn,
                Security.ticker,
                Security.name,
            )
            .join(Security, Security.id == subq.c.security_id)
            .where(subq.c.rn <= limit_per_security)
            .order_by(subq.c.security_id, subq.c.rn)
        )

        by_security: dict[int, dict[int, tuple[Decimal, str, str]]] = {}
        for row in rows:
            security_id = row.security_id
            if security_id not in by_security:
                by_security[security_id] = {}
            by_security[security_id][row.rn] = (row.close, row.ticker, row.name)

        result: list[MoverOut] = []
        for candles in by_security.values():
            if len(candles) < _MIN_CANDLES_FOR_MOVER:
                continue
            latest_close, ticker, name = candles[1]
            prev_close, _, _ = candles[2]
            if prev_close == Decimal("0"):
                continue
            change_pct = float((latest_close - prev_close) / prev_close * 100)
            result.append(
                MoverOut(
                    ticker=ticker,
                    name=name,
                    close=latest_close,
                    prev_close=prev_close,
                    change_pct=change_pct,
                )
            )
        return result

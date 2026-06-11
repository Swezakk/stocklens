"""Реализация MarketHistoryRepository на AsyncSession (SQLAlchemy 2.0).

Читает исторические данные для аналитики: свечи, дивиденды, IMOEX, ключевую ставку.
Все запросы по свечам исключают сессии выходного дня.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Candle, Dividend, IndexValue, KeyRate

_IMOEX_CODE = "IMOEX"


class SqlMarketHistoryRepository:
    """Читает исторические рыночные данные из PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def close_series(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> list[tuple[date, Decimal]]:
        """Вернуть ряд цен закрытия без выходных сессий, сортировка по дате."""
        result = await self._session.execute(
            select(Candle.trade_date, Candle.close)
            .where(
                Candle.security_id == security_id,
                Candle.trade_date >= date_from,
                Candle.trade_date <= date_to,
                Candle.is_weekend_session.is_(False),
            )
            .order_by(Candle.trade_date)
        )
        return [(row.trade_date, row.close) for row in result]

    async def dividends_map(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> dict[date, Decimal]:
        """Вернуть словарь {ex_date: дивиденд} в заданном диапазоне дат."""
        result = await self._session.execute(
            select(Dividend.ex_date, Dividend.value).where(
                Dividend.security_id == security_id,
                Dividend.ex_date >= date_from,
                Dividend.ex_date <= date_to,
            )
        )
        return {row.ex_date: row.value for row in result}

    async def imoex_series(
        self,
        date_from: date,
        date_to: date,
    ) -> list[tuple[date, Decimal]]:
        """Вернуть ряд значений IMOEX (дата, close), сортировка по дате."""
        result = await self._session.execute(
            select(IndexValue.trade_date, IndexValue.close)
            .where(
                IndexValue.index_code == _IMOEX_CODE,
                IndexValue.trade_date >= date_from,
                IndexValue.trade_date <= date_to,
            )
            .order_by(IndexValue.trade_date)
        )
        return [(row.trade_date, row.close) for row in result]

    async def latest_key_rate(self) -> Decimal | None:
        """Вернуть последнее значение ключевой ставки ЦБ РФ (в процентах)."""
        result = await self._session.execute(
            select(KeyRate.rate).order_by(KeyRate.rate_date.desc()).limit(1)
        )
        return result.scalar_one_or_none()

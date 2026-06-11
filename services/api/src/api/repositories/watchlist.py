"""Реализация WatchlistRepository на AsyncSession (SQLAlchemy 2.0)."""

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Candle, Security
from stocklens_core.models.portfolio import Watchlist


class SqlWatchlistRepository:
    """Читает и записывает список наблюдения через PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_items(self) -> list[Watchlist]:
        """Вернуть все элементы вотчлиста, сортировка по added_at asc."""
        result = await self._session.execute(select(Watchlist).order_by(Watchlist.added_at))
        return list(result.scalars().all())

    async def get_by_ticker(self, ticker: str) -> Watchlist | None:
        """Найти элемент по тикеру."""
        result = await self._session.execute(select(Watchlist).where(Watchlist.ticker == ticker))
        return result.scalar_one_or_none()

    async def add(self, ticker: str) -> Watchlist:
        """Добавить тикер в вотчлист. Коммитит транзакцию.

        При нарушении unique constraint (дубль тикера) выбрасывает
        sqlalchemy.exc.IntegrityError — сервисный слой маппирует его в
        WatchlistItemExistsError перед возвратом HTTP 409.
        """
        stmt = pg_insert(Watchlist).values(ticker=ticker).returning(Watchlist)
        result = await self._session.execute(stmt)
        item = result.scalar_one()
        await self._session.commit()
        return item

    async def delete(self, ticker: str) -> bool:
        """Удалить тикер из вотчлиста. Возвращает True если строка существовала."""
        item = await self.get_by_ticker(ticker)
        if item is None:
            return False
        await self._session.delete(item)
        await self._session.commit()
        return True

    async def security_exists(self, ticker: str) -> bool:
        """Вернуть True если бумага с тикером существует в таблице securities."""
        count = await self._session.scalar(
            select(func.count()).select_from(Security).where(Security.ticker == ticker)
        )
        return (count or 0) > 0

    async def has_candles(self, ticker: str) -> bool:
        """Вернуть True если у бумаги есть хотя бы одна свеча."""
        row = await self._session.execute(
            select(Candle.id)
            .join(Security, Security.id == Candle.security_id)
            .where(Security.ticker == ticker)
            .limit(1)
        )
        return row.scalar_one_or_none() is not None

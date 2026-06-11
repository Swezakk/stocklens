"""Реализация PortfolioRepository на AsyncSession (SQLAlchemy 2.0)."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.portfolio import PortfolioPosition


class SqlPortfolioRepository:
    """Читает и записывает позиции портфеля через PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_positions(self) -> list[PortfolioPosition]:
        """Вернуть все позиции портфеля, сортировка по id."""
        result = await self._session.execute(
            select(PortfolioPosition).order_by(PortfolioPosition.id)
        )
        return list(result.scalars().all())

    async def get_position(self, security_id: int) -> PortfolioPosition | None:
        """Найти позицию по security_id."""
        result = await self._session.execute(
            select(PortfolioPosition).where(PortfolioPosition.security_id == security_id)
        )
        return result.scalar_one_or_none()

    async def upsert_position(
        self,
        security_id: int,
        quantity: int,
        avg_price: Decimal,
        opened_at: datetime,
    ) -> PortfolioPosition:
        """Создать или обновить позицию по security_id. Коммитит транзакцию."""
        stmt = (
            pg_insert(PortfolioPosition)
            .values(
                security_id=security_id,
                quantity=quantity,
                avg_price=avg_price,
                opened_at=opened_at,
            )
            .on_conflict_do_update(
                index_elements=["security_id"],
                set_={
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "opened_at": opened_at,
                },
            )
            .returning(PortfolioPosition)
        )
        result = await self._session.execute(stmt)
        position = result.scalar_one()
        await self._session.commit()
        return position

    async def delete_position(self, security_id: int) -> bool:
        """Удалить позицию по security_id. Возвращает True если строка существовала."""
        position = await self.get_position(security_id)
        if position is None:
            return False
        await self._session.delete(position)
        await self._session.commit()
        return True

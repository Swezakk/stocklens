"""Реализация SecurityRepository на AsyncSession (SQLAlchemy 2.0)."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Security


class SqlSecurityRepository:
    """Читает ценные бумаги из PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_securities(
        self,
        is_active: bool | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Security], int]:
        """Вернуть страницу ценных бумаг и общее число записей."""
        base_query = select(Security)
        if is_active is not None:
            base_query = base_query.where(Security.is_active == is_active)

        count_result = await self._session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total: int = count_result.scalar_one()

        rows_result = await self._session.execute(
            base_query.order_by(Security.ticker).limit(limit).offset(offset)
        )
        securities = list(rows_result.scalars().all())
        return securities, total

    async def get_by_ticker(self, ticker: str) -> Security | None:
        """Найти бумагу по тикеру."""
        result = await self._session.execute(select(Security).where(Security.ticker == ticker))
        return result.scalar_one_or_none()

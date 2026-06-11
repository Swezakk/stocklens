"""Реализация DividendRepository на AsyncSession (SQLAlchemy 2.0)."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.models.market import Dividend


class SqlDividendRepository:
    """Читает дивидендные выплаты из PostgreSQL."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_dividends(
        self,
        security_id: int | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Dividend], int]:
        """Вернуть страницу дивидендов и общее число записей."""
        base_query = select(Dividend)
        if security_id is not None:
            base_query = base_query.where(Dividend.security_id == security_id)

        count_result = await self._session.execute(
            select(func.count()).select_from(base_query.subquery())
        )
        total: int = count_result.scalar_one()

        rows_result = await self._session.execute(
            base_query.order_by(Dividend.ex_date.desc()).limit(limit).offset(offset)
        )
        dividends = list(rows_result.scalars().all())
        return dividends, total

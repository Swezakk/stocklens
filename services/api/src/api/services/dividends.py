"""Сервис чтения дивидендных выплат."""

from api.repositories.protocols import DividendRepository, SecurityRepository
from api.schemas.common import Page
from api.schemas.market import DividendOut


class DividendService:
    """Оркестрирует SecurityRepository и DividendRepository, маппит ORM → DTO."""

    def __init__(
        self,
        security_repo: SecurityRepository,
        dividend_repo: DividendRepository,
    ) -> None:
        self._security_repo = security_repo
        self._dividend_repo = dividend_repo

    async def list_dividends(
        self,
        ticker: str | None,
        limit: int,
        offset: int,
    ) -> Page[DividendOut]:
        """Вернуть страницу дивидендов. Неизвестный тикер — пустая страница (фильтр)."""
        security_id: int | None = None
        if ticker is not None:
            security = await self._security_repo.get_by_ticker(ticker)
            if security is None:
                return Page(items=[], total=0, limit=limit, offset=offset)
            security_id = security.id

        dividends, total = await self._dividend_repo.list_dividends(
            security_id=security_id,
            limit=limit,
            offset=offset,
        )
        items = [DividendOut.model_validate(d) for d in dividends]
        return Page(items=items, total=total, limit=limit, offset=offset)

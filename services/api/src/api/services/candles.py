"""Сервис чтения свечных данных."""

from datetime import date

from api.core.exceptions import SecurityNotFoundError
from api.repositories.protocols import CandleRepository, SecurityRepository
from api.schemas.common import Page
from api.schemas.market import CandleOut


class CandleService:
    """Проверяет существование тикера и делегирует чтение свечей репозиторию.

    Маппинг ORM → DTO выполняет CandleRepository (там же кэш) — сервис лишь
    разрешает тикер в security_id и собирает страницу.
    """

    def __init__(
        self,
        security_repo: SecurityRepository,
        candle_repo: CandleRepository,
    ) -> None:
        self._security_repo = security_repo
        self._candle_repo = candle_repo

    async def list_candles(
        self,
        ticker: str,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> Page[CandleOut]:
        """Вернуть страницу свечей для тикера.

        Raises:
            SecurityNotFoundError: если тикер не найден в БД.
        """
        security = await self._security_repo.get_by_ticker(ticker)
        if security is None:
            raise SecurityNotFoundError(ticker)

        items, total = await self._candle_repo.list_candles(
            security_id=security.id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, total=total, limit=limit, offset=offset)

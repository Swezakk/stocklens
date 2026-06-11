"""Сервис чтения ценных бумаг."""

from api.repositories.protocols import SecurityRepository
from api.schemas.common import Page
from api.schemas.market import SecurityOut


class SecurityService:
    """Оркестрирует SecurityRepository, маппит ORM → DTO."""

    def __init__(self, repo: SecurityRepository) -> None:
        self._repo = repo

    async def list_securities(
        self,
        is_active: bool | None,
        limit: int,
        offset: int,
    ) -> Page[SecurityOut]:
        """Вернуть страницу ценных бумаг."""
        securities, total = await self._repo.list_securities(is_active, limit, offset)
        items = [SecurityOut.model_validate(s) for s in securities]
        return Page(items=items, total=total, limit=limit, offset=offset)

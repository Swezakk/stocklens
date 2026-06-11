"""Сервис рыночных справочных данных: индексы, курсы валют, ключевая ставка, муверы."""

from datetime import date

from stocklens_core.enums import Currency

from api.repositories.protocols import MarketDataRepository
from api.schemas.common import Page
from api.schemas.market import CurrencyRateOut, IndexValueOut, KeyRateOut, MoverOut, MoversOut


class MarketDataService:
    """Читает справочные рыночные данные и вычисляет ранжирование муверов."""

    def __init__(self, repo: MarketDataRepository) -> None:
        self._repo = repo

    async def list_index(
        self,
        index_code: str,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> Page[IndexValueOut]:
        """Вернуть страницу значений индекса, отсортированных по дате убывания."""
        items, total = await self._repo.index_series_page(
            index_code=index_code,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, total=total, limit=limit, offset=offset)

    async def list_currency_rates(
        self,
        currency: Currency | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> Page[CurrencyRateOut]:
        """Вернуть страницу курсов валют, отсортированных по дате убывания."""
        items, total = await self._repo.currency_rates_page(
            currency=currency,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, total=total, limit=limit, offset=offset)

    async def list_key_rate(
        self,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> Page[KeyRateOut]:
        """Вернуть страницу ключевых ставок ЦБ РФ, отсортированных по дате убывания."""
        items, total = await self._repo.key_rates_page(
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return Page(items=items, total=total, limit=limit, offset=offset)

    async def get_movers(self, limit: int) -> MoversOut:
        """Вернуть топ-N лидеров роста и падения по изменению цены закрытия.

        Бумаги с менее чем 2 свечами пропускаются репозиторием.
        Разделение gainers/losers: change_pct >= 0 → gainers, < 0 → losers.
        """
        all_movers: list[MoverOut] = await self._repo.active_securities_latest_closes()

        gainers = sorted(
            (m for m in all_movers if m.change_pct >= 0),
            key=lambda m: m.change_pct,
            reverse=True,
        )[:limit]

        losers = sorted(
            (m for m in all_movers if m.change_pct < 0),
            key=lambda m: m.change_pct,
        )[:limit]

        return MoversOut(gainers=gainers, losers=losers)

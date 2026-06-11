"""Protocol-интерфейсы репозиториев.

Сервисный слой зависит от этих Protocol, а не от конкретных реализаций —
unit-тесты подменяют реализации фиктивными объектами без подъёма БД.
"""

from datetime import date
from typing import Protocol

from stocklens_core.enums import CollectorRunStatus, SentimentLabel
from stocklens_core.models.market import Dividend, Security
from stocklens_core.models.news import NewsArticle, NewsSentiment
from stocklens_core.models.operations import CollectorRun

from api.schemas.market import CandleOut


class SecurityRepository(Protocol):
    """Чтение ценных бумаг."""

    async def list_securities(
        self,
        is_active: bool | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Security], int]:
        """Вернуть страницу ценных бумаг и общее число записей."""
        ...

    async def get_by_ticker(self, ticker: str) -> Security | None:
        """Найти бумагу по тикеру."""
        ...


class CandleRepository(Protocol):
    """Чтение свечных данных с кэшированием.

    Возвращает готовые DTO, а не ORM: кэш живёт в этом слое (спека §9.7),
    а сериализуемы в Redis именно DTO, не ORM-объекты.
    """

    async def list_candles(
        self,
        security_id: int,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CandleOut], int]:
        """Вернуть страницу свечей (DTO) и общее число записей."""
        ...


class NewsRepository(Protocol):
    """Чтение новостей с тональностью и тикерами."""

    async def list_news(
        self,
        security_id: int | None,
        sentiment: SentimentLabel | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[tuple[NewsArticle, NewsSentiment | None, list[str]]], int]:
        """Вернуть страницу новостей (статья, тональность, список тикеров) и общее число."""
        ...


class DividendRepository(Protocol):
    """Чтение дивидендных выплат."""

    async def list_dividends(
        self,
        security_id: int | None,
        limit: int,
        offset: int,
    ) -> tuple[list[Dividend], int]:
        """Вернуть страницу дивидендов и общее число записей."""
        ...


class MonitoringRepository(Protocol):
    """Чтение журнала запусков сборщиков."""

    async def list_runs(
        self,
        source: str | None,
        status: CollectorRunStatus | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CollectorRun], int]:
        """Вернуть страницу запусков сборщиков (сортировка: started_at desc) и общее число."""
        ...

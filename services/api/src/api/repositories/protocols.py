"""Protocol-интерфейсы репозиториев.

Сервисный слой зависит от этих Protocol, а не от конкретных реализаций —
unit-тесты подменяют реализации фиктивными объектами без подъёма БД.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

from stocklens_core.enums import AlertKind, CollectorRunStatus, Currency, SentimentLabel
from stocklens_core.models.market import Dividend, Security
from stocklens_core.models.news import NewsArticle, NewsSentiment
from stocklens_core.models.operations import CollectorRun
from stocklens_core.models.portfolio import BotSubscription, PortfolioPosition, Watchlist

from api.schemas.market import CandleOut, CurrencyRateOut, IndexValueOut, KeyRateOut, MoverOut


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


class PortfolioRepository(Protocol):
    """Чтение и запись позиций портфеля. Каждая бумага — одна агрегированная позиция."""

    async def list_positions(self) -> list[PortfolioPosition]:
        """Вернуть все позиции портфеля."""
        ...

    async def get_position(self, security_id: int) -> PortfolioPosition | None:
        """Найти позицию по security_id."""
        ...

    async def upsert_position(
        self,
        security_id: int,
        quantity: int,
        avg_price: Decimal,
        opened_at: datetime,
    ) -> PortfolioPosition:
        """Создать или обновить позицию. Коммитит транзакцию."""
        ...

    async def delete_position(self, security_id: int) -> bool:
        """Удалить позицию. Возвращает True если строка существовала. Коммитит транзакцию."""
        ...


class MarketHistoryRepository(Protocol):
    """Чтение исторических рыночных данных для аналитики.

    Все методы исключают сессии выходного дня (is_weekend_session=True).
    """

    async def close_series(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> list[tuple[date, Decimal]]:
        """Вернуть ряд цен закрытия (дата, цена), отсортированный по дате.

        Исключает записи с is_weekend_session=True.
        """
        ...

    async def dividends_map(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> dict[date, Decimal]:
        """Вернуть словарь {ex_date: дивиденд} в заданном диапазоне дат."""
        ...

    async def imoex_series(
        self,
        date_from: date,
        date_to: date,
    ) -> list[tuple[date, Decimal]]:
        """Вернуть ряд значений индекса IMOEX (дата, close), сортировка по дате."""
        ...

    async def latest_key_rate(self) -> Decimal | None:
        """Вернуть последнее значение ключевой ставки ЦБ РФ (в процентах, напр. 16.00)."""
        ...


class MarketDataRepository(Protocol):
    """Чтение рыночных справочных данных: индексы, курсы валют, ключевая ставка, муверы."""

    async def index_series_page(
        self,
        index_code: str,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[IndexValueOut], int]:
        """Вернуть страницу значений индекса (desc by trade_date) и общее число записей."""
        ...

    async def currency_rates_page(
        self,
        currency: Currency | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CurrencyRateOut], int]:
        """Вернуть страницу курсов валют (desc by rate_date) и общее число записей."""
        ...

    async def key_rates_page(
        self,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[KeyRateOut], int]:
        """Вернуть страницу ключевых ставок (desc by rate_date) и общее число записей."""
        ...

    async def active_securities_latest_closes(
        self,
        limit_per_security: int = 2,
    ) -> list[MoverOut]:
        """Вернуть последние 2 свечи для каждой активной бумаги (только регулярные сессии).

        Возвращает MoverOut с заполненными ticker, name, close, prev_close, change_pct.
        Бумаги с менее чем 2 свечами пропускаются на уровне БД.
        """
        ...


class BotSubscriptionRepository(Protocol):
    """Чтение и запись Telegram-подписок на алерты."""

    async def list_by_chat(self, chat_id: int) -> list[BotSubscription]:
        """Вернуть все подписки для указанного chat_id."""
        ...

    async def create(
        self,
        chat_id: int,
        kind: AlertKind,
        params: dict[str, object],
    ) -> BotSubscription:
        """Создать подписку. Коммитит транзакцию."""
        ...

    async def delete(self, sub_id: int) -> bool:
        """Удалить подписку по id. Возвращает True если строка существовала. Коммитит."""
        ...


class WatchlistRepository(Protocol):
    """Чтение и запись списка наблюдения. API — единственный write-путь (спека §4)."""

    async def list_items(self) -> list[Watchlist]:
        """Вернуть все элементы вотчлиста, сортировка по added_at asc."""
        ...

    async def get_by_ticker(self, ticker: str) -> Watchlist | None:
        """Найти элемент по тикеру."""
        ...

    async def add(self, ticker: str) -> Watchlist:
        """Добавить тикер. Коммитит. При дубле — unique violation (caller ловит)."""
        ...

    async def delete(self, ticker: str) -> bool:
        """Удалить тикер. Возвращает True если строка существовала. Коммитит."""
        ...

    async def security_exists(self, ticker: str) -> bool:
        """Вернуть True если бумага с тикером существует в securities."""
        ...

    async def has_candles(self, ticker: str) -> bool:
        """Вернуть True если у бумаги есть хотя бы одна свеча."""
        ...

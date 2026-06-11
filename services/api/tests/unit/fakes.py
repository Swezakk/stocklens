"""Типизированные фейки ORM-объектов для unit-тестов.

Dataclass-замены позволяют тестировать сервисный слой без БД.
Pydantic model_validate(from_attributes=True) работает с любым объектом с атрибутами.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel


@dataclass
class FakeSecurity:
    """Подмена Security для unit-тестов."""

    id: int
    ticker: str
    name: str
    board: str
    aliases: list[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class FakeCandle:
    """Подмена Candle для unit-тестов."""

    id: int
    security_id: int
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    value: Decimal
    is_weekend_session: bool = False


@dataclass
class FakeDividend:
    """Подмена Dividend для unit-тестов."""

    id: int
    security_id: int
    ex_date: date
    value: Decimal
    currency: Currency = Currency.RUB


@dataclass
class FakeCollectorRun:
    """Подмена CollectorRun для unit-тестов."""

    id: int
    source: str
    started_at: date
    finished_at: date | None
    status: CollectorRunStatus
    records_added: int = 0
    error_message: str | None = None


@dataclass
class FakeNewsArticle:
    """Подмена NewsArticle для unit-тестов."""

    id: int
    source: str
    url: str
    title: str
    published_at: date
    summary: str | None = None


@dataclass
class FakeNewsSentiment:
    """Подмена NewsSentiment для unit-тестов."""

    id: int
    article_id: int
    label: SentimentLabel
    score: float
    model_version: str

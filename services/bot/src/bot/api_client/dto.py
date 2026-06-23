"""Pydantic-DTO, зеркалящие JSON-ответы StockLens API, которые потребляет бот (DESIGN §11).

Mirror-DTO на HTTP-границе — клиентский слой, не второй источник истины (инвариант №4):
доменные enum (AlertKind, SentimentLabel, Currency) импортируются из stocklens_core, не
переопределяются. Денежные значения — Decimal без понижения точности; даты — date,
метки времени — datetime.
"""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field
from stocklens_core.enums import AlertKind, Currency, SentimentLabel

__all__ = [
    "DigestClaim",
    "DividendOut",
    "DividendPage",
    "IndexPage",
    "IndexValue",
    "NewsOut",
    "NewsPage",
    "Page",
    "PendingAlert",
    "PortfolioSummaryOut",
    "PositionOut",
    "SentimentOut",
    "SubscriptionIn",
    "SubscriptionOut",
]


class Page[T](BaseModel):
    """Конверт постраничного ответа списковых эндпоинтов (items/total/limit/offset)."""

    items: list[T]
    total: int
    limit: int
    offset: int


class SubscriptionIn(BaseModel):
    """Тело POST /bot/subscriptions: подписка чата на алерт (зеркало api SubscriptionIn).

    Для PRICE_LEVEL ``params`` обязан содержать ключ 'level' (число) — правило валидируется
    на стороне API (единственный источник истины), бот лишь передаёт параметры.
    """

    chat_id: int
    kind: AlertKind
    params: dict[str, object] = Field(default_factory=dict)


class SubscriptionOut(BaseModel):
    """Подписка на алерт (ответ /bot/subscriptions)."""

    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object]


class PositionOut(BaseModel):
    """Позиция портфеля с текущей рыночной оценкой."""

    ticker: str
    quantity: int
    avg_price: Decimal
    opened_at: datetime
    current_price: Decimal | None
    current_value: Decimal | None
    unrealized_pnl: Decimal | None


class PortfolioSummaryOut(BaseModel):
    """Сводка по портфелю с риск-метриками и сравнением с IMOEX."""

    positions: list[PositionOut]
    total_value: Decimal
    total_cost: Decimal
    total_unrealized_pnl: Decimal
    portfolio_return_pct: float
    imoex_return_pct: float
    sharpe: float
    max_drawdown: float
    imoex_sharpe: float
    imoex_max_drawdown: float
    period_from: date
    period_to: date


class DividendOut(BaseModel):
    """Дивидендная выплата (ключ — security_id; тикер бот знает из параметра запроса)."""

    id: int
    security_id: int
    ex_date: date
    value: Decimal
    currency: Currency


class SentimentOut(BaseModel):
    """Тональность новостной статьи."""

    label: SentimentLabel
    score: float
    model_version: str


class NewsOut(BaseModel):
    """Новостная статья с тональностью и связанными тикерами."""

    id: int
    source: str
    url: str
    title: str
    summary: str | None
    published_at: datetime
    sentiment: SentimentOut | None
    tickers: list[str]


class DividendPage(Page[DividendOut]):
    """Страница дивидендных выплат."""


class NewsPage(Page[NewsOut]):
    """Страница новостей."""


class PendingAlert(BaseModel):
    """Сработавший алерт, готовый к отправке ботом (зеркало PendingAlertOut API).

    Поля по видам алертов (остальные None):
    - price_level:        level, close
    - sentiment_spike:    article_id, article_title, article_url, article_published_at
    - dividend_upcoming:  ex_date, dividend_value, dividend_currency
    """

    chat_id: int
    kind: AlertKind
    ticker: str

    level: Decimal | None = None
    close: Decimal | None = None

    article_id: int | None = None
    article_title: str | None = None
    article_url: str | None = None
    article_published_at: datetime | None = None

    ex_date: date | None = None
    dividend_value: Decimal | None = None
    dividend_currency: Currency | None = None


class DigestClaim(BaseModel):
    """Результат резервирования дайджеста (зеркало DigestClaimOut API)."""

    claimed: bool


class IndexValue(BaseModel):
    """Значение биржевого индекса за торговый день (зеркало IndexValueOut API)."""

    trade_date: date
    close: Decimal


class IndexPage(Page[IndexValue]):
    """Страница значений биржевого индекса."""

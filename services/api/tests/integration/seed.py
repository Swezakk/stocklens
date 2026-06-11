"""Вспомогательные функции для засева тестовых данных в PostgreSQL."""

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import AlertKind, CollectorRunStatus, Currency, SentimentLabel
from stocklens_core.models.market import Candle, Dividend, IndexValue, KeyRate, Security
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker
from stocklens_core.models.operations import CollectorRun
from stocklens_core.models.portfolio import BotSubscription, PortfolioPosition


async def seed_security(
    session: AsyncSession,
    ticker: str = "SBER",
    name: str = "Сбербанк",
    is_active: bool = True,
) -> Security:
    s = Security(ticker=ticker, name=name, board="TQBR", aliases=[], is_active=is_active)
    session.add(s)
    await session.flush()
    return s


async def seed_candle(
    session: AsyncSession,
    security_id: int,
    trade_date: date = date(2024, 1, 15),
) -> Candle:
    c = Candle(
        security_id=security_id,
        trade_date=trade_date,
        open=Decimal("280.00"),
        high=Decimal("285.00"),
        low=Decimal("278.00"),
        close=Decimal("283.00"),
        volume=1_000_000,
        value=Decimal("283000000.00"),
        is_weekend_session=False,
    )
    session.add(c)
    await session.flush()
    return c


async def seed_dividend(
    session: AsyncSession,
    security_id: int,
    ex_date: date = date(2024, 7, 1),
) -> Dividend:
    d = Dividend(
        security_id=security_id,
        ex_date=ex_date,
        value=Decimal("33.00"),
        currency=Currency.RUB,
    )
    session.add(d)
    await session.flush()
    return d


async def seed_collector_run(
    session: AsyncSession,
    source: str = "moex_candles",
    status: CollectorRunStatus = CollectorRunStatus.SUCCESS,
) -> CollectorRun:
    run = CollectorRun(
        source=source,
        started_at=datetime(2024, 1, 15, 9, 0, tzinfo=UTC),
        finished_at=datetime(2024, 1, 15, 9, 5, tzinfo=UTC),
        status=status,
        records_added=42,
    )
    session.add(run)
    await session.flush()
    return run


async def seed_candles_range(
    session: AsyncSession,
    security_id: int,
    dates: list[date],
    base_price: Decimal = Decimal("280.00"),
) -> list[Candle]:
    """Засеять несколько свечей для ценового ряда."""
    candles = []
    price = base_price
    for i, trade_date in enumerate(dates):
        price = price + Decimal(str(i % 5 - 2))  # небольшая вариация для ненулевого std
        c = Candle(
            security_id=security_id,
            trade_date=trade_date,
            open=price - Decimal("1.00"),
            high=price + Decimal("2.00"),
            low=price - Decimal("2.00"),
            close=price,
            volume=1_000_000,
            value=price * Decimal("1000000"),
            is_weekend_session=False,
        )
        session.add(c)
        candles.append(c)
    await session.flush()
    return candles


async def seed_index_value(
    session: AsyncSession,
    trade_date: date = date(2024, 1, 15),
    close: Decimal = Decimal("3200.00"),
    index_code: str = "IMOEX",
) -> IndexValue:
    """Засеять значение биржевого индекса."""
    iv = IndexValue(index_code=index_code, trade_date=trade_date, close=close)
    session.add(iv)
    await session.flush()
    return iv


async def seed_index_values_range(
    session: AsyncSession,
    dates: list[date],
    base_close: Decimal = Decimal("3200.00"),
    index_code: str = "IMOEX",
) -> list[IndexValue]:
    """Засеять ряд значений индекса."""
    values = []
    close = base_close
    for i, trade_date in enumerate(dates):
        close = close + Decimal(str(i % 3 - 1))
        iv = IndexValue(index_code=index_code, trade_date=trade_date, close=close)
        session.add(iv)
        values.append(iv)
    await session.flush()
    return values


async def seed_key_rate(
    session: AsyncSession,
    rate_date: date = date(2024, 1, 1),
    rate: Decimal = Decimal("16.00"),
) -> KeyRate:
    """Засеять ключевую ставку ЦБ РФ."""
    kr = KeyRate(rate_date=rate_date, rate=rate)
    session.add(kr)
    await session.flush()
    return kr


async def seed_portfolio_position(
    session: AsyncSession,
    security_id: int,
    quantity: int = 100,
    avg_price: Decimal = Decimal("280.00"),
    opened_at: datetime | None = None,
) -> PortfolioPosition:
    """Засеять позицию портфеля."""
    if opened_at is None:
        opened_at = datetime(2024, 1, 1, tzinfo=UTC)
    pos = PortfolioPosition(
        security_id=security_id,
        quantity=quantity,
        avg_price=avg_price,
        opened_at=opened_at,
    )
    session.add(pos)
    await session.flush()
    return pos


async def seed_bot_subscription(
    session: AsyncSession,
    chat_id: int = 12345,
    kind: AlertKind = AlertKind.SENTIMENT_SPIKE,
    params: dict[str, object] | None = None,
) -> BotSubscription:
    """Засеять Telegram-подписку."""
    sub = BotSubscription(
        chat_id=chat_id,
        kind=kind,
        params=params or {},
    )
    session.add(sub)
    await session.flush()
    return sub


async def seed_news_with_sentiment(
    session: AsyncSession,
    security_id: int | None = None,
    ticker: str | None = None,
    label: SentimentLabel = SentimentLabel.POSITIVE,
) -> tuple[NewsArticle, NewsSentiment]:
    article = NewsArticle(
        source="rbc",
        url=f"https://rbc.ru/news/{label}-test",
        title=f"Тестовая новость ({label})",
        published_at=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
    )
    session.add(article)
    await session.flush()

    sentiment = NewsSentiment(
        article_id=article.id,
        label=label,
        score=0.95,
        model_version="rubert-tiny2-v1",
    )
    session.add(sentiment)

    if security_id is not None:
        news_ticker = NewsTicker(article_id=article.id, security_id=security_id)
        session.add(news_ticker)

    await session.flush()
    return article, sentiment

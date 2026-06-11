"""Вспомогательные функции для засева тестовых данных в PostgreSQL."""

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel
from stocklens_core.models.market import Candle, Dividend, Security
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker
from stocklens_core.models.operations import CollectorRun


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

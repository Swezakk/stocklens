"""Реализация NewsRepository — собирает join статья + тональность + тикеры."""

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import SentimentLabel
from stocklens_core.models.market import Security
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker


class SqlNewsRepository:
    """Читает новости из PostgreSQL с left-join на sentiment и агрегацией тикеров."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_news(
        self,
        security_id: int | None,
        sentiment: SentimentLabel | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[tuple[NewsArticle, NewsSentiment | None, list[str]]], int]:
        """Вернуть страницу новостей и общее число.

        Каждый элемент — кортеж (статья, тональность | None, список тикеров).
        """
        base_query = select(NewsArticle).outerjoin(
            NewsSentiment, NewsSentiment.article_id == NewsArticle.id
        )

        if security_id is not None:
            base_query = base_query.join(NewsTicker, NewsTicker.article_id == NewsArticle.id).where(
                NewsTicker.security_id == security_id
            )

        if sentiment is not None:
            base_query = base_query.where(NewsSentiment.label == sentiment)

        if date_from is not None:
            base_query = base_query.where(NewsArticle.published_at >= date_from)
        if date_to is not None:
            base_query = base_query.where(NewsArticle.published_at <= date_to)

        count_result = await self._session.execute(
            select(func.count()).select_from(
                base_query.with_only_columns(NewsArticle.id).distinct().subquery()
            )
        )
        total: int = count_result.scalar_one()

        # PostgreSQL требует ORDER BY поля присутствовали в SELECT DISTINCT — включаем published_at.
        article_ids_result = await self._session.execute(
            base_query.with_only_columns(NewsArticle.id, NewsArticle.published_at)
            .distinct()
            .order_by(NewsArticle.published_at.desc())
            .limit(limit)
            .offset(offset)
        )
        article_ids = [row[0] for row in article_ids_result.all()]

        if not article_ids:
            return [], total

        articles_result = await self._session.execute(
            select(NewsArticle, NewsSentiment)
            .outerjoin(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(NewsArticle.id.in_(article_ids))
            .order_by(NewsArticle.published_at.desc())
        )
        article_sentiment_pairs: list[tuple[NewsArticle, NewsSentiment | None]] = [
            (row[0], row[1]) for row in articles_result.all()
        ]

        tickers_result = await self._session.execute(
            select(NewsTicker.article_id, Security.ticker)
            .join(Security, Security.id == NewsTicker.security_id)
            .where(NewsTicker.article_id.in_(article_ids))
        )
        tickers_by_article: dict[int, list[str]] = {}
        for article_id, ticker in tickers_result.all():
            tickers_by_article.setdefault(article_id, []).append(ticker)

        result: list[tuple[NewsArticle, NewsSentiment | None, list[str]]] = [
            (article, sent, tickers_by_article.get(article.id, []))
            for article, sent in article_sentiment_pairs
        ]
        return result, total

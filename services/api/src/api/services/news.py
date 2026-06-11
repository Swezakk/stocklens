"""Сервис чтения новостей."""

from datetime import date

from stocklens_core.enums import SentimentLabel
from stocklens_core.models.news import NewsArticle, NewsSentiment

from api.repositories.protocols import NewsRepository, SecurityRepository
from api.schemas.common import Page
from api.schemas.news import NewsOut, SentimentOut


class NewsService:
    """Оркестрирует SecurityRepository и NewsRepository, маппит ORM → DTO."""

    def __init__(
        self,
        security_repo: SecurityRepository,
        news_repo: NewsRepository,
    ) -> None:
        self._security_repo = security_repo
        self._news_repo = news_repo

    async def list_news(
        self,
        ticker: str | None,
        sentiment: SentimentLabel | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> Page[NewsOut]:
        """Вернуть страницу новостей. Неизвестный тикер — пустая страница (фильтр)."""
        security_id: int | None = None
        if ticker is not None:
            security = await self._security_repo.get_by_ticker(ticker)
            if security is None:
                return Page(items=[], total=0, limit=limit, offset=offset)
            security_id = security.id

        rows, total = await self._news_repo.list_news(
            security_id=security_id,
            sentiment=sentiment,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        items = [self._to_dto(article, sent, tickers) for article, sent, tickers in rows]
        return Page(items=items, total=total, limit=limit, offset=offset)

    @staticmethod
    def _to_dto(
        article: NewsArticle,
        sent: NewsSentiment | None,
        tickers: list[str],
    ) -> NewsOut:
        """Смаппить кортеж (статья, тональность, тикеры) в NewsOut."""
        sentiment_out: SentimentOut | None = None
        if sent is not None:
            sentiment_out = SentimentOut(
                label=sent.label,
                score=sent.score,
                model_version=sent.model_version,
            )

        return NewsOut(
            id=article.id,
            source=article.source,
            url=article.url,
            title=article.title,
            summary=article.summary,
            published_at=article.published_at,
            sentiment=sentiment_out,
            tickers=tickers,
        )

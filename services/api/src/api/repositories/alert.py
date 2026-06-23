"""SQL-реализации репозиториев для оценки алертов.

Каждый репозиторий реализует узкий Protocol из services/alert_evaluation.py
для конкретного вида алерта. Такое разделение сохраняет SOLID принцип единственной
ответственности и позволяет подменять каждую реализацию независимо.
"""

from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stocklens_core.enums import SentimentLabel
from stocklens_core.models.market import Candle, Dividend
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker

_ONE_DAY = timedelta(days=1)


class SqlCloseRepository:
    """Последние два дневных закрытия для оценки price_level алертов."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def latest_two_closes(self, security_id: int) -> list[Decimal]:
        """Вернуть последние два close регулярной сессии, порядок [prev, last].

        Исключает is_weekend_session=True. Берём 2 последних записи с сортировкой
        по убыванию, затем разворачиваем → [старший, младший].
        """
        result = await self._session.execute(
            select(Candle.close)
            .where(
                Candle.security_id == security_id,
                Candle.is_weekend_session.is_(False),
            )
            .order_by(Candle.trade_date.desc())
            .limit(2)
        )
        rows = [row[0] for row in result.all()]
        return list(reversed(rows))


class SqlNewsAlertRepository:
    """Негативные новости за сегодня для оценки sentiment_spike алертов."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_news_for_alert(
        self,
        security_id: int,
        today_date: date,
    ) -> list[tuple[NewsArticle, NewsSentiment | None]]:
        """Вернуть статьи с sentiment=NEGATIVE опубликованные в today_date для тикера."""
        result = await self._session.execute(
            select(NewsArticle, NewsSentiment)
            .join(NewsTicker, NewsTicker.article_id == NewsArticle.id)
            .outerjoin(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(
                NewsTicker.security_id == security_id,
                NewsSentiment.label == SentimentLabel.NEGATIVE,
                NewsArticle.published_at >= today_date,
                NewsArticle.published_at < today_date + _ONE_DAY,
            )
            .order_by(NewsArticle.published_at.desc())
        )
        return [(article, sentiment) for article, sentiment in result.all()]


class SqlDividendAlertRepository:
    """Дивиденды в диапазоне дат для оценки dividend_upcoming алертов."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_upcoming(
        self,
        security_id: int,
        date_from: date,
        date_to: date,
    ) -> list[Dividend]:
        """Вернуть дивиденды с ex_date в диапазоне [date_from, date_to]."""
        result = await self._session.execute(
            select(Dividend)
            .where(
                Dividend.security_id == security_id,
                Dividend.ex_date >= date_from,
                Dividend.ex_date <= date_to,
            )
            .order_by(Dividend.ex_date)
        )
        return list(result.scalars().all())

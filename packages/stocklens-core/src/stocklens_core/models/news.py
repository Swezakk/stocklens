"""ORM-модели новостных данных: статьи, тональность, привязка к тикерам."""

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from stocklens_core.enums import SentimentLabel
from stocklens_core.models.base import Base, str_enum_type


class NewsArticle(Base):
    """Новостная статья из RSS-фида."""

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    source: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    url: Mapped[str] = mapped_column(sa.String(1024), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class NewsSentiment(Base):
    """Результат NLP-классификации тональности новостной статьи."""

    __tablename__ = "news_sentiment"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    label: Mapped[SentimentLabel] = mapped_column(str_enum_type(SentimentLabel), nullable=False)
    score: Mapped[float] = mapped_column(sa.Float, nullable=False)
    model_version: Mapped[str] = mapped_column(sa.String(64), nullable=False)


class NewsTicker(Base):
    """Связь новостной статьи с упомянутым в ней инструментом (M:N без суррогатного PK)."""

    __tablename__ = "news_tickers"

    article_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    security_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("securities.id", ondelete="CASCADE"),
        primary_key=True,
    )

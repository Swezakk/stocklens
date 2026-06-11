"""DTO для новостных данных."""

from datetime import datetime

from pydantic import BaseModel
from stocklens_core.enums import SentimentLabel


class SentimentOut(BaseModel):
    """DTO тональности статьи."""

    label: SentimentLabel
    score: float
    model_version: str


class NewsOut(BaseModel):
    """DTO новостной статьи с тональностью и связанными тикерами.

    Собирается вручную в сервисном слое (join sentiment + tickers),
    поэтому from_attributes не используется.
    """

    id: int
    source: str
    url: str
    title: str
    summary: str | None
    published_at: datetime
    sentiment: SentimentOut | None
    tickers: list[str]

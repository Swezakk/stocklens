"""DTO для управления Telegram-подписками, оценки алертов и ежедневного дайджеста."""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field
from stocklens_core.enums import AlertKind, Currency

#: lead_days по умолчанию для dividend_upcoming — единый дефолт для валидации и оценки.
DIVIDEND_LEAD_DAYS_DEFAULT = 3


class SubscriptionIn(BaseModel):
    """Входные данные для создания подписки."""

    chat_id: int = Field(description="Telegram chat_id пользователя")
    kind: AlertKind = Field(description="Тип алерта")
    params: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Параметры алерта. Обязательные ключи по типам: "
            "price_level — 'ticker' (str) и 'level' (float > 0); "
            "sentiment_spike — 'ticker' (str); "
            "dividend_upcoming — 'ticker' (str) и опционально 'lead_days' (int 1..30, default 3)."
        ),
    )


class SubscriptionOut(BaseModel):
    """Выходные данные подписки."""

    model_config = {"from_attributes": True}

    id: int
    chat_id: int
    kind: AlertKind
    params: dict[str, object]


class PendingAlertOut(BaseModel):
    """DTO одного сработавшего алерта, готового к отправке ботом.

    Содержит структурированные данные по виду алерта — бот форматирует сообщение сам.
    Поля по видам алертов (остальные None):
    - price_level:        level, close (последний close)
    - sentiment_spike:    article_id, article_title, article_url, article_published_at
    - dividend_upcoming:  ex_date, dividend_value, dividend_currency
    """

    chat_id: int = Field(description="Telegram chat_id получателя")
    kind: AlertKind = Field(description="Вид алерта")
    ticker: str = Field(description="Тикер ценной бумаги")

    level: Decimal | None = Field(default=None, description="price_level: целевой уровень цены")
    close: Decimal | None = Field(default=None, description="price_level: последний close")

    article_id: int | None = Field(default=None, description="sentiment_spike: id статьи")
    article_title: str | None = Field(default=None, description="sentiment_spike: заголовок")
    article_url: str | None = Field(default=None, description="sentiment_spike: ссылка")
    article_published_at: datetime | None = Field(
        default=None, description="sentiment_spike: дата публикации"
    )

    ex_date: date | None = Field(default=None, description="dividend_upcoming: дата отсечки")
    dividend_value: Decimal | None = Field(
        default=None, description="dividend_upcoming: размер дивиденда"
    )
    dividend_currency: Currency | None = Field(
        default=None, description="dividend_upcoming: валюта дивиденда"
    )


class DigestClaimOut(BaseModel):
    """Результат резервирования ежедневного дайджеста."""

    claimed: bool = Field(
        description="True — дайджест зарезервирован этим вызовом; False — уже был отправлен сегодня"
    )

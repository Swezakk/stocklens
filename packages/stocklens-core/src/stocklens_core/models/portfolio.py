"""ORM-модели портфеля пользователя и Telegram-подписок."""

from datetime import datetime
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from stocklens_core.enums import AlertKind
from stocklens_core.models.base import Base, str_enum_type


class PortfolioPosition(Base):
    """Позиция в портфеле владельца сервиса."""

    __tablename__ = "portfolio_positions"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    security_id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.ForeignKey("securities.id"), nullable=False, unique=True
    )
    quantity: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    avg_price: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)


class BotSubscription(Base):
    """Подписка Telegram-пользователя на тип алертов."""

    __tablename__ = "bot_subscriptions"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    chat_id: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    kind: Mapped[AlertKind] = mapped_column(str_enum_type(AlertKind), nullable=False)
    params: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    )

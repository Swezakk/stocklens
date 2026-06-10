"""ORM-модели рыночных данных: ценные бумаги, свечи, дивиденды, индексы, курсы валют."""

from datetime import date
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from stocklens_core.enums import Currency
from stocklens_core.models.base import Base, str_enum_type


class Security(Base):
    """Инструмент Московской биржи."""

    __tablename__ = "securities"

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    ticker: Mapped[str] = mapped_column(sa.String(16), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    board: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true")
    )


class Candle(Base):
    """Дневная свеча OHLCV по инструменту."""

    __tablename__ = "candles"
    __table_args__ = (sa.UniqueConstraint("security_id", "trade_date"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    security_id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.ForeignKey("securities.id"), nullable=False
    )
    trade_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    volume: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    value: Mapped[Decimal] = mapped_column(sa.Numeric(20, 2), nullable=False)
    is_weekend_session: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false")
    )


class Dividend(Base):
    """Дивидендная выплата по инструменту."""

    __tablename__ = "dividends"
    __table_args__ = (sa.UniqueConstraint("security_id", "ex_date"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    security_id: Mapped[int] = mapped_column(
        sa.BigInteger, sa.ForeignKey("securities.id"), nullable=False
    )
    ex_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    value: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)
    currency: Mapped[Currency] = mapped_column(str_enum_type(Currency), nullable=False)


class IndexValue(Base):
    """Значение биржевого индекса за торговый день."""

    __tablename__ = "index_values"
    __table_args__ = (sa.UniqueConstraint("index_code", "trade_date"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    index_code: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    close: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)


class CurrencyRate(Base):
    """Курс валюты к рублю по данным ЦБ РФ."""

    __tablename__ = "currency_rates"
    __table_args__ = (sa.UniqueConstraint("currency", "rate_date"),)

    id: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    currency: Mapped[Currency] = mapped_column(str_enum_type(Currency), nullable=False)
    rate_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    rate: Mapped[Decimal] = mapped_column(sa.Numeric(18, 6), nullable=False)

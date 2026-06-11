"""DTO для рыночных данных: ценные бумаги, свечи, дивиденды, индексы, курсы, ставки, муверы."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel
from stocklens_core.enums import Currency


class SecurityOut(BaseModel):
    """DTO ценной бумаги."""

    model_config = {"from_attributes": True}

    id: int
    ticker: str
    name: str
    board: str
    aliases: list[str]
    is_active: bool


class CandleOut(BaseModel):
    """DTO дневной свечи OHLCV."""

    model_config = {"from_attributes": True}

    id: int
    security_id: int
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    value: Decimal
    is_weekend_session: bool


class DividendOut(BaseModel):
    """DTO дивидендной выплаты."""

    model_config = {"from_attributes": True}

    id: int
    security_id: int
    ex_date: date
    value: Decimal
    currency: Currency


class IndexValueOut(BaseModel):
    """DTO значения биржевого индекса за торговый день."""

    model_config = {"from_attributes": True}

    trade_date: date
    close: Decimal


class CurrencyRateOut(BaseModel):
    """DTO курса валюты к рублю."""

    model_config = {"from_attributes": True}

    currency: Currency
    rate_date: date
    rate: Decimal


class KeyRateOut(BaseModel):
    """DTO ключевой ставки ЦБ РФ."""

    model_config = {"from_attributes": True}

    rate_date: date
    rate: Decimal


class MoverOut(BaseModel):
    """DTO бумаги-лидера роста или падения дня."""

    ticker: str
    name: str
    close: Decimal
    prev_close: Decimal
    change_pct: float


class MoversOut(BaseModel):
    """DTO лидеров роста и падения."""

    gainers: list[MoverOut]
    losers: list[MoverOut]

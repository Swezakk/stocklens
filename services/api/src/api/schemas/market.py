"""DTO для рыночных данных: ценные бумаги, свечи, дивиденды."""

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

"""Чтение рыночных данных из БД в pandas (ml-spec §3).

Только чтение: sync SQLAlchemy 2.0 + ORM-модели из ``stocklens-core`` (инвариант #4 —
без дублирования схемы). Возвращаемые DataFrame'ы — в схеме, которую ожидает
:mod:`stocklens_ml.data.adjust`. Decimal-колонки приводятся к float для расчётов.
"""

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.models.market import Candle, Dividend, IndexValue, Security, Split

_CANDLE_COLUMNS = ["trade_date", "open", "high", "low", "close", "volume", "is_weekend_session"]
_DIVIDEND_COLUMNS = ["ex_date", "value", "currency"]
_SPLIT_COLUMNS = ["split_date", "before", "after"]
_INDEX_COLUMNS = ["trade_date", "close"]
_PRICE_COLUMNS = ("open", "high", "low", "close")


def make_session_factory(dsn: str) -> sessionmaker[Session]:
    """Фабрика sync-сессий по DSN (postgresql+psycopg://...) для обучающих скриптов."""
    engine = create_engine(dsn, pool_pre_ping=True)
    return sessionmaker(engine)


def load_candles(session: Session, ticker: str) -> pd.DataFrame:
    """Дневные свечи бумаги, отсортированные по дате (OHLC как float)."""
    stmt = (
        select(
            Candle.trade_date,
            Candle.open,
            Candle.high,
            Candle.low,
            Candle.close,
            Candle.volume,
            Candle.is_weekend_session,
        )
        .join(Security, Candle.security_id == Security.id)
        .where(Security.ticker == ticker)
        .order_by(Candle.trade_date)
    )
    frame = pd.DataFrame(session.execute(stmt).all(), columns=_CANDLE_COLUMNS)
    for column in _PRICE_COLUMNS:
        frame[column] = frame[column].astype(float)
    return frame


def load_dividends(session: Session, ticker: str) -> pd.DataFrame:
    """Дивиденды бумаги (ex_date = registryclosedate, T+1), отсортированы по дате."""
    stmt = (
        select(Dividend.ex_date, Dividend.value, Dividend.currency)
        .join(Security, Dividend.security_id == Security.id)
        .where(Security.ticker == ticker)
        .order_by(Dividend.ex_date)
    )
    frame = pd.DataFrame(session.execute(stmt).all(), columns=_DIVIDEND_COLUMNS)
    frame["value"] = frame["value"].astype(float)
    return frame


def load_splits(session: Session, ticker: str) -> pd.DataFrame:
    """Сплиты бумаги (before акций → after), отсортированы по дате."""
    stmt = (
        select(Split.split_date, Split.before, Split.after)
        .join(Security, Split.security_id == Security.id)
        .where(Security.ticker == ticker)
        .order_by(Split.split_date)
    )
    return pd.DataFrame(session.execute(stmt).all(), columns=_SPLIT_COLUMNS)


def load_index(session: Session, index_code: str) -> pd.DataFrame:
    """Значения биржевого индекса (например, IMOEX), отсортированы по дате (close как float)."""
    stmt = (
        select(IndexValue.trade_date, IndexValue.close)
        .where(IndexValue.index_code == index_code)
        .order_by(IndexValue.trade_date)
    )
    frame = pd.DataFrame(session.execute(stmt).all(), columns=_INDEX_COLUMNS)
    frame["close"] = frame["close"].astype(float)
    return frame

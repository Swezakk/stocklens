"""Integration tests for the DB loader (testcontainers PostgreSQL, schema via create_all)."""

from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import Currency
from stocklens_core.models.base import Base
from stocklens_core.models.market import Candle, Dividend, IndexValue, Security, Split
from stocklens_ml.data import loader
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration


def _seed(session: Session) -> None:
    sber = Security(ticker="SBER", name="Сбербанк", board="TQBR")
    other = Security(ticker="GAZP", name="Газпром", board="TQBR")
    session.add_all([sber, other])
    session.flush()
    session.add_all(
        [
            Candle(
                security_id=sber.id,
                trade_date=date(2024, 1, 10),
                open=Decimal("100.00"),
                high=Decimal("102.00"),
                low=Decimal("99.00"),
                close=Decimal("101.00"),
                volume=12,
                value=Decimal("1200.00"),
                is_weekend_session=False,
            ),
            Candle(
                security_id=sber.id,
                trade_date=date(2024, 1, 9),
                open=Decimal("100.00"),
                high=Decimal("101.00"),
                low=Decimal("98.00"),
                close=Decimal("100.00"),
                volume=10,
                value=Decimal("1000.00"),
                is_weekend_session=False,
            ),
            Candle(
                security_id=other.id,
                trade_date=date(2024, 1, 10),
                open=Decimal("200.00"),
                high=Decimal("200.00"),
                low=Decimal("200.00"),
                close=Decimal("200.00"),
                volume=5,
                value=Decimal("1000.00"),
                is_weekend_session=False,
            ),
            Dividend(
                security_id=sber.id,
                ex_date=date(2024, 5, 13),
                value=Decimal("33.30"),
                currency=Currency.RUB,
            ),
            Split(security_id=sber.id, split_date=date(2024, 7, 1), before=1, after=10),
            IndexValue(index_code="IMOEX", trade_date=date(2024, 1, 10), close=Decimal("3200.50")),
        ]
    )
    session.commit()


@pytest.fixture(scope="module")
def session_factory() -> Iterator[sessionmaker[Session]]:
    with PostgresContainer("postgres:16-alpine") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        engine = create_engine(url)
        Base.metadata.create_all(engine)
        factory: sessionmaker[Session] = sessionmaker(engine)
        with factory() as session:
            _seed(session)
        yield factory
        engine.dispose()


def test_load_candles_returns_sorted_float_ohlc(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        frame = loader.load_candles(session, "SBER")

    # Только SBER, отсортировано по дате, OHLC как float.
    assert frame["trade_date"].tolist() == [date(2024, 1, 9), date(2024, 1, 10)]
    assert frame["close"].tolist() == [100.0, 101.0]
    assert frame["close"].dtype == float
    assert frame["is_weekend_session"].tolist() == [False, False]
    assert list(frame.columns) == [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_weekend_session",
    ]


def test_load_dividends_returns_value_and_currency(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        frame = loader.load_dividends(session, "SBER")

    assert frame["ex_date"].tolist() == [date(2024, 5, 13)]
    assert frame["value"].tolist() == [33.30]
    assert frame["currency"].tolist() == [Currency.RUB]


def test_load_splits_returns_before_after(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        frame = loader.load_splits(session, "SBER")

    assert frame["split_date"].tolist() == [date(2024, 7, 1)]
    assert frame["before"].tolist() == [1]
    assert frame["after"].tolist() == [10]


def test_load_index_returns_float_close(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        frame = loader.load_index(session, "IMOEX")

    assert frame["trade_date"].tolist() == [date(2024, 1, 10)]
    assert frame["close"].tolist() == [3200.50]
    assert frame["close"].dtype == float


def test_load_candles_unknown_ticker_returns_empty(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        frame = loader.load_candles(session, "UNKNOWN")

    assert frame.empty
    assert list(frame.columns) == [
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_weekend_session",
    ]

"""Интеграционные тесты репозиториев: реальная PostgreSQL через testcontainers."""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from ingestor.parsing import ParsedCandle, ParsedConstituent
from ingestor.repositories import (
    collection_tickers,
    last_candle_date,
    upsert_candles,
    upsert_securities,
)
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.models.market import Candle, Security
from stocklens_core.models.portfolio import PortfolioPosition
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[3]


def _run_migrations(db_url: str) -> None:
    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="module")
def db_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        _run_migrations(url)
        yield url


@pytest.fixture()
def session_factory(db_url: str) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(db_url)
    factory: sessionmaker[Session] = sessionmaker(engine)
    yield factory
    with engine.connect() as conn:
        conn.execute(
            text("TRUNCATE candles, dividends, splits, securities RESTART IDENTITY CASCADE")
        )
        conn.commit()


@pytest.fixture()
def sber_constituent() -> ParsedConstituent:
    return ParsedConstituent(ticker="SBER", name="Сбербанк")


@pytest.fixture()
def sber_candle() -> ParsedCandle:
    return ParsedCandle(
        ticker="SBER",
        board="TQBR",
        trade_date=date(2026, 6, 2),
        open=Decimal("321.00"),
        high=Decimal("324.00"),
        low=Decimal("319.50"),
        close=Decimal("323.00"),
        volume=17_000_000,
        value=Decimal("5500000000.00"),
        is_weekend_session=False,
    )


class TestUpsertSecurities:
    def test_upsert_twice_results_in_single_row(
        self,
        session_factory: sessionmaker[Session],
        sber_constituent: ParsedConstituent,
    ) -> None:
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
        with session_factory() as s:
            count = s.query(Security).filter_by(ticker="SBER").count()
        assert count == 1

    def test_deactivate_missing_false_keeps_is_active(
        self,
        session_factory: sessionmaker[Session],
        sber_constituent: ParsedConstituent,
    ) -> None:
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
        gazp = ParsedConstituent(ticker="GAZP", name="Газпром")
        with session_factory() as s:
            upsert_securities(s, [gazp], deactivate_missing=False)
            s.commit()
        with session_factory() as s:
            sber = s.query(Security).filter_by(ticker="SBER").one()
        assert sber.is_active is True

    def test_deactivate_missing_true_deactivates_absentee(
        self,
        session_factory: sessionmaker[Session],
        sber_constituent: ParsedConstituent,
    ) -> None:
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
        gazp = ParsedConstituent(ticker="GAZP", name="Газпром")
        with session_factory() as s:
            upsert_securities(s, [gazp], deactivate_missing=True)
            s.commit()
        with session_factory() as s:
            sber = s.query(Security).filter_by(ticker="SBER").one()
        assert sber.is_active is False


class TestUpsertCandles:
    def test_upsert_twice_updates_values(
        self,
        session_factory: sessionmaker[Session],
        sber_constituent: ParsedConstituent,
        sber_candle: ParsedCandle,
    ) -> None:
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
            sec = s.query(Security).filter_by(ticker="SBER").one()
            security_id = sec.id

        with session_factory() as s:
            upsert_candles(s, security_id, [sber_candle])
            s.commit()

        updated = ParsedCandle(
            ticker="SBER",
            board="TQBR",
            trade_date=date(2026, 6, 2),
            open=Decimal("330.00"),
            high=Decimal("335.00"),
            low=Decimal("328.00"),
            close=Decimal("332.00"),
            volume=20_000_000,
            value=Decimal("6600000000.00"),
            is_weekend_session=False,
        )
        with session_factory() as s:
            upsert_candles(s, security_id, [updated])
            s.commit()

        with session_factory() as s:
            row = s.execute(
                select(Candle).where(
                    Candle.security_id == security_id,
                    Candle.trade_date == date(2026, 6, 2),
                )
            ).scalar_one()
        assert row.close == Decimal("332.00")

    def test_last_candle_date_returns_latest(
        self,
        session_factory: sessionmaker[Session],
        sber_constituent: ParsedConstituent,
        sber_candle: ParsedCandle,
    ) -> None:
        with session_factory() as s:
            upsert_securities(s, [sber_constituent], deactivate_missing=False)
            s.commit()
            sec = s.query(Security).filter_by(ticker="SBER").one()
            security_id = sec.id

        candle2 = ParsedCandle(
            ticker="SBER",
            board="TQBR",
            trade_date=date(2026, 6, 4),
            open=Decimal("319.00"),
            high=Decimal("321.00"),
            low=Decimal("318.00"),
            close=Decimal("320.00"),
            volume=15_000_000,
            value=Decimal("4800000000.00"),
            is_weekend_session=False,
        )

        with session_factory() as s:
            upsert_candles(s, security_id, [sber_candle, candle2])
            s.commit()

        with session_factory() as s:
            result = last_candle_date(s, security_id)
        assert result == date(2026, 6, 4)


class TestCollectionTickers:
    def test_collection_tickers_includes_inactive_held_ticker(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        gazp = ParsedConstituent(ticker="GAZP", name="Газпром")
        with session_factory() as s:
            upsert_securities(s, [gazp], deactivate_missing=False)
            s.commit()

        with session_factory() as s:
            gazp_sec = s.query(Security).filter_by(ticker="GAZP").one()
            gazp_sec.is_active = False
            s.add(gazp_sec)

            pp = PortfolioPosition(
                security_id=gazp_sec.id,
                quantity=100,
                avg_price=Decimal("150.00"),
                opened_at=datetime.now(UTC),
            )
            s.add(pp)
            s.commit()

        with session_factory() as s:
            tickers = collection_tickers(s)

        ticker_names = [t for t, _ in tickers]
        assert "GAZP" in ticker_names

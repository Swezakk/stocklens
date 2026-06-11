"""Интеграционные тесты backfill: параметр from= в запросах свечей."""

import json
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import responses as resp_lib
from alembic import command
from alembic.config import Config
from ingestor.collectors.moex import sync_candles
from ingestor.iss_client import MoexIssClient
from ingestor.parsing import ParsedCandle, ParsedConstituent
from ingestor.repositories import upsert_candles, upsert_securities
from ingestor.settings import IngestorSettings
from pydantic import PostgresDsn
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.models.market import Security
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[3]
_BASE = "https://iss.moex.com/iss"


def _run_migrations(db_url: str) -> None:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(cfg, "head")


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
        conn.execute(text("TRUNCATE candles, collector_runs, securities RESTART IDENTITY CASCADE"))
        conn.commit()


@pytest.fixture()
def settings(tmp_path: Path) -> IngestorSettings:
    return IngestorSettings(
        database_url=PostgresDsn("postgresql+psycopg://x:x@localhost/x"),
        tickers_universe="SBER",
        heartbeat_path=tmp_path / "heartbeat",
    )


@pytest.fixture()
def client() -> MoexIssClient:
    return MoexIssClient(sleep=lambda _: None, retry_wait_min=0.0, retry_wait_max=0.0)


def _empty_candles_response() -> str:
    _cols = ["BOARDID", "TRADEDATE", "SECID", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]
    return json.dumps(
        {
            "history": {"columns": _cols, "data": []},
            "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, 0, 100]]},
        }
    )


class TestBackfillFromParam:
    @resp_lib.activate
    def test_empty_db_requests_without_from_param(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        with session_factory() as s:
            upsert_securities(
                s, [ParsedConstituent(ticker="SBER", name="Сбербанк")], deactivate_missing=False
            )
            s.commit()

        url = f"{_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"
        resp_lib.add(resp_lib.GET, url, body=_empty_candles_response())

        sync_candles(client, session_factory, settings)

        assert len(resp_lib.calls) == 1
        request_url = resp_lib.calls[0].request.url or ""
        assert "from=" not in request_url

    @resp_lib.activate
    def test_existing_candle_requests_from_next_day(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        with session_factory() as s:
            upsert_securities(
                s, [ParsedConstituent(ticker="SBER", name="Сбербанк")], deactivate_missing=False
            )
            s.commit()
            sec = s.query(Security).filter_by(ticker="SBER").one()
            security_id = int(sec.id)

        last_known = date(2026, 6, 2)
        candle = ParsedCandle(
            ticker="SBER",
            board="TQBR",
            trade_date=last_known,
            open=Decimal("321.00"),
            high=Decimal("324.00"),
            low=Decimal("319.50"),
            close=Decimal("323.00"),
            volume=17_000_000,
            value=Decimal("5500000000.00"),
            is_weekend_session=False,
        )
        with session_factory() as s:
            upsert_candles(s, security_id, [candle])
            s.commit()

        url = f"{_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"
        resp_lib.add(resp_lib.GET, url, body=_empty_candles_response())

        sync_candles(client, session_factory, settings)

        assert len(resp_lib.calls) == 1
        request_url = resp_lib.calls[0].request.url or ""
        assert "from=2026-06-03" in request_url

"""Интеграционные тесты сборщиков MOEX: реальная PostgreSQL + мок ISS."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import responses as resp_lib
from alembic import command
from alembic.config import Config
from ingestor.collectors.moex import (
    run_all_collectors,
    sync_candles,
    sync_dividends,
    sync_securities,
)
from ingestor.iss_client import MoexIssClient
from ingestor.parsing import ParsedConstituent
from ingestor.repositories import upsert_securities
from ingestor.settings import IngestorSettings
from pydantic import PostgresDsn
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import CollectorRunStatus
from stocklens_core.models.market import Candle, Security
from stocklens_core.models.operations import CollectorRun
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
        conn.execute(
            text(
                "TRUNCATE candles, dividends, splits, collector_runs, securities "
                "RESTART IDENTITY CASCADE"
            )
        )
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


def _seed_sber(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as s:
        upsert_securities(
            s, [ParsedConstituent(ticker="SBER", name="Сбербанк")], deactivate_missing=False
        )
        s.commit()
        sec = s.query(Security).filter_by(ticker="SBER").one()
        return int(sec.id)


class TestSyncSecurities:
    @resp_lib.activate
    def test_sync_securities_applies_alias_seed(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        """После синхронизации бумага получает псевдонимы из aliases_seed.

        Без сида матчинг новостей по именам компаний («Сбер», «Сбербанк»)
        не работает — псевдонимы обязаны попадать в БД автоматически.
        """
        description_payload = {
            "description": {
                "columns": ["name", "title", "value"],
                "data": [["SHORTNAME", "Краткое наименование", "Сбербанк"]],
            }
        }
        resp_lib.add(
            resp_lib.GET,
            f"{_BASE}/securities/SBER.json",
            json=description_payload,
        )

        sync_securities(client, session_factory, settings)

        with session_factory() as s:
            sec = s.execute(select(Security).filter_by(ticker="SBER")).scalar_one()
            assert "Сбер" in sec.aliases, f"Seed aliases not applied: aliases={sec.aliases}"


class TestSyncCandles:
    @resp_lib.activate
    def test_sync_candles_writes_candles_and_success_run(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        security_id = _seed_sber(session_factory)
        url = f"{_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"

        _cols = ["BOARDID", "TRADEDATE", "SECID", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]
        payload = json.dumps(
            {
                "history": {
                    "columns": _cols,
                    "data": [
                        [
                            "TQBR",
                            "2026-06-02",
                            "SBER",
                            321.0,
                            324.0,
                            319.5,
                            323.0,
                            17_000_000,
                            5_500_000_000.0,
                        ]
                    ],
                },
                "history.cursor": {
                    "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                    "data": [[0, 1, 100]],
                },
            }
        )
        resp_lib.add(resp_lib.GET, url, body=payload)

        sync_candles(client, session_factory, settings)

        with session_factory() as s:
            rows = (
                s.execute(select(Candle).where(Candle.security_id == security_id)).scalars().all()
            )
            run = s.query(CollectorRun).filter_by(source="moex_candles").one()

        assert len(rows) == 1
        assert run.status == CollectorRunStatus.SUCCESS
        assert run.records_added > 0


class TestSyncDividends:
    @resp_lib.activate
    def test_unknown_currency_marks_run_partial(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        _seed_sber(session_factory)
        url = f"{_BASE}/securities/SBER/dividends.json"

        payload = json.dumps(
            {
                "dividends": {
                    "columns": ["secid", "isin", "registryclosedate", "value", "currencyid"],
                    "data": [["SBER", "RU123", "2023-06-01", 25.0, "XYZ"]],
                }
            }
        )
        resp_lib.add(resp_lib.GET, url, body=payload)

        sync_dividends(client, session_factory, settings)

        with session_factory() as s:
            run = s.query(CollectorRun).filter_by(source="moex_dividends").one()
        assert run.status == CollectorRunStatus.PARTIAL


class TestRunAllCollectors:
    @resp_lib.activate
    def test_one_source_failing_does_not_stop_others(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        _seed_sber(session_factory)

        candles_url = (
            f"{_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"
        )
        securities_url = f"{_BASE}/securities/SBER.json"
        index_url = f"{_BASE}/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json"
        dividends_url = f"{_BASE}/securities/SBER/dividends.json"
        splits_url = f"{_BASE}/statistics/engines/stock/splits.json"

        for _ in range(5):
            resp_lib.add(resp_lib.GET, securities_url, status=500)

        _c = ["BOARDID", "TRADEDATE", "SECID", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]
        candles_payload = json.dumps(
            {
                "history": {"columns": _c, "data": []},
                "history.cursor": {
                    "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                    "data": [[0, 0, 100]],
                },
            }
        )
        resp_lib.add(resp_lib.GET, candles_url, body=candles_payload)

        index_payload = json.dumps(
            {
                "history": {"columns": ["BOARDID", "SECID", "TRADEDATE", "CLOSE"], "data": []},
                "history.cursor": {
                    "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                    "data": [[0, 0, 100]],
                },
            }
        )
        resp_lib.add(resp_lib.GET, index_url, body=index_payload)

        resp_lib.add(
            resp_lib.GET,
            dividends_url,
            body=json.dumps({"dividends": {"columns": [], "data": []}}),
        )
        resp_lib.add(
            resp_lib.GET,
            splits_url,
            body=json.dumps(
                {"splits": {"columns": ["tradedate", "secid", "before", "after"], "data": []}}
            ),
        )

        run_all_collectors(client, session_factory, settings)

        with session_factory() as s:
            runs = s.query(CollectorRun).all()
            statuses = {r.source: r.status for r in runs}

        assert statuses["moex_securities"] == CollectorRunStatus.FAILED
        assert statuses["moex_candles"] == CollectorRunStatus.SUCCESS
        assert statuses["moex_index"] == CollectorRunStatus.SUCCESS

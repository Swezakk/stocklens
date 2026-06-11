"""Integration tests: watchlist materialisation и deactivation-guard в sync_securities.

Критический регрессионный тест: бумага из вотчлиста, НЕ входящая в состав IMOEX,
не должна получать is_active=False при деактивации после полного ответа биржи (≥30 тикеров).
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import responses as resp_lib
from alembic import command
from alembic.config import Config
from ingestor.collectors.moex import sync_securities
from ingestor.iss_client import MoexIssClient
from ingestor.parsing import ParsedConstituent
from ingestor.repositories import collection_tickers, upsert_securities, watchlist_tickers
from ingestor.settings import IngestorSettings
from pydantic import PostgresDsn
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import CollectorRunStatus
from stocklens_core.models.market import Security
from stocklens_core.models.operations import CollectorRun
from stocklens_core.models.portfolio import Watchlist
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[3]
_BASE = "https://iss.moex.com/iss"

_ANALYTICS_URL = f"{_BASE}/statistics/engines/stock/markets/index/analytics/IMOEX.json"
_ANALYTICS_CURSOR_COLS = ["INDEX", "TOTAL", "PAGESIZE"]

# Минимальное число тикеров, которое включает деактивацию в sync_securities.
_DEACTIVATION_THRESHOLD = 30


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
            text("TRUNCATE watchlist, collector_runs, securities RESTART IDENTITY CASCADE")
        )
        conn.commit()


@pytest.fixture()
def settings(tmp_path: Path) -> IngestorSettings:
    return IngestorSettings(
        database_url=PostgresDsn("postgresql+psycopg://x:x@localhost/x"),
        tickers_universe="IMOEX",
        heartbeat_path=tmp_path / "heartbeat",
    )


@pytest.fixture()
def client() -> MoexIssClient:
    return MoexIssClient(sleep=lambda _: None, retry_wait_min=0.0, retry_wait_max=0.0)


def _seed_watchlist(factory: sessionmaker[Session], ticker: str) -> None:
    """Добавить тикер в таблицу watchlist напрямую (без API — как инвариант спеки §4)."""
    with factory() as s:
        s.add(Watchlist(ticker=ticker))
        s.commit()


def _build_analytics_payload(tickers: list[str]) -> str:
    """Сформировать ответ MOEX analytics для заданного списка тикеров."""
    cols = ["indexid", "tradedate", "ticker", "shortnames", "secids", "weight", "tradingsession"]
    data = [["IMOEX", "2026-06-11", t, f"{t} компания", t, 1.0, 3] for t in tickers]
    total = len(data)
    return json.dumps(
        {
            "analytics": {"columns": cols, "data": data},
            "analytics.cursor": {
                "columns": _ANALYTICS_CURSOR_COLS,
                "data": [[0, total, 500]],
            },
        }
    )


def _build_description_payload(ticker: str, name: str) -> str:
    """Сформировать ответ /securities/{ticker}.json с SHORTNAME."""
    return json.dumps(
        {
            "description": {
                "columns": ["name", "title", "value"],
                "data": [["SHORTNAME", "Краткое наименование", name]],
            }
        }
    )


class TestDeactivationGuard:
    """Регрессионный тест: вотчлист-тикер не деактивируется при полном ответе биржи."""

    @resp_lib.activate
    def test_watchlist_ticker_stays_active_after_full_imoex_sync(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        """REGRESSION: watchlist-тикер (вне IMOEX) не получает is_active=False.

        Воспроизводит баг: deactivation выполняла notin_(index_tickers) без учёта
        вотчлиста — любой non-index тикер деактивировался сразу после появления.
        После двух-точечного фикса (upsert_securities + collection_tickers) тикер
        должен оставаться is_active=True.
        """
        watchlist_ticker = "TCSGR"
        # Незащищённая бумага вне индекса/портфеля/вотчлиста — должна деактивироваться.
        unprotected_ticker = "OLDSEC"

        with session_factory() as s:
            upsert_securities(
                s,
                [
                    ParsedConstituent(ticker=watchlist_ticker, name="Т-Технологии"),
                    ParsedConstituent(ticker=unprotected_ticker, name="Старая бумага"),
                ],
                deactivate_missing=False,
            )
            s.commit()
        _seed_watchlist(session_factory, watchlist_ticker)

        index_tickers = [f"SEC{i:02d}" for i in range(_DEACTIVATION_THRESHOLD)]
        resp_lib.add(
            resp_lib.GET,
            _ANALYTICS_URL,
            body=_build_analytics_payload(index_tickers),
        )
        resp_lib.add(
            resp_lib.GET,
            f"{_BASE}/securities/{watchlist_ticker}.json",
            body=_build_description_payload(watchlist_ticker, "Т-Технологии"),
        )

        sync_securities(client, session_factory, settings)

        with session_factory() as s:
            protected = s.execute(
                select(Security).where(Security.ticker == watchlist_ticker)
            ).scalar_one_or_none()
            unprotected = s.execute(
                select(Security).where(Security.ticker == unprotected_ticker)
            ).scalar_one_or_none()

        assert protected is not None, f"Бумага {watchlist_ticker!r} не найдена после sync"
        assert protected.is_active is True, (
            f"Бумага {watchlist_ticker!r} получила is_active=False — "
            "deactivation-guard не защитил watchlist-тикер"
        )
        # Обратная сторона: фикс не должен был отключить деактивацию вовсе.
        assert unprotected is not None
        assert unprotected.is_active is False, (
            f"Незащищённая бумага {unprotected_ticker!r} осталась активной — деактивация сломана"
        )


class TestWatchlistMaterialization:
    """Вотчлист-тикер появляется в securities после sync_securities."""

    @resp_lib.activate
    def test_new_watchlist_ticker_is_materialized_after_sync(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        """Тикер из вотчлиста, которого ещё нет в securities, создаётся при синхронизации."""
        new_ticker = "POSI"
        _seed_watchlist(session_factory, new_ticker)

        index_tickers = [f"IDX{i:02d}" for i in range(_DEACTIVATION_THRESHOLD)]
        resp_lib.add(
            resp_lib.GET,
            _ANALYTICS_URL,
            body=_build_analytics_payload(index_tickers),
        )
        resp_lib.add(
            resp_lib.GET,
            f"{_BASE}/securities/{new_ticker}.json",
            body=_build_description_payload(new_ticker, "Позитив Технологии"),
        )

        sync_securities(client, session_factory, settings)

        with session_factory() as s:
            sec = s.execute(
                select(Security).where(Security.ticker == new_ticker)
            ).scalar_one_or_none()

        assert sec is not None, f"Бумага {new_ticker!r} не материализована"
        assert sec.name == "Позитив Технологии"
        assert sec.is_active is True


class TestCollectionTickersIncludesWatchlist:
    """collection_tickers включает вотчлист-тикер, не входящий в индекс и портфель."""

    def test_collection_tickers_includes_watchlist_inactive_security(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Бумага is_active=False из вотчлиста попадает в collection_tickers."""
        ticker = "FLOT"
        with session_factory() as s:
            upsert_securities(
                s,
                [ParsedConstituent(ticker=ticker, name="Флот тест")],
                deactivate_missing=False,
            )
            s.commit()
            sec = s.execute(select(Security).where(Security.ticker == ticker)).scalar_one()
            # Деактивируем вручную — имитируем «выпал из индекса».
            sec.is_active = False
            s.commit()

        _seed_watchlist(session_factory, ticker)

        with session_factory() as s:
            tickers_list = collection_tickers(s)

        found = [t for t, _ in tickers_list if t == ticker]
        assert found, f"Тикер {ticker!r} из вотчлиста не найден в collection_tickers"


class TestWatchlistTickerMoex404:
    """Вотчлист-тикер, которого нет на MOEX (404), не ломает синхронизацию."""

    @resp_lib.activate
    def test_invalid_watchlist_ticker_does_not_fail_run(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
        client: MoexIssClient,
    ) -> None:
        """MOEX 404 на вотчлист-тикер → warning залогирован, запуск не FAILED."""
        missing_ticker = "XXXXXX"
        _seed_watchlist(session_factory, missing_ticker)

        index_tickers = [f"OK{i:02d}" for i in range(_DEACTIVATION_THRESHOLD)]
        resp_lib.add(
            resp_lib.GET,
            _ANALYTICS_URL,
            body=_build_analytics_payload(index_tickers),
        )
        # 5 попыток с retries — все возвращают 404.
        for _ in range(5):
            resp_lib.add(
                resp_lib.GET,
                f"{_BASE}/securities/{missing_ticker}.json",
                status=404,
            )

        sync_securities(client, session_factory, settings)

        with session_factory() as s:
            run = s.query(CollectorRun).filter_by(source="moex_securities").one()

        assert run.status in (
            CollectorRunStatus.SUCCESS,
            CollectorRunStatus.PARTIAL,
        ), f"Запуск завершился {run.status!r} — ожидался SUCCESS или PARTIAL"

        with session_factory() as s:
            sec = s.execute(
                select(Security).where(Security.ticker == missing_ticker)
            ).scalar_one_or_none()
        assert sec is None, "Несуществующий тикер не должен появляться в securities"


class TestWatchlistTickersRepository:
    """Юнит-проверки функции watchlist_tickers."""

    def test_watchlist_tickers_returns_seeded_tickers(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """watchlist_tickers возвращает ровно те тикеры, что добавлены в таблицу."""
        _seed_watchlist(session_factory, "VTBR")
        _seed_watchlist(session_factory, "GAZP")

        with session_factory() as s:
            result = watchlist_tickers(s)

        assert {"VTBR", "GAZP"}.issubset(result)

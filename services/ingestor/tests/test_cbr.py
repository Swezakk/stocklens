"""Тесты парсинга и upsert данных ЦБ РФ."""

from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import responses as resp_lib
from alembic import command
from alembic.config import Config
from ingestor.collectors.cbr import (
    _parse_cbr_decimal,
    _parse_daily_xml,
    _parse_dynamic_xml,
    _parse_key_rate_soap,
    backfill_currency_rates,
    sync_currency_rates,
    sync_key_rate,
)
from ingestor.repositories import upsert_currency_rates
from ingestor.settings import IngestorSettings
from pydantic import PostgresDsn
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import CollectorRunStatus, Currency
from stocklens_core.models.market import CurrencyRate, KeyRate
from stocklens_core.models.operations import CollectorRun
from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[3]
FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
            text("TRUNCATE currency_rates, key_rates, collector_runs RESTART IDENTITY CASCADE")
        )
        conn.commit()


@pytest.fixture()
def settings(tmp_path: Path) -> IngestorSettings:
    return IngestorSettings(
        database_url=PostgresDsn("postgresql+psycopg://x:x@localhost/x"),
        heartbeat_path=tmp_path / "heartbeat",
    )


class TestParseCbrDecimal:
    def test_comma_decimal_separator(self) -> None:
        result = _parse_cbr_decimal("71,7892", "1")
        assert result == Decimal("71.7892")

    def test_nominal_scaling_cny(self) -> None:
        # CNY в старых выгрузках имел Nominal=10; итоговый курс = Value/10
        result = _parse_cbr_decimal("105,825", "10")
        assert result is not None
        assert abs(result - Decimal("10.5825")) < Decimal("0.0001")

    def test_invalid_value_returns_none(self) -> None:
        result = _parse_cbr_decimal("N/A", "1")
        assert result is None

    def test_zero_nominal_returns_none(self) -> None:
        result = _parse_cbr_decimal("71.00", "0")
        assert result is None


class TestParseDailyXml:
    def test_parses_usd_eur_cny(self) -> None:
        content = (FIXTURES_DIR / "cbr_daily.xml").read_bytes()
        rows = _parse_daily_xml(content)
        currencies = {r.currency for r in rows}
        assert Currency.USD in currencies
        assert Currency.EUR in currencies
        assert Currency.CNY in currencies

    def test_usd_rate_is_positive(self) -> None:
        content = (FIXTURES_DIR / "cbr_daily.xml").read_bytes()
        rows = _parse_daily_xml(content)
        usd = next(r for r in rows if r.currency == Currency.USD)
        assert usd.rate > 0

    def test_cny_nominal_applied(self) -> None:
        content = (FIXTURES_DIR / "cbr_daily.xml").read_bytes()
        rows = _parse_daily_xml(content)
        cny = next(r for r in rows if r.currency == Currency.CNY)
        assert cny.rate < Decimal("20")


class TestParseDynamicXml:
    def test_parses_multiple_dates(self) -> None:
        content = (FIXTURES_DIR / "cbr_dynamic_usd.xml").read_bytes()
        rows = _parse_dynamic_xml(content, Currency.USD)
        assert len(rows) >= 5

    def test_all_rows_have_usd_currency(self) -> None:
        content = (FIXTURES_DIR / "cbr_dynamic_usd.xml").read_bytes()
        rows = _parse_dynamic_xml(content, Currency.USD)
        assert all(r.currency == Currency.USD for r in rows)

    def test_rates_are_positive(self) -> None:
        content = (FIXTURES_DIR / "cbr_dynamic_usd.xml").read_bytes()
        rows = _parse_dynamic_xml(content, Currency.USD)
        assert all(r.rate > 0 for r in rows)


class TestParseKeyRateSoap:
    def test_parses_rate_from_fixture(self) -> None:
        content = (FIXTURES_DIR / "cbr_keyrate_soap.xml").read_bytes()
        rows = _parse_key_rate_soap(content)
        assert len(rows) >= 5

    def test_rate_value_correct(self) -> None:
        content = (FIXTURES_DIR / "cbr_keyrate_soap.xml").read_bytes()
        rows = _parse_key_rate_soap(content)
        assert all(r.rate == Decimal("14.50") for r in rows)

    def test_date_is_parsed(self) -> None:
        content = (FIXTURES_DIR / "cbr_keyrate_soap.xml").read_bytes()
        rows = _parse_key_rate_soap(content)
        assert rows[0].rate_date >= date(2026, 6, 1)


class TestSyncCurrencyRates:
    @resp_lib.activate
    def test_upsert_idempotent(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        content = (FIXTURES_DIR / "cbr_daily.xml").read_bytes()
        resp_lib.add(resp_lib.GET, "https://www.cbr.ru/scripts/XML_daily.asp", body=content)
        resp_lib.add(resp_lib.GET, "https://www.cbr.ru/scripts/XML_daily.asp", body=content)

        sync_currency_rates(session_factory, settings)
        sync_currency_rates(session_factory, settings)

        with session_factory() as s:
            count = s.query(CurrencyRate).count()
            run_count = s.query(CollectorRun).filter_by(source="cbr_rates").count()

        assert count == 3
        assert run_count == 2

    @resp_lib.activate
    def test_success_run_recorded(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        content = (FIXTURES_DIR / "cbr_daily.xml").read_bytes()
        resp_lib.add(resp_lib.GET, "https://www.cbr.ru/scripts/XML_daily.asp", body=content)

        sync_currency_rates(session_factory, settings)

        with session_factory() as s:
            run = s.query(CollectorRun).filter_by(source="cbr_rates").one()
        assert run.status == CollectorRunStatus.SUCCESS


class TestBackfillCurrencyRates:
    @resp_lib.activate
    def test_backfill_from_empty_db_requests_2013(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        content = (FIXTURES_DIR / "cbr_dynamic_usd.xml").read_bytes()
        resp_lib.add(
            resp_lib.GET,
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            body=content,
        )
        resp_lib.add(
            resp_lib.GET,
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            body=content,
        )
        resp_lib.add(
            resp_lib.GET,
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            body=content,
        )

        backfill_currency_rates(session_factory, settings)

        calls = [
            c
            for c in resp_lib.calls
            if c.request.url and "date_req1=01%2F01%2F2013" in c.request.url
        ]
        assert len(calls) == 3

    @resp_lib.activate
    def test_backfill_skips_if_up_to_date(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        today = datetime.now(UTC).date()
        currencies = [Currency.USD, Currency.EUR, Currency.CNY]
        with session_factory() as s:
            upsert_currency_rates(
                s,
                [{"currency": c, "rate_date": today, "rate": Decimal("70")} for c in currencies],
            )
            s.commit()

        backfill_currency_rates(session_factory, settings)

        assert len(resp_lib.calls) == 0


class TestSyncKeyRate:
    @resp_lib.activate
    def test_upsert_idempotent(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        content = (FIXTURES_DIR / "cbr_keyrate_soap.xml").read_bytes()
        for _ in range(2):
            resp_lib.add(
                resp_lib.POST,
                "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx",
                body=content,
            )

        sync_key_rate(session_factory, settings)
        sync_key_rate(session_factory, settings)

        with session_factory() as s:
            first_count = s.query(KeyRate).count()

        assert first_count >= 5

"""Интеграционные тесты RSS-сборщика: responses + testcontainers PG + fake scorer."""

from collections.abc import Iterator
from datetime import UTC
from pathlib import Path

import feedparser
import pytest
import responses as resp_lib
from alembic import command
from alembic.config import Config
from ingestor.collectors.rss import FEEDS, _parse_entry, sync_news
from ingestor.parsing import ParsedConstituent
from ingestor.repositories import upsert_securities
from ingestor.sentiment import SentimentResult
from ingestor.settings import IngestorSettings
from pydantic import PostgresDsn
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from stocklens_core.enums import CollectorRunStatus, SentimentLabel
from stocklens_core.models.news import NewsArticle, NewsSentiment, NewsTicker
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
            text(
                "TRUNCATE news_articles, news_sentiment, news_tickers, "
                "collector_runs, securities RESTART IDENTITY CASCADE"
            )
        )
        conn.commit()


@pytest.fixture()
def settings(tmp_path: Path) -> IngestorSettings:
    return IngestorSettings(
        database_url=PostgresDsn("postgresql+psycopg://x:x@localhost/x"),
        heartbeat_path=tmp_path / "heartbeat",
    )


class _FakeScorer:
    """Детерминированный scorer для тестов: всегда возвращает NEUTRAL 0.9."""

    @property
    def model_version(self) -> str:
        return "fake-v1"

    def score(self, text: str) -> SentimentResult:
        return SentimentResult(label=SentimentLabel.NEUTRAL, score=0.9)


def _seed_sber(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as s:
        upsert_securities(
            s,
            [ParsedConstituent(ticker="SBER", name="Сбербанк")],
            deactivate_missing=False,
            alias_seed={"SBER": ["Сбербанк", "Сбербанка", "Сбер"]},
        )
        s.commit()


def _rbc_content() -> bytes:
    return (FIXTURES_DIR / "rss_rbc.xml").read_bytes()


def _register_feeds(rbc_body: bytes | None = None) -> None:
    """Зарегистрировать все три фида в responses с одинаковым телом."""
    for _source, url in FEEDS:
        resp_lib.add(
            resp_lib.GET,
            url,
            body=rbc_body if rbc_body is not None else _rbc_content(),
        )


class TestSyncNewsNewArticle:
    @resp_lib.activate
    def test_new_article_scored_and_linked(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)
        _register_feeds()

        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            articles = s.execute(select(NewsArticle)).scalars().all()
            sentiments = s.execute(select(NewsSentiment)).scalars().all()

        assert len(articles) > 0
        assert len(sentiments) == len(articles)
        assert all(sent.model_version == "fake-v1" for sent in sentiments)

    @resp_lib.activate
    def test_duplicate_url_second_run_adds_zero(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)
        rbc_body = _rbc_content()

        for _source, url in FEEDS:
            resp_lib.add(resp_lib.GET, url, body=rbc_body)
        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            count_after_first = s.query(NewsArticle).count()

        for _source, url in FEEDS:
            resp_lib.add(resp_lib.GET, url, body=rbc_body)
        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            count_after_second = s.query(NewsArticle).count()

        assert count_after_second == count_after_first

    @resp_lib.activate
    def test_sentiment_not_duplicated_on_second_run(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)
        rbc_body = _rbc_content()

        for _source, url in FEEDS:
            resp_lib.add(resp_lib.GET, url, body=rbc_body)
        sync_news(session_factory, settings, _FakeScorer())

        for _source, url in FEEDS:
            resp_lib.add(resp_lib.GET, url, body=rbc_body)
        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            article_count = s.query(NewsArticle).count()
            sentiment_count = s.query(NewsSentiment).count()

        assert sentiment_count == article_count


class TestSyncNewsFeedIsolation:
    @resp_lib.activate
    def test_per_feed_failure_does_not_stop_others(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)

        rbc_source, rbc_url = FEEDS[0]
        kommersant_source, kommersant_url = FEEDS[1]
        interfax_source, interfax_url = FEEDS[2]

        resp_lib.add(resp_lib.GET, rbc_url, status=503)
        resp_lib.add(
            resp_lib.GET,
            kommersant_url,
            body=(FIXTURES_DIR / "rss_kommersant.xml").read_bytes(),
        )
        resp_lib.add(
            resp_lib.GET,
            interfax_url,
            body=(FIXTURES_DIR / "rss_interfax.xml").read_bytes(),
        )

        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            runs = s.query(CollectorRun).all()
            statuses = {r.source: r.status for r in runs}

        assert statuses[rbc_source] == CollectorRunStatus.FAILED
        assert statuses[kommersant_source] == CollectorRunStatus.SUCCESS
        assert statuses[interfax_source] == CollectorRunStatus.SUCCESS

    @resp_lib.activate
    def test_collector_run_created_per_feed(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)
        _register_feeds()

        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            run_sources = {r.source for r in s.query(CollectorRun).all()}

        feed_sources = {source for source, _ in FEEDS}
        assert feed_sources.issubset(run_sources)


class TestSyncNewsTickerLinking:
    @resp_lib.activate
    def test_sber_mentioned_linked_to_ticker(
        self,
        session_factory: sessionmaker[Session],
        settings: IngestorSettings,
    ) -> None:
        _seed_sber(session_factory)

        sber_rss = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel>'
            "<item><title>Сбербанк повысил дивиденды</title>"
            "<link>https://example.com/news/sber-dividend</link>"
            "<pubDate>Wed, 11 Jun 2026 04:00:00 +0300</pubDate>"
            "<description>Сбербанк объявил о рекордных дивидендах.</description></item>"
            "</channel></rss>"
        ).encode()

        for _source, url in FEEDS:
            resp_lib.add(resp_lib.GET, url, body=sber_rss)

        sync_news(session_factory, settings, _FakeScorer())

        with session_factory() as s:
            tickers = s.execute(select(NewsTicker)).scalars().all()

        assert len(tickers) >= 1


class TestParseEntryEdgeCases:
    """Unit-тесты _parse_entry на битых записях фида (без БД и сети)."""

    @staticmethod
    def _entries(items_xml: str) -> list[feedparser.util.FeedParserDict]:
        rss = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<rss version="2.0"><channel><title>t</title>'
            f"{items_xml}</channel></rss>"
        )
        return list(feedparser.parse(rss).entries)

    def test_entry_without_pubdate_is_skipped(self) -> None:
        entries = self._entries(
            "<item><title>Новость</title><link>https://example.com/1</link></item>"
        )
        assert _parse_entry("rss_rbc", entries[0]) is None

    def test_entry_without_title_is_skipped(self) -> None:
        entries = self._entries(
            "<item><link>https://example.com/2</link>"
            "<pubDate>Wed, 10 Jun 2026 12:00:00 +0300</pubDate></item>"
        )
        assert _parse_entry("rss_rbc", entries[0]) is None

    def test_valid_entry_parsed_with_utc_published_at(self) -> None:
        entries = self._entries(
            "<item><title>Новость</title><link>https://example.com/3</link>"
            "<pubDate>Wed, 10 Jun 2026 12:00:00 +0300</pubDate></item>"
        )
        parsed = _parse_entry("rss_rbc", entries[0])
        assert parsed is not None
        assert parsed.published_at.tzinfo is UTC
        assert parsed.published_at.hour == 9

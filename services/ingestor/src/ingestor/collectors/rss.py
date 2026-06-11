"""Сборщик новостей из RSS-фидов: РБК, Коммерсантъ, Интерфакс.

Каждый фид обрабатывается в отдельном collector_run — сбой одного не прерывает остальные.
Sentiment скорируется только для новых статей: повторный url пропускается через
ON CONFLICT DO NOTHING и не попадает в список new_ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import requests
import structlog
from sqlalchemy.orm import Session, sessionmaker

from ingestor import heartbeat
from ingestor.matching import match_tickers
from ingestor.repositories import (
    get_alias_index,
    get_security_ids_by_tickers,
    insert_news_articles_returning_new,
    insert_news_tickers,
    upsert_news_sentiment,
)
from ingestor.run_journal import collector_run
from ingestor.sentiment import SentimentScorer, build_scoring_text
from ingestor.settings import IngestorSettings

log = structlog.get_logger(__name__)

FEEDS: list[tuple[str, str]] = [
    ("rss_rbc", "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"),
    ("rss_kommersant", "https://www.kommersant.ru/rss/news.xml"),
    ("rss_interfax", "https://www.interfax.ru/rss.asp"),
]

_REQUEST_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class _ParsedArticle:
    """Распарсенная статья до вставки в БД."""

    source: str
    url: str
    title: str
    summary: str | None
    published_at: datetime


def sync_news(
    session_factory: sessionmaker[Session],
    settings: IngestorSettings,
    scorer: SentimentScorer,
) -> None:
    """Обойти все RSS-фиды и сохранить новые статьи с sentiment и тикерами.

    Каждый фид запускается в собственном collector_run.
    Heartbeat обновляется после каждого фида.

    Args:
        session_factory: Фабрика синхронных SQLAlchemy-сессий.
        settings: Конфигурация ingestor.
        scorer: Классификатор тональности.
    """
    for source, url in FEEDS:
        _sync_single_feed(source, url, session_factory, scorer)
        heartbeat.touch(settings.heartbeat_path)


def _sync_single_feed(
    source: str,
    url: str,
    session_factory: sessionmaker[Session],
    scorer: SentimentScorer,
) -> None:
    """Загрузить один RSS-фид и сохранить новые статьи.

    Args:
        source: Имя источника (ключ в collector_runs).
        url: URL RSS-фида.
        session_factory: Фабрика сессий.
        scorer: Классификатор тональности.
    """
    with collector_run(session_factory, source) as journal:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        feed = feedparser.parse(response.content)

        parsed: list[_ParsedArticle] = []
        for entry in feed.entries:
            article = _parse_entry(source, entry)
            if article is not None:
                parsed.append(article)

        if not parsed:
            log.info("rss_feed_empty", source=source)
            return

        rows: list[dict[str, object]] = [
            {
                "source": a.source,
                "url": a.url,
                "title": a.title,
                "summary": a.summary,
                "published_at": a.published_at,
            }
            for a in parsed
        ]

        with session_factory() as session:
            new_ids = insert_news_articles_returning_new(session, rows)
            session.flush()
            alias_index = get_alias_index(session)
            _score_and_link(session, new_ids, parsed, alias_index, scorer)
            session.commit()

        journal.add_records(len(new_ids))
        log.info("rss_feed_synced", source=source, new_articles=len(new_ids))


def _parse_entry(
    source: str,
    entry: feedparser.util.FeedParserDict,
) -> _ParsedArticle | None:
    """Распарсить одну запись RSS в доменный объект.

    entry.published_parsed — UTC time.struct_time (feedparser всегда нормализует к UTC).

    Args:
        source: Имя источника.
        entry: Запись из feedparser.

    Returns:
        _ParsedArticle или None если у записи нет обязательных полей.
    """
    url: str = str(getattr(entry, "link", "") or "").strip()
    title: str = str(getattr(entry, "title", "") or "").strip()

    if not url or not title:
        log.warning("rss_entry_missing_fields", source=source, url=url)
        return None

    parsed_time = getattr(entry, "published_parsed", None)
    if parsed_time is None:
        log.warning("rss_entry_no_pubdate", source=source, url=url)
        return None

    published_at = datetime(
        parsed_time[0],
        parsed_time[1],
        parsed_time[2],
        parsed_time[3],
        parsed_time[4],
        parsed_time[5],
        tzinfo=UTC,
    )

    raw_summary = getattr(entry, "summary", None)
    summary: str | None = str(raw_summary).strip() if raw_summary else None
    if summary == "":
        summary = None

    return _ParsedArticle(
        source=source,
        url=url,
        title=title,
        summary=summary,
        published_at=published_at,
    )


def _score_and_link(
    session: Session,
    new_ids: list[int],
    parsed: list[_ParsedArticle],
    alias_index: dict[str, str],
    scorer: SentimentScorer,
) -> None:
    """Скорировать sentiment и привязать тикеры для новых статей.

    Args:
        session: Открытая сессия (без commit).
        new_ids: Список id только что вставленных статей.
        parsed: Список статей в том же порядке, что и при вставке.
        alias_index: Индекс псевдоним→тикер для сопоставления тикеров.
        scorer: Классификатор тональности.
    """
    if not new_ids:
        return

    url_to_article = {a.url: a for a in parsed}
    new_urls = [a.url for a in parsed[: len(new_ids)]]

    for article_id, url in zip(new_ids, new_urls, strict=False):
        article = url_to_article.get(url)
        if article is None:
            continue

        scoring_text = build_scoring_text(article.title, article.summary)
        result = scorer.score(scoring_text)

        upsert_news_sentiment(
            session,
            article_id=article_id,
            label=result.label,
            score=result.score,
            model_version=scorer.model_version,
        )

        tickers = match_tickers(scoring_text, alias_index)
        if tickers:
            ticker_ids = get_security_ids_by_tickers(session, tickers)
            insert_news_tickers(session, article_id, list(ticker_ids.values()))

"""Unit-тесты чистых хелперов страницы «Новости» (DESIGN.md §9, §10.3).

Покрывается только presentation-логика (нормализация диапазона дат, сведение мульти-
выбора тональности к серверному фильтру, клиентское сужение, формат времени МСК, markdown
строки ленты, честная подпись корпуса). Сам Streamlit-layout не unit-тестируется (DESIGN).
"""

from datetime import UTC, date, datetime

from dashboard.api_client.dto import NewsOut, SentimentOut
from dashboard.pages.news import (
    _corpus_caption,
    _feed_caption,
    _filter_by_sentiments,
    _format_published_at,
    _iso_or_none,
    _news_row_markdown,
    _resolve_date_range,
    _resolve_sentiment_filter,
)
from stocklens_core.enums import SentimentLabel


def _article(
    *,
    article_id: int = 1,
    title: str = "Заголовок",
    url: str = "https://example.com/news/1",
    source: str = "Интерфакс",
    published_at: datetime | None = None,
    label: SentimentLabel | None = SentimentLabel.POSITIVE,
    tickers: list[str] | None = None,
) -> NewsOut:
    """Собрать NewsOut для тестов хелперов ленты/корпуса."""
    sentiment = (
        SentimentOut(label=label, score=0.8, model_version="rubert-tiny2-v1")
        if label is not None
        else None
    )
    return NewsOut(
        id=article_id,
        source=source,
        url=url,
        title=title,
        summary=None,
        published_at=published_at or datetime(2026, 6, 20, 9, 30, tzinfo=UTC),
        sentiment=sentiment,
        tickers=tickers if tickers is not None else ["SBER"],
    )


def test_resolve_date_range_returns_pair_for_full_range() -> None:
    result = _resolve_date_range((date(2026, 6, 1), date(2026, 6, 20)))

    assert result == (date(2026, 6, 1), date(2026, 6, 20))


def test_resolve_date_range_open_right_for_partial_selection() -> None:
    result = _resolve_date_range((date(2026, 6, 1),))

    assert result == (date(2026, 6, 1), None)


def test_resolve_date_range_handles_single_date_value() -> None:
    result = _resolve_date_range(date(2026, 6, 1))

    assert result == (date(2026, 6, 1), date(2026, 6, 1))


def test_resolve_date_range_empty_tuple_returns_none_pair() -> None:
    assert _resolve_date_range(()) == (None, None)


def test_resolve_date_range_non_date_value_returns_none_pair() -> None:
    assert _resolve_date_range(None) == (None, None)


def test_resolve_sentiment_filter_single_passes_to_server() -> None:
    assert _resolve_sentiment_filter([SentimentLabel.NEGATIVE]) is SentimentLabel.NEGATIVE


def test_resolve_sentiment_filter_empty_is_none() -> None:
    assert _resolve_sentiment_filter([]) is None


def test_resolve_sentiment_filter_multiple_is_none_for_client_side() -> None:
    selected = [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE]

    assert _resolve_sentiment_filter(selected) is None


def test_filter_by_sentiments_narrows_to_chosen_labels() -> None:
    articles = [
        _article(article_id=1, label=SentimentLabel.POSITIVE),
        _article(article_id=2, label=SentimentLabel.NEUTRAL),
        _article(article_id=3, label=SentimentLabel.NEGATIVE),
    ]

    result = _filter_by_sentiments(
        articles,
        [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE],
    )

    assert [article.id for article in result] == [1, 3]


def test_filter_by_sentiments_excludes_unscored_when_active() -> None:
    articles = [
        _article(article_id=1, label=SentimentLabel.POSITIVE),
        _article(article_id=2, label=None),
    ]

    result = _filter_by_sentiments(
        articles,
        [SentimentLabel.POSITIVE, SentimentLabel.NEUTRAL],
    )

    assert [article.id for article in result] == [1]


def test_filter_by_sentiments_passthrough_for_single_or_empty() -> None:
    articles = [_article(article_id=1, label=SentimentLabel.NEUTRAL)]

    assert _filter_by_sentiments(articles, []) == articles
    assert _filter_by_sentiments(articles, [SentimentLabel.POSITIVE]) == articles


def test_format_published_at_converts_utc_to_moscow() -> None:
    utc_dt = datetime(2026, 6, 20, 9, 30, tzinfo=UTC)

    assert _format_published_at(utc_dt) == "20.06.2026 12:30"


def test_news_row_markdown_links_title_and_includes_meta() -> None:
    article = _article(
        title="Сбербанк отчитался",
        url="https://example.com/news/42",
        source="Интерфакс",
        tickers=["SBER", "GAZP"],
    )

    markdown = _news_row_markdown(article)

    assert "[Сбербанк отчитался](https://example.com/news/42)" in markdown
    assert "Интерфакс" in markdown
    assert "20.06.2026 12:30" in markdown
    assert "`SBER`" in markdown
    assert "`GAZP`" in markdown
    assert "sentiment-chip" in markdown


def test_news_row_markdown_without_sentiment_omits_chip() -> None:
    article = _article(label=None)

    markdown = _news_row_markdown(article)

    assert "sentiment-chip" not in markdown


def test_news_row_markdown_escapes_untrusted_rss_html() -> None:
    """RSS-поля (title/source/url) экранируются: stored-HTML-инъекция не доходит до DOM."""
    article = _article(
        title='<img src=x onerror="alert(1)">',
        source="<script>steal()</script>",
        url="https://example.com/a?x=1&y=2",
        tickers=["SBER"],
    )

    markdown = _news_row_markdown(article)

    assert "<img" not in markdown
    assert "<script>" not in markdown
    assert "onerror" in markdown  # текст остался, но как экранированная сущность
    assert "&lt;img" in markdown
    assert "&lt;script&gt;" in markdown
    assert "&amp;y=2" in markdown


def test_feed_caption_counts_displayed_rows_not_page_total() -> None:
    """Конец диапазона считается от числа показанных строк, не от page.total (честность §9)."""
    # На странице 50 серверных статей, после клиентского сужения видно 12.
    assert _feed_caption(offset=0, shown=12, total=320) == "Показаны 1–12 из 320"


def test_feed_caption_offset_shifts_displayed_range() -> None:
    assert _feed_caption(offset=50, shown=50, total=320) == "Показаны 51–100 из 320"


def test_feed_caption_empty_page_reports_zero() -> None:
    """Пустая (после фильтра) страница — честный «0 из total», пагинация всё равно рисуется."""
    assert _feed_caption(offset=100, shown=0, total=320) == "Показано 0 из 320"


def test_corpus_caption_reports_real_sample_size_and_period() -> None:
    caption = _corpus_caption(
        total=350,
        shown=350,
        truncated=False,
        date_from=date(2026, 6, 1),
        date_to=date(2026, 6, 20),
    )

    assert "350 из 350" in caption
    assert "01.06.2026–20.06.2026" in caption
    assert "усеч" not in caption


def test_corpus_caption_marks_truncation_explicitly() -> None:
    caption = _corpus_caption(
        total=2000,
        shown=1000,
        truncated=True,
        date_from=date(2026, 6, 1),
        date_to=date(2026, 6, 20),
    )

    assert "1000 из 2000" in caption
    assert "усеч" in caption


def test_corpus_caption_without_period_omits_range() -> None:
    caption = _corpus_caption(
        total=10,
        shown=10,
        truncated=False,
        date_from=None,
        date_to=None,
    )

    assert "10 из 10" in caption
    assert " за " not in caption


def test_iso_or_none_serializes_date_or_returns_none() -> None:
    assert _iso_or_none(date(2026, 6, 1)) == "2026-06-01"
    assert _iso_or_none(None) is None

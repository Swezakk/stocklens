"""Тесты слоя fetch: DTO-парсинг через кэш-обёртку и границы fetch_news_corpus.

st.cache_data вне Streamlit-runtime пишет в кэш и шлёт «missing ScriptRunContext»
в лог — это шум, не падение (pytest без filterwarnings=error). Autouse-фикстура
чистит кэш между тестами, чтобы страницы корпуса не протекали между сценариями.
"""

from collections.abc import Iterator
from typing import Any, cast

import pytest
import streamlit as st
from dashboard.api_client.client import ApiClient
from dashboard.api_client.dto import IndexValuePage, NewsOut, NewsPage
from dashboard.api_client.fetch import fetch_index, fetch_news, fetch_news_corpus


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Iterator[None]:
    """Сбросить cache_data и cache_resource до и после каждого теста (изоляция)."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


class _FakeClient:
    """Подмена ApiClient: возвращает заранее заданные страницы новостей по offset."""

    def __init__(self, total: int) -> None:
        self._total = total
        self.calls: list[int] = []

    def _article(self, idx: int) -> NewsOut:
        return NewsOut.model_validate(
            {
                "id": idx,
                "source": "Интерфакс",
                "url": f"https://example.com/news/{idx}",
                "title": f"Новость {idx}",
                "summary": None,
                "published_at": "2026-06-20T09:30:00+00:00",
                "sentiment": None,
                "tickers": ["SBER"],
            }
        )

    def get_news(
        self,
        ticker: str | None = None,
        sentiment: Any = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> NewsPage:
        """Отдать срез [offset, offset+limit) из total статей."""
        self.calls.append(offset)
        end = min(offset + limit, self._total)
        items = [self._article(i) for i in range(offset, end)]
        return NewsPage(items=items, total=self._total, limit=limit, offset=offset)

    def get_index(
        self,
        index_code: str = "IMOEX",
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> IndexValuePage:
        """Минимальная одностраничная выдача индекса для проверки парсинга через обёртку."""
        return IndexValuePage.model_validate(
            {
                "items": [{"trade_date": "2026-06-20", "close": "3210.55"}],
                "total": 1,
                "limit": limit,
                "offset": offset,
            }
        )


def _as_client(fake: _FakeClient) -> ApiClient:
    """Привести duck-typed подмену к ApiClient для типизированных кэш-обёрток."""
    return cast(ApiClient, fake)


def test_fetch_index_parses_dto_through_wrapper() -> None:
    client = _FakeClient(total=0)

    page = fetch_index(_as_client(client))

    assert page.total == 1
    assert str(page.items[0].close) == "3210.55"


def test_fetch_news_returns_single_page() -> None:
    client = _FakeClient(total=5)

    page = fetch_news(_as_client(client))

    assert page.total == 5
    assert len(page.items) == 5


def test_corpus_stops_at_total_when_below_ceiling() -> None:
    client = _FakeClient(total=350)

    articles, truncated, total = fetch_news_corpus(_as_client(client), max_articles=1000)

    assert total == 350
    assert len(articles) == 350
    assert truncated is False
    assert client.calls == [0, 200]


def test_corpus_stops_at_ceiling_and_marks_truncated() -> None:
    client = _FakeClient(total=900)

    articles, truncated, total = fetch_news_corpus(_as_client(client), max_articles=500)

    assert total == 900
    assert len(articles) == 500
    assert truncated is True


def test_corpus_ceiling_not_multiple_of_page_size_trims_exactly() -> None:
    client = _FakeClient(total=900)

    articles, truncated, total = fetch_news_corpus(_as_client(client), max_articles=450)

    assert len(articles) == 450
    assert truncated is True
    assert total == 900


def test_corpus_reports_real_total_even_when_truncated() -> None:
    client = _FakeClient(total=2000)

    _articles, truncated, total = fetch_news_corpus(_as_client(client), max_articles=1000)

    assert total == 2000
    assert truncated is True


def test_corpus_empty_returns_zero() -> None:
    client = _FakeClient(total=0)

    articles, truncated, total = fetch_news_corpus(_as_client(client), max_articles=1000)

    assert articles == []
    assert truncated is False
    assert total == 0

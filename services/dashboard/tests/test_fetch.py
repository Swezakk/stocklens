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
from dashboard.api_client.dto import (
    CandleOut,
    CandlePage,
    IndexValuePage,
    NewsOut,
    NewsPage,
    OptimizeResult,
    SecurityOut,
    SecurityPage,
)
from dashboard.api_client.fetch import (
    fetch_all_securities,
    fetch_candles_window,
    fetch_index,
    fetch_index_window,
    fetch_news,
    fetch_news_corpus,
    fetch_optimize,
)


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
        self.optimize_calls = 0

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

    def _security(self, idx: int) -> SecurityOut:
        return SecurityOut(
            id=idx,
            ticker=f"TCK{idx}",
            name=f"Бумага {idx}",
            board="TQBR",
            aliases=[],
            is_active=True,
        )

    def _candle(self, idx: int) -> CandleOut:
        return CandleOut.model_validate(
            {
                "id": idx,
                "security_id": 1,
                "trade_date": "2026-06-20",
                "open": "100.00",
                "high": "101.00",
                "low": "99.00",
                "close": "100.50",
                "volume": 1000,
                "value": "100500.00",
                "is_weekend_session": False,
            }
        )

    def get_securities(
        self,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SecurityPage:
        """Отдать срез [offset, offset+limit) из total бумаг (для проверки добор-цикла)."""
        self.calls.append(offset)
        if limit > 200:
            raise AssertionError("per-request limit must be <= 200 (API _MAX_LIMIT)")
        end = min(offset + limit, self._total)
        items = [self._security(idx) for idx in range(offset, end)]
        return SecurityPage(items=items, total=self._total, limit=limit, offset=offset)

    def get_candles(
        self,
        ticker: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> CandlePage:
        """Отдать срез [offset, offset+limit) из total свечей окна (для проверки добор-цикла)."""
        self.calls.append(offset)
        if limit > 200:
            raise AssertionError("per-request limit must be <= 200 (API _MAX_LIMIT)")
        end = min(offset + limit, self._total)
        items = [self._candle(idx) for idx in range(offset, end)]
        return CandlePage(items=items, total=self._total, limit=limit, offset=offset)

    def _index_item(self, idx: int) -> dict[str, str]:
        return {"trade_date": "2026-06-20", "close": f"{3200 + idx}.55"}

    def get_index(
        self,
        index_code: str = "IMOEX",
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> IndexValuePage:
        """Отдать срез [offset, offset+limit) из total значений индекса (для добор-цикла)."""
        self.calls.append(offset)
        if limit > 200:
            raise AssertionError("per-request limit must be <= 200 (API _MAX_LIMIT)")
        end = min(offset + limit, self._total)
        items = [self._index_item(idx) for idx in range(offset, end)]
        return IndexValuePage.model_validate(
            {"items": items, "total": self._total, "limit": limit, "offset": offset}
        )

    def optimize(self, request: dict[str, Any]) -> OptimizeResult:
        """Минимальный результат оптимизации; счётчик ловит повторный солвер-вызов."""
        self.optimize_calls += 1
        return OptimizeResult.model_validate(
            {
                "strategy": request["strategy"],
                "requested_strategy": request["strategy"],
                "weights": {"SBER": 1.0},
                "expected_return": 0.12,
                "volatility": 0.2,
                "sharpe": 0.6,
                "frontier": [{"volatility": 0.2, "expected_return": 0.12}],
                "equal_weight_sharpe": 0.5,
                "imoex_sharpe": 0.4,
            }
        )


def _as_client(fake: _FakeClient) -> ApiClient:
    """Привести duck-typed подмену к ApiClient для типизированных кэш-обёрток."""
    return cast(ApiClient, fake)


def test_fetch_index_parses_dto_through_wrapper() -> None:
    client = _FakeClient(total=3)

    page = fetch_index(_as_client(client))

    assert page.total == 3
    assert len(page.items) == 3
    assert str(page.items[0].close) == "3200.55"


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
    # Цикл обязан остановиться на странице, перешагнувшей потолок (500), а не добирать весь
    # корпус: [0, 200, 400] = 3 страницы (600 ≥ 500). Без early-stop было бы [0,200,400,600,800].
    assert client.calls == [0, 200, 400]


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


def test_fetch_optimize_parses_dto_through_wrapper() -> None:
    client = _FakeClient(total=0)

    result = fetch_optimize(_as_client(client))

    assert result.weights == {"SBER": 1.0}
    assert len(result.frontier) == 1


def test_fetch_optimize_caches_on_repeated_params() -> None:
    """Повторный вызов с теми же параметрами не дёргает солвер заново (st.cache_data)."""
    client = _FakeClient(total=0)

    fetch_optimize(_as_client(client), period_days=365)
    fetch_optimize(_as_client(client), period_days=365)

    assert client.optimize_calls == 1


def test_fetch_all_securities_pages_through_until_total() -> None:
    """Список бумаг >200 добирается страницами по 200 без HTTP 422 (per-request limit ≤ 200)."""
    client = _FakeClient(total=250)

    securities = fetch_all_securities(_as_client(client), is_active=True)

    assert len(securities) == 250
    assert {s.ticker for s in securities} == {f"TCK{i}" for i in range(250)}
    # Две страницы: [0..200), [200..250); per-request limit 200 не упирается в _MAX_LIMIT.
    assert client.calls == [0, 200]


def test_fetch_all_securities_single_page_when_under_limit() -> None:
    """Список ≤200 берётся одной страницей — без лишнего второго запроса."""
    client = _FakeClient(total=120)

    securities = fetch_all_securities(_as_client(client))

    assert len(securities) == 120
    assert client.calls == [0]


def test_fetch_candles_window_pages_through_year_window() -> None:
    """Годовое окно (>200 торговых дней) добирается страницами, не упираясь в HTTP 422."""
    client = _FakeClient(total=250)

    candles = fetch_candles_window(
        _as_client(client),
        ticker="SBER",
        date_from="2025-06-22",
        date_to="2026-06-22",
    )

    assert len(candles) == 250
    assert client.calls == [0, 200]


def test_fetch_candles_window_empty_returns_empty_list() -> None:
    """Пустое окно свечей — один запрос, пустой список (без зацикливания)."""
    client = _FakeClient(total=0)

    candles = fetch_candles_window(_as_client(client), ticker="SBER")

    assert candles == []
    assert client.calls == [0]


def test_fetch_index_window_pages_through_year_window() -> None:
    """Годовое окно индекса (>200 значений) добирается страницами — регресс «1 год» 422.

    Закрывает Critical: overview ранее слал limit=period_days (365>200 → 422). Окно задаётся
    датами, per-request limit фиксирован 200 (фейк ассертит ≤200), цикл добирает до total.
    """
    client = _FakeClient(total=250)

    values = fetch_index_window(
        _as_client(client),
        index_code="IMOEX",
        date_from="2025-06-22",
        date_to="2026-06-22",
    )

    assert len(values) == 250
    assert client.calls == [0, 200]

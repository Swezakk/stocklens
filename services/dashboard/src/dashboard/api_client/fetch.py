"""Слой кэширования fetch поверх ApiClient (DESIGN.md §8).

Streamlit перезапускает скрипт на каждое взаимодействие, поэтому каждый GET оборачивается
в ``st.cache_data``: ключ кэша — только ``(path, нормализованные params)``, токен в ключ
НЕ входит (читается внутри клиента из session_state), чтобы ротация токена не сбрасывала
кэш и auth не смешивался с ключом данных.

``ApiClient`` держится singleton-ом в ``st.cache_resource`` и передаётся в кэш-функции
underscore-параметром ``_client`` — Streamlit его не хэширует (httpx.Client нехэшируем),
а singleton гарантирует один пул соединений на сессию.

Провайдеры токена и хук 401 инжектируются в ``get_client`` тоже underscore-параметрами:
дашборд-приложение (app.py) подставляет реальные функции auth.py позже, без жёсткой
зависимости этого модуля от ещё не существующего auth.py.
"""

import streamlit as st
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel

from dashboard.api_client.client import ApiClient, OnUnauthorized, TokenProvider
from dashboard.api_client.dto import (
    BacktestResultOut,
    CandlePage,
    CollectorRunPage,
    CurrencyRatePage,
    DividendPage,
    IndexValuePage,
    KeyRatePage,
    MoversOut,
    NewsOut,
    NewsPage,
    PortfolioSummaryOut,
    SecurityPage,
)
from dashboard.settings import get_settings

#: TTL и потолки читаются один раз на импорте — значения имеют дефолты в settings.
_SETTINGS = get_settings()
_CACHE_TTL_SECONDS = _SETTINGS.cache_ttl_seconds

#: Размер страницы при добивании корпуса новостей (DESIGN §9: _MAX_LIMIT API = 200).
_NEWS_PAGE_LIMIT = 200


@st.cache_resource
def get_client(
    _token_provider: TokenProvider,
    _on_unauthorized: OnUnauthorized,
) -> ApiClient:
    """Singleton ApiClient на сессию (st.cache_resource).

    Провайдеры — underscore-параметры: Streamlit их не хэширует, поэтому ресурс
    остаётся истинным singleton-ом и не завязан на конкретную реализацию auth.
    """
    settings = get_settings()
    return ApiClient(
        base_url=settings.api_base_url,
        api_prefix=settings.api_prefix,
        timeout=settings.request_timeout_seconds,
        token_provider=_token_provider,
        on_unauthorized=_on_unauthorized,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_index(
    _client: ApiClient,
    index_code: str = "IMOEX",
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> IndexValuePage:
    """Кэш-обёртка GET /data/index."""
    return _client.get_index(
        index_code=index_code,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_currency_rates(
    _client: ApiClient,
    currency: Currency | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> CurrencyRatePage:
    """Кэш-обёртка GET /data/currency-rates."""
    return _client.get_currency_rates(
        currency=currency,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_key_rate(
    _client: ApiClient,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> KeyRatePage:
    """Кэш-обёртка GET /data/key-rate."""
    return _client.get_key_rate(
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_movers(_client: ApiClient, limit: int = 5) -> MoversOut:
    """Кэш-обёртка GET /data/movers."""
    return _client.get_movers(limit=limit)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_securities(
    _client: ApiClient,
    is_active: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> SecurityPage:
    """Кэш-обёртка GET /data/securities."""
    return _client.get_securities(is_active=is_active, limit=limit, offset=offset)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_candles(
    _client: ApiClient,
    ticker: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> CandlePage:
    """Кэш-обёртка GET /data/candles."""
    return _client.get_candles(
        ticker=ticker,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_dividends(
    _client: ApiClient,
    ticker: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> DividendPage:
    """Кэш-обёртка GET /data/dividends."""
    return _client.get_dividends(ticker=ticker, limit=limit, offset=offset)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_news(
    _client: ApiClient,
    ticker: str | None = None,
    sentiment: SentimentLabel | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> NewsPage:
    """Кэш-обёртка GET /data/news (одна страница)."""
    return _client.get_news(
        ticker=ticker,
        sentiment=sentiment,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_collector_runs(
    _client: ApiClient,
    source: str | None = None,
    status: CollectorRunStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> CollectorRunPage:
    """Кэш-обёртка GET /monitoring/runs."""
    return _client.get_collector_runs(
        source=source,
        status=status,
        limit=limit,
        offset=offset,
    )


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_portfolio_summary(
    _client: ApiClient,
    period_days: int = 365,
) -> PortfolioSummaryOut:
    """Кэш-обёртка GET /portfolio/summary."""
    return _client.get_portfolio_summary(period_days=period_days)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_backtest(_client: ApiClient, months_back: int = 12) -> BacktestResultOut:
    """Кэш-обёртка GET /portfolio/backtest."""
    return _client.get_backtest(months_back=months_back)


def fetch_news_corpus(
    _client: ApiClient,
    ticker: str | None = None,
    sentiment: SentimentLabel | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_articles: int | None = None,
) -> tuple[list[NewsOut], bool, int]:
    """Собрать period-bounded корпус новостей пагинированным циклом (DESIGN §9).

    Корпус нужен для агрегатов тональности (динамика тона, частотные слова). ``/data/news``
    пагинирован (страница ≤ 200), поэтому добираем страницами до достижения ``total`` ИЛИ
    жёсткого потолка ``max_articles`` (дефолт — news_corpus_max_articles из settings).

    Возвращает ``(articles, truncated, total)``:
    - ``articles`` — статьи, обрезанные ровно до потолка (потолок не обязан быть кратен 200);
    - ``truncated`` — True, если реальный ``total`` превышает потолок (усечение явное);
    - ``total`` — настоящий размер выборки по версии API (для честной подписи графика, §9).

    Усечение не молчаливое: вызывающая страница показывает реальный ``total`` и период.
    """
    ceiling = max_articles if max_articles is not None else _SETTINGS.news_corpus_max_articles

    articles: list[NewsOut] = []
    offset = 0
    total = 0
    while True:
        page = fetch_news(
            _client,
            ticker=ticker,
            sentiment=sentiment,
            date_from=date_from,
            date_to=date_to,
            limit=_NEWS_PAGE_LIMIT,
            offset=offset,
        )
        total = page.total
        articles.extend(page.items)
        offset += _NEWS_PAGE_LIMIT
        if not page.items or len(articles) >= total or len(articles) >= ceiling:
            break

    truncated = total > ceiling
    return articles[:ceiling], truncated, total

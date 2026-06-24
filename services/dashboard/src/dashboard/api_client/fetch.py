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

from collections.abc import Callable

import streamlit as st
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel

from dashboard.api_client.client import ApiClient, OnUnauthorized, TokenProvider
from dashboard.api_client.dto import (
    BacktestResultOut,
    CandleOut,
    CandlePage,
    CollectorRunPage,
    CurrencyRatePage,
    DividendOut,
    DividendPage,
    IndexValueOut,
    IndexValuePage,
    KeyRatePage,
    MoversOut,
    NewsOut,
    NewsPage,
    OptimizationStrategy,
    OptimizeRequest,
    OptimizeResult,
    Page,
    PortfolioSummaryOut,
    SecurityOut,
    SecurityPage,
    VolatilityForecastHistoryOut,
)
from dashboard.settings import get_settings

#: TTL и потолки читаются один раз на импорте — значения имеют дефолты в settings.
_SETTINGS = get_settings()
_CACHE_TTL_SECONDS = _SETTINGS.cache_ttl_seconds

#: Размер страницы при добивании корпуса новостей (DESIGN §9: _MAX_LIMIT API = 200).
_NEWS_PAGE_LIMIT = 200

#: Размер страницы пагинированных добор-циклов (API _MAX_LIMIT = 200: limit>200 → HTTP 422).
#: Любой per-request limit обязан быть ≤ 200, иначе FastAPI Query(le=200) отклоняет запрос
#: валидацией ДО выполнения зависимости. Поэтому крупные выборки добираются страницами.
_API_PAGE_LIMIT = 200

#: Жёсткий потолок страниц добор-цикла (защита от рантэвей-пагинации, как у корпуса новостей).
#: 50 страниц × 200 = 10000 записей — заведомо выше любой дневной выборки дашборда.
_MAX_PAGES = 50


def _collect_pages[T](fetch_page: Callable[[int], Page[T]]) -> list[T]:
    """Собрать все элементы пагинированного эндпоинта добор-циклом по offset (DESIGN §9).

    Каждая страница запрашивается с ``offset`` кратным ``_API_PAGE_LIMIT``; цикл идёт до
    исчерпания (``len(items) >= total`` или пустая страница) либо до жёсткого потолка
    ``_MAX_PAGES`` (защита от рантэвей-пагинации). Per-request limit фиксирован 200 внутри
    ``fetch_page``, поэтому ни один запрос не упрётся в HTTP 422 (API _MAX_LIMIT=200).
    """
    items: list[T] = []
    offset = 0
    for _ in range(_MAX_PAGES):
        page = fetch_page(offset)
        items.extend(page.items)
        offset += _API_PAGE_LIMIT
        if not page.items or len(items) >= page.total:
            break
    return items


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
def fetch_volatility_forecast_history(
    _client: ApiClient,
    ticker: str,
    lookback: int = 90,
) -> VolatilityForecastHistoryOut:
    """Кэш-обёртка GET /predict/volatility/history (не пагинирован — один ответ с рядом точек)."""
    return _client.get_volatility_forecast_history(ticker=ticker, lookback=lookback)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_securities(
    _client: ApiClient,
    is_active: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> SecurityPage:
    """Кэш-обёртка GET /data/securities."""
    return _client.get_securities(is_active=is_active, limit=limit, offset=offset)


def fetch_all_securities(
    _client: ApiClient,
    is_active: bool | None = None,
) -> list[SecurityOut]:
    """Собрать ВЕСЬ список бумаг пагинированным циклом (per-request limit ≤ 200, §9).

    Активная MOEX-вселенная (~200+ бумаг) не умещается в одну страницу API (_MAX_LIMIT=200),
    а запрос limit>200 отклоняется валидацией FastAPI (HTTP 422). Поэтому страницы по 200
    добираются до достижения реального ``total`` — без молчаливого усечения списка тикеров.

    Функция НЕ кэшируется (возвращает list, а не Page): кэшируется единичная страница
    ``fetch_securities``, как ``fetch_news_corpus`` композирует ``fetch_news``.
    """
    return _collect_pages(
        lambda offset: fetch_securities(
            _client, is_active=is_active, limit=_API_PAGE_LIMIT, offset=offset
        )
    )


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


def fetch_candles_window(
    _client: ApiClient,
    ticker: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[CandleOut]:
    """Собрать ВСЕ свечи окна ``[date_from, date_to]`` пагинацией (per-request limit ≤ 200).

    Годовое окно ≈ 250 торговых дней > 200, а limit>200 → HTTP 422. Сервер уже фильтрует по
    окну дат, поэтому его ``total`` равен числу свечей окна — добор страницами даёт ровно
    окно, не больше. Без кэша: композирует кэшируемую ``fetch_candles``.
    """
    return _collect_pages(
        lambda offset: fetch_candles(
            _client,
            ticker=ticker,
            date_from=date_from,
            date_to=date_to,
            limit=_API_PAGE_LIMIT,
            offset=offset,
        )
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


def fetch_dividends_all(
    _client: ApiClient,
    ticker: str | None = None,
) -> list[DividendOut]:
    """Собрать ВСЮ историю дивидендов бумаги пагинацией (per-request limit ≤ 200).

    ``/data/dividends`` не фильтруется по дате (вся история), у долго-листингованной бумаги
    выплат может быть >200, а limit>200 → HTTP 422. Окно периода применяется на странице
    после фетча (``_dividends_in_window``). Без кэша: композирует кэшируемую ``fetch_dividends``.
    """
    return _collect_pages(
        lambda offset: fetch_dividends(_client, ticker=ticker, limit=_API_PAGE_LIMIT, offset=offset)
    )


def fetch_index_window(
    _client: ApiClient,
    index_code: str = "IMOEX",
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[IndexValueOut]:
    """Собрать ВСЕ значения индекса окна ``[date_from, date_to]`` пагинацией (limit ≤ 200).

    Годовое окно индекса ≈ 250 торговых дней > 200, а limit>200 → HTTP 422. Окно задаётся
    датами (не record-count): сервер фильтрует по окну, его ``total`` равен числу значений
    окна. Без кэша: композирует кэшируемую ``fetch_index``.
    """
    return _collect_pages(
        lambda offset: fetch_index(
            _client,
            index_code=index_code,
            date_from=date_from,
            date_to=date_to,
            limit=_API_PAGE_LIMIT,
            offset=offset,
        )
    )


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


@st.cache_data(ttl=_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_optimize(
    _client: ApiClient,
    period_days: int = 365,
    strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
) -> OptimizeResult:
    """Кэш-обёртка POST /portfolio/optimize (дорогой солвер Марковица, §8).

    Ключ кэша — примитивные нормализованные параметры (``period_days`` / ``strategy``), а не
    сам ``OptimizeRequest``: pydantic-модель не хэшируема для ``st.cache_data`` без
    ``hash_funcs``. Запрос собирается внутри из этих примитивов (текущие позиции владельца,
    ``tickers=None``) — повторный rerun отдаёт кэш, а не пересчитывает солвер заново.
    """
    request = OptimizeRequest(period_days=period_days, strategy=strategy)
    return _client.optimize(request.model_dump(mode="json"))


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

"""Эндпоинты рыночных и новостных данных."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query, Request
from stocklens_core.enums import Currency, SentimentLabel

from api.core.cache import RedisCache
from api.core.db import RedisDep, SessionDep
from api.core.pagination import PageDep
from api.core.settings import ApiSettings
from api.repositories.candle import SqlCandleRepository
from api.repositories.dividend import SqlDividendRepository
from api.repositories.market_data import SqlMarketDataRepository
from api.repositories.news import SqlNewsRepository
from api.repositories.security import SqlSecurityRepository
from api.schemas.common import Page
from api.schemas.market import (
    CandleOut,
    CurrencyRateOut,
    DividendOut,
    IndexValueOut,
    KeyRateOut,
    MoversOut,
    SecurityOut,
)
from api.schemas.news import NewsOut
from api.services.candles import CandleService
from api.services.dividends import DividendService
from api.services.market_data import MarketDataService
from api.services.news import NewsService
from api.services.securities import SecurityService

router = APIRouter(prefix="/api/v1", tags=["data"])

IsActiveDep = Annotated[bool | None, Query(description="Фильтр по активности")]
TickerOptionalDep = Annotated[str | None, Query(description="Фильтр по тикеру")]
TickerRequiredDep = Annotated[str, Query(description="Тикер инструмента, например SBER")]
SentimentDep = Annotated[SentimentLabel | None, Query(description="Фильтр по тональности")]
DateFromDep = Annotated[date | None, Query(description="Начало диапазона дат")]
DateToDep = Annotated[date | None, Query(description="Конец диапазона дат")]
CurrencyDep = Annotated[Currency | None, Query(description="Фильтр по валюте")]
IndexCodeDep = Annotated[str, Query(description="Код индекса, например IMOEX")]
MoverLimitDep = Annotated[
    int,
    Query(ge=1, le=50, description="Количество позиций в списке лидеров роста/падения (1–50)"),
]


def _settings(request: Request) -> ApiSettings:
    """Получить ApiSettings из app.state (читаются один раз при старте)."""
    settings: ApiSettings = request.app.state.settings
    return settings


@router.get(
    "/data/securities",
    response_model=Page[SecurityOut],
    summary="Список ценных бумаг",
    description="Возвращает постраничный список инструментов MOEX с фильтрацией по is_active.",
)
async def list_securities(
    session: SessionDep,
    page: PageDep,
    is_active: IsActiveDep = None,
) -> Page[SecurityOut]:
    """GET /data/securities — постраничный список ценных бумаг."""
    repo = SqlSecurityRepository(session)
    service = SecurityService(repo)
    return await service.list_securities(
        is_active=is_active,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/candles",
    response_model=Page[CandleOut],
    summary="Свечи по тикеру",
    description="Возвращает постраничные свечи OHLCV для указанного тикера. Тикер обязателен.",
)
async def list_candles(
    request: Request,
    session: SessionDep,
    redis: RedisDep,
    page: PageDep,
    ticker: TickerRequiredDep,
    date_from: DateFromDep = None,
    date_to: DateToDep = None,
) -> Page[CandleOut]:
    """GET /data/candles — свечи для тикера. 404 если тикер не найден."""
    settings = _settings(request)
    cache = RedisCache(redis)
    security_repo = SqlSecurityRepository(session)
    candle_repo = SqlCandleRepository(
        session=session,
        cache=cache,
        ttl=settings.cache_ttl_candles_seconds,
    )
    service = CandleService(security_repo=security_repo, candle_repo=candle_repo)
    return await service.list_candles(
        ticker=ticker,
        date_from=date_from,
        date_to=date_to,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/news",
    response_model=Page[NewsOut],
    summary="Новости",
    description="Возвращает постраничные новости с тональностью. Тикер и sentiment — фильтры.",
)
async def list_news(
    session: SessionDep,
    page: PageDep,
    ticker: TickerOptionalDep = None,
    sentiment: SentimentDep = None,
    date_from: DateFromDep = None,
    date_to: DateToDep = None,
) -> Page[NewsOut]:
    """GET /data/news — новости с тональностью и связанными тикерами."""
    security_repo = SqlSecurityRepository(session)
    news_repo = SqlNewsRepository(session)
    service = NewsService(security_repo=security_repo, news_repo=news_repo)
    return await service.list_news(
        ticker=ticker,
        sentiment=sentiment,
        date_from=date_from,
        date_to=date_to,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/dividends",
    response_model=Page[DividendOut],
    summary="Дивиденды",
    description="Возвращает постраничные дивидендные выплаты. Тикер — опциональный фильтр.",
)
async def list_dividends(
    session: SessionDep,
    page: PageDep,
    ticker: TickerOptionalDep = None,
) -> Page[DividendOut]:
    """GET /data/dividends — дивиденды с опциональным фильтром по тикеру."""
    security_repo = SqlSecurityRepository(session)
    dividend_repo = SqlDividendRepository(session)
    service = DividendService(security_repo=security_repo, dividend_repo=dividend_repo)
    return await service.list_dividends(
        ticker=ticker,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/index",
    response_model=Page[IndexValueOut],
    summary="Значения биржевого индекса",
    description=(
        "Возвращает постраничные значения индекса (по умолчанию IMOEX), "
        "отсортированные по дате убывания."
    ),
)
async def list_index(
    session: SessionDep,
    page: PageDep,
    index_code: IndexCodeDep = "IMOEX",
    date_from: DateFromDep = None,
    date_to: DateToDep = None,
) -> Page[IndexValueOut]:
    """GET /data/index — значения биржевого индекса с пагинацией."""
    repo = SqlMarketDataRepository(session)
    service = MarketDataService(repo)
    return await service.list_index(
        index_code=index_code,
        date_from=date_from,
        date_to=date_to,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/currency-rates",
    response_model=Page[CurrencyRateOut],
    summary="Курсы валют к рублю",
    description="Возвращает постраничные курсы валют ЦБ РФ, отсортированные по дате убывания.",
)
async def list_currency_rates(
    session: SessionDep,
    page: PageDep,
    currency: CurrencyDep = None,
    date_from: DateFromDep = None,
    date_to: DateToDep = None,
) -> Page[CurrencyRateOut]:
    """GET /data/currency-rates — курсы валют с опциональной фильтрацией."""
    repo = SqlMarketDataRepository(session)
    service = MarketDataService(repo)
    return await service.list_currency_rates(
        currency=currency,
        date_from=date_from,
        date_to=date_to,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/key-rate",
    response_model=Page[KeyRateOut],
    summary="Ключевая ставка ЦБ РФ",
    description=(
        "Возвращает постраничную историю ключевой ставки, отсортированную по дате убывания."
    ),
)
async def list_key_rate(
    session: SessionDep,
    page: PageDep,
    date_from: DateFromDep = None,
    date_to: DateToDep = None,
) -> Page[KeyRateOut]:
    """GET /data/key-rate — история ключевой ставки ЦБ РФ."""
    repo = SqlMarketDataRepository(session)
    service = MarketDataService(repo)
    return await service.list_key_rate(
        date_from=date_from,
        date_to=date_to,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/data/movers",
    response_model=MoversOut,
    summary="Лидеры роста и падения",
    description=(
        "Возвращает топ-N активных бумаг с наибольшим ростом и падением цены "
        "по сравнению с предыдущим торговым днём (выходные сессии исключены)."
    ),
)
async def get_movers(
    session: SessionDep,
    limit: MoverLimitDep = 5,
) -> MoversOut:
    """GET /data/movers — лидеры роста и падения дня."""
    repo = SqlMarketDataRepository(session)
    service = MarketDataService(repo)
    return await service.get_movers(limit=limit)

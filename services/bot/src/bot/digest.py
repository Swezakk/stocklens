"""Сбор данных дайджеста: IMOEX + портфель + ближайшие отсечки + негативные новости (DESIGN §11).

Оркестрация поверх API-клиента (расчётной бизнес-логики у бота нет): тянет IMOEX, сводку
портфеля, по каждой бумаге портфеля — дивиденды (отбор окна отсечек), негативные новости
(отбор по тикерам портфеля). Все запросы идут конкурентно (asyncio.gather). Чистые отборы
(select_*) unit-тестируемы; gather_digest тестируется с замоканным клиентом.
"""

import asyncio
from collections.abc import Sequence
from datetime import date, timedelta

from stocklens_core.enums import SentimentLabel

from bot.api_client.client import ApiClient
from bot.api_client.dto import DividendOut, DividendPage, IndexValue, NewsOut
from bot.digest_model import DigestData, UpcomingDividend

#: Окно ближайших дивидендных отсечек (дней) и пределы вывода дайджеста.
_DIVIDEND_WINDOW_DAYS = 7
_NEWS_FETCH_LIMIT = 30
_DIGEST_DIVIDENDS_SHOWN = 5
_DIGEST_NEWS_SHOWN = 5

#: Запрашиваем 2 записи IMOEX (сортировка по дате убывания): [0]=вчера, [1]=позавчера.
_IMOEX_FETCH_LIMIT = 2
_IMOEX_MIN_FOR_YESTERDAY = 1
_IMOEX_MIN_FOR_PRIOR = 2


def select_upcoming(
    dividends: Sequence[DividendOut], ticker: str, today: date, within_days: int
) -> list[UpcomingDividend]:
    """Отобрать отсечки бумаги в окне [today, today+within_days] по возрастанию даты."""
    horizon = today + timedelta(days=within_days)
    chosen = [
        UpcomingDividend(
            ticker=ticker, ex_date=item.ex_date, value=item.value, currency=item.currency
        )
        for item in dividends
        if today <= item.ex_date <= horizon
    ]
    return sorted(chosen, key=lambda item: item.ex_date)


def select_portfolio_news(news: Sequence[NewsOut], tickers: frozenset[str]) -> list[NewsOut]:
    """Отобрать новости, пересекающиеся по тикерам с портфелем (по убыванию даты публикации)."""
    chosen = [article for article in news if tickers.intersection(article.tickers)]
    return sorted(chosen, key=lambda article: article.published_at, reverse=True)


async def gather_digest(client: ApiClient, today: date) -> DigestData:
    """Собрать DigestData: IMOEX + сводка портфеля + ближайшие отсечки + негативные новости."""
    summary, imoex_values = await asyncio.gather(
        client.get_portfolio_summary(),
        client.get_index(index_code="IMOEX", limit=_IMOEX_FETCH_LIMIT),
    )
    tickers = [position.ticker for position in summary.positions]
    dividends, negative_news = await asyncio.gather(
        _gather_dividends(client, tickers, today),
        _gather_news(client, frozenset(tickers)),
    )
    imoex_yesterday, imoex_prior = _split_imoex(imoex_values)
    return DigestData(
        summary=summary,
        dividends=dividends,
        negative_news=negative_news,
        imoex_yesterday=imoex_yesterday,
        imoex_prior=imoex_prior,
    )


def _split_imoex(
    values: list[IndexValue],
) -> tuple[IndexValue | None, IndexValue | None]:
    """Вернуть (вчера, позавчера) из списка, отсортированного по дате убывания."""
    yesterday = values[0] if len(values) >= _IMOEX_MIN_FOR_YESTERDAY else None
    prior = values[1] if len(values) >= _IMOEX_MIN_FOR_PRIOR else None
    return yesterday, prior


async def _gather_dividends(
    client: ApiClient, tickers: Sequence[str], today: date
) -> list[UpcomingDividend]:
    """Конкурентно запросить дивиденды по бумагам портфеля и отобрать ближайшие отсечки."""
    if not tickers:
        return []
    pages: list[DividendPage] = await asyncio.gather(
        *(client.get_dividends(ticker=ticker) for ticker in tickers)
    )
    upcoming: list[UpcomingDividend] = []
    for ticker, page in zip(tickers, pages, strict=True):
        upcoming.extend(select_upcoming(page.items, ticker, today, _DIVIDEND_WINDOW_DAYS))
    upcoming.sort(key=lambda item: item.ex_date)
    return upcoming[:_DIGEST_DIVIDENDS_SHOWN]


async def _gather_news(client: ApiClient, tickers: frozenset[str]) -> list[NewsOut]:
    """Запросить негативные новости и отобрать пересекающиеся с портфелем (топ N свежих)."""
    if not tickers:
        return []
    page = await client.get_news(sentiment=SentimentLabel.NEGATIVE, limit=_NEWS_FETCH_LIMIT)
    return select_portfolio_news(page.items, tickers)[:_DIGEST_NEWS_SHOWN]

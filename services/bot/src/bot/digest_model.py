"""Структуры данных дайджеста (DESIGN §11).

Вынесены отдельно от ``digest`` (async-сбор) и ``formatting`` (чистый рендер), чтобы оба
зависели от лёгких дата-классов, а не друг от друга. Поля — уже отобранные значения для
рендера; вся фильтрация (окно отсечек, пересечение тикеров) сделана при сборе.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from stocklens_core.enums import Currency

from bot.api_client.dto import NewsOut, PortfolioSummaryOut


@dataclass(frozen=True)
class UpcomingDividend:
    """Ближайшая дивидендная отсечка по бумаге портфеля (тикер известен из запроса)."""

    ticker: str
    ex_date: date
    value: Decimal
    currency: Currency


@dataclass(frozen=True)
class DigestData:
    """Собранные данные дайджеста: сводка портфеля, ближайшие отсечки, негативные новости."""

    summary: PortfolioSummaryOut
    dividends: Sequence[UpcomingDividend]
    negative_news: Sequence[NewsOut]

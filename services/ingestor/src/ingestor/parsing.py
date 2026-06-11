"""Парсеры ответов MOEX ISS в доменные объекты.

Все датаклассы заморожены (frozen=True) — неизменяемость гарантирует отсутствие
случайных мутаций после парсинга.

Decimal: все числовые поля получаем через Decimal(str(x)), чтобы избежать
потери точности при конвертации из JSON float.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import structlog
from stocklens_core.enums import Currency

log = structlog.get_logger(__name__)

_WEEKEND_WEEKDAY_MIN = 5

# «SUR» — устаревший код рубля в старых данных ISS, семантически == RUB.
_CURRENCY_MAP: dict[str, Currency] = {
    "RUB": Currency.RUB,
    "SUR": Currency.RUB,
    "USD": Currency.USD,
    "EUR": Currency.EUR,
    "CNY": Currency.CNY,
}


@dataclass(frozen=True)
class ParsedCandle:
    """Дневная свеча, распарсенная из блока history MOEX ISS."""

    ticker: str
    board: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    value: Decimal
    is_weekend_session: bool


@dataclass(frozen=True)
class ParsedDividend:
    """Дивидендная выплата, распарсенная из блока dividends MOEX ISS.

    ex_date (дата отсечки реестра) соответствует полю registryclosedate в ISS.
    В российской практике T+1: registryclosedate — дата фиксации реестра,
    ex_date в биржевом смысле — на один торговый день раньше. Здесь используется
    registryclosedate как есть, так как именно эта дата присутствует в ISS API.
    """

    ticker: str
    ex_date: date
    value: Decimal
    currency: Currency | None


@dataclass(frozen=True)
class ParsedSplit:
    """Дробление / консолидация акций, распарсенная из блока splits MOEX ISS."""

    ticker: str
    split_date: date
    before: int
    after: int


@dataclass(frozen=True)
class ParsedConstituent:
    """Компонент биржевого индекса (тикер + краткое название)."""

    ticker: str
    name: str


def parse_candle(row: dict[str, object]) -> ParsedCandle | None:
    """Распарсить строку блока history в ParsedCandle.

    Строки без торгов (null OHLC) пропускаются — возвращает None.
    is_weekend_session = True если день недели суббота (5) или воскресенье (6).

    Args:
        row: Словарь {column: value} из блока history ISS.

    Returns:
        ParsedCandle или None если данных нет (нет торгов).
    """
    raw_open = row.get("OPEN")
    raw_high = row.get("HIGH")
    raw_low = row.get("LOW")
    raw_close = row.get("CLOSE")

    if raw_open is None or raw_high is None or raw_low is None or raw_close is None:
        return None

    trade_date = date.fromisoformat(str(row["TRADEDATE"]))

    return ParsedCandle(
        ticker=str(row["SECID"]),
        board=str(row["BOARDID"]),
        trade_date=trade_date,
        open=Decimal(str(raw_open)),
        high=Decimal(str(raw_high)),
        low=Decimal(str(raw_low)),
        close=Decimal(str(raw_close)),
        volume=int(str(row["VOLUME"])),
        value=Decimal(str(row["VALUE"])),
        is_weekend_session=trade_date.weekday() >= _WEEKEND_WEEKDAY_MIN,
    )


def parse_dividend(ticker: str, row: dict[str, object]) -> ParsedDividend:
    """Распарсить строку блока dividends в ParsedDividend.

    Неизвестный код валюты → currency=None (вызывающий логирует warning).

    Args:
        ticker: Тикер инструмента (используется вместо secid из строки).
        row: Словарь {column: value} из блока dividends ISS.

    Returns:
        ParsedDividend с currency=None при неизвестном коде валюты.
    """
    currency_code = str(row.get("currencyid", ""))
    currency = _CURRENCY_MAP.get(currency_code)

    return ParsedDividend(
        ticker=ticker,
        ex_date=date.fromisoformat(str(row["registryclosedate"])),
        value=Decimal(str(row["value"])),
        currency=currency,
    )


def parse_split(row: dict[str, object]) -> ParsedSplit:
    """Распарсить строку блока splits в ParsedSplit.

    Args:
        row: Словарь {column: value} из блока splits ISS.

    Returns:
        ParsedSplit.
    """
    return ParsedSplit(
        ticker=str(row["secid"]),
        split_date=date.fromisoformat(str(row["tradedate"])),
        before=int(str(row["before"])),
        after=int(str(row["after"])),
    )


def parse_constituent(row: dict[str, object]) -> ParsedConstituent:
    """Распарсить строку блока analytics (IMOEX constituents) в ParsedConstituent.

    Колонки ISS: ticker (тикер), shortnames (краткое название), secids (secid).

    Args:
        row: Словарь {column: value} из блока analytics ISS.

    Returns:
        ParsedConstituent.
    """
    return ParsedConstituent(
        ticker=str(row["ticker"]),
        name=str(row["shortnames"]),
    )

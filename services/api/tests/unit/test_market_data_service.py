"""Unit-тесты MarketDataService и логики ранжирования муверов.

Фейковые репозитории реализуют MarketDataRepository Protocol без БД.
"""

from datetime import date
from decimal import Decimal

import pytest
from api.schemas.common import Page
from api.schemas.market import CurrencyRateOut, IndexValueOut, KeyRateOut, MoverOut, MoversOut
from api.services.market_data import MarketDataService
from stocklens_core.enums import Currency


def _mover(ticker: str, name: str, close: str, prev_close: str, change_pct: float) -> MoverOut:
    return MoverOut(
        ticker=ticker,
        name=name,
        close=Decimal(close),
        prev_close=Decimal(prev_close),
        change_pct=change_pct,
    )


class FakeMarketDataRepo:
    """Фейковая реализация MarketDataRepository для unit-тестов."""

    def __init__(
        self,
        index_items: list[IndexValueOut] | None = None,
        index_total: int = 0,
        currency_items: list[CurrencyRateOut] | None = None,
        currency_total: int = 0,
        key_rate_items: list[KeyRateOut] | None = None,
        key_rate_total: int = 0,
        movers_raw: list[MoverOut] | None = None,
    ) -> None:
        self._index_items = index_items or []
        self._index_total = index_total
        self._currency_items = currency_items or []
        self._currency_total = currency_total
        self._key_rate_items = key_rate_items or []
        self._key_rate_total = key_rate_total
        self._movers_raw = movers_raw or []

    async def index_series_page(
        self,
        index_code: str,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[IndexValueOut], int]:
        return self._index_items, self._index_total

    async def currency_rates_page(
        self,
        currency: Currency | None,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CurrencyRateOut], int]:
        return self._currency_items, self._currency_total

    async def key_rates_page(
        self,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[KeyRateOut], int]:
        return self._key_rate_items, self._key_rate_total

    async def active_securities_latest_closes(
        self,
        limit_per_security: int = 2,
    ) -> list[MoverOut]:
        return self._movers_raw


@pytest.mark.asyncio
async def test_list_index_returns_page() -> None:
    """list_index возвращает Page[IndexValueOut] с корректными полями."""
    items = [
        IndexValueOut(trade_date=date(2024, 1, 15), close=Decimal("3200.00")),
        IndexValueOut(trade_date=date(2024, 1, 14), close=Decimal("3180.00")),
    ]
    repo = FakeMarketDataRepo(index_items=items, index_total=10)
    service = MarketDataService(repo)

    result = await service.list_index(
        index_code="IMOEX",
        date_from=None,
        date_to=None,
        limit=50,
        offset=0,
    )

    assert isinstance(result, Page)
    assert result.total == 10
    assert len(result.items) == 2
    assert result.items[0].trade_date == date(2024, 1, 15)
    assert result.items[0].close == Decimal("3200.00")


@pytest.mark.asyncio
async def test_list_index_empty_returns_empty_page() -> None:
    """list_index с пустым репо возвращает Page с пустым items."""
    repo = FakeMarketDataRepo(index_items=[], index_total=0)
    service = MarketDataService(repo)

    result = await service.list_index("IMOEX", None, None, 50, 0)

    assert result.total == 0
    assert result.items == []


@pytest.mark.asyncio
async def test_list_currency_rates_returns_page() -> None:
    """list_currency_rates возвращает Page[CurrencyRateOut]."""
    items = [
        CurrencyRateOut(
            currency=Currency.USD, rate_date=date(2024, 1, 15), rate=Decimal("89.50")
        ),
    ]
    repo = FakeMarketDataRepo(currency_items=items, currency_total=1)
    service = MarketDataService(repo)

    result = await service.list_currency_rates(
        currency=Currency.USD, date_from=None, date_to=None, limit=50, offset=0
    )

    assert result.total == 1
    assert result.items[0].currency == Currency.USD
    assert result.items[0].rate == Decimal("89.50")


@pytest.mark.asyncio
async def test_list_currency_rates_no_filter_returns_all_currencies() -> None:
    """list_currency_rates без фильтра currency передаёт None в репо."""
    repo = FakeMarketDataRepo(currency_items=[], currency_total=0)
    service = MarketDataService(repo)

    result = await service.list_currency_rates(None, None, None, 50, 0)

    assert result.items == []


@pytest.mark.asyncio
async def test_list_key_rate_returns_page() -> None:
    """list_key_rate возвращает Page[KeyRateOut]."""
    items = [KeyRateOut(rate_date=date(2024, 1, 1), rate=Decimal("16.00"))]
    repo = FakeMarketDataRepo(key_rate_items=items, key_rate_total=5)
    service = MarketDataService(repo)

    result = await service.list_key_rate(None, None, 50, 0)

    assert result.total == 5
    assert result.items[0].rate == Decimal("16.00")


@pytest.mark.asyncio
async def test_get_movers_ranks_gainers_and_losers() -> None:
    """get_movers возвращает top-N гейнеров (desc) и лузеров (asc) по change_pct."""
    raw = [
        _mover("SBER", "Сбербанк", "300", "280", 7.14),
        _mover("GAZP", "Газпром", "150", "160", -6.25),
        _mover("LKOH", "Лукойл", "7000", "6900", 1.45),
        _mover("MGNT", "Магнит", "5000", "5200", -3.85),
        _mover("YNDX", "Яндекс", "2500", "2400", 4.17),
    ]
    repo = FakeMarketDataRepo(movers_raw=raw)
    service = MarketDataService(repo)

    result = await service.get_movers(limit=2)

    assert isinstance(result, MoversOut)
    assert len(result.gainers) == 2
    assert result.gainers[0].ticker == "SBER"
    assert result.gainers[1].ticker == "YNDX"
    assert len(result.losers) == 2
    assert result.losers[0].ticker == "GAZP"
    assert result.losers[1].ticker == "MGNT"


@pytest.mark.asyncio
async def test_get_movers_limit_larger_than_securities() -> None:
    """get_movers с limit > количества бумаг возвращает все доступные по категории."""
    raw = [
        _mover("SBER", "Сбербанк", "300", "280", 7.14),
        _mover("GAZP", "Газпром", "150", "160", -6.25),
    ]
    repo = FakeMarketDataRepo(movers_raw=raw)
    service = MarketDataService(repo)

    result = await service.get_movers(limit=5)

    assert len(result.gainers) == 1
    assert len(result.losers) == 1


@pytest.mark.asyncio
async def test_get_movers_empty_repo_returns_empty_lists() -> None:
    """get_movers с пустым репо возвращает пустые gainers и losers."""
    repo = FakeMarketDataRepo(movers_raw=[])
    service = MarketDataService(repo)

    result = await service.get_movers(limit=5)

    assert result.gainers == []
    assert result.losers == []


@pytest.mark.asyncio
async def test_get_movers_zero_change_pct_appears_in_gainers_not_losers() -> None:
    """Бумага с change_pct=0 попадает в gainers (не отрицательная), не в losers."""
    raw = [
        _mover("FLAT", "Без движения", "100", "100", 0.0),
        _mover("DOWN", "Падение", "90", "100", -10.0),
    ]
    repo = FakeMarketDataRepo(movers_raw=raw)
    service = MarketDataService(repo)

    result = await service.get_movers(limit=5)

    gainers_tickers = [m.ticker for m in result.gainers]
    losers_tickers = [m.ticker for m in result.losers]
    assert "FLAT" in gainers_tickers
    assert "FLAT" not in losers_tickers

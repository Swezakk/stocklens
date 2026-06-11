"""Unit-тесты BacktestService: равновзвешенный бэктест vs IMOEX.

Фейковые репозитории реализуют MarketHistoryRepository и PortfolioRepository
без БД. Проверяем математику и edge-cases.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from api.core.exceptions import InsufficientDataError
from api.schemas.portfolio import BacktestResultOut
from api.services.backtest import BacktestService
from stocklens_core.models.portfolio import PortfolioPosition


class FakePortfolioRepo:
    """Фейковый PortfolioRepository для unit-тестов бэктеста."""

    def __init__(self, positions: list[PortfolioPosition]) -> None:
        self._positions = positions

    async def list_positions(self) -> list[PortfolioPosition]:
        return self._positions

    async def get_position(self, security_id: int) -> PortfolioPosition | None:
        return next((p for p in self._positions if p.security_id == security_id), None)

    async def upsert_position(
        self, security_id: int, quantity: int, avg_price: Decimal, opened_at: datetime
    ) -> PortfolioPosition:
        raise NotImplementedError

    async def delete_position(self, security_id: int) -> bool:
        raise NotImplementedError


class FakeMarketHistoryRepo:
    """Фейковый MarketHistoryRepository для unit-тестов бэктеста."""

    def __init__(
        self,
        close_series: dict[int, list[tuple[date, Decimal]]] | None = None,
        dividends: dict[int, dict[date, Decimal]] | None = None,
        imoex: list[tuple[date, Decimal]] | None = None,
        key_rate: Decimal | None = None,
    ) -> None:
        self._close_series = close_series or {}
        self._dividends = dividends or {}
        self._imoex = imoex or []
        self._key_rate = key_rate

    async def close_series(
        self, security_id: int, date_from: date, date_to: date
    ) -> list[tuple[date, Decimal]]:
        return self._close_series.get(security_id, [])

    async def dividends_map(
        self, security_id: int, date_from: date, date_to: date
    ) -> dict[date, Decimal]:
        return self._dividends.get(security_id, {})

    async def imoex_series(self, date_from: date, date_to: date) -> list[tuple[date, Decimal]]:
        return self._imoex

    async def latest_key_rate(self) -> Decimal | None:
        return self._key_rate


def _make_position(security_id: int, quantity: int = 100) -> PortfolioPosition:
    """Создать PortfolioPosition без реальной ORM-сессии."""
    pos = cast(
        PortfolioPosition,
        type(
            "FakePosition",
            (),
            {
                "security_id": security_id,
                "quantity": quantity,
                "avg_price": Decimal("100.00"),
                "opened_at": datetime(2024, 1, 1, tzinfo=UTC),
            },
        )(),
    )
    return pos


def _prices(start: Decimal, n: int, step: Decimal = Decimal("1.00")) -> list[tuple[date, Decimal]]:
    """Сгенерировать ценовой ряд из n точек с шагом step."""
    dates = [date(2024, 1, d + 1) for d in range(n)]
    price = start
    result = []
    for d in dates:
        result.append((d, price))
        price += step
    return result


@pytest.mark.asyncio
async def test_backtest_returns_result_for_valid_portfolio() -> None:
    """backtest возвращает BacktestResultOut при валидном портфеле с историей."""
    prices = _prices(Decimal("100.00"), 10)
    imoex = [(d, Decimal("3000") + Decimal(str(i * 10))) for i, (d, _) in enumerate(prices)]

    portfolio_repo = FakePortfolioRepo([_make_position(1)])
    market_repo = FakeMarketHistoryRepo(
        close_series={1: prices},
        imoex=imoex,
        key_rate=Decimal("16.00"),
    )
    service = BacktestService(portfolio_repo, market_repo)

    result = await service.run(months_back=1)

    assert isinstance(result, BacktestResultOut)
    assert result.months_back == 1
    assert result.period_from < result.period_to
    assert len(result.equity_curve) >= 2
    assert all(p.portfolio > 0 for p in result.equity_curve)


@pytest.mark.asyncio
async def test_backtest_empty_portfolio_raises_insufficient_data() -> None:
    """backtest с пустым портфелем возвращает 422 InsufficientDataError."""
    portfolio_repo = FakePortfolioRepo([])
    market_repo = FakeMarketHistoryRepo()
    service = BacktestService(portfolio_repo, market_repo)

    with pytest.raises(InsufficientDataError) as exc_info:
        await service.run(months_back=3)

    assert "портфель" in exc_info.value.detail.lower() or "позиц" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_backtest_ticker_no_candles_in_window_is_skipped() -> None:
    """Тикер без свечей в окне не ломает бэктест — просто пропускается."""
    prices_sec1 = _prices(Decimal("100.00"), 10)
    imoex = [(d, Decimal("3000")) for d, _ in prices_sec1]

    portfolio_repo = FakePortfolioRepo([
        _make_position(1),
        _make_position(2),  # нет свечей для sec_id=2
    ])
    market_repo = FakeMarketHistoryRepo(
        close_series={1: prices_sec1, 2: []},
        imoex=imoex,
    )
    service = BacktestService(portfolio_repo, market_repo)

    result = await service.run(months_back=1)

    assert isinstance(result, BacktestResultOut)
    assert len(result.equity_curve) >= 2


@pytest.mark.asyncio
async def test_backtest_all_tickers_no_candles_raises_insufficient_data() -> None:
    """Если ни один тикер не имеет свечей — InsufficientDataError."""
    portfolio_repo = FakePortfolioRepo([_make_position(1)])
    market_repo = FakeMarketHistoryRepo(close_series={1: []})
    service = BacktestService(portfolio_repo, market_repo)

    with pytest.raises(InsufficientDataError):
        await service.run(months_back=3)


@pytest.mark.asyncio
async def test_backtest_equal_weight_buy_and_hold_returns_positive_for_rising_prices() -> None:
    """Равновзвешенный бэктест показывает положительную доходность на растущих ценах."""
    prices = _prices(Decimal("100.00"), 10, step=Decimal("5.00"))  # +5 каждый день
    imoex = [(d, Decimal("3000.00")) for d, _ in prices]  # flat IMOEX

    portfolio_repo = FakePortfolioRepo([_make_position(1)])
    market_repo = FakeMarketHistoryRepo(close_series={1: prices}, imoex=imoex)
    service = BacktestService(portfolio_repo, market_repo)

    result = await service.run(months_back=1)

    assert result.portfolio_return_pct > 0


@pytest.mark.asyncio
async def test_backtest_months_back_beyond_data_uses_available_history() -> None:
    """months_back > доступной истории использует всю доступную историю."""
    prices = _prices(Decimal("100.00"), 5)  # всего 5 точек
    imoex = [(d, Decimal("3000.00")) for d, _ in prices]

    portfolio_repo = FakePortfolioRepo([_make_position(1)])
    market_repo = FakeMarketHistoryRepo(close_series={1: prices}, imoex=imoex)
    service = BacktestService(portfolio_repo, market_repo)

    # months_back=120 >> 5 доступных точек, не должно падать
    result = await service.run(months_back=120)

    assert isinstance(result, BacktestResultOut)
    assert len(result.equity_curve) >= 2


@pytest.mark.asyncio
async def test_backtest_equity_curve_starts_at_one() -> None:
    """Кривая капитала начинается с 1.0 (нормированный начальный капитал)."""
    prices = _prices(Decimal("100.00"), 5)
    imoex = [(d, Decimal("3000.00")) for d, _ in prices]

    portfolio_repo = FakePortfolioRepo([_make_position(1)])
    market_repo = FakeMarketHistoryRepo(close_series={1: prices}, imoex=imoex)
    service = BacktestService(portfolio_repo, market_repo)

    result = await service.run(months_back=1)

    assert result.equity_curve[0].portfolio == pytest.approx(1.0)
    assert result.equity_curve[0].imoex == pytest.approx(1.0)

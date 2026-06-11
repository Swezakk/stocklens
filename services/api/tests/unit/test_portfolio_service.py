"""Unit-тесты PortfolioService с фиктивными репозиториями (без БД)."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from api.core.exceptions import (
    InsufficientDataError,
    PositionNotFoundError,
    SecurityNotFoundError,
)
from api.repositories.protocols import (
    MarketHistoryRepository,
    PortfolioRepository,
    SecurityRepository,
)
from api.schemas.portfolio import OptimizeRequest, PositionIn
from api.services.portfolio import PortfolioService
from stocklens_core.models.market import Security
from stocklens_core.models.portfolio import PortfolioPosition

from tests.unit.fakes import FakeSecurity


@dataclass
class FakePosition:
    """Подмена PortfolioPosition для unit-тестов."""

    id: int
    security_id: int
    quantity: int
    avg_price: Decimal
    opened_at: datetime


def _fake_security(ticker: str = "SBER", sec_id: int = 1) -> Security:
    return cast(Security, FakeSecurity(id=sec_id, ticker=ticker, name="Тест", board="TQBR"))


def _fake_position(security_id: int = 1) -> PortfolioPosition:
    return cast(
        PortfolioPosition,
        FakePosition(
            id=1,
            security_id=security_id,
            quantity=100,
            avg_price=Decimal("280.00"),
            opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )


class _SecurityFound:
    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        sec = _fake_security()
        return [sec], 1

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return _fake_security(ticker)


class _SecurityNotFound:
    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return [], 0

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return None


@dataclass
class _PortfolioRepoWithPosition:
    positions: list[PortfolioPosition] = field(default_factory=lambda: [_fake_position()])

    async def list_positions(self) -> list[PortfolioPosition]:
        return list(self.positions)

    async def get_position(self, security_id: int) -> PortfolioPosition | None:
        for p in self.positions:
            if p.security_id == security_id:
                return p
        return None

    async def upsert_position(
        self,
        security_id: int,
        quantity: int,
        avg_price: Decimal,
        opened_at: datetime,
    ) -> PortfolioPosition:
        pos = _fake_position(security_id)
        return pos

    async def delete_position(self, security_id: int) -> bool:
        return True


class _PortfolioRepoEmpty:
    async def list_positions(self) -> list[PortfolioPosition]:
        return []

    async def get_position(self, security_id: int) -> PortfolioPosition | None:
        return None

    async def upsert_position(
        self,
        security_id: int,
        quantity: int,
        avg_price: Decimal,
        opened_at: datetime,
    ) -> PortfolioPosition:
        return _fake_position(security_id)

    async def delete_position(self, security_id: int) -> bool:
        return False


class _MarketHistorySingleDay:
    """Только одна дата в истории — вызов summary должен дать InsufficientDataError."""

    async def close_series(
        self, security_id: int, date_from: date, date_to: date
    ) -> list[tuple[date, Decimal]]:
        return [(date(2024, 6, 1), Decimal("300.00"))]

    async def dividends_map(
        self, security_id: int, date_from: date, date_to: date
    ) -> dict[date, Decimal]:
        return {}

    async def imoex_series(self, date_from: date, date_to: date) -> list[tuple[date, Decimal]]:
        return [(date(2024, 6, 1), Decimal("3200.00"))]

    async def latest_key_rate(self) -> Decimal | None:
        return Decimal("16.00")


class _MarketHistoryEmpty:
    async def close_series(
        self, security_id: int, date_from: date, date_to: date
    ) -> list[tuple[date, Decimal]]:
        return []

    async def dividends_map(
        self, security_id: int, date_from: date, date_to: date
    ) -> dict[date, Decimal]:
        return {}

    async def imoex_series(self, date_from: date, date_to: date) -> list[tuple[date, Decimal]]:
        return []

    async def latest_key_rate(self) -> Decimal | None:
        return None


def _make_service(
    security_repo: SecurityRepository,
    portfolio_repo: PortfolioRepository,
    market_repo: MarketHistoryRepository,
) -> PortfolioService:
    return PortfolioService(
        security_repo=security_repo,
        portfolio_repo=portfolio_repo,
        market_history_repo=market_repo,
    )


async def test_upsert_position_resolves_ticker_and_returns_position() -> None:
    """upsert_position: тикер найден → возвращает PositionOut."""
    svc = _make_service(
        _SecurityFound(),
        _PortfolioRepoEmpty(),
        _MarketHistoryEmpty(),
    )
    data = PositionIn(
        ticker="SBER",
        quantity=100,
        avg_price=Decimal("280.00"),
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = await svc.upsert_position(data)
    assert result.ticker == "SBER"
    assert result.quantity == 100


async def test_upsert_position_raises_404_for_unknown_ticker() -> None:
    """upsert_position: тикер не найден → SecurityNotFoundError."""
    svc = _make_service(
        _SecurityNotFound(),
        _PortfolioRepoEmpty(),
        _MarketHistoryEmpty(),
    )
    data = PositionIn(
        ticker="UNKNOWN",
        quantity=10,
        avg_price=Decimal("100.00"),
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(SecurityNotFoundError) as exc_info:
        await svc.upsert_position(data)

    assert exc_info.value.ticker == "UNKNOWN"
    assert exc_info.value.status == 404


async def test_delete_position_raises_404_when_position_absent() -> None:
    """delete_position: позиции нет → PositionNotFoundError."""
    svc = _make_service(
        _SecurityFound(),
        _PortfolioRepoEmpty(),
        _MarketHistoryEmpty(),
    )
    with pytest.raises(PositionNotFoundError) as exc_info:
        await svc.delete_position("SBER")

    assert exc_info.value.ticker == "SBER"
    assert exc_info.value.status == 404


async def test_delete_position_raises_404_for_unknown_ticker() -> None:
    """delete_position: тикер неизвестен → SecurityNotFoundError."""
    svc = _make_service(
        _SecurityNotFound(),
        _PortfolioRepoWithPosition(),
        _MarketHistoryEmpty(),
    )
    with pytest.raises(SecurityNotFoundError):
        await svc.delete_position("UNKNOWN")


async def test_summary_raises_422_on_single_day_history() -> None:
    """summary: одна дата в истории → InsufficientDataError 422."""
    svc = _make_service(
        _SecurityFound(),
        _PortfolioRepoWithPosition(),
        _MarketHistorySingleDay(),
    )
    with pytest.raises(InsufficientDataError) as exc_info:
        await svc.summary(period_days=30)

    assert exc_info.value.status == 422


async def test_optimize_raises_422_on_less_than_two_tickers() -> None:
    """optimize: один тикер → InsufficientDataError 422."""
    svc = _make_service(
        _SecurityFound(),
        _PortfolioRepoEmpty(),
        _MarketHistoryEmpty(),
    )
    with pytest.raises(InsufficientDataError) as exc_info:
        await svc.optimize(OptimizeRequest(tickers=["SBER"], period_days=365))

    assert exc_info.value.status == 422

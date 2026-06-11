"""Unit-тесты CandleService с фиктивными репозиториями (без БД).

FakeSecurity / FakeCandle — dataclass с теми же атрибутами что у ORM-классов.
Pydantic from_attributes=True принимает любой объект с нужными атрибутами.
cast() в фейках — единственная точка приведения типов (test-only boundary).
"""

from datetime import date
from decimal import Decimal
from typing import cast

import pytest
from api.core.exceptions import SecurityNotFoundError
from api.repositories.protocols import CandleRepository, SecurityRepository
from api.schemas.market import CandleOut
from api.services.candles import CandleService
from stocklens_core.models.market import Security

from tests.unit.fakes import FakeSecurity


def _fake_security(ticker: str = "SBER") -> Security:
    return cast(Security, FakeSecurity(id=1, ticker=ticker, name="Тест", board="TQBR"))


def _candle_out(security_id: int = 1) -> CandleOut:
    return CandleOut(
        id=10,
        security_id=security_id,
        trade_date=date(2024, 1, 15),
        open=Decimal("280.00"),
        high=Decimal("285.00"),
        low=Decimal("278.00"),
        close=Decimal("283.00"),
        volume=1_000_000,
        value=Decimal("283000000.00"),
        is_weekend_session=False,
    )


class _SecurityRepoFound:
    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return [], 0

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return _fake_security(ticker)


class _SecurityRepoNotFound:
    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return [], 0

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return None


class _CandleRepoWithData:
    async def list_candles(
        self,
        security_id: int,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CandleOut], int]:
        return [_candle_out(security_id)], 1


class _CandleRepoEmpty:
    async def list_candles(
        self,
        security_id: int,
        date_from: date | None,
        date_to: date | None,
        limit: int,
        offset: int,
    ) -> tuple[list[CandleOut], int]:
        return [], 0


def _service(
    security_repo: SecurityRepository,
    candle_repo: CandleRepository,
) -> CandleService:
    return CandleService(security_repo=security_repo, candle_repo=candle_repo)


async def test_list_candles_returns_page_for_known_ticker() -> None:
    svc = _service(_SecurityRepoFound(), _CandleRepoWithData())
    page = await svc.list_candles("SBER", None, None, 50, 0)

    assert page.total == 1
    assert len(page.items) == 1
    assert page.items[0].security_id == 1


async def test_list_candles_returns_404_for_unknown_ticker() -> None:
    svc = _service(_SecurityRepoNotFound(), _CandleRepoEmpty())

    with pytest.raises(SecurityNotFoundError) as exc_info:
        await svc.list_candles("UNKNOWN", None, None, 50, 0)

    assert exc_info.value.ticker == "UNKNOWN"
    assert exc_info.value.status == 404


async def test_list_candles_returns_empty_page_when_no_data() -> None:
    svc = _service(_SecurityRepoFound(), _CandleRepoEmpty())
    page = await svc.list_candles("SBER", None, None, 50, 0)

    assert page.total == 0
    assert page.items == []


async def test_list_candles_passes_pagination_to_repo() -> None:
    received: dict[str, int] = {}

    class _TrackingCandleRepo:
        async def list_candles(
            self,
            security_id: int,
            date_from: date | None,
            date_to: date | None,
            limit: int,
            offset: int,
        ) -> tuple[list[CandleOut], int]:
            received["limit"] = limit
            received["offset"] = offset
            return [], 0

    svc = _service(_SecurityRepoFound(), _TrackingCandleRepo())
    await svc.list_candles("SBER", None, None, 25, 100)

    assert received["limit"] == 25
    assert received["offset"] == 100

"""Unit-тесты WatchlistService с фиктивными репозиториями (без БД)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from api.core.exceptions import WatchlistItemExistsError, WatchlistItemNotFoundError
from api.schemas.watchlist import WatchlistStatus
from api.services.watchlist import WatchlistService
from sqlalchemy.exc import IntegrityError
from stocklens_core.models.portfolio import Watchlist


@dataclass
class _FakeWatchlistItem:
    """Dataclass-замена Watchlist — совместима по атрибутам с Protocol и схемой."""

    id: int
    ticker: str
    added_at: datetime


def _make_watchlist_item(ticker: str, added_at: datetime) -> Watchlist:
    """Создать подмену Watchlist без привязки к SQLAlchemy-сессии."""
    return cast(Watchlist, _FakeWatchlistItem(id=1, ticker=ticker, added_at=added_at))


@dataclass
class _FakeWatchlistRepo:
    """Подмена WatchlistRepository для unit-тестов."""

    items: list[Watchlist] = field(default_factory=list)
    securities: set[str] = field(default_factory=set)
    candles: set[str] = field(default_factory=set)
    raise_integrity_on_add: bool = False

    async def list_items(self) -> list[Watchlist]:
        return list(self.items)

    async def get_by_ticker(self, ticker: str) -> Watchlist | None:
        for item in self.items:
            if item.ticker == ticker:
                return item
        return None

    async def add(self, ticker: str) -> Watchlist:
        if self.raise_integrity_on_add:
            raise IntegrityError("unique", {}, Exception("duplicate"))
        item = _make_watchlist_item(ticker, datetime.now(tz=UTC))
        self.items.append(item)
        return item

    async def delete(self, ticker: str) -> bool:
        before = len(self.items)
        self.items = [i for i in self.items if i.ticker != ticker]
        return len(self.items) < before

    async def security_exists(self, ticker: str) -> bool:
        return ticker in self.securities

    async def has_candles(self, ticker: str) -> bool:
        return ticker in self.candles


_GRACE = 3600
_NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)


def _fixed_clock() -> datetime:
    return _NOW


def _make_service(repo: _FakeWatchlistRepo) -> WatchlistService:
    return WatchlistService(repo=repo, grace_seconds=_GRACE, clock=_fixed_clock)


async def test_list_items_ready_when_security_and_candles_exist() -> None:
    """list_items: бумага + свечи → статус READY, has_data=True."""
    added_at = _NOW - timedelta(hours=5)
    item = _make_watchlist_item("SBER", added_at)
    repo = _FakeWatchlistRepo(
        items=[item],
        securities={"SBER"},
        candles={"SBER"},
    )
    result = await _make_service(repo).list_items()

    assert len(result) == 1
    assert result[0].status == WatchlistStatus.READY
    assert result[0].has_data is True


async def test_list_items_pending_when_added_recently_no_data() -> None:
    """list_items: только что добавлен, данных нет → статус PENDING."""
    added_at = _NOW - timedelta(seconds=10)
    item = _make_watchlist_item("POSI", added_at)
    repo = _FakeWatchlistRepo(items=[item])

    result = await _make_service(repo).list_items()

    assert result[0].status == WatchlistStatus.PENDING
    assert result[0].has_data is False


async def test_list_items_not_found_when_grace_expired_no_data() -> None:
    """list_items: grace истёк, данных нет → статус NOT_FOUND."""
    added_at = _NOW - timedelta(seconds=_GRACE + 1)
    item = _make_watchlist_item("XXXXXX", added_at)
    repo = _FakeWatchlistRepo(items=[item])

    result = await _make_service(repo).list_items()

    assert result[0].status == WatchlistStatus.NOT_FOUND
    assert result[0].has_data is False


async def test_add_item_raises_409_on_duplicate() -> None:
    """add_item: IntegrityError из репо → WatchlistItemExistsError 409."""
    repo = _FakeWatchlistRepo(raise_integrity_on_add=True)
    with pytest.raises(WatchlistItemExistsError) as exc_info:
        await _make_service(repo).add_item("SBER")
    assert exc_info.value.status == 409
    assert "SBER" in exc_info.value.detail


async def test_add_item_returns_pending_for_fresh_ticker() -> None:
    """add_item: новый тикер возвращает WatchlistItemOut в статусе PENDING."""
    repo = _FakeWatchlistRepo()
    result = await _make_service(repo).add_item("GAZP")
    assert result.ticker == "GAZP"
    assert result.status == WatchlistStatus.PENDING


async def test_remove_item_raises_404_when_absent() -> None:
    """remove_item: тикера нет в вотчлисте → WatchlistItemNotFoundError 404."""
    repo = _FakeWatchlistRepo()
    with pytest.raises(WatchlistItemNotFoundError) as exc_info:
        await _make_service(repo).remove_item("NONEXISTENT")
    assert exc_info.value.status == 404
    assert "NONEXISTENT" in exc_info.value.detail


async def test_remove_item_succeeds_when_present() -> None:
    """remove_item: тикер присутствует → удаляется без исключений."""
    item = _make_watchlist_item("LKOH", _NOW - timedelta(hours=1))
    repo = _FakeWatchlistRepo(items=[item])
    await _make_service(repo).remove_item("LKOH")
    assert len(repo.items) == 0

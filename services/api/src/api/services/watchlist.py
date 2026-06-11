"""Сервис управления списком наблюдения: добавление, удаление, статус."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError

from api.core.exceptions import WatchlistItemExistsError, WatchlistItemNotFoundError
from api.repositories.protocols import WatchlistRepository
from api.schemas.watchlist import WatchlistItemOut, WatchlistStatus

ClockFn = Callable[[], datetime]


def _utcnow() -> datetime:
    """Вернуть текущее время UTC."""
    return datetime.now(tz=UTC)


class WatchlistService:
    """Управляет списком наблюдения и вычисляет статус материализации каждого тикера.

    Зависит от Protocol-интерфейса — unit-тесты подменяют репозиторий.
    Принимает опциональный clock для детерминированного тестирования статуса PENDING.
    """

    def __init__(
        self,
        repo: WatchlistRepository,
        grace_seconds: int,
        clock: ClockFn = _utcnow,
    ) -> None:
        self._repo = repo
        self._grace = timedelta(seconds=grace_seconds)
        self._clock = clock

    async def list_items(self) -> list[WatchlistItemOut]:
        """Вернуть все элементы вотчлиста с вычисленным статусом."""
        items = await self._repo.list_items()
        result: list[WatchlistItemOut] = []
        for item in items:
            status, has_data = await self._derive_status(item.ticker, item.added_at)
            result.append(
                WatchlistItemOut(
                    ticker=item.ticker,
                    added_at=item.added_at,
                    status=status,
                    has_data=has_data,
                )
            )
        return result

    async def add_item(self, ticker: str) -> WatchlistItemOut:
        """Добавить тикер в вотчлист.

        Raises:
            WatchlistItemExistsError: тикер уже присутствует в вотчлисте.
        """
        try:
            item = await self._repo.add(ticker)
        except IntegrityError as exc:
            raise WatchlistItemExistsError(ticker) from exc
        status, has_data = await self._derive_status(item.ticker, item.added_at)
        return WatchlistItemOut(
            ticker=item.ticker,
            added_at=item.added_at,
            status=status,
            has_data=has_data,
        )

    async def remove_item(self, ticker: str) -> None:
        """Удалить тикер из вотчлиста.

        Удаление НЕ удаляет бумагу или свечи — ingestor просто перестанет включать
        её в следующую синхронизацию; deactivation произойдёт при следующем IMOEX-обновлении.

        Raises:
            WatchlistItemNotFoundError: тикер отсутствует в вотчлисте.
        """
        deleted = await self._repo.delete(ticker)
        if not deleted:
            raise WatchlistItemNotFoundError(ticker)

    async def _derive_status(
        self,
        ticker: str,
        added_at: datetime,
    ) -> tuple[WatchlistStatus, bool]:
        """Вычислить статус и флаг наличия данных для тикера.

        Returns:
            (WatchlistStatus, has_data) — has_data=True только при READY.
        """
        security_ok = await self._repo.security_exists(ticker)
        candles_ok = await self._repo.has_candles(ticker) if security_ok else False

        if security_ok and candles_ok:
            return WatchlistStatus.READY, True

        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        added = added_at if added_at.tzinfo is not None else added_at.replace(tzinfo=UTC)

        if now - added < self._grace:
            return WatchlistStatus.PENDING, False

        return WatchlistStatus.NOT_FOUND, False

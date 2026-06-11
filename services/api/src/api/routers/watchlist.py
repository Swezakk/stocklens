"""Эндпоинты управления списком наблюдения."""

from fastapi import APIRouter
from starlette.responses import Response

from api.core.db import SessionDep
from api.core.settings import get_settings
from api.repositories.watchlist import SqlWatchlistRepository
from api.schemas.watchlist import WatchlistItemIn, WatchlistItemOut
from api.services.watchlist import WatchlistService

router = APIRouter(prefix="/api/v1/watchlist", tags=["watchlist"])


def _service(session: SessionDep) -> WatchlistService:
    """Собрать WatchlistService из зависимостей запроса."""
    settings = get_settings()
    return WatchlistService(
        repo=SqlWatchlistRepository(session),
        grace_seconds=settings.watchlist_grace_seconds,
    )


@router.get(
    "",
    response_model=list[WatchlistItemOut],
    summary="Список наблюдения",
    description="Возвращает все тикеры вотчлиста с вычисленным статусом материализации.",
)
async def list_watchlist(session: SessionDep) -> list[WatchlistItemOut]:
    """GET /watchlist — все элементы вотчлиста."""
    return await _service(session).list_items()


@router.post(
    "",
    response_model=WatchlistItemOut,
    status_code=201,
    summary="Добавить тикер в список наблюдения",
    description=(
        "Добавляет тикер. ingestor материализует бумагу при следующей синхронизации. "
        "409 если тикер уже в вотчлисте."
    ),
)
async def add_watchlist_item(session: SessionDep, body: WatchlistItemIn) -> WatchlistItemOut:
    """POST /watchlist — добавить тикер."""
    return await _service(session).add_item(body.ticker)


@router.delete(
    "/{ticker}",
    status_code=204,
    summary="Удалить тикер из списка наблюдения",
    description=(
        "Удаляет тикер из вотчлиста. "
        "Бумага и свечи в БД остаются — ingestor перестанет собирать при след. синхронизации. "
        "404 если тикер не найден в вотчлисте."
    ),
)
async def delete_watchlist_item(session: SessionDep, ticker: str) -> Response:
    """DELETE /watchlist/{ticker} — удалить тикер, 204 при успехе."""
    await _service(session).remove_item(ticker.strip().upper())
    return Response(status_code=204)

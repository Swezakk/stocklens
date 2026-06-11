"""Эндпоинты управления портфелем: позиции, сводка, оптимизация, бэктест."""

from typing import Annotated

from fastapi import APIRouter, Query
from starlette.responses import Response

from api.core.db import SessionDep
from api.repositories.market_history import SqlMarketHistoryRepository
from api.repositories.portfolio import SqlPortfolioRepository
from api.repositories.security import SqlSecurityRepository
from api.schemas.portfolio import (
    BacktestResultOut,
    OptimizeRequest,
    OptimizeResult,
    PortfolioSummaryOut,
    PositionIn,
    PositionOut,
)
from api.services.backtest import BacktestService
from api.services.portfolio import PortfolioService

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])

PeriodDaysDep = Annotated[
    int,
    Query(ge=2, le=3650, description="Глубина истории в днях (2–3650)"),
]
MonthsBackDep = Annotated[
    int,
    Query(ge=1, le=120, description="Глубина бэктеста в месяцах (1–120)"),
]


def _service(session: SessionDep) -> PortfolioService:
    """Собрать PortfolioService из зависимостей запроса."""
    return PortfolioService(
        security_repo=SqlSecurityRepository(session),
        portfolio_repo=SqlPortfolioRepository(session),
        market_history_repo=SqlMarketHistoryRepository(session),
    )


@router.get(
    "/positions",
    response_model=list[PositionOut],
    summary="Список позиций портфеля",
    description="Возвращает все позиции с текущей рыночной оценкой и нереализованным PnL.",
)
async def list_positions(session: SessionDep) -> list[PositionOut]:
    """GET /portfolio/positions — все позиции."""
    return await _service(session).list_positions()


@router.post(
    "/positions",
    response_model=PositionOut,
    summary="Создать или обновить позицию",
    description=(
        "Upsert позиции по тикеру: если позиция по тикеру уже есть — обновляет, "
        "иначе создаёт. 404 если тикер неизвестен."
    ),
)
async def upsert_position(session: SessionDep, body: PositionIn) -> PositionOut:
    """POST /portfolio/positions — upsert позиции."""
    return await _service(session).upsert_position(body)


@router.delete(
    "/positions/{ticker}",
    status_code=204,
    summary="Удалить позицию",
    description="Удалить позицию по тикеру. 404 если тикер неизвестен или позиции нет.",
)
async def delete_position(session: SessionDep, ticker: str) -> Response:
    """DELETE /portfolio/positions/{ticker} — удалить позицию, 204 при успехе."""
    await _service(session).delete_position(ticker)
    return Response(status_code=204)


@router.get(
    "/summary",
    response_model=PortfolioSummaryOut,
    summary="Сводка портфеля с риск-метриками",
    description=(
        "Вычисляет риск-метрики (Шарп, max drawdown) и сравнивает с IMOEX "
        "за указанный период. 422 если истории котировок недостаточно."
    ),
)
async def portfolio_summary(
    session: SessionDep,
    period_days: PeriodDaysDep = 365,
) -> PortfolioSummaryOut:
    """GET /portfolio/summary — сводка с риск-метриками."""
    return await _service(session).summary(period_days)


@router.post(
    "/optimize",
    response_model=OptimizeResult,
    summary="Оптимизация портфеля (Марковиц)",
    description=(
        "Возвращает веса max-Sharpe и min-vol портфелей, эффективную границу и Шарп IMOEX. "
        "422 если менее 2 тикеров или истории котировок недостаточно."
    ),
)
async def optimize_portfolio(
    session: SessionDep,
    body: OptimizeRequest,
) -> OptimizeResult:
    """POST /portfolio/optimize — оптимизация Марковица."""
    return await _service(session).optimize(body)


@router.get(
    "/backtest",
    response_model=BacktestResultOut,
    summary="Бэктест равновзвешенного портфеля",
    description=(
        "Симулирует равновзвешенный buy-and-hold для текущих позиций портфеля "
        "за months_back месяцев и сравнивает с IMOEX. "
        "422 если портфель пуст или данных котировок недостаточно."
    ),
)
async def backtest(
    session: SessionDep,
    months_back: MonthsBackDep = 12,
) -> BacktestResultOut:
    """GET /portfolio/backtest — бэктест vs IMOEX."""
    svc = BacktestService(
        portfolio_repo=SqlPortfolioRepository(session),
        market_history_repo=SqlMarketHistoryRepository(session),
    )
    return await svc.run(months_back=months_back)

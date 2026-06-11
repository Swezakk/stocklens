"""Сервис управления портфелем: CRUD позиций, сводка, оптимизация Марковица.

CPU-bound аналитика выполняется через run_in_threadpool (asyncio-friendly).
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial

import numpy as np
import pandas as pd
from pypfopt.exceptions import OptimizationError
from starlette.concurrency import run_in_threadpool
from stocklens_core.models.portfolio import PortfolioPosition

from api.analytics import optimization as opt
from api.analytics.returns import total_returns
from api.analytics.risk import equity_curve, max_drawdown, sharpe_ratio
from api.core.exceptions import (
    InsufficientDataError,
    InvalidStrategyParamsError,
    PositionNotFoundError,
    SecurityNotFoundError,
)
from api.repositories.protocols import (
    MarketHistoryRepository,
    PortfolioRepository,
    SecurityRepository,
)
from api.schemas.portfolio import (
    FrontierPoint,
    OptimizationStrategy,
    OptimizeRequest,
    OptimizeResult,
    PortfolioSummaryOut,
    PositionIn,
    PositionOut,
)

_INSUFFICIENT_HISTORY_MSG = (
    "Недостаточно истории котировок для расчёта риск-метрик: "
    "нужно не менее 2 совмещённых торговых дней"
)
_MIN_ALIGNED_DAYS = 2
_MIN_TICKERS_FOR_OPTIMIZE = 2
_MIN_VALID_IMOEX_PRICES = 3


class PortfolioService:
    """Управляет позициями портфеля и вычисляет риск-метрики.

    Зависит от Protocol-интерфейсов — unit-тесты подменяют реализации.
    """

    def __init__(
        self,
        security_repo: SecurityRepository,
        portfolio_repo: PortfolioRepository,
        market_history_repo: MarketHistoryRepository,
    ) -> None:
        self._security_repo = security_repo
        self._portfolio_repo = portfolio_repo
        self._market_history_repo = market_history_repo

    async def list_positions(self) -> list[PositionOut]:
        """Вернуть все позиции с текущей рыночной оценкой."""
        positions = await self._portfolio_repo.list_positions()
        return [await self._enrich_position(pos) for pos in positions]

    async def upsert_position(self, data: PositionIn) -> PositionOut:
        """Создать или обновить позицию.

        Raises:
            SecurityNotFoundError: если тикер не найден в БД.
        """
        security = await self._security_repo.get_by_ticker(data.ticker)
        if security is None:
            raise SecurityNotFoundError(data.ticker)

        position = await self._portfolio_repo.upsert_position(
            security_id=security.id,
            quantity=data.quantity,
            avg_price=data.avg_price,
            opened_at=data.opened_at,
        )
        return await self._enrich_position(position, ticker=data.ticker)

    async def delete_position(self, ticker: str) -> None:
        """Удалить позицию по тикеру.

        Raises:
            SecurityNotFoundError: если тикер не известен в БД.
            PositionNotFoundError: если позиция по тикеру отсутствует в портфеле.
        """
        security = await self._security_repo.get_by_ticker(ticker)
        if security is None:
            raise SecurityNotFoundError(ticker)

        deleted = await self._portfolio_repo.delete_position(security.id)
        if not deleted:
            raise PositionNotFoundError(ticker)

    async def summary(self, period_days: int) -> PortfolioSummaryOut:
        """Вычислить сводку портфеля с риск-метриками за указанный период.

        Raises:
            InsufficientDataError: если менее 2 совмещённых торговых дней.
        """
        positions = await self._portfolio_repo.list_positions()
        date_to = datetime.now(tz=UTC).date()
        date_from = date_to - timedelta(days=period_days)

        annual_rate = await self._resolve_annual_rate()

        position_out_list, security_ids = await self._build_position_list(positions)

        if not positions:
            return self._empty_summary(position_out_list, date_from, date_to)

        close_series_by_id, divs_by_id = await self._fetch_market_data(
            security_ids, date_from, date_to
        )
        imoex_series = await self._market_history_repo.imoex_series(date_from, date_to)

        try:
            portfolio_returns, common_dates = await run_in_threadpool(
                partial(
                    _compute_portfolio_returns,
                    close_series_by_id=close_series_by_id,
                    divs_by_id=divs_by_id,
                    positions=positions,
                )
            )
        except ValueError as exc:
            raise InsufficientDataError(str(exc)) from exc

        if len(portfolio_returns) < _MIN_ALIGNED_DAYS:
            raise InsufficientDataError(_INSUFFICIENT_HISTORY_MSG)

        imoex_returns = await run_in_threadpool(
            partial(
                _compute_benchmark_returns,
                imoex_series=imoex_series,
                common_dates=common_dates,
            )
        )

        sharpe_val, mdd_val = await run_in_threadpool(
            partial(_compute_risk_metrics, returns=portfolio_returns, annual_rate=annual_rate)
        )

        imoex_sharpe, imoex_mdd = await run_in_threadpool(
            partial(_compute_risk_metrics, returns=imoex_returns, annual_rate=annual_rate)
        )

        total_value, total_cost, pnl, return_pct, imoex_return_pct = _compute_portfolio_totals(
            position_out_list, portfolio_returns, imoex_returns
        )

        return PortfolioSummaryOut(
            positions=position_out_list,
            total_value=total_value,
            total_cost=total_cost,
            total_unrealized_pnl=pnl,
            portfolio_return_pct=return_pct,
            imoex_return_pct=imoex_return_pct,
            sharpe=sharpe_val,
            max_drawdown=mdd_val,
            imoex_sharpe=imoex_sharpe,
            imoex_max_drawdown=imoex_mdd,
            period_from=date_from,
            period_to=date_to,
        )

    async def optimize(self, request: OptimizeRequest) -> OptimizeResult:
        """Оптимизировать портфель по выбранной стратегии Марковица.

        Raises:
            InsufficientDataError: менее 2 тикеров или недостаточно истории.
            InvalidStrategyParamsError: отсутствует обязательный параметр стратегии
                или целевое значение недостижимо (нефeasible).
        """
        tickers = request.tickers
        if tickers is None:
            positions = await self._portfolio_repo.list_positions()
            security_ids = [p.security_id for p in positions]
            id_to_ticker = await self._resolve_tickers(security_ids)
            tickers = list(id_to_ticker.values())

        if len(tickers) < _MIN_TICKERS_FOR_OPTIMIZE:
            raise InsufficientDataError("Для оптимизации портфеля нужно не менее 2 тикеров")

        date_to = datetime.now(tz=UTC).date()
        date_from = date_to - timedelta(days=request.period_days)
        annual_rate = await self._resolve_annual_rate()

        prices_data: dict[str, list[tuple[date, Decimal]]] = {}
        for ticker in tickers:
            security = await self._security_repo.get_by_ticker(ticker)
            if security is None:
                raise SecurityNotFoundError(ticker)
            series = await self._market_history_repo.close_series(security.id, date_from, date_to)
            prices_data[ticker] = series

        imoex_series = await self._market_history_repo.imoex_series(date_from, date_to)

        try:
            result = await run_in_threadpool(
                partial(
                    _run_optimization,
                    prices_data=prices_data,
                    imoex_series=imoex_series,
                    annual_rate=annual_rate,
                    strategy=request.strategy,
                    target_return=request.target_return,
                    target_volatility=request.target_volatility,
                    risk_aversion=request.risk_aversion,
                )
            )
        except ValueError as exc:
            msg = str(exc)
            if "не менее 2" in msg:
                raise InsufficientDataError(msg) from exc
            raise InvalidStrategyParamsError(msg) from exc
        except OptimizationError as exc:
            raise InvalidStrategyParamsError(
                f"Целевое значение недостижимо для выбранной стратегии: {exc}"
            ) from exc
        return result

    async def _enrich_position(
        self,
        position: PortfolioPosition,
        ticker: str | None = None,
    ) -> PositionOut:
        """Обогатить позицию текущей ценой и PnL."""
        if ticker is None:
            id_to_ticker = await self._resolve_tickers([position.security_id])
            ticker = id_to_ticker.get(position.security_id, "UNKNOWN")

        today = datetime.now(tz=UTC).date()
        yesterday = today - timedelta(days=1)
        series = await self._market_history_repo.close_series(
            position.security_id, yesterday - timedelta(days=7), today
        )

        current_price: Decimal | None = series[-1][1] if series else None
        current_value: Decimal | None = (
            current_price * position.quantity if current_price is not None else None
        )
        unrealized_pnl: Decimal | None = (
            (current_price - position.avg_price) * position.quantity
            if current_price is not None
            else None
        )

        return PositionOut(
            ticker=ticker,
            quantity=position.quantity,
            avg_price=position.avg_price,
            opened_at=position.opened_at,
            current_price=current_price,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
        )

    async def _build_position_list(
        self, positions: list[PortfolioPosition]
    ) -> tuple[list[PositionOut], list[int]]:
        """Построить список PositionOut с разрешением тикеров."""
        security_ids = [p.security_id for p in positions]
        id_to_ticker = await self._resolve_tickers(security_ids)

        position_out_list: list[PositionOut] = []
        for pos in positions:
            ticker = id_to_ticker.get(pos.security_id, "UNKNOWN")
            position_out = await self._enrich_position(pos, ticker=ticker)
            position_out_list.append(position_out)

        return position_out_list, security_ids

    async def _resolve_tickers(self, security_ids: list[int]) -> dict[int, str]:
        """Разрешить security_id → ticker через SecurityRepository."""
        result: dict[int, str] = {}
        for sec_id in security_ids:
            securities, _ = await self._security_repo.list_securities(None, 10000, 0)
            for sec in securities:
                if sec.id == sec_id:
                    result[sec_id] = sec.ticker
                    break
        return result

    async def _resolve_annual_rate(self) -> float:
        """Получить годовую безрисковую ставку как дробь. 0.0 если данных нет."""
        key_rate = await self._market_history_repo.latest_key_rate()
        return float(key_rate) / 100.0 if key_rate is not None else 0.0

    async def _fetch_market_data(
        self,
        security_ids: list[int],
        date_from: date,
        date_to: date,
    ) -> tuple[dict[int, list[tuple[date, Decimal]]], dict[int, dict[date, Decimal]]]:
        """Загрузить свечи и дивиденды для всех позиций."""
        close_series_by_id: dict[int, list[tuple[date, Decimal]]] = {}
        divs_by_id: dict[int, dict[date, Decimal]] = {}
        for sec_id in security_ids:
            close_series_by_id[sec_id] = await self._market_history_repo.close_series(
                sec_id, date_from, date_to
            )
            divs_by_id[sec_id] = await self._market_history_repo.dividends_map(
                sec_id, date_from, date_to
            )
        return close_series_by_id, divs_by_id

    def _empty_summary(
        self,
        positions: list[PositionOut],
        date_from: date,
        date_to: date,
    ) -> PortfolioSummaryOut:
        """Сводка для пустого портфеля."""
        return PortfolioSummaryOut(
            positions=positions,
            total_value=Decimal("0"),
            total_cost=Decimal("0"),
            total_unrealized_pnl=Decimal("0"),
            portfolio_return_pct=0.0,
            imoex_return_pct=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            imoex_sharpe=0.0,
            imoex_max_drawdown=0.0,
            period_from=date_from,
            period_to=date_to,
        )


def _compute_portfolio_returns(
    close_series_by_id: dict[int, list[tuple[date, Decimal]]],
    divs_by_id: dict[int, dict[date, Decimal]],
    positions: list[PortfolioPosition],
) -> tuple[np.ndarray, list[date]]:
    """Вычислить взвешенные дневные доходности портфеля.

    Позиции без данных в заданном периоде исключаются из расчёта —
    не ломают пересечение дат для остальных.
    Веса — доля рыночной стоимости каждой позиции на каждую дату.
    Возвращает (portfolio_returns, common_dates).
    """
    if not close_series_by_id:
        return np.array([]), []

    non_empty = {k: v for k, v in close_series_by_id.items() if len(v) >= _MIN_ALIGNED_DAYS}
    if not non_empty:
        return np.array([]), []

    series_list = list(non_empty.values())
    date_sets = [{d for d, _ in s} for s in series_list]
    common_dates_set: set[date] = date_sets[0].copy()
    for ds in date_sets[1:]:
        common_dates_set &= ds
    common_dates = sorted(common_dates_set)

    close_series_by_id = non_empty

    if len(common_dates) < _MIN_ALIGNED_DAYS:
        return np.array([]), common_dates

    sec_ids = list(close_series_by_id.keys())
    prices_matrix: dict[int, np.ndarray] = {}
    for sec_id in sec_ids:
        date_map = {d: float(p) for d, p in close_series_by_id[sec_id]}
        prices_matrix[sec_id] = np.array([date_map[d] for d in common_dates])

    qty_map = {p.security_id: p.quantity for p in positions}
    portfolio_returns = np.zeros(len(common_dates) - 1)

    for sec_id in sec_ids:
        prices = prices_matrix[sec_id]
        qty = qty_map.get(sec_id, 0)
        div_by_date = divs_by_id.get(sec_id, {})
        div_by_index: dict[int, Decimal] = {
            common_dates.index(d): v for d, v in div_by_date.items() if d in common_dates_set
        }
        returns = total_returns(prices, div_by_index)
        weights = prices[:-1] * qty
        total_weight = sum(prices_matrix[sid][:-1] * qty_map.get(sid, 0) for sid in sec_ids)
        weight_frac = np.where(total_weight > 0, weights / total_weight, 1.0 / len(sec_ids))
        portfolio_returns += weight_frac * returns

    return portfolio_returns, common_dates


def _compute_benchmark_returns(
    imoex_series: list[tuple[date, Decimal]],
    common_dates: list[date],
) -> np.ndarray:
    """Извлечь доходности IMOEX на общих датах портфеля."""
    if len(common_dates) < _MIN_ALIGNED_DAYS or not imoex_series:
        return np.zeros(max(0, len(common_dates) - 1))

    imoex_map = {d: float(p) for d, p in imoex_series}
    imoex_prices = np.array([imoex_map.get(d, float("nan")) for d in common_dates])
    valid_mask = ~np.isnan(imoex_prices)

    if valid_mask.sum() < _MIN_ALIGNED_DAYS:
        return np.zeros(len(common_dates) - 1)

    imoex_filled = imoex_prices.copy()
    for i in range(1, len(imoex_filled)):
        if np.isnan(imoex_filled[i]):
            imoex_filled[i] = imoex_filled[i - 1]

    result: np.ndarray = np.diff(imoex_filled) / imoex_filled[:-1]
    return result


def _compute_risk_metrics(
    returns: np.ndarray,
    annual_rate: float,
) -> tuple[float, float]:
    """Вычислить коэффициент Шарпа и максимальную просадку."""
    if len(returns) < _MIN_ALIGNED_DAYS:
        return 0.0, 0.0
    try:
        sharpe_val = sharpe_ratio(returns, annual_rate)
    except ValueError:
        sharpe_val = 0.0
    curve = equity_curve(returns)
    mdd = max_drawdown(curve)
    return sharpe_val, mdd


def _compute_portfolio_totals(
    positions: list[PositionOut],
    portfolio_returns: np.ndarray,
    imoex_returns: np.ndarray,
) -> tuple[Decimal, Decimal, Decimal, float, float]:
    """Вычислить совокупные показатели портфеля."""
    total_value = sum(
        (p.current_value for p in positions if p.current_value is not None),
        Decimal("0"),
    )
    total_cost = sum(
        (p.avg_price * p.quantity for p in positions),
        Decimal("0"),
    )
    total_pnl = sum(
        (p.unrealized_pnl for p in positions if p.unrealized_pnl is not None),
        Decimal("0"),
    )

    portfolio_return_pct = float(np.prod(1.0 + portfolio_returns) - 1.0) * 100.0
    imoex_return_pct = (
        float(np.prod(1.0 + imoex_returns) - 1.0) * 100.0 if len(imoex_returns) > 0 else 0.0
    )

    return total_value, total_cost, total_pnl, portfolio_return_pct, imoex_return_pct


def _run_optimization(
    prices_data: dict[str, list[tuple[date, Decimal]]],
    imoex_series: list[tuple[date, Decimal]],
    annual_rate: float,
    strategy: OptimizationStrategy = OptimizationStrategy.MAX_SHARPE,
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_aversion: float | None = None,
) -> OptimizeResult:
    """Синхронная обёртка для CPU-bound оптимизации (вызывается через threadpool).

    Raises:
        ValueError: недостаточно тикеров/дат или отсутствует обязательный параметр стратегии.
        pypfopt.OptimizationError: целевое значение нефeasible — caller маппирует в 422.
    """
    if len(prices_data) < _MIN_TICKERS_FOR_OPTIMIZE:
        raise ValueError("Для оптимизации портфеля нужно не менее 2 тикеров")

    common_dates_set: set[date] | None = None
    for series in prices_data.values():
        dates = {d for d, _ in series}
        if common_dates_set is None:
            common_dates_set = dates
        else:
            common_dates_set &= dates

    if common_dates_set is None or len(common_dates_set) < _MIN_ALIGNED_DAYS:
        raise ValueError(
            "Недостаточно истории котировок для оптимизации: нужно не менее 2 совмещённых дат"
        )

    common_dates = sorted(common_dates_set)
    data_dict: dict[str, list[float]] = {}
    for ticker, series in prices_data.items():
        date_map = {d: float(p) for d, p in series}
        data_dict[ticker] = [date_map[d] for d in common_dates]

    prices_df = pd.DataFrame(data_dict, index=pd.DatetimeIndex(common_dates))

    weights = opt.build_weights_for_strategy(
        prices_df,
        strategy=strategy,
        annual_rate=annual_rate,
        target_return=target_return,
        target_volatility=target_volatility,
        risk_aversion=risk_aversion,
    )
    ret, vol, sharpe_val = opt.compute_portfolio_performance(prices_df, weights, annual_rate)

    frontier_raw = opt.compute_frontier_points(prices_df)
    frontier = [FrontierPoint(volatility=v, expected_return=r) for v, r in frontier_raw]

    n = len(prices_data)
    equal_w = dict.fromkeys(prices_data, 1.0 / n)
    _, _, equal_sharpe = opt.compute_portfolio_performance(prices_df, equal_w, annual_rate)
    imoex_sharpe_val = _compute_imoex_sharpe(imoex_series, annual_rate)

    return OptimizeResult(
        strategy=strategy,
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe_val,
        frontier=frontier,
        equal_weight_sharpe=equal_sharpe,
        imoex_sharpe=imoex_sharpe_val,
    )


def _compute_imoex_sharpe(
    imoex_series: list[tuple[date, Decimal]],
    annual_rate: float,
) -> float:
    """Вычислить Шарп для IMOEX как бенчмарк."""
    if len(imoex_series) < _MIN_VALID_IMOEX_PRICES:
        return 0.0
    prices = np.array([float(p) for _, p in imoex_series])
    returns = np.diff(prices) / prices[:-1]
    try:
        return sharpe_ratio(returns, annual_rate)
    except ValueError:
        return 0.0

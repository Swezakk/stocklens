"""Сервис бэктеста равновзвешенного портфеля vs IMOEX.

Берёт текущие позиции портфеля, симулирует buy-and-hold с равными весами
за months_back месяцев, вычисляет риск-метрики и кривую капитала.

Архитектурные решения:
- Тикер без свечей в окне пропускается (не ломает расчёт остальных).
- months_back > доступной истории → используется вся доступная история.
- Пустой портфель → InsufficientDataError (HTTP 422).
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial

import numpy as np
from starlette.concurrency import run_in_threadpool

from api.analytics.returns import total_returns
from api.analytics.risk import equity_curve, max_drawdown, sharpe_ratio
from api.core.exceptions import InsufficientDataError
from api.repositories.protocols import MarketHistoryRepository, PortfolioRepository
from api.schemas.portfolio import BacktestResultOut, EquityPointOut

_MIN_ALIGNED_DAYS = 2
_MONTHS_TO_DAYS_APPROX = 30


class BacktestService:
    """Вычисляет результаты гипотетического бэктеста на текущем составе портфеля."""

    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        market_history_repo: MarketHistoryRepository,
    ) -> None:
        self._portfolio_repo = portfolio_repo
        self._market_history_repo = market_history_repo

    async def run(self, months_back: int) -> BacktestResultOut:
        """Запустить бэктест за months_back месяцев.

        Raises:
            InsufficientDataError: пустой портфель или ни одна бумага не имеет данных.
        """
        positions = await self._portfolio_repo.list_positions()
        if not positions:
            raise InsufficientDataError("Бэктест невозможен: портфель не содержит позиций")

        date_to = datetime.now(tz=UTC).date()
        date_from = date_to - timedelta(days=months_back * _MONTHS_TO_DAYS_APPROX)

        security_ids = [p.security_id for p in positions]
        close_series_by_id: dict[int, list[tuple[date, Decimal]]] = {}
        divs_by_id: dict[int, dict[date, Decimal]] = {}

        for sec_id in security_ids:
            close_series_by_id[sec_id] = await self._market_history_repo.close_series(
                sec_id, date_from, date_to
            )
            divs_by_id[sec_id] = await self._market_history_repo.dividends_map(
                sec_id, date_from, date_to
            )

        imoex_raw = await self._market_history_repo.imoex_series(date_from, date_to)
        key_rate = await self._market_history_repo.latest_key_rate()
        annual_rate = float(key_rate) / 100.0 if key_rate is not None else 0.0

        try:
            portfolio_returns, common_dates = await run_in_threadpool(
                partial(
                    _compute_equal_weight_returns,
                    close_series_by_id=close_series_by_id,
                    divs_by_id=divs_by_id,
                )
            )
        except ValueError as exc:
            raise InsufficientDataError(str(exc)) from exc

        if len(portfolio_returns) < _MIN_ALIGNED_DAYS:
            raise InsufficientDataError(
                "Недостаточно истории котировок для бэктеста: нужно не менее 2 торговых дней"
            )

        imoex_returns = await run_in_threadpool(
            partial(
                _compute_imoex_returns,
                imoex_series=imoex_raw,
                common_dates=common_dates,
            )
        )

        p_sharpe, p_mdd, p_curve = await run_in_threadpool(
            partial(_compute_metrics_and_curve, returns=portfolio_returns, annual_rate=annual_rate)
        )
        i_sharpe, i_mdd, i_curve = await run_in_threadpool(
            partial(_compute_metrics_and_curve, returns=imoex_returns, annual_rate=annual_rate)
        )

        portfolio_return_pct = float(np.prod(1.0 + portfolio_returns) - 1.0) * 100.0
        imoex_return_pct = float(np.prod(1.0 + imoex_returns) - 1.0) * 100.0

        equity_points = [
            EquityPointOut(
                date=common_dates[i] if i < len(common_dates) else common_dates[-1],
                portfolio=float(p_curve[i]),
                imoex=float(i_curve[i]),
            )
            for i in range(len(p_curve))
        ]

        actual_from = common_dates[0] if common_dates else date_from
        actual_to = common_dates[-1] if common_dates else date_to

        return BacktestResultOut(
            months_back=months_back,
            period_from=actual_from,
            period_to=actual_to,
            portfolio_return_pct=portfolio_return_pct,
            imoex_return_pct=imoex_return_pct,
            portfolio_sharpe=p_sharpe,
            imoex_sharpe=i_sharpe,
            portfolio_max_drawdown=p_mdd,
            imoex_max_drawdown=i_mdd,
            equity_curve=equity_points,
        )


def _compute_equal_weight_returns(
    close_series_by_id: dict[int, list[tuple[date, Decimal]]],
    divs_by_id: dict[int, dict[date, Decimal]],
) -> tuple[np.ndarray, list[date]]:
    """Вычислить равновзвешенные дневные доходности портфеля.

    Бумаги с менее чем 2 точками пропускаются.
    Возвращает (returns, common_dates).
    """
    non_empty = {k: v for k, v in close_series_by_id.items() if len(v) >= _MIN_ALIGNED_DAYS}
    if not non_empty:
        return np.array([]), []

    date_sets = [{d for d, _ in s} for s in non_empty.values()]
    common_dates_set: set[date] = date_sets[0].copy()
    for ds in date_sets[1:]:
        common_dates_set &= ds
    common_dates = sorted(common_dates_set)

    if len(common_dates) < _MIN_ALIGNED_DAYS:
        return np.array([]), common_dates

    sec_ids = list(non_empty.keys())
    equal_weight = 1.0 / len(sec_ids)
    portfolio_returns = np.zeros(len(common_dates) - 1)

    for sec_id in sec_ids:
        date_map = {d: float(p) for d, p in non_empty[sec_id]}
        prices = np.array([date_map[d] for d in common_dates])
        div_by_date = divs_by_id.get(sec_id, {})
        div_by_index: dict[int, Decimal] = {
            common_dates.index(d): v for d, v in div_by_date.items() if d in common_dates_set
        }
        returns = total_returns(prices, div_by_index)
        portfolio_returns += equal_weight * returns

    return portfolio_returns, common_dates


def _compute_imoex_returns(
    imoex_series: list[tuple[date, Decimal]],
    common_dates: list[date],
) -> np.ndarray:
    """Извлечь доходности IMOEX на общих датах портфеля (forward-fill пропусков)."""
    if len(common_dates) < _MIN_ALIGNED_DAYS or not imoex_series:
        return np.zeros(max(0, len(common_dates) - 1))

    imoex_map = {d: float(p) for d, p in imoex_series}
    imoex_prices = np.array([imoex_map.get(d, float("nan")) for d in common_dates])

    for i in range(1, len(imoex_prices)):
        if np.isnan(imoex_prices[i]):
            imoex_prices[i] = imoex_prices[i - 1]

    if np.isnan(imoex_prices[0]):
        return np.zeros(len(common_dates) - 1)

    result: np.ndarray = np.diff(imoex_prices) / imoex_prices[:-1]
    return result


def _compute_metrics_and_curve(
    returns: np.ndarray,
    annual_rate: float,
) -> tuple[float, float, np.ndarray]:
    """Вычислить Шарп, max drawdown и кривую капитала."""
    if len(returns) < _MIN_ALIGNED_DAYS:
        curve = np.array([1.0, 1.0])
        return 0.0, 0.0, curve
    try:
        sharpe_val = sharpe_ratio(returns, annual_rate)
    except ValueError:
        sharpe_val = 0.0
    curve = equity_curve(returns)
    mdd = max_drawdown(curve)
    return sharpe_val, mdd, curve

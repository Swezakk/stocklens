"""Оптимизация портфеля через PyPortfolioOpt (синхронный модуль).

Использует Ledoit-Wolf как единственный оценщик ковариации — sample_cov
ненадёжен на близкокоррелированных бумагах (SBER/SBERP и аналогах).
"""

import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from api.analytics.constants import TRADING_DAYS_PER_YEAR

_MIN_TICKERS_FOR_OPTIMIZATION = 2
_MIN_PRICES_FOR_OPTIMIZATION = 2


def build_max_sharpe_weights(
    prices_df: pd.DataFrame,
    annual_rate_fraction: float,
) -> dict[str, float]:
    """Найти веса портфеля с максимальным коэффициентом Шарпа.

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).
            Должен содержать не менее 2 тикеров и не менее 2 значений цен.
        annual_rate_fraction: годовая безрисковая ставка как десятичная дробь.

    Returns:
        Словарь {ticker: вес} с суммой весов ≈ 1.

    Raises:
        ValueError: если менее 2 тикеров, менее 2 цен или ковариация вырожденная.
        OptimizationError: если оптимизатор не может найти решение.
    """
    _validate_prices_df(prices_df)
    mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe(risk_free_rate=annual_rate_fraction)
    weights: dict[str, float] = ef.clean_weights()
    return weights


def build_min_volatility_weights(
    prices_df: pd.DataFrame,
) -> dict[str, float]:
    """Найти веса портфеля с минимальной волатильностью.

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).
            Должен содержать не менее 2 тикеров и не менее 2 значений цен.

    Returns:
        Словарь {ticker: вес} с суммой весов ≈ 1.

    Raises:
        ValueError: если менее 2 тикеров или менее 2 цен.
        OptimizationError: если оптимизатор не может найти решение.
    """
    _validate_prices_df(prices_df)
    mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()
    ef = EfficientFrontier(mu, cov)
    ef.min_volatility()
    weights: dict[str, float] = ef.clean_weights()
    return weights


def compute_frontier_points(
    prices_df: pd.DataFrame,
    n: int = 20,
) -> list[tuple[float, float]]:
    """Вычислить точки эффективной границы (волатильность, доходность).

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).
        n: количество точек на границе.

    Returns:
        Список кортежей (волатильность, ожидаемая_доходность).
        Точки с ошибкой оптимизации пропускаются.
    """
    _validate_prices_df(prices_df)
    mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()

    mu_min = float(mu.min())
    mu_max = float(mu.max())

    if mu_min >= mu_max:
        return []

    target_returns = [mu_min + (mu_max - mu_min) * i / (n - 1) for i in range(n)]

    points: list[tuple[float, float]] = []
    for target_ret in target_returns:
        try:
            ef = EfficientFrontier(mu, cov)
            ef.efficient_return(target_return=target_ret)
            vol, ret, _ = ef.portfolio_performance()
            points.append((float(vol), float(ret)))
        except (OptimizationError, ValueError):
            continue

    return points


def compute_portfolio_performance(
    prices_df: pd.DataFrame,
    weights: dict[str, float],
    annual_rate_fraction: float,
) -> tuple[float, float, float]:
    """Вычислить характеристики портфеля: (доходность, волатильность, Шарп).

    Args:
        prices_df: DataFrame с ценами тикеров.
        weights: словарь {ticker: вес}.
        annual_rate_fraction: годовая безрисковая ставка.

    Returns:
        Кортеж (expected_return, volatility, sharpe).
    """
    _validate_prices_df(prices_df)
    mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()
    ef = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))
    ef.set_weights(weights)
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=annual_rate_fraction, verbose=False)
    return float(ret), float(vol), float(sharpe)


def _validate_prices_df(prices_df: pd.DataFrame) -> None:
    """Проверить входной DataFrame перед оптимизацией.

    Raises:
        ValueError: если менее 2 тикеров или менее 2 строк (цен).
    """
    if prices_df.shape[1] < _MIN_TICKERS_FOR_OPTIMIZATION:
        raise ValueError("Для оптимизации портфеля нужно не менее 2 тикеров")
    if prices_df.shape[0] < _MIN_PRICES_FOR_OPTIMIZATION:
        raise ValueError("Для оптимизации портфеля нужно не менее 2 значений цен")

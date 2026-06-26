"""Оптимизация портфеля через PyPortfolioOpt (синхронный модуль).

Использует Ledoit-Wolf как единственный оценщик ковариации — sample_cov
ненадёжен на близкокоррелированных бумагах (SBER/SBERP и аналогах).
"""

import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from api.analytics.constants import TRADING_DAYS_PER_YEAR
from api.schemas.portfolio import OptimizationStrategy

_MIN_TICKERS_FOR_OPTIMIZATION = 2
_MIN_PRICES_FOR_OPTIMIZATION = 2


def _expected_returns(prices_df: pd.DataFrame) -> "pd.Series[float]":
    """Вычислить ожидаемые годовые доходности методом mean_historical_return.

    Единственный источник mu для всех функций модуля — гарантирует, что
    проверка feasibility и солвер используют идентичный вектор ожидаемых доходностей.

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).

    Returns:
        pd.Series с ожидаемыми годовыми доходностями для каждого тикера.
    """
    return expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)


def max_sharpe_is_feasible(prices_df: pd.DataFrame, annual_rate_fraction: float) -> bool:
    """Проверить, достижима ли max-Sharpe оптимизация для данных цен и ставки.

    Зеркалит условие ValueError в pypfopt EfficientFrontier.max_sharpe:
    исключение поднимается при max(mu) <= risk_free_rate, поэтому feasibility —
    строгое неравенство max(mu) > rf. Равенство → False (недостижимо).

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).
        annual_rate_fraction: годовая безрисковая ставка как десятичная дробь.

    Returns:
        True если хотя бы один актив обгоняет безрисковую ставку, иначе False.
    """
    mu = _expected_returns(prices_df)
    return float(mu.max()) > annual_rate_fraction


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
    mu = _expected_returns(prices_df)
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
    mu = _expected_returns(prices_df)
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
    mu = _expected_returns(prices_df)
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
    mu = _expected_returns(prices_df)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()
    ef = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))
    ef.set_weights(weights)
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=annual_rate_fraction, verbose=False)
    return float(ret), float(vol), float(sharpe)


def build_weights_for_strategy(
    prices_df: pd.DataFrame,
    strategy: OptimizationStrategy,
    annual_rate: float,
    target_return: float | None = None,
    target_volatility: float | None = None,
    risk_aversion: float | None = None,
) -> dict[str, float]:
    """Найти веса портфеля для заданной стратегии оптимизации.

    Создаёт свежий EfficientFrontier на каждый вызов — объект не переиспользуется
    (pypfopt EF хранит состояние после вызова objective-метода).

    Args:
        prices_df: DataFrame (index=dates, columns=tickers, values=close).
        strategy: Стратегия из OptimizationStrategy.
        annual_rate: Годовая безрисковая ставка (десятичная дробь).
        target_return: Целевая доходность (обязателен для TARGET_RETURN).
        target_volatility: Целевой риск (обязателен для TARGET_RISK).
        risk_aversion: Коэффициент неприятия риска (обязателен для MAX_UTILITY).

    Returns:
        Словарь {ticker: вес} с суммой весов ≈ 1.

    Raises:
        ValueError: Отсутствует обязательный параметр стратегии.
        OptimizationError: Задача нефeasible (напр. target_return выше максимума).
    """
    _validate_prices_df(prices_df)
    mu = _expected_returns(prices_df)
    cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_PER_YEAR).ledoit_wolf()
    ef = EfficientFrontier(mu, cov)

    if strategy == OptimizationStrategy.MAX_SHARPE:
        ef.max_sharpe(risk_free_rate=annual_rate)
    elif strategy == OptimizationStrategy.MIN_VOLATILITY:
        ef.min_volatility()
    elif strategy == OptimizationStrategy.TARGET_RETURN:
        if target_return is None:
            raise ValueError(
                "Стратегия TARGET_RETURN требует параметр target_return (годовая доходность)"
            )
        ef.efficient_return(target_return=target_return)
    elif strategy == OptimizationStrategy.TARGET_RISK:
        if target_volatility is None:
            raise ValueError(
                "Стратегия TARGET_RISK требует параметр target_volatility (целевой риск)"
            )
        ef.efficient_risk(target_volatility=target_volatility)
    elif strategy == OptimizationStrategy.MAX_UTILITY:
        if risk_aversion is None:
            raise ValueError("Стратегия MAX_UTILITY требует параметр risk_aversion (коэффициент λ)")
        ef.max_quadratic_utility(risk_aversion=risk_aversion)

    weights: dict[str, float] = ef.clean_weights()
    return weights


def _validate_prices_df(prices_df: pd.DataFrame) -> None:
    """Проверить входной DataFrame перед оптимизацией.

    Raises:
        ValueError: если менее 2 тикеров или менее 2 строк (цен).
    """
    if prices_df.shape[1] < _MIN_TICKERS_FOR_OPTIMIZATION:
        raise ValueError("Для оптимизации портфеля нужно не менее 2 тикеров")
    if prices_df.shape[0] < _MIN_PRICES_FOR_OPTIMIZATION:
        raise ValueError("Для оптимизации портфеля нужно не менее 2 значений цен")

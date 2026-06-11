"""Риск-метрики портфеля: безрисковая ставка, коэффициент Шарпа, просадка.

Все функции синхронные; не зависят от HTTP/БД.
"""

import math

import numpy as np

from api.analytics.constants import TRADING_DAYS_PER_YEAR

_MIN_RETURNS_FOR_SHARPE = 2
_ZERO_STD_THRESHOLD = 1e-14


def daily_risk_free(
    annual_rate_fraction: float,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Вычислить дневную безрисковую ставку через компаундирование.

    Формула: (1 + annual_rate_fraction)^(1/trading_days) - 1.
    Намеренно НЕ используется линейное деление annual/days — оно даёт
    систематическую погрешность при высоких ставках (>10%).

    Args:
        annual_rate_fraction: годовая ставка как десятичная дробь (0.16 для 16%).
        trading_days: торговых дней в году (по умолчанию 252).

    Returns:
        Дневная безрисковая ставка.
    """
    return math.pow(1.0 + annual_rate_fraction, 1.0 / trading_days) - 1.0


def sharpe_ratio(
    returns: np.ndarray,
    annual_rate_fraction: float,
) -> float:
    """Вычислить аннуализированный коэффициент Шарпа.

    Формула: mean(excess) / std(returns, ddof=1) * sqrt(TRADING_DAYS_PER_YEAR),
    где excess = returns - daily_risk_free(annual_rate_fraction).

    Args:
        returns: массив дневных доходностей (n >= 2).
        annual_rate_fraction: годовая безрисковая ставка как десятичная дробь.

    Returns:
        Аннуализированный коэффициент Шарпа.

    Raises:
        ValueError: если len(returns) < 2 или std == 0 (нет вариации в доходностях).
    """
    if len(returns) < _MIN_RETURNS_FOR_SHARPE:
        raise ValueError("Для расчёта коэффициента Шарпа нужно не менее 2 значений доходностей")

    std_val = float(np.std(returns, ddof=1))
    if std_val < _ZERO_STD_THRESHOLD:
        raise ValueError(
            "Стандартное отклонение доходностей равно нулю — коэффициент Шарпа неопределён"
        )

    rf_daily = daily_risk_free(annual_rate_fraction)
    excess = returns - rf_daily
    return float(np.mean(excess) / std_val * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Вычислить максимальную просадку по кривой доходности портфеля.

    Формула: min((equity - running_max) / running_max), где running_max —
    накопленный максимум до текущей точки.

    Args:
        equity_curve: массив значений портфеля (n >= 1).

    Returns:
        Максимальная просадка как отрицательное число (<= 0).
        0.0 если кривая монотонно возрастает.
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min())


def equity_curve(returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
    """Построить кривую капитала из ряда дневных доходностей.

    Кривая ВКЛЮЧАЕТ стартовую точку `initial` перед применением доходностей —
    иначе просадка первого дня (относительно стартового капитала) теряется
    в max_drawdown.

    Args:
        returns: массив дневных доходностей длиной n.
        initial: начальная стоимость портфеля (по умолчанию 1.0).

    Returns:
        Массив значений длиной n+1: [initial, initial*(1+r_0), ...].
    """
    grown: np.ndarray = initial * np.cumprod(1.0 + returns)
    return np.concatenate([np.array([initial]), grown])

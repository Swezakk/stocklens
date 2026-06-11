"""Расчёт доходностей по ценовым рядам.

Все функции принимают и возвращают numpy-массивы; не зависят от HTTP/БД.
"""

from datetime import date
from decimal import Decimal

import numpy as np

_MIN_PRICES_FOR_RETURNS = 2


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Вычислить логарифмические доходности: r_t = ln(P_t / P_{t-1}).

    Args:
        prices: массив цен длиной n (n >= 2).

    Returns:
        Массив длиной n-1.

    Raises:
        ValueError: если в массиве меньше двух элементов.
    """
    if len(prices) < _MIN_PRICES_FOR_RETURNS:
        raise ValueError("Для расчёта доходностей нужно не менее 2 значений цен")
    return np.diff(np.log(prices))


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Вычислить простые доходности: r_t = (P_t - P_{t-1}) / P_{t-1}.

    Args:
        prices: массив цен длиной n (n >= 2).

    Returns:
        Массив длиной n-1.

    Raises:
        ValueError: если в массиве меньше двух элементов.
    """
    if len(prices) < _MIN_PRICES_FOR_RETURNS:
        raise ValueError("Для расчёта доходностей нужно не менее 2 значений цен")
    result: np.ndarray = np.diff(prices) / prices[:-1]
    return result


def total_returns(
    prices: np.ndarray,
    dividends_by_index: dict[int, Decimal],
) -> np.ndarray:
    """Вычислить полные доходности с учётом дивидендов.

    r_t = (P_t + D_t) / P_{t-1} - 1, где D_t — дивиденд с ex_date == дата t
    (иначе 0). Индекс dividends_by_index соответствует позиции в массиве prices
    (0-based: 0 соответствует первому дню, т.е. t=1).

    Args:
        prices: массив цен длиной n (n >= 2).
        dividends_by_index: словарь {индекс_в_prices: дивиденд}. Индекс 0 = P[0],
            индекс 1 = P[1], и т.д. Дивиденд учитывается при расчёте r_t,
            где t = индекс (P[t] — числитель).

    Returns:
        Массив полных доходностей длиной n-1.

    Raises:
        ValueError: если в массиве меньше двух элементов.
    """
    if len(prices) < _MIN_PRICES_FOR_RETURNS:
        raise ValueError("Для расчёта доходностей нужно не менее 2 значений цен")

    dividends = np.zeros(len(prices), dtype=float)
    for idx, div_value in dividends_by_index.items():
        dividends[idx] = float(div_value)

    numerator = prices[1:] + dividends[1:]
    result: np.ndarray = numerator / prices[:-1] - 1.0
    return result


def align_dates(
    series_a: list[tuple[date, Decimal]],
    series_b: list[tuple[date, Decimal]],
) -> tuple[list[date], np.ndarray, np.ndarray]:
    """Выровнять два ценовых ряда по общим датам (inner join).

    Args:
        series_a: список (дата, цена), отсортированный по дате.
        series_b: список (дата, цена), отсортированный по дате.

    Returns:
        Кортеж (общие_даты, цены_a, цены_b) по общим датам.
    """
    map_a = {d: float(p) for d, p in series_a}
    map_b = {d: float(p) for d, p in series_b}
    common = sorted(map_a.keys() & map_b.keys())
    prices_a = np.array([map_a[d] for d in common])
    prices_b = np.array([map_b[d] for d in common])
    return common, prices_a, prices_b

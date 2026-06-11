"""Unit-тесты аналитических функций с аналитически известными ответами.

TDD-цикл: Red (этот файл написан до вызова функций) → Green (функции реализованы).
Все ответы проверяются через математически вычисленные значения.
"""

import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from api.analytics.constants import TRADING_DAYS_PER_YEAR
from api.analytics.optimization import (
    build_max_sharpe_weights,
    build_min_volatility_weights,
    build_weights_for_strategy,
    compute_frontier_points,
)
from api.analytics.returns import log_returns, simple_returns, total_returns
from api.analytics.risk import daily_risk_free, equity_curve, max_drawdown, sharpe_ratio
from api.schemas.portfolio import OptimizationStrategy
from pypfopt import expected_returns as pf_expected_returns


def test_log_returns_known_values() -> None:
    """ln(110/100) = ln(1.1) ≈ 0.09531, ln(121/110) = ln(1.1) ≈ 0.09531."""
    prices = np.array([100.0, 110.0, 121.0])
    result = log_returns(prices)

    assert len(result) == 2
    assert math.isclose(result[0], math.log(1.1), rel_tol=1e-9)
    assert math.isclose(result[1], math.log(1.1), rel_tol=1e-9)


def test_log_returns_raises_on_single_price() -> None:
    with pytest.raises(ValueError, match="не менее 2"):
        log_returns(np.array([100.0]))


def test_simple_returns_known_values() -> None:
    """(110-100)/100 = 0.10, (90-110)/110 = -0.1818..."""
    prices = np.array([100.0, 110.0, 90.0])
    result = simple_returns(prices)

    assert len(result) == 2
    assert math.isclose(result[0], 0.10, rel_tol=1e-9)
    assert math.isclose(result[1], -20.0 / 110.0, rel_tol=1e-9)


def test_simple_returns_raises_on_single_price() -> None:
    with pytest.raises(ValueError, match="не менее 2"):
        simple_returns(np.array([100.0]))


def test_total_returns_without_dividends_equals_simple_returns() -> None:
    """При нулевых дивидендах total_returns == simple_returns."""
    prices = np.array([100.0, 110.0, 121.0])
    total = total_returns(prices, dividends_by_index={})
    simple = simple_returns(prices)

    np.testing.assert_allclose(total, simple, rtol=1e-9)


def test_total_returns_with_dividend_on_date() -> None:
    """P[0]=100, P[1]=110, dividend on index 1 = 5.
    r_1 = (110 + 5) / 100 - 1 = 0.15.
    """
    prices = np.array([100.0, 110.0, 120.0])
    dividends: dict[int, Decimal] = {1: Decimal("5.00")}
    result = total_returns(prices, dividends)

    assert math.isclose(result[0], 0.15, rel_tol=1e-9)
    # r_2 без дивиденда: (120 - 110) / 110 = 0.0909...
    assert math.isclose(result[1], 10.0 / 110.0, rel_tol=1e-9)


def test_total_returns_raises_on_single_price() -> None:
    with pytest.raises(ValueError, match="не менее 2"):
        total_returns(np.array([100.0]), {})


def test_daily_risk_free_compound_formula() -> None:
    """16% ставка: (1.16)^(1/252) - 1.

    Аналитическое значение ≈ 0.0005823...
    Линейное приближение 0.16/252 ≈ 0.000635 — другое число, тест провалится если
    использовать linear division.
    """
    annual_rate = 0.16
    result = daily_risk_free(annual_rate)

    expected = math.pow(1.16, 1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    assert math.isclose(result, expected, rel_tol=1e-12)

    # Явная проверка: результат НЕ равен linear approximation
    linear_approx = annual_rate / TRADING_DAYS_PER_YEAR
    assert not math.isclose(result, linear_approx, rel_tol=1e-4), (
        f"daily_risk_free должен использовать компаундирование, не деление: "
        f"compound={result}, linear={linear_approx}"
    )


def test_daily_risk_free_zero_rate() -> None:
    """При нулевой ставке daily_risk_free = 0."""
    assert daily_risk_free(0.0) == 0.0


def test_daily_risk_free_custom_trading_days() -> None:
    """Проверить с явным числом торговых дней."""
    result = daily_risk_free(0.10, trading_days=365)
    expected = math.pow(1.10, 1.0 / 365) - 1.0
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_sharpe_ratio_flat_positive_returns_zero_rate() -> None:
    """При постоянной доходности 0.001 и нулевой ставке Шарп = mean/std * sqrt(252).

    Ряд: 100 одинаковых значений 0.001.
    mean = 0.001, std (ddof=1) = 0, ValueError.
    Поэтому создаём ряд с небольшой вариацией, но известным mean и std.
    """
    # Ряд [0.001, 0.003] * 50 — mean = 0.002, std(ddof=1) ≈ известна
    n = 100
    returns = np.array([0.001 if i % 2 == 0 else 0.003 for i in range(n)])
    rate = 0.0
    result = sharpe_ratio(returns, rate)

    mean_val = float(np.mean(returns))  # = 0.002
    std_val = float(np.std(returns, ddof=1))
    expected = mean_val / std_val * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_sharpe_ratio_known_mean_std() -> None:
    """Ряд с известными mean=0.002, std≈0.001: Шарп=mean/std*sqrt(252).

    Используем точные значения для аналитической проверки.
    """
    # mean=0.002, std=0.001 => Шарп = 0.002/0.001 * sqrt(252) = 2 * 15.875 ≈ 31.75
    np.random.seed(42)
    # Нормальное распределение с заданными параметрами
    returns = np.array([0.002 + (0.001 * x) for x in [-1, 1] * 50])  # mean=0.002, std≈0.001
    rate = 0.0
    result = sharpe_ratio(returns, rate)

    mean_val = float(np.mean(returns))
    std_val = float(np.std(returns, ddof=1))
    expected = mean_val / std_val * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_sharpe_ratio_raises_on_single_return() -> None:
    with pytest.raises(ValueError, match="не менее 2"):
        sharpe_ratio(np.array([0.001]), 0.0)


def test_sharpe_ratio_raises_on_zero_variance() -> None:
    """Одинаковые доходности → std == 0 → ValueError."""
    returns = np.full(10, 0.002)
    with pytest.raises(ValueError, match="нулю"):
        sharpe_ratio(returns, 0.0)


def test_max_drawdown_known_sequence() -> None:
    """Кривая [1, 2, 1.5, 3, 1.5]:
    После пика 2: (1.5-2)/2 = -0.25
    После пика 3: (1.5-3)/3 = -0.5
    MDD = -0.5.
    """
    curve = np.array([1.0, 2.0, 1.5, 3.0, 1.5])
    result = max_drawdown(curve)
    assert math.isclose(result, -0.5, rel_tol=1e-9)


def test_max_drawdown_monotonic_increase_is_zero() -> None:
    """Монотонно растущая кривая → просадки нет → 0.0."""
    curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = max_drawdown(curve)
    assert result == 0.0


def test_max_drawdown_single_drop() -> None:
    """[2, 1] → (1-2)/2 = -0.5."""
    curve = np.array([2.0, 1.0])
    result = max_drawdown(curve)
    assert math.isclose(result, -0.5, rel_tol=1e-9)


def test_equity_curve_known_values() -> None:
    """returns=[0.1, 0.1]: equity = [1.0, 1.1, 1.21] (включая стартовую точку)."""
    returns = np.array([0.1, 0.1])
    result = equity_curve(returns, initial=1.0)

    assert len(result) == 3
    assert math.isclose(result[0], 1.0, rel_tol=1e-9)
    assert math.isclose(result[1], 1.1, rel_tol=1e-9)
    assert math.isclose(result[2], 1.21, rel_tol=1e-9)


def test_equity_curve_with_initial_value() -> None:
    """returns=[0.5]: equity = [100.0, 150.0] с initial=100.0 (старт включён)."""
    returns = np.array([0.5])
    result = equity_curve(returns, initial=100.0)
    assert math.isclose(result[0], 100.0, rel_tol=1e-9)
    assert math.isclose(result[1], 150.0, rel_tol=1e-9)


def test_max_drawdown_captures_first_day_drop() -> None:
    """Просадка первого дня от стартового капитала должна учитываться.

    returns=[-0.5, 0.1]: кривая [1.0, 0.5, 0.55], MDD = (0.5-1.0)/1.0 = -0.5.
    Без стартовой точки в кривой просадка первого дня терялась бы (давала 0).
    """
    returns = np.array([-0.5, 0.1])
    result = max_drawdown(equity_curve(returns))
    assert math.isclose(result, -0.5, rel_tol=1e-9)


def _make_uncorrelated_prices_df() -> pd.DataFrame:
    """Два некоррелированных синтетических актива равной волатильности.

    Актив A: синусоида, Актив B: косинусоида — ортогональные сигналы.
    Используется для проверки min-vol ≈ 50/50.
    """
    np.random.seed(0)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    # Два независимых ряда с одинаковыми статистическими свойствами
    rng = np.random.default_rng(42)
    a_returns = rng.normal(0.0005, 0.01, n)
    b_returns = rng.normal(0.0005, 0.01, n)
    # Делаем их некоррелированными через ортогонализацию
    b_returns -= np.corrcoef(a_returns, b_returns)[0, 1] * a_returns
    b_returns = b_returns / np.std(b_returns) * 0.01

    prices_a = 100.0 * np.cumprod(1.0 + a_returns)
    prices_b = 100.0 * np.cumprod(1.0 + b_returns)

    return pd.DataFrame({"ASSET_A": prices_a, "ASSET_B": prices_b}, index=dates)


def _make_near_identical_prices_df() -> pd.DataFrame:
    """Два почти идентичных актива — тест на устойчивость shrinkage."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    base_returns = rng.normal(0.0005, 0.01, n)
    prices_base = 100.0 * np.cumprod(1.0 + base_returns)
    # Второй актив — почти тот же с микро-шумом
    prices_similar = prices_base * (1.0 + rng.normal(0, 0.001, n))

    return pd.DataFrame(
        {"ASSET_A": prices_base, "ASSET_B": prices_similar},
        index=dates,
    )


def test_min_volatility_uncorrelated_equal_variance_near_50_50() -> None:
    """Два некоррелированных актива равной волатильности → min-vol ≈ 50/50."""
    prices_df = _make_uncorrelated_prices_df()
    weights = build_min_volatility_weights(prices_df)

    total = sum(weights.values())
    assert math.isclose(total, 1.0, abs_tol=1e-4), f"Сумма весов = {total}, ожидалось 1.0"

    w_a = weights.get("ASSET_A", 0.0)
    w_b = weights.get("ASSET_B", 0.0)
    # При некоррелированных и равной дисперсии ожидается ~50/50 с допуском 15%
    assert abs(w_a - 0.5) < 0.15, f"Вес ASSET_A={w_a}, ожидалось ≈0.5"
    assert abs(w_b - 0.5) < 0.15, f"Вес ASSET_B={w_b}, ожидалось ≈0.5"


def test_near_identical_assets_weights_sum_to_one_no_crash() -> None:
    """Почти одинаковые активы: Ledoit-Wolf не крашится, веса сумм ≈ 1."""
    prices_df = _make_near_identical_prices_df()
    # Не должно бросить исключение (shrinkage обрабатывает вырождение)
    weights = build_min_volatility_weights(prices_df)

    total = sum(weights.values())
    assert math.isclose(total, 1.0, abs_tol=1e-4), f"Сумма весов = {total}"


def test_max_sharpe_weights_sum_to_one() -> None:
    """max_sharpe: сумма весов ≈ 1."""
    prices_df = _make_uncorrelated_prices_df()
    weights = build_max_sharpe_weights(prices_df, annual_rate_fraction=0.16)

    total = sum(weights.values())
    assert math.isclose(total, 1.0, abs_tol=1e-4), f"Сумма весов = {total}"


def test_build_min_volatility_raises_on_single_ticker() -> None:
    """Менее 2 тикеров → ValueError."""
    n = 50
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame({"SBER": np.linspace(100, 120, n)}, index=dates)

    with pytest.raises(ValueError, match="не менее 2 тикеров"):
        build_min_volatility_weights(df)


def test_build_max_sharpe_raises_on_single_price_row() -> None:
    """Менее 2 строк цен → ValueError."""
    dates = pd.date_range("2023-01-01", periods=1, freq="B")
    df = pd.DataFrame({"SBER": [100.0], "LKOH": [5000.0]}, index=dates)

    with pytest.raises(ValueError, match="не менее 2 значений цен"):
        build_max_sharpe_weights(df, annual_rate_fraction=0.0)


def test_build_weights_for_strategy_max_sharpe_vs_min_vol_differ() -> None:
    """MAX_SHARPE и MIN_VOLATILITY дают разные веса на реалистичных данных."""
    prices_df = _make_uncorrelated_prices_df()
    w_sharpe = build_weights_for_strategy(
        prices_df, OptimizationStrategy.MAX_SHARPE, annual_rate=0.16
    )
    w_minvol = build_weights_for_strategy(
        prices_df, OptimizationStrategy.MIN_VOLATILITY, annual_rate=0.16
    )
    assert w_sharpe != w_minvol, "MAX_SHARPE и MIN_VOLATILITY не должны давать идентичные веса"
    assert math.isclose(sum(w_sharpe.values()), 1.0, abs_tol=1e-4)
    assert math.isclose(sum(w_minvol.values()), 1.0, abs_tol=1e-4)


def test_build_weights_for_strategy_target_return_missing_param_raises() -> None:
    """TARGET_RETURN без target_return → ValueError с RU-сообщением."""
    prices_df = _make_uncorrelated_prices_df()
    with pytest.raises(ValueError, match="TARGET_RETURN"):
        build_weights_for_strategy(prices_df, OptimizationStrategy.TARGET_RETURN, annual_rate=0.16)


def test_build_weights_for_strategy_target_risk_missing_param_raises() -> None:
    """TARGET_RISK без target_volatility → ValueError."""
    prices_df = _make_uncorrelated_prices_df()
    with pytest.raises(ValueError, match="TARGET_RISK"):
        build_weights_for_strategy(prices_df, OptimizationStrategy.TARGET_RISK, annual_rate=0.16)


def test_build_weights_for_strategy_max_utility_missing_param_raises() -> None:
    """MAX_UTILITY без risk_aversion → ValueError."""
    prices_df = _make_uncorrelated_prices_df()
    with pytest.raises(ValueError, match="MAX_UTILITY"):
        build_weights_for_strategy(prices_df, OptimizationStrategy.MAX_UTILITY, annual_rate=0.16)


def test_build_weights_for_strategy_target_return_valid() -> None:
    """TARGET_RETURN с разумным target_return → веса суммируются ≈ 1."""
    prices_df = _make_uncorrelated_prices_df()
    mu = pf_expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_PER_YEAR)
    feasible_target = float(mu.mean())
    weights = build_weights_for_strategy(
        prices_df,
        OptimizationStrategy.TARGET_RETURN,
        annual_rate=0.16,
        target_return=feasible_target,
    )
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-4)


def test_frontier_points_returns_list() -> None:
    """compute_frontier_points возвращает непустой список кортежей."""
    prices_df = _make_uncorrelated_prices_df()
    points = compute_frontier_points(prices_df, n=5)

    assert isinstance(points, list)
    # Не все точки могут разрешиться, но хотя бы одна должна быть
    assert len(points) >= 1
    for vol, ret in points:
        assert isinstance(vol, float)
        assert isinstance(ret, float)

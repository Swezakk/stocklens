"""Фичи и таргет волатильности (ml-spec §4.1–4.3).

Чистые функции над pandas. Фичи (Паркинсон, HAR-регрессоры) — только трейлинговые окна
(данные ≤ t, без утечек). Таргет — forward-looking по построению (Σ будущих квадратов
доходностей), это допустимо: предсказываем именно его.
"""

import math

import numpy as np
import pandas as pd

from stocklens_ml.config import HORIZON_DAYS

#: Коэффициент Паркинсона 1/(4·ln2) ≈ 0.3607.
_PARKINSON_COEF = 1.0 / (4.0 * math.log(2.0))
#: Окна HAR в торговых днях: день / неделя / месяц (ml-spec §4.3).
_WEEK_WINDOW = 5
_MONTH_WINDOW = 22


def parkinson_variance(candles: pd.DataFrame) -> pd.Series:
    """Дневной range-based прокси дисперсии (Паркинсон, §4.1): (1/(4·ln2))·(ln(H/L))².

    Инвариантен к сплит-коррекции (общий множитель сокращается в ln(H/L)).
    """
    ratio = np.log(candles["high"].to_numpy(dtype=float) / candles["low"].to_numpy(dtype=float))
    return pd.Series(_PARKINSON_COEF * ratio**2, index=candles.index, name="parkinson_var")


def har_regressors(parkinson_var: pd.Series) -> pd.DataFrame:
    """HAR-регрессоры (§4.3): средние Паркинсон-дисперсии за 1/5/22 торговых дня (включая t).

    Трейлинговые окна; ``min_periods`` равно ширине окна → warm-up в виде NaN, без утечек.
    """
    return pd.DataFrame(
        {
            "rv_d": parkinson_var,
            "rv_w": parkinson_var.rolling(_WEEK_WINDOW, min_periods=_WEEK_WINDOW).mean(),
            "rv_m": parkinson_var.rolling(_MONTH_WINDOW, min_periods=_MONTH_WINDOW).mean(),
        }
    )


def realized_variance_target(returns: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Return-based таргет (§4.2): RV_target_t = Σ_{k=1..H} r²_{t+k}.

    Forward-сумма квадратов будущих доходностей; последние ``horizon`` строк → NaN
    (полного будущего окна нет). Это таргет (не фича), forward-looking допустим.
    """
    squared = returns.astype(float) ** 2
    target = squared.shift(-1)
    for step in range(2, horizon + 1):
        target = target + squared.shift(-step)
    target.name = "rv_target"
    return target

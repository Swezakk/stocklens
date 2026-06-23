"""Наивные baseline'ы волатильности (ml-spec §5.3).

RW-RV — «волатильность завтра = волатильность вчера»: прогноз дисперсии на следующие H дней
равен последней наблюдённой H-дневной реализованной дисперсии (трейлинг). Любая модель
засчитывается, только если бьёт этот baseline по среднему QLIKE (§6.2).
"""

import pandas as pd

from stocklens_ml.config import HORIZON_DAYS


def rw_rv_forecast(returns: pd.Series, horizon: int = HORIZON_DAYS) -> pd.Series:
    """Прогноз RW-RV: Σ квадратов доходностей за прошлые ``horizon`` дней (трейлинг, без утечек).

    Выровнен по индексу с return-based RV-таргетом: прогноз в точке t сравнивается с
    реализованной дисперсией следующих H дней (RV_target_t).
    """
    squared = returns.astype(float) ** 2
    return squared.rolling(horizon, min_periods=horizon).sum().rename("rw_rv_forecast")

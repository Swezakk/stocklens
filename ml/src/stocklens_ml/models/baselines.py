"""Наивные baseline'ы волатильности (§5.3) и тренда (§5.4).

RW-RV — «волатильность завтра = волатильность вчера»: прогноз дисперсии на следующие H дней
равен последней наблюдённой H-дневной реализованной дисперсии (трейлинг). Always-up — «рынок
всегда растёт»: P(up)=1 на каждом горизонте. Любая модель засчитывается, только если бьёт
свой baseline (волатильность — QLIKE, §6.2; тренд — accuracy/F1/ROC-AUC, §6.2).
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


def always_up_forecast(index: pd.Index) -> pd.Series:
    """Прогноз always-up baseline тренда (ml-spec §5.4): P(up)=1 на каждой строке ``index``.

    Выровнен по переданному индексу для прямого сравнения с метками тренда. Модель тренда
    засчитывается, только если бьёт этот baseline по accuracy/F1/ROC-AUC (§6.2).
    """
    return pd.Series(1.0, index=index, name="always_up_forecast")

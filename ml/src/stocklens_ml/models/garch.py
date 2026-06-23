"""GARCH(1,1) прогноз волатильности через пакет arch (ml-spec §5.1).

Класс метода — параметрическая эконометрика (условная дисперсия, оценка QMLE), НЕ ML
(Engle 1982 / Bollerslev 1986). Доходности фитятся в процентах (×100) для устойчивости
оптимизатора arch; прогноз дисперсии возвращается в долях² (÷100²), чтобы совпадать по
шкале с return-based RV-таргетом (§4.2, §6.2).
"""

import numpy as np
import pandas as pd
from arch import arch_model

from stocklens_ml.config import HORIZON_DAYS

#: Масштаб доходностей в проценты перед фитом (рекомендация arch против плохой сходимости).
_RETURN_SCALE = 100.0
#: Минимум наблюдений для устойчивого фита GARCH.
_MIN_OBSERVATIONS = 100


def forecast_variance(returns: pd.Series, horizon: int = HORIZON_DAYS) -> float:
    """Прогноз дисперсии кумулятивной ``horizon``-дневной доходности по GARCH(1,1).

    Возвращает дисперсию в долях² (для QLIKE); волатильность = sqrt(результат). Под капотом:
    фит на доходностях ×100, ``forecast(horizon)`` даёт условные дисперсии h.1..h.H (percent²),
    их сумма = дисперсия H-дневной доходности (при mean='Constant'), деление на 100² → доли².
    """
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"GARCH: недостаточно наблюдений для фита ({len(clean)} < {_MIN_OBSERVATIONS})"
        )
    scaled = clean * _RETURN_SCALE
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="t")
    result = model.fit(disp="off")
    forecast = result.forecast(horizon=horizon, method="analytic", reindex=False)
    variance_percent2 = float(np.asarray(forecast.variance)[-1].sum())
    return variance_percent2 / (_RETURN_SCALE**2)

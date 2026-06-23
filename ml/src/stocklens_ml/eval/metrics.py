"""Метрики оценки прогноза волатильности (ml-spec §6.2).

QLIKE — основная метрика (на дисперсиях, устойчива к шуму прокси, асимметрична: сильнее
штрафует недопрогноз). RMSE — дополнительная. Оба отбрасывают невалидные пары (NaN, и для
QLIKE — неположительные дисперсии: ln и деление требуют > 0).
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def qlike(realized_var: npt.ArrayLike, forecast_var: npt.ArrayLike) -> float:
    """Средний QLIKE (Patton 2011, нормированная форма): mean(rv/h − ln(rv/h) − 1).

    Аргументы — **дисперсии** (не волатильности), в одной шкале (доли², ml-spec §6.2).
    Меньше — лучше; 0 при идеальном прогнозе. Пары с NaN или неположительными значениями
    исключаются (ln и деление определены только для положительных дисперсий).
    """
    realized = np.asarray(realized_var, dtype=np.float64)
    forecast = np.asarray(forecast_var, dtype=np.float64)
    mask = np.isfinite(realized) & np.isfinite(forecast) & (realized > 0.0) & (forecast > 0.0)
    if not mask.any():
        raise ValueError(
            "QLIKE: нет валидных пар дисперсий (нужны конечные положительные значения)"
        )
    ratio = realized[mask] / forecast[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Среднеквадратичная ошибка; пары с NaN исключаются."""
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not mask.any():
        raise ValueError("RMSE: нет валидных пар")
    error = actual_array[mask] - predicted_array[mask]
    return float(np.sqrt(np.mean(error**2)))

"""Модель HAR-RV волатильности (ml-spec §5.2).

Класс метода — обычная линейная регрессия (OLS), НЕ машинное обучение (Corsi 2009).
Прогнозирует return-based RV_target из трёх HAR-регрессоров на Паркинсон-прокси
(rv_d/rv_w/rv_m, §4.3). Параметры оцениваются только на train-окне (без утечек, §8.5).
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression

#: Порядок регрессоров HAR — фиксирован для согласованности fit/predict.
HAR_REGRESSORS = ["rv_d", "rv_w", "rv_m"]


class HarRvModel:
    """HAR-RV: OLS-регрессия RV_target ~ rv_d + rv_w + rv_m. Обучается на валидных строках."""

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, regressors: pd.DataFrame, target: pd.Series) -> "HarRvModel":
        """Обучить OLS на строках без NaN в регрессорах и таргете (warm-up/хвост исключаются)."""
        features = regressors[HAR_REGRESSORS].to_numpy(dtype=float)
        observed = np.asarray(target, dtype=float)
        mask = np.isfinite(features).all(axis=1) & np.isfinite(observed)
        if not mask.any():
            raise ValueError("HAR-RV: нет валидных строк для обучения (все NaN)")
        self._model.fit(features[mask], observed[mask])
        return self

    def predict(self, regressors: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Прогноз дисперсии по регрессорам (строки должны быть без NaN)."""
        features = regressors[HAR_REGRESSORS].to_numpy(dtype=float)
        return np.asarray(self._model.predict(features), dtype=np.float64)

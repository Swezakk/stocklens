"""Serving-адаптер модели тренда: сборка фич + инференс нативного CatBoost (ml-spec §8.4).

Зеркало features.py для волатильности: фрейм фич собирается тем же кодом, что при обучении
(``build_trend_frame`` из ``stocklens_ml``) — без train/serve skew. Инференс отдаёт P(up)
(вероятность роста, а не метку класса pyfunc-обёртки; тикет 3455b248) и SHAP-вклады фич
через нативный CatBoost, без пакета shap (он тянет numba/llvmlite, конфликтующие с
Python 3.12 + numpy 2.x).
"""

from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from stocklens_ml.features.assemble import TREND_FEATURE_COLUMNS, build_trend_frame

from api.ml.bundle import TrendShapResult

__all__ = ["TREND_FEATURE_COLUMNS", "CatBoostTrendPredictor", "build_serving_trend_frame"]


def build_serving_trend_frame(
    candles: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    train_start: date,
    horizon: int,
) -> pd.DataFrame:
    """Собрать фрейм фич (``trade_date`` + TREND_FEATURE_COLUMNS + ``trend_target``) для инференса.

    ``trend_target`` (forward) на инференсе не используется — прогноз берётся как-of последней
    строки; ``horizon`` влияет только на него, поэтому serving-фичи от него не зависят.
    """
    if candles.empty:
        return pd.DataFrame(columns=["trade_date", *TREND_FEATURE_COLUMNS, "trend_target"])
    return build_trend_frame(candles, dividends, splits, train_start=train_start, horizon=horizon)


class CatBoostTrendPredictor:
    """Адаптер нативного CatBoost-классификатора тренда: predict_proba (P(up)) + SHAP-вклады."""

    def __init__(self, model: CatBoostClassifier) -> None:
        self._model = model

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Вероятность роста P(up) на каждую строку (второй столбец predict_proba)."""
        proba = self._model.predict_proba(x)[:, 1]
        return np.asarray(proba, dtype=np.float64)

    def shap(self, x: pd.DataFrame) -> TrendShapResult:
        """SHAP-вклады фич (ml-spec §8.4) через нативный CatBoost.

        На инференсе истинная метка ``y`` неизвестна — ``Pool`` создаётся без неё. SHAP-значения
        зависят только от модели и матрицы фич ``X`` (не от меток), поэтому результат идентичен
        обучающему ``shap_values`` с метками. Для бинарной классификации ``type="ShapValues"``
        возвращает 2D-массив формы (n_samples, n_features+1): последний столбец — базовое
        значение (логит). Вклады — срез без него; базовое значение одинаково по строкам, берём
        из первой.
        """
        raw = self._model.get_feature_importance(Pool(x), type="ShapValues")
        shap = np.asarray(raw, dtype=np.float64)
        contribs = shap[:, :-1]
        base_value = float(shap[0, -1])
        return TrendShapResult(
            contribs=contribs,
            base_value=base_value,
            feature_names=list(self._model.feature_names_),
        )

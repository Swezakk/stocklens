"""Модель тренда на CatBoost (ml-spec §5.4).

Бинарный классификатор направления цены через ``horizon`` дней (P(up)). Гиперпараметры —
из спеки (§5.4), вынесены в keyword-аргументы конструктора, чтобы тесты обучали крошечное
число итераций. Вклады фич — нативным ``get_feature_importance(type="ShapValues")``, без
пакета shap (он тянет numba/llvmlite, конфликтующие с Python 3.12 + numpy 2.x).
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier, Pool

#: Гиперпараметры CatBoost тренда по умолчанию (ml-spec §5.4).
_DEFAULT_ITERATIONS = 600
_DEFAULT_DEPTH = 4
_DEFAULT_LEARNING_RATE = 0.03
_DEFAULT_LOSS_FUNCTION = "Logloss"
_DEFAULT_EVAL_METRIC = "AUC"
_DEFAULT_L2_LEAF_REG = 6.0
_DEFAULT_RANDOM_SEED = 42
_DEFAULT_AUTO_CLASS_WEIGHTS = "Balanced"
_DEFAULT_EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class TrendHyperparams:
    """Перебираемые гиперпараметры CatBoost тренда (ml-spec §5.4).

    Поля — только те, что финализируются на walk-forward (число деревьев, глубина, шаг
    обучения, L2-регуляризация). Значения по умолчанию — стартовые из спеки §5.4 (источник
    истины — модульные константы ``_DEFAULT_*``, чтобы не дублировать 600/4/0.03/6.0). Не
    включает loss/eval-метрику/веса классов/seed: они зафиксированы и не тюнятся.
    """

    iterations: int = _DEFAULT_ITERATIONS
    depth: int = _DEFAULT_DEPTH
    learning_rate: float = _DEFAULT_LEARNING_RATE
    l2_leaf_reg: float = _DEFAULT_L2_LEAF_REG


class TrendShap(NamedTuple):
    """Результат SHAP-разложения тренда: вклады фич, базовое значение и имена фич."""

    contribs: npt.NDArray[np.float64]
    base_value: float
    feature_names: list[str]


class TrendModel:
    """CatBoost-классификатор тренда: fit → predict_proba (P(up)) → shap_values (вклады фич)."""

    def __init__(
        self,
        iterations: int = _DEFAULT_ITERATIONS,
        depth: int = _DEFAULT_DEPTH,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        loss_function: str = _DEFAULT_LOSS_FUNCTION,
        eval_metric: str = _DEFAULT_EVAL_METRIC,
        l2_leaf_reg: float = _DEFAULT_L2_LEAF_REG,
        random_seed: int = _DEFAULT_RANDOM_SEED,
        auto_class_weights: str = _DEFAULT_AUTO_CLASS_WEIGHTS,
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> None:
        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function=loss_function,
            eval_metric=eval_metric,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            auto_class_weights=auto_class_weights,
            early_stopping_rounds=early_stopping_rounds,
            # Иначе CatBoost при fit пишет служебный каталог catboost_info/ в CWD.
            allow_writing_files=False,
        )

    def fit(
        self,
        x_tr: pd.DataFrame,
        y_tr: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "TrendModel":
        """Обучить классификатор; при наличии ``eval_set`` берётся лучшая итерация (early stop)."""
        self._model.fit(
            x_tr,
            y_tr,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
            verbose=False,
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Вероятность роста P(up) на каждую строку (второй столбец predict_proba)."""
        proba = self._model.predict_proba(x)[:, 1]
        return np.asarray(proba, dtype=np.float64)

    def shap_values(self, x: pd.DataFrame, y: pd.Series) -> TrendShap:
        """SHAP-вклады фич (ml-spec §8.4) через нативный CatBoost.

        Для бинарной классификации ``type="ShapValues"`` возвращает 2D-массив формы
        (n_samples, n_features+1): последний столбец — базовое значение (логит). Вклады —
        срез без него; базовое значение одинаково по строкам, берём из первой.
        """
        raw = self._model.get_feature_importance(Pool(x, y), type="ShapValues")
        shap = np.asarray(raw, dtype=np.float64)
        contribs = shap[:, :-1]
        base_value = float(shap[0, -1])
        return TrendShap(contribs=contribs, base_value=base_value, feature_names=self.feature_names)

    @property
    def feature_names(self) -> list[str]:
        """Имена фич в порядке обучения (для согласования вкладов SHAP и serving-матрицы)."""
        return list(self._model.feature_names_)

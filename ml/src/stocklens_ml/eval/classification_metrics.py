"""Метрики оценки прогноза тренда (ml-spec §6.2): accuracy, F1, ROC-AUC.

Accuracy и F1 — по бинарным меткам, полученным порогом ``DEFAULT_THRESHOLD`` над P(up);
ROC-AUC — по сырым вероятностям (порог не применяется, оценивается ранжирование). Все три
отбрасывают пары с NaN. Модуль зависит от scikit-learn и потому живёт под [train]-extra: API
его НЕ импортирует (serving-слой берёт только metrics.py без sklearn).
"""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#: Порог решения P(up) → класс «вверх» (ml-spec §5.4): p_up ≥ порога ⇒ метка 1.
DEFAULT_THRESHOLD = 0.5

#: ROC-AUC определён только при ≥ двух классах в выборке.
_MIN_CLASSES_FOR_ROC = 2


def _finite_mask(
    y_true: npt.NDArray[np.float64], p_up: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Маска пар без NaN/inf — общая дисциплина с metrics.py."""
    return np.isfinite(y_true) & np.isfinite(p_up)


def accuracy(
    y_true: npt.ArrayLike, p_up: npt.ArrayLike, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Доля верных меток (sklearn ``accuracy_score``) при пороге ``threshold`` над P(up)."""
    truth = np.asarray(y_true, dtype=np.float64)
    proba = np.asarray(p_up, dtype=np.float64)
    mask = _finite_mask(truth, proba)
    if not mask.any():
        raise ValueError("accuracy: нет валидных пар (нужны конечные значения)")
    labels = (proba[mask] >= threshold).astype(np.int64)
    return float(accuracy_score(truth[mask].astype(np.int64), labels))


def f1(y_true: npt.ArrayLike, p_up: npt.ArrayLike, threshold: float = DEFAULT_THRESHOLD) -> float:
    """F1 по класса «вверх» (sklearn ``f1_score``) при пороге ``threshold`` над P(up)."""
    truth = np.asarray(y_true, dtype=np.float64)
    proba = np.asarray(p_up, dtype=np.float64)
    mask = _finite_mask(truth, proba)
    if not mask.any():
        raise ValueError("F1: нет валидных пар (нужны конечные значения)")
    labels = (proba[mask] >= threshold).astype(np.int64)
    return float(f1_score(truth[mask].astype(np.int64), labels, zero_division=0.0))


def roc_auc(y_true: npt.ArrayLike, p_up: npt.ArrayLike) -> float:
    """ROC-AUC по сырым P(up) (sklearn ``roc_auc_score``, без порога — оценка ранжирования).

    Метрика не определена, если в выборке один класс — проверяем явно до вызова sklearn,
    чтобы вернуть русскоязычное доменное сообщение (sklearn бросает английское).
    """
    truth = np.asarray(y_true, dtype=np.float64)
    proba = np.asarray(p_up, dtype=np.float64)
    mask = _finite_mask(truth, proba)
    if not mask.any():
        raise ValueError("ROC-AUC: нет валидных пар (нужны конечные значения)")
    if len(np.unique(truth[mask])) < _MIN_CLASSES_FOR_ROC:
        raise ValueError("ROC-AUC не определён: в выборке один класс")
    return float(roc_auc_score(truth[mask].astype(np.int64), proba[mask]))

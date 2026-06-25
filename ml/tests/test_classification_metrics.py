"""Unit-tests for trend classification metrics (ml-spec §6.2): accuracy, F1, ROC-AUC."""

import numpy as np
import pytest
from stocklens_ml.eval import classification_metrics


def test_accuracy_perfect_separation() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    p_up = np.array([0.1, 0.4, 0.6, 0.9])

    assert classification_metrics.accuracy(y_true, p_up) == pytest.approx(1.0)


def test_accuracy_counts_thresholded_errors() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    # 0.5-порог: вторая метка предсказана как 1 (ошибка) → 3 из 4 верны.
    p_up = np.array([0.1, 0.7, 0.6, 0.9])

    assert classification_metrics.accuracy(y_true, p_up) == pytest.approx(0.75)


def test_f1_known_value() -> None:
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    # Предсказано положительными: [1, 0, 1, 0] → TP=1, FP=1, FN=1.
    p_up = np.array([0.9, 0.2, 0.8, 0.1])

    # precision = 1/2, recall = 1/2 → F1 = 0.5.
    assert classification_metrics.f1(y_true, p_up) == pytest.approx(0.5)


def test_roc_auc_perfect_ranking_on_raw_probabilities() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    p_up = np.array([0.1, 0.2, 0.8, 0.9])

    assert classification_metrics.roc_auc(y_true, p_up) == pytest.approx(1.0)


def test_roc_auc_handles_tied_probabilities_spanning_both_classes() -> None:
    # Несколько одинаковых p_up на разных классах → ничьи усредняются (AUC=0.5).
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    p_up = np.array([0.5, 0.5, 0.5, 0.5])

    assert classification_metrics.roc_auc(y_true, p_up) == pytest.approx(0.5)


def test_roc_auc_raises_for_single_class_sample() -> None:
    y_true = np.array([1.0, 1.0, 1.0])
    p_up = np.array([0.2, 0.6, 0.9])

    with pytest.raises(ValueError, match="ROC-AUC не определён: в выборке один класс"):
        classification_metrics.roc_auc(y_true, p_up)


def test_metrics_ignore_nan_pairs() -> None:
    y_true = np.array([0.0, np.nan, 1.0, 1.0])
    p_up = np.array([0.1, 0.9, 0.8, 0.7])

    # Пара с NaN исключается → остаются три верно предсказанные строки.
    assert classification_metrics.accuracy(y_true, p_up) == pytest.approx(1.0)


def test_accuracy_raises_when_no_valid_pairs() -> None:
    with pytest.raises(ValueError, match="accuracy"):
        classification_metrics.accuracy(np.array([np.nan]), np.array([0.5]))

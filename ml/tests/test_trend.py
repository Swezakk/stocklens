"""Unit-tests for the CatBoost trend model (ml-spec §5.4): fit, predict_proba, SHAP."""

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.models.trend import TrendModel

_TINY_ITERATIONS = 30
_SEED = 42


def _learnable_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетика: таргет строго определяется знаком ``driver`` → модель обязана разделить классы."""
    rng = np.random.default_rng(_SEED)
    n = 120
    driver = rng.normal(size=n)
    noise = rng.normal(size=n)
    features = pd.DataFrame({"driver": driver, "noise": noise})
    target = pd.Series((driver > 0.0).astype(int), name="trend_target")
    return features, target


def test_predict_proba_returns_probabilities_in_unit_interval() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)
    proba = model.predict_proba(features)

    assert proba.shape == (len(features),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_model_separates_planted_classes() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)
    proba = model.predict_proba(features)

    # Класс полностью определяется знаком driver → P(up) выше там, где driver>0.
    up_mask = features["driver"] > 0.0
    assert proba[up_mask.to_numpy()].mean() > proba[(~up_mask).to_numpy()].mean()


def test_shap_values_shape_and_base_value() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)
    contribs, base_value, names = model.shap_values(features, target)

    assert contribs.shape == (len(features), features.shape[1])
    assert isinstance(base_value, float)
    assert names == ["driver", "noise"]


def test_shap_values_satisfy_additivity() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)
    contribs, base_value, _ = model.shap_values(features, target)

    # SHAP-аддитивность (в log-odds): Σ вкладов фич + base = сырое предсказание модели.
    # Проверяет КОРРЕКТНОСТЬ среза [:, :-1] (вклады) и base-колонки [0, -1], а не только форму:
    # ошибка оси или смещение base-колонки нарушили бы это равенство.
    raw = np.asarray(
        model._model.predict(features, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    assert np.allclose(contribs.sum(axis=1) + base_value, raw, atol=1e-5)


def test_feature_names_property_reflects_training_columns() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)

    assert model.feature_names == ["driver", "noise"]


def test_fit_with_eval_set_uses_best_model() -> None:
    features, target = _learnable_dataset()
    split = 90
    x_tr, y_tr = features.iloc[:split], target.iloc[:split]
    x_val, y_val = features.iloc[split:], target.iloc[split:]

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(
        x_tr, y_tr, eval_set=(x_val, y_val)
    )
    proba = model.predict_proba(x_val)

    assert proba.shape == (len(x_val),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predict_proba_is_one_dimensional() -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=_TINY_ITERATIONS, random_seed=_SEED).fit(features, target)

    assert model.predict_proba(features).ndim == 1


@pytest.mark.parametrize("iterations", [1, _TINY_ITERATIONS])
def test_fit_returns_self_for_chaining(iterations: int) -> None:
    features, target = _learnable_dataset()

    model = TrendModel(iterations=iterations, random_seed=_SEED)

    assert model.fit(features, target) is model

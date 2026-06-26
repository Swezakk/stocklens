"""Unit-тест serving-адаптера тренда: P(up) и корректность переписанного SHAP-среза (без сети).

Главное, что route-stub не покрывает: SHAP считается без истинных меток (``y`` на инференсе
неизвестен), поэтому ``Pool`` создаётся без них. Тест доказывает, что (1) label-less ``Pool``
для ``ShapValues`` действительно возвращает форму (1, n_features+1) — эмпирическое
подтверждение допущения адаптера; (2) переписанный срез аддитивен: сумма вкладов + базовое
значение совпадает с сырым логитом модели (RawFormulaVal). Обучается крошечный, но обучаемый
CatBoost с фиксированным сидом — без сети и MLflow.
"""

from datetime import date
from pathlib import Path

import mlflow.catboost
import numpy as np
import pandas as pd
from api.ml.trend import TREND_FEATURE_COLUMNS, CatBoostTrendPredictor, build_serving_trend_frame
from catboost import CatBoostClassifier, Pool

_FEATURE_NAMES = ["f0", "f1", "f2"]
_RANDOM_SEED = 42


def _learnable_frame(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Маленький обучаемый фрейм: метка — знак линейной комбинации фич + лёгкий шум."""
    rng = np.random.default_rng(_RANDOM_SEED)
    features = rng.normal(size=(n, len(_FEATURE_NAMES)))
    signal = features @ np.array([1.5, -1.0, 0.5])
    noise = rng.normal(scale=0.3, size=n)
    labels = (signal + noise > 0.0).astype(int)
    x = pd.DataFrame(features, columns=_FEATURE_NAMES)
    y = pd.Series(labels, name="trend_target")
    return x, y


def _trained_predictor() -> tuple[CatBoostTrendPredictor, CatBoostClassifier, pd.DataFrame]:
    x, y = _learnable_frame()
    model = CatBoostClassifier(
        iterations=30,
        depth=3,
        learning_rate=0.1,
        random_seed=_RANDOM_SEED,
        # Иначе CatBoost при fit пишет служебный каталог catboost_info/ в CWD.
        allow_writing_files=False,
    )
    model.fit(x, y, verbose=False)
    return CatBoostTrendPredictor(model), model, x


def test_predict_proba_returns_probabilities_in_unit_interval() -> None:
    predictor, _, x = _trained_predictor()

    proba = predictor.predict_proba(x)

    assert proba.shape == (len(x),)
    assert proba.dtype == np.float64
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_label_less_pool_returns_shap_with_base_column() -> None:
    _, model, x = _trained_predictor()
    single_row = x.iloc[[0]]

    raw = model.get_feature_importance(Pool(single_row), type="ShapValues")
    shap = np.asarray(raw, dtype=np.float64)

    # Бинарная классификация: (n_samples, n_features+1), последний столбец — базовое значение.
    assert shap.shape == (1, len(_FEATURE_NAMES) + 1)


def test_shap_is_additive_to_raw_logit_without_labels() -> None:
    predictor, model, x = _trained_predictor()

    result = predictor.shap(x)
    reconstructed = result.contribs.sum(axis=1) + result.base_value
    raw_logit = model.predict(x, prediction_type="RawFormulaVal")

    assert result.contribs.shape == (len(x), len(_FEATURE_NAMES))
    assert result.feature_names == _FEATURE_NAMES
    np.testing.assert_allclose(reconstructed, raw_logit, atol=1e-5)


def _trend_columns_frame(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Обучаемый фрейм с реальными именами признаков тренда (TREND_FEATURE_COLUMNS).

    Нужен round-trip-тесту: он проверяет, что имена признаков артефакта после перезагрузки
    совпадают со схемой обучения — это фиксирует выравнивание SHAP feature<->value.
    """
    rng = np.random.default_rng(_RANDOM_SEED)
    features = rng.normal(size=(n, len(TREND_FEATURE_COLUMNS)))
    signal = features[:, 0] * 1.5 - features[:, 1]
    labels = (signal + rng.normal(scale=0.3, size=n) > 0.0).astype(int)
    x = pd.DataFrame(features, columns=TREND_FEATURE_COLUMNS)
    y = pd.Series(labels, name="trend_target")
    return x, y


def test_native_artifact_round_trips_to_working_predictor(tmp_path: Path) -> None:
    """Тикет 3455b248 (сквозной шов): нативно залогированный CatBoost перезагружается в предиктор,
    отдающий P(up) и SHAP, а не метки класса pyfunc.

    Без моков — реальные ``mlflow.catboost.log_model`` → ``load_model`` через локальный
    sqlite-MLflow. Последний assert замыкает выравнивание SHAP feature<->value: имена признаков
    перезагруженного артефакта совпадают со схемой обучения тренда (тихая порча при расхождении
    порядка колонок train/serve была бы поймана).
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    x, y = _trend_columns_frame()
    model = CatBoostClassifier(
        iterations=30, depth=3, random_seed=_RANDOM_SEED, allow_writing_files=False
    )
    model.fit(x, y, verbose=False)
    with mlflow.start_run():
        info = mlflow.catboost.log_model(model, name="model")

    loaded = mlflow.catboost.load_model(info.model_uri)
    predictor = CatBoostTrendPredictor(loaded)

    proba = predictor.predict_proba(x)
    shap = predictor.shap(x)

    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert shap.contribs.shape == (len(x), len(TREND_FEATURE_COLUMNS))
    assert list(loaded.feature_names_) == TREND_FEATURE_COLUMNS


def test_empty_candles_yield_typed_empty_trend_frame() -> None:
    frame = build_serving_trend_frame(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        train_start=date(2022, 4, 1),
        horizon=5,
    )

    assert frame.empty
    assert list(frame.columns) == ["trade_date", *TREND_FEATURE_COLUMNS, "trend_target"]

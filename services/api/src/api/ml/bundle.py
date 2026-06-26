"""Загруженные ML-модели в app.state (ml-spec §8.1).

ModelBundle хранит модель волатильности (обязательная) и модель тренда (опциональная) +
метаданные версии. readiness проверяет только волатильность — промах тренда не валит
готовность сервиса (инвариант «ошибка одного источника не валит остальные»). Сервисы зависят
от ``VolatilityPredictor`` / ``TrendPredictor`` (Protocol) — unit-тесты подменяют их стабами
без подъёма MLflow.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class VolatilityPredictor(Protocol):
    """Прогнозист волатильности: фрейм фич (SERVING_FEATURES) → дисперсия (доли²), форма (1,)."""

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Вернуть прогноз дисперсии H-дневной доходности как-of последней строки фрейма."""
        ...


class TrendShapResult(NamedTuple):
    """Результат SHAP-разложения тренда: вклады фич, базовое значение и имена фич.

    Живёт в serving-слое (а не импортируется из ``stocklens_ml``), чтобы адаптер был
    самодостаточным и не зависел от тренировочного модуля (зеркало features.py, где
    serving тянет только ``assemble``).
    """

    contribs: npt.NDArray[np.float64]
    base_value: float
    feature_names: list[str]


class TrendPredictor(Protocol):
    """Прогнозист тренда: фрейм фич (TREND_FEATURE_COLUMNS) → P(up) и SHAP-вклады фич."""

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Вернуть вероятность роста P(up) на каждую строку фрейма, форма (n,)."""
        ...

    def shap(self, x: pd.DataFrame) -> TrendShapResult:
        """Вернуть SHAP-разложение P(up) для строк фрейма (вклады + базовое значение)."""
        ...


@dataclass(frozen=True)
class LoadedVolatilityModel:
    """Модель волатильности из реестра + метаданные версии (ml-spec §7.2, §8.3)."""

    predictor: VolatilityPredictor
    model_version: str
    method: str
    metrics: Mapping[str, float]
    horizon_days: int


@dataclass(frozen=True)
class LoadedTrendModel:
    """Модель тренда из реестра + метаданные версии (ml-spec §7.3, §8.4).

    Нативный CatBoost-артефакт не несёт метрик/метода — только версию реестра и горизонт
    (из ``stocklens_ml.config.HORIZON_DAYS``, единый источник истины для горизонта тренда).
    """

    predictor: TrendPredictor
    model_version: str
    horizon_days: int


@dataclass
class ModelBundle:
    """Контейнер загруженных моделей в app.state.ml. Тренд — опционален (None при промахе)."""

    volatility: LoadedVolatilityModel | None = None
    trend: LoadedTrendModel | None = None

    def ready(self) -> bool:
        """Готовность ML: загружена обязательная модель волатильности (тренд не блокирует)."""
        return self.volatility is not None

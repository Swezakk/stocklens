"""Загруженные ML-модели в app.state (ml-spec §8.1).

ModelBundle хранит модель волатильности + метаданные версии. Тренд отложен (модель не
зарегистрирована) — поле ``volatility`` единственное обязательное; readiness проверяет только
его. Сервис зависит от ``VolatilityPredictor`` (Protocol) — unit-тесты подменяют его стабом
без подъёма MLflow.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class VolatilityPredictor(Protocol):
    """Прогнозист волатильности: фрейм фич (SERVING_FEATURES) → дисперсия (доли²), форма (1,)."""

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Вернуть прогноз дисперсии H-дневной доходности как-of последней строки фрейма."""
        ...


@dataclass(frozen=True)
class LoadedVolatilityModel:
    """Модель волатильности из реестра + метаданные версии (ml-spec §7.2, §8.3)."""

    predictor: VolatilityPredictor
    model_version: str
    method: str
    metrics: Mapping[str, float]
    horizon_days: int


@dataclass
class ModelBundle:
    """Контейнер загруженных моделей в app.state.ml. Тренд — отложен (None)."""

    volatility: LoadedVolatilityModel | None = None

    def ready(self) -> bool:
        """Готовность ML: загружена обязательная модель волатильности (тренд не блокирует)."""
        return self.volatility is not None

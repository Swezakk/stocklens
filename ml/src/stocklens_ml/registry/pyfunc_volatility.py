"""MLflow-обёртка модели волатильности для serving (ml-spec §2.1, §7.1).

«Models from Code» (MLflow 3.x): модель определяется ЭТИМ скриптом, а не пиклится по ссылке
на класс. При загрузке MLflow исполняет файл и берёт инстанс из ``set_model()`` — поэтому
артефакт самодостаточен и грузится в API-контейнере, где пакета ``stocklens_ml`` нет
(см. ревью контракта serving: cloudpickle сериализует класс по módule-пути, что сломало бы
загрузку). Импортируются только mlflow/numpy/pandas/arch — без ``stocklens_ml``.

Состояние конкретной версии (метод-победитель, метрики vs baseline, HAR-коэффициенты) не
хардкодится в статическом скрипте, а читается из артефакта ``state`` в ``load_context`` —
так каждая зарегистрированная версия воспроизводит свою замороженную конфигурацию.

Инлайн-формула GARCH дублирует ~6 строк из :mod:`stocklens_ml.models.garch` осознанно (ради
самодостаточности артефакта); дрейф ловит ``tests/test_pyfunc_volatility.py``.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch import arch_model
from mlflow.models import infer_signature, set_model
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import log_model
from mlflow.pyfunc.model import PythonModel, PythonModelContext

#: Порядок HAR-регрессоров — синхронизирован с stocklens_ml.models.har.HAR_REGRESSORS.
_HAR_REGRESSORS = ["rv_d", "rv_w", "rv_m"]
#: Масштаб доходностей в проценты перед фитом GARCH (синхрон с models.garch._RETURN_SCALE).
_RETURN_SCALE = 100.0
#: Минимум наблюдений для устойчивого фита GARCH.
_MIN_GARCH_OBS = 100
#: Горизонт прогноза по умолчанию (торговые дни; ml-spec §4, D8).
_DEFAULT_HORIZON = 5

#: Имена методов-победителей (ml-spec §8.3).
METHOD_GARCH = "garch"
METHOD_HAR = "har_rv"

#: Контракт входного фрейма serving (единый источник для API): доходности + HAR-регрессоры.
SERVING_FEATURES = ["r", *_HAR_REGRESSORS]

#: pip-зависимости serving-артефакта: инференс не требует stocklens_ml/sklearn/mlflow-extras.
_SERVING_REQUIREMENTS = ["arch>=7.2", "numpy>=2.1", "pandas>=2.2"]
#: Имя артефакта со state конкретной версии модели.
_STATE_ARTIFACT = "state"


class VolatilityModel(PythonModel):
    """Прогноз дисперсии H-дневной доходности: GARCH (рефит на окне) или HAR (линейный).

    GARCH не несёт переносимого состояния — переобучается на переданном окне доходностей при
    каждом инференсе (корректная эконометрическая практика). HAR несёт замороженные OLS-
    коэффициенты, оценённые на обучающем окне. Метод и метрики читаются из state-артефакта.
    """

    def __init__(
        self,
        method: str | None = None,
        metrics: dict[str, float] | None = None,
        har_coef: list[float] | None = None,
        har_intercept: float | None = None,
        horizon: int = _DEFAULT_HORIZON,
    ) -> None:
        self.method = method
        self.metrics = metrics or {}
        self.har_coef = None if har_coef is None else np.asarray(har_coef, dtype=float)
        self.har_intercept = har_intercept
        self.horizon = horizon

    def load_context(self, context: PythonModelContext) -> None:
        """Восстановить состояние версии (метод, метрики, HAR-коэффициенты) из артефакта state."""
        state = json.loads(Path(context.artifacts[_STATE_ARTIFACT]).read_text(encoding="utf-8"))
        self.method = state["method"]
        self.metrics = state["metrics"]
        self.horizon = int(state["horizon"])
        coef = state.get("har_coef")
        self.har_coef = None if coef is None else np.asarray(coef, dtype=float)
        self.har_intercept = state.get("har_intercept")

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> npt.NDArray[np.float64]:
        """MLflow-вход: делегирует чистому forecast() (context/params не используются)."""
        return self.forecast(model_input)

    def forecast(self, model_input: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Дисперсия (доли²) как-of последней строки фрейма фич; форма (1,). API берёт sqrt."""
        if self.method == METHOD_GARCH:
            return self._forecast_garch(model_input)
        if self.method == METHOD_HAR:
            return self._forecast_har(model_input)
        raise ValueError(f"VolatilityModel: неизвестный метод прогноза {self.method!r}")

    def _forecast_garch(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        clean = frame["r"].dropna().to_numpy(dtype=float)
        if len(clean) < _MIN_GARCH_OBS:
            raise ValueError(
                f"GARCH: недостаточно наблюдений для фита ({len(clean)} < {_MIN_GARCH_OBS})"
            )
        scaled = clean * _RETURN_SCALE
        fitted = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="t").fit(
            disp="off"
        )
        forecast = fitted.forecast(horizon=self.horizon, method="analytic", reindex=False)
        variance_percent2 = float(np.asarray(forecast.variance)[-1].sum())
        return np.array([variance_percent2 / (_RETURN_SCALE**2)], dtype=np.float64)

    def _forecast_har(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        if self.har_coef is None or self.har_intercept is None:
            raise ValueError("HAR: коэффициенты модели не загружены")
        regressors = frame[_HAR_REGRESSORS].to_numpy(dtype=float)
        finite_rows = np.isfinite(regressors).all(axis=1)
        if not finite_rows.any():
            raise ValueError("HAR: нет валидной строки регрессоров для прогноза")
        as_of = regressors[finite_rows][-1]
        variance = float(as_of @ self.har_coef + self.har_intercept)
        return np.array([variance], dtype=np.float64)


def log_volatility_model(
    method: str,
    metrics: dict[str, float],
    horizon: int,
    input_example: pd.DataFrame,
    *,
    har_coef: list[float] | None = None,
    har_intercept: float | None = None,
    registered_model_name: str | None = None,
) -> ModelInfo:
    """Залогировать serving-обёртку через «Models from Code» (требует активный MLflow-run).

    State версии (метод, метрики, HAR-коэффициенты) пишется во временный артефакт и
    упаковывается с моделью; ``python_model`` — путь к этому скрипту, поэтому загрузка не
    зависит от наличия ``stocklens_ml`` в среде инференса.
    """
    signature = infer_signature(input_example, np.array([0.0], dtype=np.float64))
    state = {
        "method": method,
        "metrics": metrics,
        "horizon": horizon,
        "har_coef": har_coef,
        "har_intercept": har_intercept,
    }
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        model_info: ModelInfo = log_model(
            name="model",
            python_model=str(Path(__file__)),
            artifacts={_STATE_ARTIFACT: str(state_path)},
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            pip_requirements=_SERVING_REQUIREMENTS,
        )
    return model_info


set_model(VolatilityModel())

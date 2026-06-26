"""Интеграционный smoke реестра MLflow (§11.3 / тикет a1c4f7e2): сквозной путь serving.

Доказывает связку, ранее проверенную только по частям:
«клиент логирует/регистрирует → PG-backed MLflow-сервер (`--serve-artifacts`) →
прод-loader API грузит модель по алиасу `models:/<name>@production`».

Три несущих элемента сетапа (каждый — реальный найденный сюрприз, см. тикет):
1. Образ сервера — `services/mlflow/Dockerfile` (mlflow 3.14.0 + psycopg2-binary):
   стоковый `ghcr.io/mlflow/mlflow` без psycopg2 не стартует с PostgreSQL-бэкендом.
2. Бэкенд — `postgresql+psycopg2://` (НЕ psycopg3): на psycopg3 ломается резолюция
   `models:/<name>@<alias>` (`operator does not exist: integer = character varying`).
3. `MLFLOW_SERVER_ALLOWED_HOSTS=localhost:*,127.0.0.1:*`: иначе 403 (DNS-rebinding guard)
   на динамическом порту testcontainers — mlflow сверяет host:port по fnmatch, и голый
   `localhost` НЕ матчит `localhost:<port>`.

TEST 1 гоняет реальный прод-loader (`api.ml.loader.load_bundle`) против живого сервера —
именно тут прячется интеграционный риск. TEST 2 — критерий готовности §11.3: реальный
`lifespan` приложения грузит модели из реестра → `/health/ready` отдаёт 200.
"""

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import mlflow.catboost
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from api.core.auth.settings import AuthSettings
from api.core.settings import ApiSettings
from api.main import create_app
from api.ml.loader import load_bundle
from asgi_lifespan import LifespanManager
from catboost import CatBoostClassifier
from httpx import ASGITransport, AsyncClient
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.postgres import PostgresContainer

import mlflow
from mlflow import MlflowClient

pytestmark = pytest.mark.integration

#: Тег локально собираемого образа MLflow-сервера (см. services/mlflow/Dockerfile).
_IMAGE_TAG = "stocklens-mlflow:test"
#: Сетевые алиасы внутри общей docker-сети контейнеров smoke-стека.
_PG_ALIAS = "pgmlflow"
_MLFLOW_ALIAS = "mlflow"
#: Порт MLflow-сервера внутри контейнера.
_MLFLOW_PORT = 5000
#: Имена зарегистрированных моделей и алиас — должны совпадать с дефолтами ApiSettings.
_VOLATILITY_MODEL = "stocklens-volatility"
_TREND_MODEL = "stocklens-trend"
_ALIAS = "production"
#: Дисперсия стаб-модели волатильности (доли²); loader её не вызывает, но предикт обязателен.
_STUB_VARIANCE = 0.0009
#: Сид для воспроизводимого обучения крошечного CatBoost тренда.
_RANDOM_SEED = 42
#: Таймаут ожидания готовности MLflow-сервера (сек.).
_SERVER_READY_TIMEOUT_SECONDS = 90.0
#: Интервал опроса /health MLflow-сервера (сек.).
_SERVER_POLL_INTERVAL_SECONDS = 1.0
#: HTTP 200 — сервер готов.
_HTTP_OK = 200
#: Заглушки DSN: loader.load_bundle не обращается к БД/Redis, нужны лишь валидные ApiSettings.
_DUMMY_DB_DSN = "postgresql+asyncpg://stocklens:stocklens@localhost:5432/stocklens"
_DUMMY_REDIS_DSN = "redis://localhost:6379/0"


class StubVol(PythonModel):
    """Стаб модели волатильности для pyfunc-артефакта (метод/метрики/горизонт как у прода).

    На уровне модуля — чтобы cloudpickle сериализовал её по module-пути и обратная загрузка
    в том же процессе восстановила класс. Атрибуты ``method``/``metrics``/``horizon`` читает
    прод-loader через ``unwrap_python_model()`` (api.ml.loader._load_volatility).
    """

    method = "garch"
    metrics: ClassVar[dict[str, float]] = {"qlike": 0.7}
    horizon = 5

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Фиксированная дисперсия на каждую строку (loader предикт не вызывает)."""
        return np.full(len(model_input), _STUB_VARIANCE, dtype=np.float64)


def _build_mlflow_image() -> None:
    """Собрать образ MLflow-сервера из services/mlflow/Dockerfile (идемпотентно, слой-кэш)."""
    repo_root = Path(__file__).resolve().parents[4]
    dockerfile = repo_root / "services" / "mlflow" / "Dockerfile"
    context = repo_root / "services" / "mlflow"
    try:
        subprocess.run(
            ["docker", "build", "-t", _IMAGE_TAG, "-f", str(dockerfile), str(context)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Сборка образа {_IMAGE_TAG} упала: {exc.stderr}") from exc


def _wait_for_mlflow(base_url: str) -> None:
    """Опрашивать /health MLflow-сервера до 200 либо упасть по таймауту с диагностикой."""
    deadline = time.monotonic() + _SERVER_READY_TIMEOUT_SECONDS
    last_error = "ответ не получен"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == _HTTP_OK:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError) as exc:
            last_error = str(exc)
        time.sleep(_SERVER_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"MLflow-сервер не стал готов за {_SERVER_READY_TIMEOUT_SECONDS:.0f}с: {last_error}"
    )


def _learnable_trend_frame(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Маленький обучаемый фрейм: метка — знак линейной комбинации фич + лёгкий шум."""
    rng = np.random.default_rng(_RANDOM_SEED)
    feature_names = ["f0", "f1", "f2"]
    features = rng.normal(size=(n, len(feature_names)))
    signal = features @ np.array([1.5, -1.0, 0.5])
    labels = (signal + rng.normal(scale=0.3, size=n) > 0.0).astype(int)
    x = pd.DataFrame(features, columns=feature_names)
    y = pd.Series(labels, name="trend_target")
    return x, y


def _register_models(base_url: str) -> None:
    """Зарегистрировать обе модели против сервера + алиас production (psycopg2-путь алиасов)."""
    mlflow.set_tracking_uri(base_url)
    client = MlflowClient(tracking_uri=base_url)

    with mlflow.start_run():
        vol_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=StubVol(),
            registered_model_name=_VOLATILITY_MODEL,
        )
    client.set_registered_model_alias(
        _VOLATILITY_MODEL, _ALIAS, str(vol_info.registered_model_version)
    )

    x, y = _learnable_trend_frame()
    trend_model = CatBoostClassifier(
        iterations=20,
        depth=3,
        learning_rate=0.1,
        random_seed=_RANDOM_SEED,
        # Иначе CatBoost при fit пишет служебный каталог catboost_info/ в CWD.
        allow_writing_files=False,
    )
    trend_model.fit(x, y, verbose=False)
    with mlflow.start_run():
        trend_info = mlflow.catboost.log_model(
            trend_model,
            name="model",
            registered_model_name=_TREND_MODEL,
        )
    client.set_registered_model_alias(
        _TREND_MODEL, _ALIAS, str(trend_info.registered_model_version)
    )


@pytest.fixture(scope="module")
def mlflow_registry() -> Iterator[str]:
    """Поднять PG + MLflow (`--serve-artifacts`), зарегистрировать обе модели, вернуть base URL.

    PG и MLflow делят docker-сеть — сервер достаёт PostgreSQL по алиасу. Teardown останавливает
    оба контейнера и удаляет сеть (PG останавливается даже если старт MLflow/регистрация упали).
    """
    _build_mlflow_image()

    network = Network()
    network.create()
    postgres = (
        PostgresContainer(
            "postgres:16-alpine",
            username="mlflow",
            password="mlflow",
            dbname="mlflow",
            driver="psycopg2",
        )
        .with_network(network)
        .with_network_aliases(_PG_ALIAS)
    )
    mlflow_server: DockerContainer | None = None
    try:
        postgres.start()
        backend = f"postgresql+psycopg2://mlflow:mlflow@{_PG_ALIAS}:5432/mlflow"
        mlflow_server = (
            DockerContainer(_IMAGE_TAG)
            .with_network(network)
            .with_network_aliases(_MLFLOW_ALIAS)
            .with_exposed_ports(_MLFLOW_PORT)
            .with_env("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*")
            .with_command(
                "mlflow server"
                f" --backend-store-uri {backend}"
                " --artifacts-destination /mlflow/artifacts --serve-artifacts"
                f" --host 0.0.0.0 --port {_MLFLOW_PORT}"
            )
        )
        mlflow_server.start()
        base_url = (
            f"http://{mlflow_server.get_container_host_ip()}"
            f":{mlflow_server.get_exposed_port(_MLFLOW_PORT)}"
        )
        _wait_for_mlflow(base_url)
        _register_models(base_url)
        yield base_url
    finally:
        if mlflow_server is not None:
            mlflow_server.stop()
        postgres.stop()
        network.remove()


def _loader_settings(tracking_uri: str) -> ApiSettings:
    """ApiSettings для loader.load_bundle: реальны только ML-поля, DSN БД/Redis — заглушки."""
    return ApiSettings.model_validate(
        {
            "database_url_async": _DUMMY_DB_DSN,
            "redis_url": _DUMMY_REDIS_DSN,
            "mlflow_tracking_uri": tracking_uri,
            "ml_required_for_ready": True,
            # Happy-path грузится с первой попытки; малое число — чтобы не висеть 30с при сбое.
            "ml_load_attempts": 2,
            "ml_load_interval_seconds": 1.0,
        }
    )


def test_load_bundle_loads_both_models_by_alias_via_serve_artifacts(
    mlflow_registry: str,
) -> None:
    """Прод-loader грузит обе модели по алиасу через serve-artifacts на psycopg2-бэкенде.

    Реальный ``api.ml.loader.load_bundle`` против живого PG-backed сервера: волатильность
    (pyfunc + unwrap → method/version) и тренд (нативный CatBoost) резолвятся по
    ``models:/<name>@production`` и скачиваются через `mlflow-artifacts://`-прокси.
    """
    settings = _loader_settings(mlflow_registry)

    bundle = load_bundle(settings)

    assert bundle.ready() is True
    assert bundle.volatility is not None
    assert bundle.volatility.model_version == "1"
    assert bundle.volatility.method == "garch"
    assert bundle.trend is not None
    assert bundle.trend.model_version == "1"


async def test_health_ready_returns_200_when_models_loaded_from_registry(
    mlflow_registry: str,
    test_settings: ApiSettings,
    test_auth_settings: AuthSettings,
) -> None:
    """§11.3: реальный lifespan грузит модели из реестра → /health/ready = 200, models = ok.

    Предпочтённый путь (не суррогат): приложение поднимается через ``LifespanManager``, и
    именно lifespan делает реальный ``load_bundle`` из MLflow-контейнера, реальное ожидание
    схемы (мигрированный pg_container) и реальный Redis. Без подмены app.state.ml и без
    dependency-overrides — проверяется ровно тот шов, ради которого существует §11.3.
    """
    settings = ApiSettings.model_validate(
        {
            "database_url_async": str(test_settings.database_url_async),
            "redis_url": str(test_settings.redis_url),
            "mlflow_tracking_uri": mlflow_registry,
            "ml_required_for_ready": True,
            "ml_load_attempts": 2,
            "ml_load_interval_seconds": 1.0,
        }
    )
    app = create_app()
    app.state.settings = settings
    app.state.auth_settings = test_auth_settings

    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.get("/api/v1/health/ready")

    assert response.status_code == _HTTP_OK
    body = response.json()
    assert body["models"] == "ok"
    assert body["status"] == "ready"

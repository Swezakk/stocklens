"""Общие фикстуры тестов ML-проекта.

Изолирует глобальное состояние MLflow между тестами. ``mlflow.set_experiment`` кэширует
активный эксперимент в ДВУХ местах: module-global ``_active_experiment_id`` И переменную
окружения ``MLFLOW_EXPERIMENT_ID`` (плюс ``MLFLOW_EXPERIMENT_NAME``). Смена tracking URI этот
кэш НЕ сбрасывает. Без сброса тест, выставивший эксперимент на своей временной sqlite-БД,
оставляет id в окружении; последующий тест (со своей пустой БД и вызовом ``start_run`` без
``set_experiment``) читает stale ``MLFLOW_EXPERIMENT_ID`` через ``_get_experiment_id_from_env``
и падает «No Experiment with id=… exists». Фикстура завершает активный прогон, обнуляет
module-global и снимает обе переменные окружения после каждого теста, делая порядок прогонки
независимым (иначе утечка проявляется лишь при определённом алфавитном порядке файлов и
является скрыто-флаки).
"""

import os
from collections.abc import Iterator

import pytest
from mlflow.tracking import fluent

import mlflow

#: Переменные окружения, в которые MLflow кэширует активный эксперимент (контракт mlflow.fluent).
_MLFLOW_EXPERIMENT_ENV_VARS = ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME")


@pytest.fixture(autouse=True)
def _reset_mlflow_active_experiment() -> Iterator[None]:
    """Сбросить активный прогон, module-global и env-переменные эксперимента MLflow после теста."""
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()
    fluent._active_experiment_id = None
    for env_var in _MLFLOW_EXPERIMENT_ENV_VARS:
        os.environ.pop(env_var, None)

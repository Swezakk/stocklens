"""Продвижение моделей в реестре MLflow по алиасам (ml-spec §7.1, §12).

Stages устарели — используем алиасы. После регистрации лучший прогон помечается
``champion`` (автоматически в обучении, D6); прод-алиас ``production`` ставится вручную
после ревью метрик в MLflow UI (рунбук §12). API грузит модель по
``models:/<name>@<alias>``; откат — переназначение ``production`` на прежнюю версию.
"""

from mlflow import MlflowClient

#: Алиас лучшей по walk-forward QLIKE версии (ставится обучением автоматически).
CHAMPION_ALIAS = "champion"
#: Прод-алиас (ставится вручную после ревью; именно по нему грузит модель API).
PRODUCTION_ALIAS = "production"


def set_alias(client: MlflowClient, name: str, alias: str, version: int | str) -> None:
    """Назначить алиас версии модели; перенос с прежней версии — атомарный."""
    client.set_registered_model_alias(name=name, alias=alias, version=str(version))


def mark_champion(client: MlflowClient, name: str, version: int | str) -> None:
    """Пометить версию как ``champion`` (лучшая по walk-forward QLIKE; D6)."""
    set_alias(client, name, CHAMPION_ALIAS, version)


def promote_to_production(client: MlflowClient, name: str, version: int | str) -> None:
    """Перевести версию в ``production`` — ручной шаг рунбука §12 после ревью метрик."""
    set_alias(client, name, PRODUCTION_ALIAS, version)

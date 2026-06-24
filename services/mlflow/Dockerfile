# Тонкий слой над официальным образом MLflow: добавляем драйвер PostgreSQL.
# Официальный ghcr.io/mlflow/mlflow его НЕ содержит — без драйвера сервер с PostgreSQL
# backend-store не стартует (mlflow/mlflow#9513). Версия пинится под клиент 3.14.0
# (ml/pyproject mlflow>=3.1,<4), чтобы схема реестра совпадала.
#
# psycopg2, НЕ psycopg(v3): в MLflow 3.x колонка `workspace` стала ведущим компонентом PK
# во всех таблицах реестра (PR #20657), и на диалекте psycopg3 SQLAlchemy биндит
# integer-параметр `model_versions.version` с `::VARCHAR`-кастом → PostgreSQL строг к типам
# (`operator does not exist: integer = character varying`), из-за чего падает ЛЮБАЯ операция
# по версии модели: set/get alias и резолюция `models:/<name>@<alias>` (путь инференса API).
# psycopg2 биндит integer корректно (аналог mlflow/mlflow#9151). Проверено на проде:
# get_model_version и set_registered_model_alias проходят на psycopg2 поверх той же схемы.
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir "psycopg2-binary>=2.9"

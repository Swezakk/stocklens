# Тонкий слой над официальным образом MLflow: добавляем драйвер PostgreSQL psycopg (v3).
# Официальный ghcr.io/mlflow/mlflow его НЕ содержит — без драйвера сервер с backend-store
# postgresql+psycopg:// не стартует (mlflow/mlflow#9513). Версия пинится под клиент 3.14.0
# (ml/pyproject mlflow>=3.1,<4), чтобы схема реестра совпадала.
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir "psycopg[binary]>=3.3"

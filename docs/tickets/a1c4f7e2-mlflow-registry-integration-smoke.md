---
id: a1c4f7e2
title: "MLflow-реестр: интеграционный smoke client→PG-server→register→load"
status: resolved
priority: medium
component: services/api
discovered: 2026-06-24
discovered-from: [5g-mlflow-serving]
---

## Что отложено

Связка «клиент логирует → PG-backed MLflow-сервер хранит → API грузит по
`models:/stocklens-volatility@<alias>`» проверена **по частям, но не вместе**:

- логика log/register/load — на sqlite-бэкенде (`ml/tests/test_pyfunc_volatility.py`,
  `test_train_volatility.py`);
- сам сервер на PostgreSQL + `/health` — отдельным end-to-end-стартом (53 таблицы схемы,
  psycopg 3.3.4), без регистрации модели через него.

Не покрыт совместный путь: резолв артефактов через `--serve-artifacts`
(`mlflow-artifacts://`) и загрузка по алиасу с PG-реестра. Именно там прячутся
интеграционные сюрпризы.

## Почему отложено

Это §11.3 спеки (smoke интеграции) — относится к фазе API-serving (`services/api/src/api/ml/`),
которой ещё нет. Для 5g (оффлайн-обучение + инфраструктура реестра) совместный smoke не
требуется; разнесённая проверка достаточна.

## Критерий готовности (фаза API-serving, §8/§11.3)

- [ ] Поднять `mlflow` в compose, `register_champion(...)` против него (PG backend).
- [ ] `mlflow.pyfunc.load_model("models:/stocklens-volatility@champion")` + `predict` —
      из процесса БЕЗ `stocklens_ml` (среда API-образа).
- [ ] Проверить резолв артефактов через сервер (`--serve-artifacts`), не прямой доступ к тому.
- [ ] Включить в CI smoke compose-стека (§11.3): `mlflow` healthy → API `lifespan` грузит
      модель → `/health/ready` = 200.

## Resolution (2026-06-24, фаза D) — путь доказан вручную на проде

Совместный путь проверен **на проде целиком**: тренер залогировал → PG-backed mlflow-сервер
сохранил → прод-API загрузил `models:/stocklens-volatility@production` через
`--serve-artifacts` и отдал реальный прогноз (`POST /predict/volatility` SBER → GARCH,
v1, QLIKE 0.698 vs baseline 1.589). Артефакт-резолюция через serve-artifacts работает.
Ключевое открытие: на PostgreSQL нужен **psycopg2**, не psycopg3 (иначе
`operator does not exist: integer = character varying` на любой операции по версии модели).
**Остаётся:** автоматизировать как CI smoke compose-стека (§11.3) — пока проверка ручная.

## Resolution (2026-06-26)

Совместный путь автоматизирован как testcontainers-smoke
(`services/api/tests/integration/test_mlflow_registry_smoke.py`,
`@pytest.mark.integration`). Сценарий: фикстура собирает образ `stocklens-mlflow:test`
из `services/mlflow/Dockerfile` → поднимает PostgreSQL 16 + MLflow-сервер
(`--serve-artifacts`) в общей docker-сети → регистрирует обе модели против сервера
(StubVol pyfunc → `stocklens-volatility`, крошечный CatBoost → `stocklens-trend`, обеим
алиас `production`) → прод-loader `api.ml.loader.load_bundle` грузит обе по
`models:/<name>@production` через serve-artifacts на psycopg2-бэкенде → `/health/ready`
отдаёт 200.

Два теста закрывают оба критерия §11.3:
- **TEST 1** (`test_load_bundle_loads_both_models_by_alias_via_serve_artifacts`):
  реальный `load_bundle` против живого сервера; `bundle.ready()` True,
  волатильность (version `1`, method `garch`) и тренд (version `1`) загружены —
  это и есть интеграционный риск, ранее проверявшийся лишь на sqlite.
- **TEST 2** (`test_health_ready_returns_200_when_models_loaded_from_registry`):
  предпочтённый путь — приложение поднимается через реальный `lifespan`
  (`LifespanManager`), именно он делает `load_bundle` из MLflow-контейнера; без подмены
  `app.state.ml`. GET `/health/ready` → 200, `models = ok`, `status = ready`.

Прогон — в существующем CI-job `api` (testcontainers + Docker уже доступны на раннере;
integration-тесты не деселектятся — дефолтного `-m "not integration"` нет). Правок `ci.yml`
не потребовалось. **Важно:** слой-кэш делает пересборку идемпотентной **локально**, но между
ранами GitHub Actions persistent-кэша слоёв нет — каждый api-CI-ран тянет базовый
`ghcr.io/mlflow/mlflow` (~1GB) холодно. Оптимизация (pull прод-образа / GHA-кэш / вынос в
отдельный job) — [#c7a2f9e4](c7a2f9e4-mlflow-smoke-cold-image-build-ci-cost.md).

Три несущих элемента сетапа (load-bearing, иначе путь не воспроизводится) зафиксированы
в docstring теста:
- **образ с psycopg2** — стоковый `ghcr.io/mlflow/mlflow` без `psycopg2-binary` не
  стартует с PostgreSQL-бэкендом; собираем свежий `stocklens-mlflow:test`;
- **существующая БД `mlflow`** — `PostgresContainer(... dbname="mlflow",
  driver="psycopg2")` создаёт её, а бэкенд-URI сервера — `postgresql+psycopg2://`
  (psycopg3 ломает резолюцию `models:/<name>@<alias>`);
- **`MLFLOW_SERVER_ALLOWED_HOSTS=localhost:*,127.0.0.1:*`** — без host:port-маски
  DNS-rebinding guard MLflow отдаёт 403 на динамическом порту testcontainers.

Проверено: `pytest …/test_mlflow_registry_smoke.py -m integration` → 2 passed (~18с);
полный `pytest services/api/tests -m integration` → 95 passed, без регрессий;
mypy strict и ruff (0.15.19) check/format — зелёные.

---
id: c7a2f9e4
title: "MLflow-smoke собирает образ холодно на каждом CI-ране api (~1GB базы)"
status: open
priority: low
component: ci
discovered: 2026-06-26
discovered-from: [a1c4f7e2-mlflow-registry-smoke]
tags: ["ci", "performance", "testcontainers", "mlflow"]
---

# c7a2f9e4: холодная сборка mlflow-образа в CI-джобе api

## Что обнаружено

Интеграционный smoke реестра MLflow ([#a1c4f7e2](a1c4f7e2-mlflow-registry-integration-smoke.md))
собирает образ `stocklens-mlflow:test` из `services/mlflow/Dockerfile` прямо в фикстуре
(`_build_mlflow_image`, `docker build`). Локально слой-кэш Docker делает повторную сборку
мгновенной — но **между ранами GitHub Actions persistent-кэша слоёв нет**: каждый прогон
api-джобы тянет базовый `ghcr.io/mlflow/mlflow:v3.14.0` (~1GB) и ставит `psycopg2-binary`
заново. Это добавляет минуты к **каждому** пушу/PR, затрагивающему api-джобу.

## Почему это проблема

`@pytest.mark.integration`-тесты api-джобы не деселектятся (нет дефолтного `-m "not
integration"`), значит smoke идёт на каждом CI-ране api — и платит за холодную сборку каждый
раз. Функционально корректно, но дорого по wall-clock CI.

## Что можно сделать (варианты, требует решения)

1. **Тянуть прод-образ вместо сборки** — `docker pull ghcr.io/swezakk/stocklens-mlflow:<tag>`
   (публикуется job `publish`), фикстура использует готовый тег. Один pull вместо build;
   но связывает тест с реестром и актуальностью опубликованного образа.
2. **GHA-кэш слоёв** — `docker/build-push-action` с `cache-from/to: type=gha` в шаге
   перед pytest; фикстура использует готовый тег. Хермитично (тестирует реальный Dockerfile),
   кэш переживает раны.
3. **Вынести smoke в отдельный/реже идущий job** (nightly или only-on-main), оставив
   быструю api-джобу без него.

Предпочтительно (2): сохраняет хермитичность сборки из Dockerfile и убирает холодный pull.

## Acceptance criteria

- Образ MLflow не собирается холодно с нуля на каждом api-CI-ране (pull/cache/вынос).
- Smoke по-прежнему идёт в CI и проверяет реальный psycopg2+serve-artifacts путь.
- `services/mlflow/Dockerfile` остаётся источником истины образа.

## Sources

- `services/api/tests/integration/test_mlflow_registry_smoke.py::_build_mlflow_image`
  (`docker build` в фикстуре).
- `.github/workflows/ci.yml` — job `api` (`pytest services/api/tests`, integration не
  деселектится).
- Обнаружено: review закрытия [#a1c4f7e2](a1c4f7e2-mlflow-registry-integration-smoke.md).

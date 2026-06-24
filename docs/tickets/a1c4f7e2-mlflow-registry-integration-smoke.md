---
id: a1c4f7e2
title: "MLflow-реестр: интеграционный smoke client→PG-server→register→load"
status: open
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

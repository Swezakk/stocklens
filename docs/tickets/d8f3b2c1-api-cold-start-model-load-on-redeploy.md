---
id: d8f3b2c1
title: "api не догружает модель при полном редеплое (cold-start) — решить depends_on"
status: open
priority: medium
component: deploy
discovered: 2026-06-24
discovered-from: [phase-d-ml-activation]
---

## Проблема

У `api` в [docker-compose.prod.yml](../../docker-compose.prod.yml) **нет**
`depends_on: mlflow` (mlflow намеренно аддитивен). При ЛЮБОМ полном редеплое, где
`api` и `mlflow` пересоздаются вместе, `api` стартует раньше, чем `mlflow` дойдёт до
healthy (~45с), а `load_bundle` (внутри `lifespan`) ограничен ~12с
(`ML_LOAD_ATTEMPTS=3`×`ML_LOAD_INTERVAL_SECONDS=2` + `MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`).
Итог: `load_bundle` сдаётся → `/health/ready` = `models:unavailable`, ML молча degraded,
**пока api не перезапустить вручную** (`docker restart …-api-1`) уже после mlflow healthy.

Это **не падение** — `ML_REQUIRED_FOR_READY=false` держит стек healthy (dashboard/bot
поднимаются), ML просто не работает до ручного рестарта. Подтверждено в фазе D: модель
загрузилась только после ручного `docker restart` api, когда mlflow был уже healthy.

## Развилка (решает владелец — это trade-off связности)

1. **`depends_on: mlflow: condition: service_healthy` на api** — api ждёт mlflow →
   `load_bundle` резолвит `@production` с первой попытки, авто-загрузка без ручного рестарта.
   Минус: критический путь (api→dashboard→bot) связывается с mlflow — если mlflow когда-то
   не дойдёт до healthy, весь стек не поднимется (теряется текущая graceful-degradation:
   «ML сломан, но core жив»). Риск OOM-loop mlflow, который делал это опасным, СНЯТ
   (psycopg2 + `mem_limit 1536m`, стабилен).
2. **Оставить как есть** + операционное правило: после полного редеплоя `docker restart`
   api, когда mlflow healthy. Сохраняет изоляцию ML-сбоев от core, но требует ручного шага
   (легко забыть → ML тихо degraded).

## Критерий готовности

- [ ] Решение по развилке зафиксировано (с владельцем).
- [ ] Если (1): добавить `depends_on` + проверить полный редеплой → `models:ok` без ручного
      рестарта; если (2): записать правило в `docs/deploy.md` (рунбук редеплоя).

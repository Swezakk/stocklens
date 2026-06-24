---
id: b9d3e5a8
title: "Прод-топология MLflow-реестра: куда логирует переобучение и как ревьюить"
status: resolved
priority: medium
component: deploy
discovered: 2026-06-24
discovered-from: [5g-mlflow-serving]
---

## Что решить

Прод-сервис `mlflow` в [docker-compose.prod.yml](../../docker-compose.prod.yml) — только
`expose` (наружу не публикуется). Прод-API грузит модель из **прод-реестра**
(`models:/stocklens-volatility@production`), значит:

- ручное переобучение (`ml/README.md`, рунбук §12) должно логировать в **прод**-MLflow,
  а не в локальный dev-сервер — иначе прод-реестр пуст и API нечего грузить;
- прод-MLflow достижим только внутри docker-сети VPS, поэтому и логирование, и ревью метрик
  в UI требуют доступа извне.

Развилка (решить на фазе деплоя E):

1. **SSH-туннель** к `mlflow:5000` на время переобучения/ревью (ничего наружу не публикуется,
   максимально закрыто; неудобно для регулярного UI).
2. **Traefik-роут** на поддомен (как для API/дашборда) с авторизацией — удобный UI, но
   расширяет периметр; нужна защита (basic-auth/oauth) на роут.

## Почему отложено

Это вопрос фазы деплоя (E), а не реализации ML-слоя. Compose для `mlflow` уже готов; выбор
доступа не влияет на код 5g.

## Критерий готовности (фаза деплоя)

- [ ] Зафиксировать в `docs/deploy.md`: переобучение логирует в прод-MLflow (через выбранный
      доступ), `MLFLOW_TRACKING_URI` рунбука = прод-сервер.
- [ ] Реализовать доступ (туннель-инструкция ИЛИ Traefik-роут + авторизация).
- [ ] Создать external-том `stocklens_mlflow_artifacts` на VPS до первого деплоя mlflow.
- [ ] Прогнать первичное обучение против прод-БД → проверить, что прод-API грузит champion.

## Resolution (2026-06-24, фаза D) — основное снято

Доступ для обучения решён **третьим вариантом** (не туннель / не Traefik): one-off контейнер
на compose-сети VPS (`…_default`, репо из `/etc/dokploy/compose/…/code`,
`--mlflow-uri http://mlflow:5000`; детали — память deploy). Сделано: external-том
`stocklens_mlflow_artifacts` создан; первичное обучение против прод-БД прошло (GARCH бьёт
baseline), `stocklens-volatility` v1 зарегистрирована (alias `champion`+`production`),
прод-API грузит `@production` и отдаёт реальный прогноз. Потребовалось: mlflow
`--allowed-hosts` (security-middleware 3.x) + **psycopg2** (psycopg3 ломает реестр на PG).
UI-ревью метрик решён **ad-hoc SSH-туннелем** (выбор владельца, не Traefik): команда
форвардит порт mlflow с IP контейнера, `http://localhost:5000` в браузере; задокументировано
в `docs/deploy.md` (раздел «MLflow — реестр моделей и ревью метрик»). Тикет закрыт.

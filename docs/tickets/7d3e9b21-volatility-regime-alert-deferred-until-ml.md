---
id: 7d3e9b21
title: "volatility_regime: оценка алерта отложена до ML-слоя"
status: open
priority: low
component: services/api
discovered: 2026-06-23
discovered-from: [b2-alerts]
---

## Что отложено

Вид алерта `AlertKind.VOLATILITY_REGIME` объявлен в `stocklens-core`, но **не оценивается**:

- `AlertEvaluationService.collect_pending()` (`services/api/src/api/services/alert_evaluation.py`)
  пропускает его явным `continue`.
- Парсер бота (`services/bot/src/bot/subscriptions.py`) не даёт на него подписаться
  (возвращает «отложен до ML»).

## Почему

Алерт «смена режима волатильности» требует ML-прогнозов волатильности
(`predictions`, `kind=volatility`) — ML-слой (§8 спеки) ещё не реализован, каталога
`services/api/src/api/ml/` нет.

## Критерий готовности (когда появится ML-инференс волатильности)

- [ ] Определить контракт params (порог режима) в `schemas/bot.py` + валидацию в `BotSubscriptionService`.
- [ ] Добавить `_evaluate_volatility_regime()` в `AlertEvaluationService`: прогноз vs порог.
- [ ] Дедуп-ключ `alert:vol:{chat_id}:{ticker}:{YYYY-MM-DD}` (TTL 86400).
- [ ] Разрешить подписку в парсере бота + `format_alert` ветку.
- [ ] Unit + интеграционные тесты ветки; обновить спеку §11 (убрать пометку «отложен»).

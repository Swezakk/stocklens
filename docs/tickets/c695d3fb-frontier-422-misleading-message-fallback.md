---
id: c695d3fb
title: Эффективная граница — вводящее в заблуждение сообщение «нужно ≥ 2 бумаги» при нерешаемом max_sharpe
status: resolved
priority: medium
component: dashboard
discovered: 2026-06-26
resolved: 2026-06-27
discovered-from: []
tags: ["portfolio", "optimization", "ux", "diagnostics", "demo-data"]
---

# c695d3fb: Эффективная граница — вводящее в заблуждение сообщение «нужно ≥ 2 бумаги» при нерешаемом max_sharpe

## What was observed

На странице «Портфель» с тремя позициями (AFKS, TATN, MAGN — то есть `3 ≥ 2`)
секция «Эффективная граница» показывает заглушку
«Недостаточно данных для построения эффективной границы (нужно ≥ 2 бумаги)».
Условие по количеству бумаг формально выполнено, но граница не строится.

Фактическая причина — не количество тикеров. Цепочка:

1. Дашборд запрашивает `/portfolio/optimize` со стратегией по умолчанию `MAX_SHARPE`
   (`services/dashboard/src/dashboard/api_client/fetch.py:356`).
2. На сервере вызывается `ef.max_sharpe(risk_free_rate=annual_rate)`
   (`services/api/src/api/analytics/optimization.py:168`), где
   `annual_rate = ключевая ставка ЦБ / 100` (`services/api/src/api/services/portfolio.py:293-296`).
3. В pypfopt стоит жёсткий guard
   (`.venv/.../pypfopt/efficient_frontier/efficient_frontier.py:244-247`):
   ```python
   if max(self.expected_returns) <= risk_free_rate:
       raise ValueError(
           "at least one of the assets must have an expected return exceeding the risk-free rate"
       )
   ```
   Касательный (tangency) портфель невозможен, если ни у одной бумаги годовая
   историческая доходность не превышает безрисковую ставку.
4. `ValueError` маппится в `InvalidStrategyParamsError` → HTTP 422
   (`services/api/src/api/services/portfolio.py:220-224`).
5. Дашборд трактует **любой** 422 от оптимизации как «пустое состояние»
   (`services/dashboard/src/dashboard/pages/portfolio.py:243-258`) и рисует общую
   подпись `_EMPTY_FRONTIER` = «нужно ≥ 2 бумаги»
   (`services/dashboard/src/dashboard/pages/portfolio.py:529-533`).

Реальный кейс срабатывания: портфель убыточный (доходность −28.91%, Sharpe −1.84,
все три бумаги снижаются за период), безрисковая ставка = высокая ключевая ставка ЦБ,
поэтому `max(годовая доходность) ≤ ставка` и `max_sharpe` отказывается решать задачу.

## Why it is a problem

- **Диагностируемость / UX.** К одной и той же подписи «нужно ≥ 2 бумаги» сходятся
  минимум три разных причины 422 (мало совмещённых торговых дат → `InsufficientDataError`;
  нерешаемый солвер → `InvalidStrategyParamsError`; пустая граница при `mu_min ≥ mu_max`),
  плюс ветка `result.frontier == []` без 422
  (`services/dashboard/src/dashboard/pages/portfolio.py:510-512`). Владелец видит причину,
  которой нет (тикеров достаточно), и не понимает, что делать. Нарушает проектное правило
  «4xx — по-русски, с сущностью и причиной» и принцип диагностируемости из CLAUDE.md.
- **Продуктовая логика.** На убыточном рынке max-Sharpe в принципе нефизичен (депозит под
  ключевую ставку выгоднее любого набора падающих акций). Минимизация волатильности
  (`min_volatility`) при этом решается всегда — она не требует превышения ставки, — но
  владельцу её не предлагают: секция просто пустеет.

## Why it is not a duplicate

- [#f4a7c2e1](f4a7c2e1-trend-model-no-edge-negative-result.md) — про отрицательный
  результат trend-модели (ML edge); этот тикет про оптимизатор Марковица и отображение
  ошибки в дашборде. Разные слой и компонент.

## What probably needs to be done

Три задачи в рамках одного тикета:

1. **Различать причины 422 (диагностика/UX).** Перестать сливать все 422 от
   `/portfolio/optimize` в общую подпись «нужно ≥ 2 бумаги». Пробрасывать осмысленное
   сообщение из `detail` ответа (RFC 9457 Problem Details) и показывать его, например:
   «Граница не строится: ни одна бумага портфеля не обгоняет ключевую ставку ЦБ за период».
   Точку различения держать в API (отдельные доменные исключения / коды) либо на дашборде
   по тексту `detail` — выбрать в плане. *requires verification* какие именно доменные
   подтипы 422 завести, чтобы не размазывать бизнес-правило между API и фронтом
   (бэкенд — источник истины).
2. **Фолбэк на min-volatility, когда max_sharpe нефизичен.** Когда max-Sharpe нерешаем
   (`max(mu) ≤ rf`), считать и показывать min-volatility-портфель и его границу
   (`services/api/src/api/analytics/optimization.py:46-68` + `compute_frontier_points`),
   с явной пометкой в UI, что max-Sharpe недоступен при текущей ставке. Решить, делать ли
   фолбэк автоматически в сервисе или предложить владельцу выбор стратегии в дашборде.
   *requires verification* поведения `compute_frontier_points` на полностью отрицательных
   `mu` (точки `efficient_return` должны решаться при long-only bounds (0,1)).
3. **Подобрать значения демо-портфеля, чтобы оптимизатор работал (витрина/скриншоты).**
   Для демо-данных (витрина README + скриншоты, см. коммит 61830ee) подобрать такой набор
   тикеров / период / позиции, чтобы хотя бы у одной бумаги годовая историческая доходность
   превышала ключевую ставку и max-Sharpe был решаем — тогда эффективная граница рисуется
   на демо-портфеле. Зафиксировать подобранный набор там, где задаётся демо/seed-портфель,
   и обновить скриншот витрины. *requires verification* где именно живёт демо/seed-портфель
   (seed-скрипт / фикстура дашборда) — на момент заведения тикета не локализовано.

## Acceptance criteria

- При нерешаемом max-Sharpe дашборд показывает осмысленную причину (не «нужно ≥ 2 бумаги»),
  отличимую от случаев «<2 тикеров» и «мало истории»; тексты — по-русски, с причиной.
- Реализован фолбэк/выбор min-volatility: на убыточном портфеле секция «Эффективная граница»
  не пустеет, а показывает решаемую границу либо внятную альтернативу.
- На демо-портфеле витрины эффективная граница строится; скриншот обновлён.
- Регрессионные тесты: unit на сервис `optimize()` (ветка `max(mu) ≤ rf` → ожидаемое
  поведение фолбэка/сообщения, а не общий 422) и на дашбордную ветку рендера причины;
  существующие тесты `services/api/tests/unit/test_analytics.py`,
  `services/api/tests/integration/test_portfolio_routes.py`,
  `services/dashboard/tests/` зелёные.

## Sources

- `services/dashboard/src/dashboard/pages/portfolio.py:243-258` — `_is_empty_state_error` / `_render_load_failure` (все 422 → пустое состояние).
- `services/dashboard/src/dashboard/pages/portfolio.py:504-533` — `_render_frontier_section` / `_load_optimize`, единая подпись `_EMPTY_FRONTIER`.
- `services/dashboard/src/dashboard/pages/portfolio.py:98` — текст `_EMPTY_FRONTIER`.
- `services/api/src/api/services/portfolio.py:175-229` — `optimize()` и маппинг `ValueError`/`OptimizationError` → 422.
- `services/api/src/api/services/portfolio.py:293-296` — `_resolve_annual_rate` (ключевая ставка как безрисковая).
- `services/api/src/api/analytics/optimization.py:19-107` — `build_max_sharpe_weights` / `build_min_volatility_weights` / `compute_frontier_points`.
- pypfopt `EfficientFrontier.max_sharpe`: guard `max(expected_returns) <= risk_free_rate` → `ValueError`.
- Коммит 61830ee — витрина README + скриншоты (демо-данные).

## Resolution (2026-06-27)

Задачи 1 и 2 выполнены; задача 3 осознанно отложена в новую фичу.

- **Задача 1 — различить причины 422.** Дашборд `_render_load_failure` на пустом 422
  показывает реальный серверный `detail` (RFC 9457, русский текст с причиной — backend
  источник истины), хардкод секции остаётся graceful-фолбэком при дженерик-`detail`.
  Касается всех трёх секций (позиции/equity/граница). Коммит `b615f7f`.
- **Задача 2 — фолбэк на min-volatility.** Когда max-Sharpe нерешаем, API авто-фолбэчит
  на min-volatility и возвращает рабочий портфель (HTTP 200) с полями
  `requested_strategy` + `fallback_reason` вместо 422. Два пути фолбэка: предикат
  `max(mu) ≤ rf` (точное зеркало pypfopt-гарда, единый `_expected_returns`) и
  `try/except OptimizationError` для численной нестабильности солвера у границы (без
  магических margin). Дашборд рисует честный баннер `render_info` над графиком,
  независимо от пустоты фронтира. Коммиты `5b7f2ae` (API) + `38bb4b0` (e2e-тест) +
  `b615f7f` (dashboard).
- **Задача 3 — подбор демо-портфеля под витрину — отложена.** По решению владельца
  свёрнута в новую фичу капитал-аллокации (ввод суммы/срока/стратегии → Марковиц
  подбирает тикеры), которая сама даст оптимизируемый портфель для скриншотов. Не
  потеряна — переходит в скоуп фичи.

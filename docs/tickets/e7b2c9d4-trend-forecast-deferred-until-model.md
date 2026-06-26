---
id: e7b2c9d4
title: "Прогноз тренда (P↑ + SHAP) отложен до trend-модели"
status: open
priority: low
component: services/api
discovered: 2026-06-24
discovered-from: [m3-forecasts-page]
---

## Что отложено

Часть «Тренд» страницы дашборда «Прогнозы» (ml-spec §10): `P(up)` с доверительной подачей,
**SHAP-объяснение** (bar вкладов фич) и направление. Сейчас на странице — честная статичная
строка (`forecasts.py::_render_trend_note`), без интерактивной заглушки.

Реализована только волатильность (прогноз vs факт + QLIKE; коммиты cccf1c5 API + f34cd26
дашборд). Тренд-вертикали нет:

- нет `/predict/trend` (роутер `predict.py` — только волатильность);
- trend-модель не обучена и не зарегистрирована (реестр пуст по `stocklens-trend`).

## Почему отложено

Тренд-модель — отдельная вертикаль (ml-spec **D1/§8.3**: CatBoost price-only + baseline
«всегда вверх»). M1 покрыл только волатильность (GARCH/HAR). Без обученной модели
`P(up)`/SHAP взять неоткуда; фейковая заглушка нарушила бы «backend — источник истины».

## Критерий готовности (фаза trend-модели, §8.3/§11.2)

- [ ] Обучить trend-модель walk-forward + baseline-гейт (регистрировать только если бьёт
      «всегда вверх»; D6), алиас `production` — как у волатильности.
- [ ] `POST /predict/trend` → `prob_up ∈ [0,1]` + `shap` (вклады фич; сумма + base ≈ margin,
      additivity) + направление; тест `predict_trend_returns_prob_and_shap` (§11.2).
- [ ] Загрузка trend-модели в `ModelBundle` (lifespan) рядом с волатильностью; readiness-гейт.
- [ ] Дашборд: секция «Тренд» на странице «Прогнозы» — `P(up)` доверительной подачей
      (не «сигнал»), SHAP-bar, направление; три ветки сетевого вызова.
- [ ] Снять статичную строку `_render_trend_note`; обновить DESIGN.md §10.4 + ml-spec §10.

## Update (2026-06-26, фазы T1–T4)

- [x] `POST /predict/trend` → `prob_up` + `shap` (list[ShapContribution], additivity) +
      `direction` реализован; загрузка нативного CatBoost в `ModelBundle` (lifespan),
      readiness-гейт — тренд неблокирующий. Serving **код-комплит** (T2).
- [ ] Обучение/регистрация — **trend-модель не имеет edge**: walk-forward + гиперпоиск на
      прод-данных дали mean ROC-AUC ≈ 0.49 (< 0.5), baseline-гейт корректно НЕ зарегистрировал.
      См. [#f4a7c2e1](f4a7c2e1-trend-model-no-edge-negative-result.md). Активация (реестр →
      `production` → деплой → дашборд) **отложена** до sentiment-фичи + достаточной глубины
      новостей. Заглушка `_render_trend_note` остаётся честной.

---
id: 3455b248
title: "Trend-артефакт: signature декларирует P(up) float, нативный pyfunc.predict отдаёт класс {0,1}"
status: resolved
priority: medium
component: ml
discovered: 2026-06-25
discovered-from: ["#e7b2c9d4"]
tags: ["ml", "mlflow", "serving", "catboost", "signature", "contract"]
---

# 3455b248: Trend-артефакт: signature декларирует P(up) float, нативный pyfunc.predict отдаёт класс {0,1}

## Resolution

Serving-вертикаль T2 грузит trend-модель через `mlflow.catboost.load_model` и берёт P(up) из
`predict_proba(...)[:, 1]` (минуя pyfunc `.predict()`, отдающий метку класса), SHAP считается
on-demand нативным `get_feature_importance` — декларированная signature совпадает с инференсом.

## What was observed

`train_trend.register_champion` логирует CatBoost-классификатор **нативным**
`mlflow.catboost.log_model` (как требует контракт фазы T1: нативный flavor, не pyfunc-обёртка).
Signature выводится из `model.predict_proba(example)` → output-схема артефакта = float P(up)
(вероятность роста).

Но pyfunc-загрузка нативного flavor (`mlflow.pyfunc.load_model(uri).predict(...)`) вызывает
у `CatBoostClassifier` метод `.predict()`, который для бинарной классификации возвращает
**метки классов**, а не вероятности. Проверено эмпирически на зарегистрированной версии:

```
PREDICT_OUTPUT: [1. 0. 0. 1. 0.]
UNIQUE: [0.0, 1.0]
```

То есть декларированная output-схема артефакта (float P(up) ∈ [0,1]) не совпадает с тем, что
вернёт pyfunc-инференс по этому артефакту (дискретный класс 0/1).

## Why it is a problem

Контракт serving — `prob_up ∈ [0,1]` (ml-spec §11.2, см. критерий готовности в
[#e7b2c9d4](e7b2c9d4-trend-forecast-deferred-until-model.md)). API-слой, грузящий trend-модель
по `models:/stocklens-trend@production`, рассчитывает получить вероятность для доверительной
подачи и SHAP. Если он пойдёт через pyfunc `.predict()`, получит дискретный класс — тихая
потеря информации (нет калибровки/порога), а декларированная signature будет «врать» о форме
выхода. Корректность serving-контракта, не текущей оффлайн-фазы.

В текущей фазе T1 это **не баг**: задача явно предписывает нативный `mlflow.catboost` flavor и
signature из `predict_proba`; отклоняться сейчас нельзя. Тесты T1 (версия + алиас через
`MlflowClient`) этот зазор не видят — артефакт грузится корректно, регистрация проходит.
Проблема материализуется только на фазе serving.

## Why it is not a duplicate

- [#e7b2c9d4](e7b2c9d4-trend-forecast-deferred-until-model.md) — про **отсутствие**
  trend-вертикали в API (нет `/predict/trend`, модель не обучена/не зарегистрирована). Этот
  тикет — про **зазор контракта уже зарегистрированного** артефакта: как именно serving-слой
  должен извлечь P(up), чтобы декларированная signature не расходилась с инференсом. Разные
  корневые причины: там — «фичи нет», тут — «способ инференса по существующему flavor».

## What probably needs to be done

На фазе serving (не сейчас) выбрать один из вариантов и реализовать его в API-`ml/`:

- Грузить нативный CatBoost-flavor через `mlflow.catboost.load_model(uri)` (вернёт
  `CatBoostClassifier`) и вызывать `predict_proba(...)[:, 1]` — P(up) напрямую, минуя
  pyfunc `.predict()`; либо
- Залогировать тонкую pyfunc-обёртку поверх native-модели, чья `predict()` возвращает
  `predict_proba(...)[:, 1]`, и выводить signature из неё — тогда `pyfunc.predict()` сразу
  даёт P(up) (ценой ещё одного flavor в артефакте — требует verification, не противоречит ли
  контракту «нативный артефакт»).

Решение «raw-model + predict_proba» vs «pyfunc-обёртка» — за фазой serving; обновить ml-spec
§11.2, если выберут обёртку.

## Acceptance criteria

- API-инференс trend по `models:/stocklens-trend@production` возвращает `prob_up ∈ [0,1]`
  (float), а не дискретный класс.
- Декларированная output-signature артефакта совпадает с тем, что фактически возвращает
  выбранный путь инференса (тест на форму/диапазон выхода).
- SHAP-вклады извлекаются по той же загруженной модели (additivity-инвариант, §11.2).

## Sources

- `ml/src/stocklens_ml/training/train_trend.py` — `register_champion`: `infer_signature(example,
  model.predict_proba(example))` + `mlflow.catboost.log_model(model._model, name="model", ...)`.
- Эмпирическая проверка (эта сессия): native pyfunc `.predict()` на зарегистрированной версии
  `stocklens-trend` вернул `{0.0, 1.0}` (метки), не вероятности.
- ml-spec §11.2 (контракт `/predict/trend`: `prob_up ∈ [0,1]` + SHAP).

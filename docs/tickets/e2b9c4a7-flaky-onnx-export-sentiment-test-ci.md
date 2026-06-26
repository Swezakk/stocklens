---
id: e2b9c4a7
title: "Флакозный ONNX-export в TestOnnxSentimentScorer — сетевой сбой роняет CI ingestor"
status: open
priority: low
component: services/ingestor
discovered: 2026-06-26
discovered-from: [ci-run-28247473823]
tags: ["ci", "flaky", "sentiment", "onnx", "huggingface"]
---

# e2b9c4a7: флакозный ONNX-export sentiment-теста в CI

## Что обнаружено

`services/ingestor/tests/test_sentiment.py::TestOnnxSentimentScorer` (3 теста:
`test_positive_news_classified`, `test_negative_news_classified`,
`test_model_version_matches_id`) экспортирует HF-модель
`cointegrated/rubert-tiny-sentiment-balanced` в ONNX через `python -m optimum.exporters.onnx`
**в момент прогона**. На CI-ране 28247473823 эта subprocess-команда вернула exit 1 → 3 ERROR,
джоба `ingestor` упала (хотя `92 passed`, coverage 84%). **Rerun без единой правки кода —
зелёный**, что подтверждает: сбой транзиентный (скачивание модели с HuggingFace + экспорт —
сетезависимы), не детерминированный слом.

## Почему это проблема

Тест ходит в сеть (HuggingFace Hub) и запускает тяжёлый ONNX-export в каждом CI-ране ingestor.
Кэш `.pytest-onnx-cache/` локально помогает, но между ранами GitHub Actions persistent-кэша
нет — каждый ран качает+экспортит заново, и любой сетевой/HF-хаб хиккап роняет джобу.
Ложные красные CI вынуждают ручные rerun'ы и заглушают настоящие падения ingestor.

## Что можно сделать (варианты, требует решения)

1. **Кэшировать экспортированный ONNX между ранами** — `actions/cache` на каталог
   `.pytest-onnx-cache/` по ключу = id модели; export только при cache-miss.
2. **Ретрай экспорта** — обернуть `optimum.exporters.onnx` в retry с backoff (сетевые сбои
   HF транзиентны); фикстура падает лишь после N попыток.
3. **Предзагрузка модели в образ/шаг CI** — экспорт в отдельном шаге с ретраем до pytest,
   тест использует готовый артефакт.

Предпочтительно (1)+(2): кэш убирает повтор скачивания, ретрай страхует cache-miss.

## Acceptance criteria

- Транзиентный сетевой сбой HF/optimum не роняет джобу `ingestor` (ретрай или кэш).
- ONNX-sentiment по-прежнему реально проверяется (не мок, не skip по умолчанию).
- Нет повторного скачивания модели на каждом ране при наличии кэша.

## Sources

- `services/ingestor/tests/test_sentiment.py::TestOnnxSentimentScorer` (export в фикстуре).
- CI-ран `28247473823`: 3 ERROR `subprocess.CalledProcessError` на `optimum.exporters.onnx`;
  rerun той же джобы — success (доказательство транзиентности).
- Обнаружено: верификация CI при закрытии тикетов d2b8e5f3/1c33eb15/a1c4f7e2 (эта сессия).

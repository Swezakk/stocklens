---
id: d2b8e5f3
title: "train_trend.run()/train_volatility.run() не изолируют падение одного тикера"
status: open
priority: medium
component: ml
discovered: 2026-06-26
discovered-from: [trend-T4-prod-training]
tags: ["ml", "robustness", "training", "invariant"]
---

# d2b8e5f3: per-ticker изоляция в run() обоих тренеров

## Что обнаружено

`train_trend.run()` и `train_volatility.run()` перебирают тикеры **без try/except вокруг
тикера**: `build_ticker_frame → evaluate_frame → log_run` идут напрямую. Падение на одном
тикере роняет **весь прогон** — модель не регистрируется.

Реальные триггеры (обнаружены при T4 на проде):
- `roc_auc` бросает `ValueError` на single-class concatenated test (тонкий/новый листинг);
- `CatBoostError` (НЕ подкласс `ValueError`) на single-class **train**-фолде —
  `Target contains only one unique value`.

В T4 это обошли подмножеством тикеров ≥1000 свечей (исключив HEAD/YDEX/SVCB/UGLD), но при
еженедельном переобучении (ml-spec §12) на растущем/меняющемся наборе тикеров один
вырожденный тикер обвалит весь retrain.

## Почему это проблема

Нарушает проектный инвариант «ошибка одного источника не валит остальные» (CLAUDE.md, правила
данных). `tune_trend.evaluate_grid` **уже** изолирует per-ticker (skip+log
`ticker_skipped`, ловит `ValueError | ArithmeticError | CatBoostError`) — тренеры отстали.

## Что нужно сделать

Добавить per-ticker try/except в `run()` обоих тренеров, зеркаля изоляцию из
`tune_trend.evaluate_grid`: на падении тикера — лог (`ticker_skipped` с причиной) и
`continue`, остальные тикеры оцениваются, прогон завершается и регистрирует champion по
выжившим. Регрессионный тест: один тикер с single-class фолдом пропускается, остальные дают
модель.

## Sources

- `ml/src/stocklens_ml/training/train_trend.py::run` (нет try/except в цикле тикеров);
  то же в `train_volatility.py::run`.
- Паттерн изоляции для зеркалирования — `ml/src/stocklens_ml/training/tune_trend.py`
  (`evaluate_grid`, `except (ValueError, ArithmeticError, CatBoostError)`).
- Обнаружено: T4-обучение тренда (эта сессия), [#f4a7c2e1](f4a7c2e1-trend-model-no-edge-negative-result.md).

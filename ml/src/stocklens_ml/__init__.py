"""StockLens ML — оффлайн-проект обучения моделей волатильности и тренда.

Слои: data (чтение/коррекция котировок) → features (фичи и таргеты) → models
(GARCH/HAR/CatBoost) → eval (walk-forward, QLIKE) → registry (MLflow) → training (CLI).
Контракт — docs/specs/2026-06-23-stocklens-ml-spec.md. Sync-проект (как ingestor).
"""

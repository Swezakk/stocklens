# StockLens

Персональный аналитический веб-сервис по российскому фондовому рынку:
ежедневный сбор котировок Мосбиржи и финансовых новостей → детерминированная
аналитика и ML-прогнозы → веб-дашборд + Telegram-алерты.

Проект выполняет двойную роль: семестровый проект курса «Python & Data Science»
(архитектурная рамка DataPulse) и реальный инструмент для частного инвестора.

## Что умеет (целевой объём)

- **Мониторинг рынка**: дневные свечи ~45 бумаг индекса IMOEX, дивиденды, курсы ЦБ —
  источник MOEX ISS API (официальный, без ключа).
- **Новости и sentiment**: RSS РБК / Коммерсантъ / Интерфакс, тональность —
  rubert-tiny2.
- **Честный ML**: прогноз волатильности (GARCH / HAR-RV против naive baseline,
  walk-forward валидация), вероятностная классификация тренда (CatBoost + SHAP).
  Точечный прогноз цены сознательно не делается — см. спеку, раздел «Не-цели».
- **Портфель**: P&L против IMOEX, риск-метрики, оптимизация Марковица, бэктест.
- **Telegram-бот**: утренний дайджест и алерты по бумагам портфеля.

## Архитектура

Микросервисы в Docker Compose: `db` (PostgreSQL 16) · `redis` · `ingestor`
(APScheduler, сбор и backfill) · `api` (FastAPI, async) · `dashboard` (Streamlit) ·
`bot` (aiogram) · `mlflow` (эксперименты и реестр моделей).

Полный дизайн — [docs/specs/2026-06-11-stocklens-design.md](docs/specs/2026-06-11-stocklens-design.md).
Правила работы с кодом — [CLAUDE.md](CLAUDE.md).

## Статус

Репозиторий на этапе bootstrap: утверждённая дизайн-спека, конфигурация качества
(ruff, mypy, pre-commit, CI). Реализация идёт по фазам:

1. `packages/stocklens-core` — модели данных, enum'ы, настройки
2. Alembic-миграции + PostgreSQL в Compose
3. `services/ingestor` — сбор MOEX
4. RSS + sentiment, ЦБ РФ
5. `services/api` — FastAPI
6. `services/dashboard`, `services/bot`, ML-пайплайн, деплой (Dokploy)

## Запуск

Появится вместе с первым сервисом: целевая команда — `docker compose up --build`.

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

Утверждённая дизайн-спека и конфигурация качества (ruff, mypy, pre-commit, CI)
на месте. Реализация идёт по фазам:

1. `packages/stocklens-core` — модели данных, enum'ы, настройки — **готово**
2. Alembic-миграции + PostgreSQL в Compose — **готово**
3. `services/ingestor` — сбор MOEX — **готово**
4. RSS + sentiment, ЦБ РФ — **готово**
5. `services/api` — FastAPI
6. `services/dashboard`, `services/bot`, ML-пайплайн, деплой (Dokploy)

## Запуск

Скелет стека (БД → миграции → сборщик MOEX) поднимается одной командой;
остальные сервисы добавятся по фазам:

```bash
cp .env.example .env             # задать DB_PASSWORD и прочие секреты
docker compose up -d --build     # db → migrations (alembic) → ingestor
```

При первом старте ingestor выполняет backfill всей истории котировок
(~46 бумаг IMOEX, ≤1 запрос/сек к MOEX ISS — порядка 25 минут) и курсов ЦБ,
далее работает по расписанию: свечи и индекс — 23:55 МСК; справочник бумаг,
дивиденды и сплиты — 08:00; курсы и ключевая ставка ЦБ — 13:00;
RSS-новости (РБК, Коммерсантъ, Интерфакс) с sentiment-скорингом
(rubert-tiny, ONNX) — каждые 30 минут. Новости архива не имеют —
прод-инстанс должен работать непрерывно.

Миграции с хоста (опционально, против compose-БД):

```bash
uv sync
DATABASE_URL=postgresql+psycopg://stocklens:<пароль>@localhost:5432/stocklens \
  uv run alembic upgrade head
```

Интеграционные тесты (нужен Docker):

```bash
uv run pytest tests -m integration                            # миграции
uv run --project services/ingestor pytest services/ingestor/tests  # ingestor
```

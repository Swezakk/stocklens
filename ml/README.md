# stocklens-ml — оффлайн ML-проект (волатильность)

Оффлайн-обучение и оценка моделей волатильности StockLens: данные из БД → фичи →
walk-forward-валидация → регистрация лучшей модели в реестр MLflow. Контракт разработки —
[docs/specs/2026-06-23-stocklens-ml-spec.md](../docs/specs/2026-06-23-stocklens-ml-spec.md),
методология «для новичка» — […-ml-primer.md](../docs/specs/2026-06-23-stocklens-ml-primer.md).

Проект **синхронный** (как ingestor — батчевая обработка), Python 3.12, uv. Serving
(инференс из API) — отдельный пакет `services/api/src/api/ml/` (поздняя фаза).

## Состав

| Слой | Модуль | Назначение |
|------|--------|------------|
| Данные | `data/loader.py`, `data/adjust.py` | чтение свечей/дивидендов/сплитов; split-adjust, total-return, исключение weekend-сессий |
| Фичи | `features/volatility.py`, `technical.py`, `assemble.py` | Паркинсон-прокси, HAR-регрессоры, RV-таргет, сборка фрейма без утечек |
| Модели | `models/garch.py`, `har.py`, `baselines.py` | GARCH(1,1) (эконометрика), HAR-RV (OLS), baseline RW-RV |
| Оценка | `eval/metrics.py`, `walk_forward.py` | QLIKE/RMSE, `TimeSeriesSplit(gap=5)` walk-forward |
| Реестр | `registry/pyfunc_volatility.py`, `promote.py` | serving-обёртка (Models-from-Code), алиасы champion/production |
| Обучение | `training/train_volatility.py` | CLI: фичи → walk-forward → MLflow → регистрация champion |

## Локальная разработка

```bash
uv sync --project ml
uv run --project ml pytest ml/tests                         # тесты (loader — testcontainers, нужен Docker)
uv run --project ml mypy ml/src/stocklens_ml ml/tests       # типизация (strict)
uvx ruff check ml && uvx ruff format --check ml             # линт/формат
```

## Рунбук переобучения (ml-spec §12)

Ручной еженедельный прогон. MLflow-сервер поднимается стеком (`docker compose up -d mlflow`,
backend — PostgreSQL, см. [docker-compose.yml](../docker-compose.yml) §2.3 спеки). Данные —
из прод-БД на **чтение** (там история котировок).

> **Прод vs dev.** Прод-API грузит модель из **прод**-реестра, поэтому переобучение для прода
> должно логировать туда же: `MLFLOW_TRACKING_URI` указывает на прод-MLflow (доступ — туннель
> или Traefik-роут, решается на фазе деплоя, тикет `docs/tickets/b9d3e5a8-…`). Локальный
> `localhost:5000` ниже — для dev-прогонов и отладки.

```bash
# 1. Окружение
uv sync --project ml

# 2. Оценка + регистрация champion (DATABASE_URL — read-only DSN прод-БД; MLFLOW — адрес сервера)
DATABASE_URL='postgresql+psycopg://user:pass@host:5432/stocklens' \
MLFLOW_TRACKING_URI='http://localhost:5000' \
  uv run --project ml python -m stocklens_ml.training.train_volatility \
    --tickers SBER GAZP LKOH --n-splits 5 --mlflow-uri http://localhost:5000
```

Что делает скрипт:

1. По каждому тикеру: фичи → walk-forward (QLIKE/RMSE для baseline RW-RV, HAR-RV, GARCH),
   лог прогона в эксперимент `volatility`.
2. Выбирает **агрегатного победителя** — метод с минимальным средним QLIKE по тикерам.
3. **Baseline-гейт (D6):** если ни один метод не бьёт baseline по среднему QLIKE —
   регистрации нет (лог `no_model_beats_baseline`), не молча.
4. Иначе: фит победителя на полном окне, лог serving-обёртки, регистрация версии
   `stocklens-volatility` и пометка алиасом `champion`.

GARCH переносимого состояния не несёт (рефит на окне при инференсе); HAR несёт пулинговые
OLS-коэффициенты по всем тикерам — оба грузятся в API без зависимости от `stocklens_ml`.

### Продвижение в production (ручное, после ревью метрик в MLflow UI)

`champion` ставится автоматически; прод-алиас `production` (по нему грузит модель API) —
вручную, осознанно:

```python
from mlflow import MlflowClient
from stocklens_ml.registry.promote import promote_to_production

client = MlflowClient()  # MLFLOW_TRACKING_URI должен указывать на сервер
promote_to_production(client, "stocklens-volatility", version="<номер версии champion>")
```

### Откат

Переназначить `production` на предыдущую версию — мгновенно, без передеплоя API
(API грузит модель по алиасу при рестарте):

```python
promote_to_production(client, "stocklens-volatility", version="<предыдущая версия>")
```

## Методологические инварианты (ml-spec)

- Валидация **только** walk-forward / `TimeSeriesSplit` (`gap = HORIZON_DAYS`); случайный
  K-fold запрещён — утечка через препроцессинг главный риск.
- Каждая модель сравнивается с naive baseline; в реестр идёт только бьющая baseline.
- Обучение на данных **пост-2022** (`TRAIN_START = 2022-04-01`, структурный разрыв).
- Weekend-сессии MOEX исключаются из обучения (`is_weekend_session`).
- Точечный прогноз цены не делаем — это не-цель спеки (прогнозируем дисперсию/волатильность).

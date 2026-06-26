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
| Обучение | `training/train_volatility.py`, `train_trend.py` | CLI: фичи → walk-forward → MLflow → регистрация champion (волатильность / тренд) |

## Локальная разработка

`[train]` extra — зависимости обучения (sklearn; позже catboost/shap). В образ API они не
тянутся (serving зависит от stocklens-ml без `[train]`), поэтому для ml/ всегда `--extra train`:

```bash
uv sync --project ml --extra train
uv run --project ml --extra train pytest ml/tests           # тесты (loader — testcontainers, нужен Docker)
uv run --project ml --extra train mypy ml/src/stocklens_ml ml/tests   # типизация (strict)
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
# 1. Окружение (с extra train — обучение использует sklearn)
uv sync --project ml --extra train

# 2. Оценка + регистрация champion (DATABASE_URL — read-only DSN прод-БД; MLFLOW — адрес сервера)
DATABASE_URL='postgresql+psycopg://user:pass@host:5432/stocklens' \
MLFLOW_TRACKING_URI='http://localhost:5000' \
  uv run --project ml --extra train python -m stocklens_ml.training.train_volatility \
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

## Рунбук переобучения тренда (ml-spec §5.4, §6.2, §12)

Симметричен волатильности, но прогнозирует **направление** цены через горизонт (P(up))
CatBoost-классификатором. Та же дисциплина: walk-forward `TimeSeriesSplit(gap=HORIZON_DAYS)`,
сравнение с naive baseline, регистрация только бьющей baseline версии.

```bash
uv sync --project ml --extra train

DATABASE_URL='postgresql+psycopg://user:pass@host:5432/stocklens' \
MLFLOW_TRACKING_URI='http://localhost:5000' \
  uv run --project ml --extra train python -m stocklens_ml.training.train_trend \
    --tickers SBER GAZP LKOH --n-splits 5 --mlflow-uri http://localhost:5000
```

Что делает скрипт:

1. По каждому тикеру: фичи тренда → walk-forward (accuracy/F1/ROC-AUC для CatBoost и always-up
   baseline), лог прогона в эксперимент `trend`. Forward-таргет считается на горизонт `HORIZON_DAYS`;
   хвостовые строки без будущего close (NaN-таргет) отбрасываются перед обучением.
2. Выбирает **агрегатного победителя** — метод с максимальным средним **ROC-AUC** по тикерам.
3. **Baseline-гейт по ROC-AUC, а не accuracy (D6):** always-up baseline даёт accuracy базовой
   ставки (53–57 %) при ROC-AUC = 0.5. Гейт берёт ROC-AUC (инвариант к балансу классов): модель
   засчитывается, только если её средний ROC-AUC **строго** превышает 0.5. Иначе регистрации
   нет (лог `no_model_beats_baseline`), не молча.
4. Иначе: финальный фит CatBoost на полном окне (early-stop валидационный хвост с H-дневным purge
   против утечки), лог **нативным** `mlflow.catboost`, регистрация версии `stocklens-trend` и
   пометка алиасом `champion`.

Артефакт тренда — нативный CatBoost (а не pyfunc-обёртка волатильности): модель самодостаточна,
переносимого ручного состояния не несёт. Продвижение в `production` и откат — те же команды
`promote_to_production`, что и для волатильности, с именем `stocklens-trend`.

### Подбор гиперпараметров тренда (ml-spec §5.4)

Стартовые гиперпараметры спеки (`iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=6.0`)
дали средний walk-forward ROC-AUC ≈ 0.49 по тикерам — модель **не** обошла always-up baseline
(0.5). Спека §5.4 санкционирует финализацию гиперпараметров на walk-forward (консервативно:
малая глубина, сильная регуляризация — данных мало, риск переобучения высок). Скрипт `tune_trend`
выполняет эту финализацию.

```bash
DATABASE_URL='postgresql+psycopg://user:pass@host:5432/stocklens' \
MLFLOW_TRACKING_URI='http://localhost:5000' \
  uv run --project ml --extra train python -m stocklens_ml.training.tune_trend \
    --tickers SBER GAZP LKOH --n-splits 5 --mlflow-uri http://localhost:5000
```

Что делает скрипт:

1. Грузит фрейм каждого тикера **один раз** (DB-чтение — медленная часть), затем переоценивает
   его под каждый конфиг малой консервативной сетки (`depth ∈ {2,3,4} × l2_leaf_reg ∈ {6.0, 12.0}`,
   `learning_rate=0.03`, `iterations=800` с early-stop — 6 конфигов). Сетка узкая намеренно:
   глубокие/слабо регуляризованные варианты исключены, чтобы не подгонять под walk-forward (bias
   отбора).
2. По каждому конфигу — walk-forward `evaluate_trend` (CatBoost + always-up baseline), среднее
   ROC-AUC/accuracy/F1 по тикерам. Тикер, упавший на одном конфиге (например одноклассовая
   test-выборка → ROC-AUC ValueError), пропускается с логом `ticker_skipped` — sweep не падает.
3. Каждый конфиг логируется отдельным прогоном в эксперимент **`trend-tuning`** (параметры +
   средние метрики). **Регистрации модели НЕТ** — это только поиск. В конце — отсортированный по
   ROC-AUC лог и событие `tuning_complete` с лучшим конфигом, его средним ROC-AUC и булевым
   `beats_baseline` (строгое превышение 0.5).

Выбранный конфиг затем используется для реальной регистрации champion одним из двух способов:
зашить его в дефолты `TrendHyperparams` (`models/trend.py`) — тогда CLI `train_trend` подхватит
его автоматически; либо передать программно в `train_trend.register_champion(..., hyperparams=...)`.
`tune_trend` сам в реестр не пишет — он лишь даёт данные для решения.

### Сравнение классов моделей (sanity check)

Тренд-CatBoost на стартовых гиперпараметрах дал средний walk-forward ROC-AUC ≈ 0.49 — модель не
обошла always-up baseline (0.5). Чтобы доказать, что слаб **сигнал**, а не конкретная модель,
скрипт `compare_trend_models` прогоняет ДРУГИЕ классы моделей на ТЕХ ЖЕ фичах и том же
walk-forward.

```bash
DATABASE_URL='postgresql+psycopg://user:pass@host:5432/stocklens' \
MLFLOW_TRACKING_URI='http://localhost:5000' \
  uv run --project ml --extra train python -m stocklens_ml.training.compare_trend_models \
    --tickers SBER GAZP LKOH --n-splits 5 --mlflow-uri http://localhost:5000
```

Три семьи с **фиксированными** конфигами (НИКАКОГО тюнинга альтернатив): `catboost` (REFERENCE,
стартовые гиперпараметры — internal-consistency проверка ≈0.49), `logreg` (`C=1.0`, L2 через
`l1_ratio=0.0`, `max_iter=1000`; per-fold `StandardScaler` фитится на train-строках — антиутечка),
`random_forest` (`n_estimators=200`, `max_depth=4`, `class_weight="balanced"`). Фрейм чистится от
warm-up строк с NaN в фичах (CatBoost терпит NaN, sklearn — нет; отбор по доступности фич, не по
таргету). Каждая семья логируется отдельным прогоном в эксперимент **`trend-model-comparison`**
(конфиг + средние ROC-AUC/accuracy/F1). **Регистрации модели НЕТ** — это одноразовый
подтверждающий эксперимент. В конце — событие `comparison_complete` со списком всех семей и их
превышением baseline.

## Методологические инварианты (ml-spec)

- Валидация **только** walk-forward / `TimeSeriesSplit` (`gap = HORIZON_DAYS`); случайный
  K-fold запрещён — утечка через препроцессинг главный риск.
- Каждая модель сравнивается с naive baseline; в реестр идёт только бьющая baseline.
- Обучение на данных **пост-2022** (`TRAIN_START = 2022-04-01`, структурный разрыв).
- Weekend-сессии MOEX исключаются из обучения (`is_weekend_session`).
- Точечный прогноз цены не делаем — это не-цель спеки (прогнозируем дисперсию/волатильность).

# StockLens — ML-блок: контракт разработки

**Статус:** контракт первого захода ML (волатильность + тренд + serving).
**Дата:** 2026-06-23.
**Подчинён** главной спеке [docs/specs/2026-06-11-stocklens-design.md](2026-06-11-stocklens-design.md)
(§8 ML-пайплайн, §9 API). Расхождение этого документа с главной спекой — баг документа.
**Парный документ:** [2026-06-23-stocklens-ml-primer.md](2026-06-23-stocklens-ml-primer.md) —
пояснительная записка «что и почему» простым языком (для владельца, не контракт).

Этот документ — **что и как строить**: раскладка кода, контракты данных, точные формулы
фич и моделей, протокол валидации, конвенции MLflow, API-эндпоинты, критерии приёмки и
тест-план. Всё ниже — для исполняющего агента; формулы и API-сигнатуры **выверены через
context7/web** (см. §13 «Источники»), не по памяти.

---

## 1. Назначение и границы первого захода

### 1.1. В объёме (этот заход)

1. **Модель волатильности** (§8.1, основная задача) — прогноз реализованной волатильности
   на горизонте 5 торговых дней. Два метода + baseline, сравнение по QLIKE.
2. **Модель тренда** (§8.3, вспомогательная) — классификация направления (вверх/вниз) на
   5 торговых дней. **Вариант price-only** (без sentiment-фичи) — осознанное отклонение от
   §8.3, см. §1.3.
3. **Инфраструктура**: сервис `mlflow` (7-й по инварианту §1 главной спеки), оффлайн
   uv-проект `ml/`, EDA-ноутбуки.
4. **Serving**: пакет `api/ml/`, эндпоинты `POST /predict/volatility` и `POST /predict/trend`
   (§9.7), запись в `predictions`, гейт `health/ready` на загруженные модели (§9.5).
5. **Разблокировка** алерта `volatility_regime` (тикет
   `docs/tickets/7d3e9b21-volatility-regime-alert-deferred-until-ml.md`).
6. **Дашборд** страница 4 «Прогнозы» (§10): прогноз волатильности vs факт + метрики vs
   baseline; вероятность тренда + SHAP; версия модели.

### 1.2. Вне объёма (отдельные заходы)

- **Sentiment-фича тренда** — отложена: новости в проде глубиной ~14 дней (RSS без архива,
  невосстановимы). Добавляется отдельным коммитом, когда лента углубится.
- **Точечный прогноз цены/доходности** — **не-цель** по §2 главной спеки (дневная
  доходность ≈ случайное блуждание). Не добавлять.
- **Автоматизация переобучения** — переобучение ручное (§8.5). Cron/оркестрация — вне объёма.

### 1.3. Зафиксированные решения (decisions)

| ID | Решение | Обоснование |
|----|---------|-------------|
| **D1** | Волатильность: **GARCH(1,1)** + **HAR-RV** (на дневном range-based прокси) + baseline RW-RV. Тренд: **CatBoost** price-only + baseline «всегда вверх». | §8.1/§8.3; sentiment-фича отложена (§1.2). |
| **D2** ✅ согласовано | `predictions` пишет **API при инференсе** (идемпотентный upsert). Инвариант §2 расширен на `predictions` (главная спека §4 + `CLAUDE.md` инв. #2 обновлены). | §8.5: «версия пишется в `predictions`»; инференс в API (§9.6). Единственный write-путь сохраняется. |
| **D3** | `health/ready` гейтится на загруженные модели (§9.5). Загрузка в `lifespan` с ретраями по образцу `_wait_for_schema`. Флаг `ML_REQUIRED_FOR_READY` (дефолт `true`). | §9.5 перечисляет модели как обязательную зависимость наравне с БД. Флаг даёт аварийный обход. |
| **D4** | Дневной прокси волатильности для HAR — **range-based (Паркинсон из H/L)**, НЕ скользящее std доходностей. | Скользящее std — само многодневное окно → коллинеарность регрессоров HAR, убивает мультигоризонтный сигнал (Corsi 2009; verified). Паркинсон ~5× эффективнее квадрата доходности. |
| **D5** | MLflow: продвижение через **алиасы** (`champion`/`production`), НЕ stages (deprecated). Нативный `mlflow.catboost`; GARCH/HAR — `mlflow.pyfunc.PythonModel`. | verified: stages deprecated, заменены алиасами; нативного flavor для `arch` нет. |
| **D6** | Валидация: **walk-forward `TimeSeriesSplit`**, expanding window, **`gap=5`**. QLIKE (primary) + RMSE для волатильности; accuracy + F1 + ROC-AUC vs «всегда вверх» для тренда. | §8.5 (walk-forward only); `gap≥горизонт` против утечки на перекрывающихся таргетах (verified). |
| **D7** | Обучение — оффлайн uv-проект `ml/` (repo-root), ручной запуск; `mlflow` server — сервис Compose. | §5/§8.5 главной спеки. |
| **D8** | Universe и период: ~47 текущих бумаг (`securities`); обучение на **пост-2022** как основной вариант (структурный разрыв 28.02–24.03.2022). | §8.1; данные подтверждены: candles 2013→2026, ~108k строк. |

---

## 2. Раскладка репозитория

### 2.1. Оффлайн uv-проект `ml/` (repo-root)

```
ml/
  pyproject.toml            # uv-проект stocklens-ml (Python 3.12, .python-version)
  uv.lock
  README.md                 # рунбук переобучения (§12)
  src/stocklens_ml/
    __init__.py
    config.py               # pydantic-settings: DB DSN (read-only), MLflow URI, имена/алиасы
                            #   моделей, HORIZON_DAYS=5, TRAIN_START=2022-04-01
    data/
      loader.py             # чтение candles/dividends/splits/index из БД в pandas (sync)
      adjust.py             # split-adjust, total-return c дивидендами, исключение weekend
    features/
      returns.py            # скорректированные лог-доходности
      volatility.py         # дневной прокси (Паркинсон), RV-таргет, HAR-регрессоры
      technical.py          # RSI(14), MACD(12,26,9), z-score объёма — для тренда
      assemble.py           # сборка фрейма фич per-ticker, контроль отсутствия утечек
    models/
      garch.py              # обёртка arch_model GARCH(1,1), 5-дневный форкаст
      har.py                # HAR-RV на дневном range-based прокси (OLS)
      baselines.py          # RW-RV (волатильность), «всегда вверх» (тренд)
      trend.py              # CatBoostClassifier + извлечение SHAP
    eval/
      metrics.py            # qlike(), rmse(), accuracy/f1/roc_auc
      walk_forward.py       # харнесс TimeSeriesSplit(gap=5), expanding
    registry/
      pyfunc_volatility.py  # VolatilityModel(mlflow.pyfunc.PythonModel)
      promote.py            # register_model + set_registered_model_alias
    training/
      train_volatility.py   # CLI: фичи → walk-forward → MLflow log → register → alias
      train_trend.py        # CLI: фичи → walk-forward → MLflow log → register → alias
  notebooks/
    01_eda.ipynb            # EDA: распределения, разрыв 2022, дивгэпы, сплиты, Parkinson vs r^2
    02_volatility.ipynb     # разбор модели волатильности
    03_trend.ipynb          # разбор тренда + SHAP
  artifacts/                # .gitignore — локальный fallback joblib
  tests/
    test_features_no_leakage.py
    test_volatility_features.py
    test_technical.py
    test_metrics_qlike.py
    test_walk_forward_gap.py
    test_baselines.py
```

**Зависимости `ml/pyproject.toml`:** `pandas`, `numpy`, `arch`, `scikit-learn`, `catboost`,
`shap`, `mlflow`, `statsmodels` (OLS для HAR), `sqlalchemy`, `psycopg` (чтение БД, sync),
`stocklens-core` (editable, ORM-модели и енумы), `joblib`. Dev: `pytest`, `mypy`, `ruff`.

**Стиль:** sync (как ingestor — батчевая обработка). Mypy strict, ruff (`B,UP,SIM,PL,I`),
файлы ≤400 строк, функции ≤60.

### 2.2. Serving-пакет `services/api/src/api/ml/`

```
api/ml/
  __init__.py
  loader.py        # MlflowClient: load по models:/<name>@<alias> в lifespan → app.state
  bundle.py        # ModelBundle: загруженные модели + метаданные (версия, метрики)
  volatility.py    # инференс-адаптер: ticker → фичи (из БД) → 5-дневная волатильность
  trend.py         # инференс-адаптер: ticker → фичи → P(up) + SHAP-вклады
  features.py      # переиспользование расчёта фич (импорт из stocklens-ml ИЛИ зеркало — см. §6.4)
```

Доменные исключения — в `api/core/exceptions.py`: `ModelNotLoadedError` (503/readiness),
`InsufficientHistoryError` (422, RU: «Не удалось построить прогноз: по тикеру {ticker}
недостаточно истории»). Роутер `routers/predict.py`, сервис
`services/prediction_service.py`, репозиторий `repositories/prediction_repo.py`, DTO
`schemas/predict.py`.

### 2.3. Сервис `mlflow` в Compose (7-й по инварианту)

`docker-compose.yml` и `docker-compose.prod.yml`:

```yaml
mlflow-db-init:                          # one-shot: идемпотентно создаёт БД mlflow
  image: postgres:16
  entrypoint: ["sh", "-c"]
  command:
    - >
      psql "$DATABASE_ADMIN_URL" -tc "SELECT 1 FROM pg_database WHERE datname='mlflow'"
      | grep -q 1 || psql "$DATABASE_ADMIN_URL" -c "CREATE DATABASE mlflow"
  environment:
    DATABASE_ADMIN_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/postgres
  depends_on:
    db: { condition: service_healthy }
  restart: "no"

mlflow:
  image: ghcr.io/mlflow/mlflow:v3.1.4    # пин-версия; verified API соответствует 3.1.4
  command: >
    mlflow server
    --backend-store-uri postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/mlflow
    --artifacts-destination /mlflow/artifacts
    --serve-artifacts
    --host 0.0.0.0 --port 5000
  volumes:
    - mlflow_artifacts:/mlflow/artifacts
  depends_on:
    db: { condition: service_healthy }
    mlflow-db-init: { condition: service_completed_successfully }
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health').status==200 else 1)"]
    interval: 15s
    timeout: 5s
    retries: 5
```

- **Bootstrap БД `mlflow` — через one-shot `mlflow-db-init`, НЕ через `initdb.d`.** Скрипты
  `docker-entrypoint-initdb.d` Postgres выполняются **только при первой инициализации пустого
  тома**; прод использует уже заполненный external-том `stocklens_pgdata`, поэтому init-скрипт
  там **никогда не запустится**. `mlflow-db-init` идемпотентно создаёт БД на любом (в т.ч.
  существующем) томе и завершается; `mlflow` стартует после него
  (`service_completed_successfully`).
- Backend store — **отдельная БД `mlflow`** на том же сервисе `db`. Артефакты — том
  `mlflow_artifacts`.
- `--serve-artifacts`: API тянет артефакты модели через сервер MLflow по `models:/`-URI, без
  прямого доступа к тому. API ходит в `http://mlflow:5000`.
- **Healthcheck — реальный блок** (GET `/health`), а не комментарий: на него опирается smoke
  §11.3.
- Прод: том `mlflow_artifacts` — external (как `stocklens_pgdata`), не теряется при пересборке.

---

## 3. Контракт данных (источники)

Все таблицы — `stocklens-core` (ORM), пишет только ingestor (§2 главной спеки). ML читает
**только на чтение**.

| Таблица | Поля (используемые) | Натуральный ключ |
|---------|---------------------|------------------|
| `candles` | `security_id`, `trade_date`, `open/high/low/close` (Numeric 18,6), `volume` (BigInteger), `is_weekend_session` (bool) | `(security_id, trade_date)` |
| `dividends` | `security_id`, `ex_date`, `value` (Numeric 18,6), `currency` | `(security_id, ex_date)` |
| `splits` | `security_id`, `split_date`, `before` (int), `after` (int) | `(security_id, split_date)` |
| `index_values` | `index_code`, `trade_date`, `close` | `(index_code, trade_date)` |
| `securities` | `id`, `ticker`, `is_active` | `ticker` |

Подтверждённая глубина (прод, 2026-06-23): `candles` 108 005 строк, 2013-03-25 → 2026-06-19,
47 бумаг; `dividends` 483, 2013→2025.

### 3.1. Обязательные преобразования перед расчётом фич

Порядок строгий: **сплит-коррекция → исключение weekend-сессий → total-return c дивидендами →
обрезка по `TRAIN_START`**. Weekend исключается **до** расчёта доходностей: иначе доходность
сессии после выходного считалась бы относительно аномальной weekend-сессии.

1. **Сплит-коррекция (D4-смежное).** `candles` хранят сырые цены. При сплите цена
   скачкообразно ломает доходность (прецедент TRNFP 1:100, см. docstring
   `Split` в [market.py:68-73](../../packages/stocklens-core/src/stocklens_core/models/market.py#L68-L73)).
   `Split.before` акций становятся `Split.after` (1 акция → 100 ⇒ `before=1, after=100`).
   Цены **строго до** `split_date` умножаются на коэффициент `before/after`, чтобы стать
   сопоставимыми с пост-сплит ценами. Коэффициент кумулятивный при нескольких сплитах.
   Без коррекции лог-доходность в день сплита = ложный обвал/взлёт.

2. **Исключение weekend-сессий.** Строки `is_weekend_session = true` (сессии выходного дня
   MOEX, торгуются с 03.2025) **исключаются** из обучения и расчёта фич (§8.1, риск-таблица
   главной спеки) **до расчёта доходностей**. Календарь фич строится по обычным торговым дням.

3. **Total-return с дивидендами.** Доходность **в день дивидендного гэпа** (биржевая ex-date)
   корректируется на величину дивиденда. **Семантика `ex_date`:** ingestor пишет
   `ex_date = MOEX registryclosedate` — это **T+1** (на один торговый день позже истинной
   ex-date, см.
   [parsing.py:47-55](../../services/ingestor/src/ingestor/parsing.py#L47-L55)).
   Гэп наблюдается на торговой сессии **истинной ex-date = `ex_date − 1 торговый день`**.
   Total-return доходность: `r_t = ln((close_t + div_adj_t) / close_{t-1})`, где `div_adj_t`
   = дивиденд на акцию (в той же сплит-шкале; приведённый к рублям; не-рублёвые дивиденды
   конвертируются по `currency_rates` на дату или исключаются с пометкой в EDA). Дивиденд
   относится к торговому дню истинной ex-date.

4. **Survivorship-оговорка.** `securities` содержит только текущие конституенты IMOEX +
   watchlist/портфель (ingestor `sync_securities`). Бумаги, покинувшие индекс, не
   бэкфилятся → universe смещён в сторону «выживших». Фиксируется в README модели и в EDA;
   расширение universe — вне объёма.

---

## 4. Фичи и таргеты — точные определения

Горизонт `H = 5` торговых дней. Все скользящие окна — по торговым дням (после исключения
weekend). Обозначения: `C_t` — скорректированный close, `H_t/L_t` — high/low,
`r_t = ln(C_t / C_{t-1})` — скорректированная (split+дивиденд) лог-доходность.

### 4.1. Дневной прокси дисперсии (вход регрессоров HAR)

**Паркинсон (range-based, D4):** дневная дисперсия для **регрессоров HAR**
```
σ²_P,t = (1 / (4·ln 2)) · (ln(H_t / L_t))²          ;  1/(4·ln2) ≈ 0.3607
```
Range-based прокси выбран как **дневной вход HAR-регрессоров** (≈5× эффективнее квадрата
доходности; D4). Квадрат дневной лог-доходности `r_t²` — документированный fallback/сравнение
(в EDA). **Скользящее std доходностей в качестве дневного прокси HAR запрещено (D4)** —
само многодневное окно, ломает мультигоризонтный сигнал. Это прокси для **регрессоров**, не
для таргета: таргет — return-based (§4.2, по §8.1).

### 4.2. Таргет волатильности (return-based, §8.1)

Главная спека §8.1 фиксирует таргет как реализованную волатильность следующих 5 дней
по **std лог-доходностей** (Паркинсон-оценка — только в EDA). Для QLIKE нужна дисперсия,
поэтому таргет — **return-based реализованная дисперсия** кумулятивной 5-дневной доходности:
```
RV_target_t = Σ_{k=1..H} r²_{t+k}              (return-based 5-дневная реализованная дисперсия)
```
`Σ r²` — стандартная несмещённая оценка дисперсии 5-дневной доходности при нулевом среднем
(тот же конъюнктурно-несмещённый прокси, который требует QLIKE, §6.2). Прогноз модели —
оценка `RV_target_t`. Отображаемая волатильность = `sqrt(RV_target)` (стандартное отклонение
кумулятивной 5-дневной доходности). `predictions.value` хранит **волатильность**
`sqrt(прогноз дисперсии)` (доля, не проценты; колонка `sa.Float`), `value > 0`.

> Согласованность: GARCH прогнозирует дисперсию дневных доходностей и суммирует в 5-дневную
> (§5.1) — тоже return-based; HAR (§5.2) прогнозирует тот же return-based `RV_target` из
> Паркинсон-регрессоров. Обе модели и baseline (§5.3) и QLIKE (§6.2) работают на одной
> return-based шкале дисперсии. Паркинсон-оценка таргета считается в EDA для сравнения (§8.1).

### 4.3. Регрессоры HAR (модель §5.2)

Дневной Паркинсон-прокси `σ²_P` (§4.1) агрегируется в три **средних** (нормировка 1/h,
нестед-окна, включая день t); HAR прогнозирует return-based `RV_target` (§4.2) из этих
регрессоров:
```
RV^(d)_t = σ²_P,t
RV^(w)_t = (1/5)  · Σ_{i=0..4}  σ²_P,(t−i)
RV^(m)_t = (1/22) · Σ_{i=0..21} σ²_P,(t−i)
```

### 4.4. Фичи тренда (price-only)

Все — на скорректированных ценах/доходностях, только из прошлого (лаг ≥ 1 относительно дня
прогноза):

| Фича | Определение |
|------|-------------|
| Лаги доходности | `r_{t}`, `r_{t−1}`, `r_{t−2}`, `r_{t−3}`, `r_{t−4}` |
| `RSI(14)` | Wilder RSI на 14 торговых дней |
| `MACD(12,26,9)` | линия MACD = EMA12 − EMA26; сигнал = EMA9(MACD); гистограмма = MACD − сигнал |
| z-score объёма | `(volume_t − mean_20) / std_20` по 20 торговым дням |
| RV(5) | `sqrt(Σ_{i=0..4} σ²_P,(t−i))` — реализованная волатильность за 5 дней (фича-состояние) |

**Sentiment-агрегаты — НЕ включаются** в этом заходе (D1; добавляются позже отдельным
коммитом со схемой фичи и переобучением).

### 4.5. Таргет тренда

Направление кумулятивной доходности на `H` дней:
```
y_t = 1, если ln(C_{t+H} / C_t) > 0, иначе 0      (вверх / вниз)
```
Граница ровно на 0 (нулевая доходность → класс «вниз»); порог фиксирован, не подбирается.

### 4.6. Контроль утечек (инвариант)

- Любая фича в день `t` использует только данные `≤ t`. Таргет смотрит в `t+1..t+H`.
- Скейлеры/нормировки/импьютеры (если есть) **фитятся только на train-окне** (§8.5). Для
  GARCH/HAR — параметры оцениваются только на train; для CatBoost — никакого глобального
  скейлинга по всей выборке.
- Тест `test_features_no_leakage.py`: подмена будущих значений `NaN`/случайными не меняет
  ни одну фичу в точке `t` (assert поэлементного равенства).

---

## 5. Модели и конфигурации

### 5.1. Волатильность — GARCH(1,1) (`arch`)

**Класс метода:** параметрическая эконометрика (условная дисперсия), оценка MLE/QMLE — **не
ML** (Engle 1982 / Bollerslev 1986).

```python
from arch import arch_model
# returns_pct: дневные лог-доходности в ПРОЦЕНТАХ (×100) — иначе плохая сходимость оптимизатора
am = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="t")
res = am.fit(disp="off")
fc = res.forecast(horizon=5, method="analytic", reindex=False)
# fc.variance — колонки h.1..h.5: h.k = k-шаговая условная дисперсия r в t+k
var_5d_pct2 = fc.variance.iloc[-1].sum()      # дисперсия кумулятивной 5-дневной доходности (percent²)
vol_5d = (var_5d_pct2 ** 0.5) / 100.0         # обратно в доли (÷100)
```

- **Масштаб ×100 вручную** (не `rescale=True`) — единицы предсказуемы. Деление на 100 на
  выходе — обязательно (связано с масштабом).
- `dist="t"` (Student-t) — тяжёлые хвосты доходностей акций.
- `method="analytic"` валиден для GARCH (линеен по квадратам остатков).
- Сумма `h.1..h.5` корректна при `mean="Constant"` (нет AR-кросс-членов).

### 5.2. Волатильность — HAR-RV (OLS)

**Класс метода:** обычная линейная регрессия (OLS) — **не ML** (Corsi 2009). Реализация —
`statsmodels.OLS` либо `sklearn.LinearRegression`.

```
RV_target_t ≈ β0 + β_d·RV^(d)_t + β_w·RV^(w)_t + β_m·RV^(m)_t
```
Это HAR-RV на **дневном range-based прокси** (Паркинсон, §4.1) вместо внутридневной RV
(интрадей-данных нет). Имя **HAR-RV-X не используем**: в литературе оно означает иное — HAR
с добавленными внешними регрессорами (VIX, jump-компоненты), а не «HAR на дневном прокси».
Оценка — OLS; из-за перекрытия окон в EDA приводятся HAC/Newey-West стандартные ошибки
(информативно; оценка остаётся OLS).

### 5.3. Волатильность — baseline RW-RV

«Волатильность завтра = волатильность вчера»: прогноз 5-дневной return-based дисперсии =
последняя наблюдённая 5-дневная return-based реализованная дисперсия (`Σ r²` за прошлые
5 торговых дней) — та же шкала, что у таргета (§4.2). Любая модель **засчитывается, только
если бьёт RW-RV по среднему QLIKE на walk-forward** (D6, §8.5).

### 5.4. Тренд — CatBoost (price-only)

**Класс метода:** машинное обучение — градиентный бустинг на деревьях (это единственный ML
в заходе).

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=600, depth=4, learning_rate=0.03,
    loss_function="Logloss", eval_metric="AUC",
    l2_leaf_reg=6.0, random_seed=42,
    auto_class_weights="Balanced",        # дисбаланс вверх/вниз
    early_stopping_rounds=50,
)
model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
p_up = model.predict_proba(X)[:, 1]       # колонка 1 = P(up)
```
Параметры — стартовые; финальные подбираются на walk-forward (консервативно: малая глубина,
сильная регуляризация — данных мало, риск переобучения высок). Baseline — «всегда вверх»
(predict 1 всегда), ожидание модели 52–55% accuracy; результат позиционируется как
вероятностная оценка, **не торговый сигнал** (§8.3).

### 5.5. SHAP для тренда

```python
from catboost import Pool
shap = model.get_feature_importance(Pool(X, y, cat_features=None), type="ShapValues")
# binary → 2D (n_samples, n_features+1); последняя колонка = base value
base_value = float(shap[0, -1])
contribs = shap[:, :-1]
# для одного сэмпла i → JSON {feature: вклад}
sample = {name: float(v) for name, v in zip(model.feature_names_, contribs[i])}
```
**Важно (verified):** 2D-раскладка верна для бинарной классификации (наш случай). Для
мультикласса был бы 3D — не переиспользовать срез `[:, :-1]` для мультикласса.
Альтернатива — `shap.TreeExplainer(model).shap_values(X)` (без base-колонки; base в
`explainer.expected_value`) — **не** срезать последнюю колонку у этого вывода.

---

## 6. Валидация (walk-forward)

### 6.1. Протокол (D6, §8.5)

- **`sklearn.model_selection.TimeSeriesSplit(n_splits=5, gap=5)`**, expanding window
  (`max_train_size=None`). `gap=5 = горизонт` — исключает из конца train-окна сэмплы, чьи
  таргеты `t+1..t+H` перекрывают test-окно (forward purging). `gap` добавлен в sklearn 0.24.
- Случайный K-fold **запрещён** (§8.5) — главная причина утечек.
- Embargo (буфер после test) в walk-forward не требуется (после test нет train-данных);
  `gap=горизонт` достаточно. Полный purge+embargo нужен только для K-fold/CPCV — здесь не
  применяется.
- Скейлеры/фичи/параметры моделей фитятся **только на train-окне каждого сплита** (§4.6).
- Для волатильности (GARCH/HAR) walk-forward — на уровне ряда одной бумаги; для тренда —
  пул бумаг с разбиением по времени (одна и та же временная граница для всех тикеров, без
  утечки между train/test по дате).

### 6.2. Метрики

| Задача | Primary | Дополнительно | Baseline-гейт |
|--------|---------|---------------|---------------|
| Волатильность | **QLIKE** (на дисперсиях) | RMSE | средний QLIKE < QLIKE(RW-RV) |
| Тренд | **ROC-AUC** | accuracy, F1 | accuracy > accuracy(«всегда вверх») и ROC-AUC > 0.5 |

**QLIKE** (нормированная форма Patton 2011, на **дисперсиях**, lower-is-better):
```
QLIKE = (1/T) · Σ_t [ σ²_proxy,t / h_t − ln(σ²_proxy,t / h_t) − 1 ]
```
где `h_t` — прогноз дисперсии, `σ²_proxy,t` — реализованная дисперсия (`RV_target`). QLIKE
устойчив к шуму прокси и асимметричен (сильнее штрафует недопрогноз дисперсии).
**Единицы:** и `h_t`, и `σ²_proxy,t` входят в QLIKE как дисперсии в **долях²** (= (5-дневная
волатильность в долях)²), согласованно с §4.2. Масштаб ×100/÷100 внутри GARCH (§5.1) —
внутренний для фита и **обязан быть снят** до сравнения по QLIKE/`RV_target` (иначе percent²
vs decimal²).
Значимость превосходства над baseline — тест Diebold-Mariano-West с Newey-West HAC по ряду
поэлементных разностей лосса (приводится в MLflow/EDA, информативно).

---

## 7. MLflow — конвенции (D5)

### 7.1. Реестр и продвижение

- Имена в реестре: `stocklens-volatility`, `stocklens-trend`.
- Эксперименты: `volatility`, `trend`. Логируются `log_params` (конфиг модели, окна фич,
  `TRAIN_START`), `log_metrics` (QLIKE/RMSE или AUC/accuracy/F1, и метрики baseline),
  `log_artifact` (графики прогноз-vs-факт, SHAP-summary, отчёт walk-forward).
- **Продвижение — алиасы, НЕ stages** (stages deprecated): после регистрации лучшего прогона
  `client.set_registered_model_alias(name, "champion", version)`; прод-алиас — `production`.
  `client.transition_model_version_stage` **не использовать**.
- Логирование: тренд — нативный `mlflow.catboost.log_model(model, name="model",
  registered_model_name="stocklens-trend", signature=...)`; волатильность — кастомный
  `mlflow.pyfunc.PythonModel` (для `arch`/HAR нет нативного flavor), лог через
  `mlflow.pyfunc.log_model(name=..., python_model=..., registered_model_name=...,
  pip_requirements=[...])`.

### 7.2. `predictions.model_version`

Строка `model_version` (String(64)) = **номер версии реестра** (incrementing int как str):
`str(mv.version)`, где `mv` — `client.create_model_version(...)` или
`client.get_model_version_by_alias(name, alias)`. Эта же строка отображается в UI (§10,
страница 4). Загрузка модели API — по алиасу: `models:/stocklens-volatility@production`.

---

## 8. Serving (API)

### 8.1. Загрузка моделей (lifespan, D3)

В `api/core/lifespan.py` после Redis — загрузка моделей из реестра в `app.state`:
```
app.state.ml = load_bundle(settings)   # api/ml/loader.py
```
`load_bundle` грузит `models:/stocklens-volatility@{alias}` и
`models:/stocklens-trend@{alias}` (`mlflow.pyfunc.load_model` / `mlflow.catboost.load_model`),
кладёт в `ModelBundle` (модель + версия + метрики). Недоступность реестра на старте —
ретраи по образцу `_wait_for_schema`
([lifespan.py:24-49](../../services/api/src/api/core/lifespan.py#L24-L49)); после
исчерпания — `ModelNotLoadedError`. При `ML_REQUIRED_FOR_READY=true` это валит readiness
(не liveness).

### 8.2. `health/ready` (§9.5)

Расширить `ReadyResponse` полем `models: Literal["ok","unavailable"]`. Модели не загружены →
`models="unavailable"` и `http_status=503` (наравне с БД, §9.5). Redis остаётся
деградируемым. При `ML_REQUIRED_FOR_READY=false` — `models` информативно, без 503.

### 8.3. Эндпоинты (§9.7)

**`POST /predict/volatility`** — тело `{ ticker: str }`. Ответ `VolatilityPredictionOut`:
```
ticker: str
predicted_for: date          # as-of дата (последний использованный close)
horizon_days: int = 5
volatility: float            # sqrt(прогноз дисперсии), decimal
model: str                   # имя модели-победителя (garch | har_rv_x)
model_version: str
metrics_vs_baseline: { qlike: float, qlike_baseline: float, rmse: float }
```

**`POST /predict/trend`** — тело `{ ticker: str }`. Ответ `TrendPredictionOut`:
```
ticker: str
predicted_for: date
horizon_days: int = 5
prob_up: float               # P(up) ∈ [0,1]
direction: Literal["up","down"]
shap: dict[str, float]       # {feature: вклад} для этого предсказания
base_value: float
model_version: str
```

- CPU-bound инференс — через `fastapi.concurrency.run_in_threadpool` (§9.2 главной спеки:
  API async, CPU-bound в threadpool).
- Нет данных по тикеру / недостаточно истории → `InsufficientHistoryError` → 422 с RU-текстом
  (§9.4: «Не удалось построить прогноз: по тикеру {ticker} недостаточно истории»).
  Неизвестный тикер → 404 (RU). `response_model` обязателен (§9 главной спеки).
- **Pydantic v2 protected-namespace:** поля `model` и `model_version` в DTO начинаются с
  `model` → выставить `model_config = ConfigDict(protected_namespaces=())` на этих DTO (или
  переименовать в `registry_version`/`model_name`), иначе Pydantic выдаёт warning. Та же
  оговорка, что для `ApiSettings` в §8.6.

### 8.4. Запись `predictions` (D2)

`PredictionService` при инференсе делает идемпотентный upsert в `predictions` по
натуральному ключу `(security_id, predicted_for, horizon_days, kind, model_version)`:
- волатильность → `kind=PredictionKind.VOLATILITY`, `value = volatility`;
- тренд → `kind=PredictionKind.TREND`, `value = prob_up`.
SHAP **не** хранится в `predictions` (возвращается в ответе; при необходимости — отдельный
артефакт). Повторный вызов в тот же день с той же версией модели — no-op (upsert).

> **✅ Согласовано (инвариант §2 расширен).** Владелец согласовал расширение write-владения
> API на `predictions`. Главная спека §4 и `CLAUDE.md` инвариант #2 обновлены: API владеет
> записью в `portfolio_positions`, `bot_subscriptions` и `predictions`. Единственный
> write-путь сохраняется (в `predictions` пишет только API — при ML-инференсе).

### 8.5. Фичи на инференсе (D — единый источник)

Расчёт фич для инференса **обязан совпадать** с обучающим (иначе train/serve skew). Вариант
по умолчанию: `ml/`-функции фич импортируются в API из пакета `stocklens-ml`
(добавить как зависимость API, только модуль `features`/`data`, без training/mlflow-кода),
либо вынести чистый расчёт фич в `stocklens-core`. Зеркалирование формул в двух местах
**запрещено** (DRY). Решение реализации фиксируется при разработке; контракт — единая
кодовая функция фич для train и serve.

### 8.6. Настройки (`ApiSettings`, pydantic-settings)

Добавить поля (env без префикса, §9.6 — чтение `os.environ` запрещено):
```
MLFLOW_TRACKING_URI: str            # http://mlflow:5000
ML_VOLATILITY_MODEL: str = "stocklens-volatility"
ML_TREND_MODEL: str = "stocklens-trend"
ML_MODEL_ALIAS: str = "production"
ML_REQUIRED_FOR_READY: bool = True
```
Поля не начинаются с `model_` (конфликт protected-namespace Pydantic v2); если понадобится —
выставить `model_config["protected_namespaces"] = ()`.

---

## 9. Алерт `volatility_regime` (разблокировка тикета 7d3e9b21)

§11 главной спеки: «смена режима волатильности (прогноз выше порога)». Определение порога:
алерт срабатывает, когда последняя прогнозируемая 5-дневная волатильность тикера превышает
квантиль `VOLATILITY_REGIME_QUANTILE` (дефолт `0.80`) распределения его прогнозов/реализаций
за trailing-окно `VOLATILITY_REGIME_LOOKBACK` (дефолт `252` торговых дня). Реализация — в
`AlertEvaluationService` (§9.1, оценка алертов в API): снять `continue`-заглушку для
`AlertKind.VOLATILITY_REGIME`
([alert_evaluation.py:101](../../services/api/src/api/services/alert_evaluation.py#L101)),
добавить `_evaluate_volatility_regime()`, бот — разрешить подписку (`subscriptions.py` парсер),
убрать `ML`-deferred-ошибку. **Обязательно обновить docstring'и**, утверждающие обратное:
модульный docstring `alert_evaluation.py` (строки 3-4: «volatility_regime не реализован…») и
docstring `collect_pending` («подписки VOLATILITY_REGIME пропускаются») — иначе дрейф
«документация vs поведение». Дедуп — Redis NX+TTL по `(chat, ticker, режим, дата)`, как у
прочих алертов. Тикет закрывается этим изменением.

---

## 10. Дашборд — страница 4 «Прогнозы» (§10)

- **Волатильность:** график прогноз vs факт (реализованная за последующие 5 дней), плашка
  метрик `QLIKE` модели vs baseline; выбор тикера.
- **Тренд:** `P(up)` с доверительной подачей (не «сигнал»), **SHAP-объяснение** (bar
  вкладов фич из ответа API), направление.
- **Версия модели** (`model_version`) — на странице (§8.5 главной спеки: версия видна в UI).
- Streamlit ходит только в API (инвариант §3); три ветки сетевого вызова (успех / ошибка
  сервера / сеть) — обязательны (§10 главной спеки). Дизайн-токены —
  [services/dashboard/DESIGN.md](../../services/dashboard/DESIGN.md).

---

## 11. Критерии приёмки и тест-план (TDD)

TDD обязателен для всей бизнес-логики (§12 главной спеки; имена тестов — английские,
по сценарию). Не покрывается: ноутбуки, конфиг MLflow-сервиса, сгенерированный код.

### 11.1. `ml/` (оффлайн)

- `test_features_no_leakage` — будущие данные не влияют на фичу в точке `t` (§4.6).
- `test_volatility_features` — Паркинсон-прокси и HAR-регрессоры считаются по формулам §4.1/4.3
  на фикстуре с известным ответом; `RV^(w)` = среднее (нормировка 1/5), не сумма.
- `test_technical` — RSI/MACD/z-объём против эталонных значений.
- `test_metrics_qlike` — QLIKE = 0 при идеальном прогнозе (`h = σ²_proxy`), > 0 иначе,
  на дисперсиях.
- `test_walk_forward_gap` — `TimeSeriesSplit(gap=5)` не допускает перекрытия train-таргетов
  с test-окном (проверка индексов).
- `test_baselines` — RW-RV и «всегда вверх» дают ожидаемые прогнозы; модель в обучении
  логирует метрику baseline.
- **Baseline-гейт:** `train_volatility.py`/`train_trend.py` регистрируют модель в реестр
  **только если** она бьёт baseline (D6); иначе — лог + отказ от регистрации (не молча).

### 11.2. API (serving)

- `predict_volatility_returns_404_for_unknown_ticker` (§12 пример).
- `predict_volatility_returns_422_for_insufficient_history`.
- `predict_volatility_persists_prediction_idempotently` — повторный вызов не плодит строки
  (upsert по натуральному ключу).
- `predict_trend_returns_prob_and_shap` — `prob_up ∈ [0,1]`, `shap` непустой, сумма вкладов
  + base ≈ margin (additivity).
- `health_ready_returns_503_when_models_not_loaded` (при `ML_REQUIRED_FOR_READY=true`).
- Инференс не блокирует event-loop (вызван через threadpool) — smoke.
- Модели в тестах — лёгкие стабы через Protocol `ModelBundle` (unit-тесты сервисов не
  поднимают MLflow; §9.1 главной спеки — repository/инференс за Protocol).

### 11.3. Smoke интеграции

- Compose поднимает `mlflow` (healthcheck зелёный); API `lifespan` грузит обе модели;
  `/health/ready` = 200 после загрузки.

---

## 12. Рунбук переобучения (`ml/README.md`)

Ручной еженедельный запуск (§8.5; автоматизация — вне объёма, помечается как направление
развития):
```
uv sync --project ml
DATABASE_URL=... MLFLOW_TRACKING_URI=http://localhost:5000 \
  uv run --project ml python -m stocklens_ml.training.train_volatility
uv run --project ml python -m stocklens_ml.training.train_trend
# Скрипт: фичи → walk-forward → MLflow log → (если бьёт baseline) register → alias champion.
# Промоушен champion→production — ручной (после ревью метрик в MLflow UI):
#   client.set_registered_model_alias("stocklens-volatility", "production", <version>)
```
Первичное обучение — против прод-БД на чтение (данные там). Откат — переключение алиаса
`production` на предыдущую версию (мгновенно, без передеплоя API: API грузит по алиасу при
рестарте; для горячего обновления — отдельный заход).

---

## 13. Источники (выверено context7/web)

- **arch (GARCH):** github.com/bashtage/arch — `forecast(horizon, method="analytic")`,
  раскладка `variance` (h.1..h.5 = k-шаговые дисперсии), масштаб ×100, сумма+sqrt для
  горизонта.
- **HAR-RV:** Corsi (2009) J. Financial Econometrics 7(2) — OLS на средних RV daily/weekly/
  monthly; дневной прокси при дневных данных — range-based (Parkinson 1980; Garman-Klass 1980).
- **QLIKE:** Patton (2011) J. Econometrics — нормированная форма на дисперсиях, robust к шуму
  прокси.
- **walk-forward:** sklearn `TimeSeriesSplit(gap=...)` (≥0.24); López de Prado AFML (purging/
  embargo) — embargo не нужен для forward-only split, `gap=горизонт` достаточно.
- **CatBoost/SHAP:** context7 `/catboost/catboost`, `/shap/shap` — `CatBoostClassifier`,
  `get_feature_importance(type="ShapValues")` (binary 2D, last col = base), `TreeExplainer`.
- **MLflow:** context7 `/mlflow/mlflow` v3.1.4 — stages **deprecated** → алиасы
  (`set_registered_model_alias`); нативный `mlflow.catboost`; `pyfunc.PythonModel` для arch;
  `models:/<name>@<alias>`; `model_version = str(mv.version)`.

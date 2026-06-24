# Деплой StockLens на VPS (Dokploy + образы из GHCR)

Рунбук прод-деплоя. Источник истины по архитектуре — спека
[2026-06-11-stocklens-design.md](specs/2026-06-11-stocklens-design.md), §14 (деплой).
План этой итерации — `~/.claude/plans/stocklens/2026-06-11-deploy-dokploy.md`.

## Принципы

- **Сборка образов — в CI** (GitHub Actions, job `publish` в `.github/workflows/ci.yml`):
  на каждый зелёный push в `main` собираются и публикуются `ghcr.io/swezakk/stocklens-api`
  и `ghcr.io/swezakk/stocklens-ingestor` (теги `latest` и `sha-<short>`). Сервер только
  `pull`'ит — тяжёлый ONNX-экспорт и установка scipy/sklearn не нагружают 4 ГБ VPS.
- **Образы публичные** — серверу не нужны креды для `pull`. Первый `publish` создаёт
  пакеты приватными: после первого зелёного деплоя один раз переключить видимость
  каждого пакета (`stocklens-api`, `stocklens-ingestor`) на **Public** в GitHub
  (Profile → Packages → пакет → Package settings → Change visibility).
- **Режим Dokploy — «Docker Compose»**, не «Stack»: обычный `docker compose up`
  поддерживает one-shot job миграций и `depends_on: service_completed_successfully`.
- **Том БД — внешний** (`stocklens_pgdata`, `external: true`): Dokploy не пересоздаёт
  его при redeploy. Новости из RSS невосстановимы — БД бэкапится host-cron'ом.

## Сервер

- `swezakk@213.171.29.124` (Ubuntu 22.04, 2 vCPU, 3.8 ГБ RAM, 30 ГБ диск).
- Соседи на хосте, которые НЕ трогаем: `xray-cascade` (порт 10443).
- Caddy (`dashboard-nir.prod-track.ru`, порты 80/443) — останавливается, чтобы
  освободить порты под Traefik (Dokploy).

## Переменные окружения (задаются в env-сторе Dokploy, не в репозитории)

| Переменная | Назначение | Пример значения |
|---|---|---|
| `DB_USER` | пользователь Postgres | `stocklens` |
| `DB_PASSWORD` | пароль Postgres (сгенерировать сильный) | `<openssl rand -base64 24>` |
| `DB_NAME` | имя БД | `stocklens` |
| `DATABASE_URL` | sync-DSN (ingestor, migrations) | `postgresql+psycopg://stocklens:<pwd>@db:5432/stocklens` |
| `DATABASE_URL_ASYNC` | async-DSN (api) | `postgresql+asyncpg://stocklens:<pwd>@db:5432/stocklens` |
| `REDIS_URL` | DSN Redis | `redis://redis:6379/0` |
| `TICKERS_UNIVERSE` | стартовая вселенная тикеров | `IMOEX` |
| `IMAGE_TAG` | тег образов GHCR | `latest` (или `sha-xxxxxxx` для пина) |
| `AUTH_SECRET` | секрет подписи JWT (api) | `<openssl rand -base64 32>` |
| `AUTH_OWNER_USERNAME` | имя владельца (api/dashboard/bot) | `admin` |
| `AUTH_OWNER_PASSWORD` | пароль владельца (api; гейт дашборда; логин бота) | `<сильный пароль>` |
| `TELEGRAM_BOT_TOKEN` | токен Telegram-бота от @BotFather (сервис `bot`) | `(секрет)` |

Хосты `db`/`redis` — это имена сервисов из `docker-compose.prod.yml` (внутренняя сеть compose).
Сервис `bot` (long-polling aiogram) потребляет `TELEGRAM_BOT_TOKEN` + `AUTH_OWNER_*`/`API_URL`
(логин в API владельческими кредами); входящих портов не имеет.

## Фаза B — подготовка сервера (обратимо)

```bash
# B1. Бэкенд iptables должен быть nf_tables (Swarm-публикация портов на legacy молча падает).
iptables --version            # ожидаем "(nf_tables)"

# B2. Swap 4 ГБ (на боксе его нет) — страховка от OOM.
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-stocklens-swap.conf
free -h                       # проверяем, что swap появился

# B3. Каталог бэкапов и внешний том БД.
sudo mkdir -p /opt/stocklens/backups
docker volume create stocklens_pgdata

# B4. Освобождаем 80/443 — останавливаем Caddy (файлы /var/www/dashboard НЕ удаляем).
sudo systemctl stop caddy
sudo systemctl disable caddy
sudo ss -tlnp | grep -E ':80 |:443 '   # должно быть пусто
docker ps --filter name=xray           # X-ray по-прежнему Up на 10443
```

## Фаза C — установка Dokploy

```bash
# Порты 80/443/3000 должны быть свободны.
curl -sSL https://dokploy.com/install.sh | sudo sh

# Сразу проверяем, что Swarm-init не задел X-ray и порты на месте.
docker info | grep -i 'swarm: active'
docker ps --filter name=xray           # X-ray Up
curl -sS -o /dev/null -w '%{http_code}\n' http://213.171.29.124:10443 || true  # доступен снаружи
```

Панель: `http://213.171.29.124:3000` — создать админ-аккаунт. Позже закрыть порт 3000
файрволом на доверенный IP.

## Фаза D — деплой ядра данных (новости — как можно раньше)

1. В Dokploy: создать проект → приложение типа **Compose** (НЕ Stack), источник —
   репозиторий `Swezakk/stocklens`, файл `docker-compose.prod.yml`.
2. Заполнить env (таблица выше). `DB_PASSWORD` — сгенерировать.
3. Deploy. Порядок поднимется сам: `db` → `migrations` (one-shot) → `redis`/`ingestor`/`api`.
4. **До первой записи новостей** — поставить cron бэкапа:

Скрипт берёт дамп через `docker exec stocklens-db pg_dump` — пароль БД ему не нужен.

```bash
# Скопировать скрипт на хост (он лежит в репозитории) и поставить ежедневный cron в 03:30 UTC.
sudo install -m 0755 scripts/pg_backup.sh /opt/stocklens/pg_backup.sh
( sudo crontab -l 2>/dev/null; echo '30 3 * * * /opt/stocklens/pg_backup.sh' ) | sudo crontab -
# Проверить ручным прогоном:
sudo /opt/stocklens/pg_backup.sh && ls -lh /opt/stocklens/backups
```

5. Верификация сбора:

```bash
docker exec stocklens-db psql -U stocklens -d stocklens -c \
  "select source, status, started_at from collector_runs order by started_at desc limit 10;"
docker exec stocklens-db psql -U stocklens -d stocklens -c \
  "select count(*) from news_items;"   # растёт со временем
```

## Фаза E — API наружу (домен + TLS)

1. Добавить A-запись DNS: `api.stocklens.prod-track.ru` → `213.171.29.124`.
2. В Dokploy на сервисе `api` задать домен `api.stocklens.prod-track.ru`, порт `8000`,
   включить Let's Encrypt (HTTPS). Traefik выпустит сертификат автоматически.
3. Проверить:

```bash
curl -sS https://api.stocklens.prod-track.ru/api/v1/health/ready
# и Swagger: https://api.stocklens.prod-track.ru/docs
```

## MLflow — реестр моделей и ревью метрик

`mlflow` развёрнут как 7-й сервис (только `expose`, наружу не публикуется). Прод-API грузит
модель волатильности из реестра по `models:/stocklens-volatility@production` (alias
выставляется при регистрации). Ревью метрик в UI — через **ad-hoc SSH-туннель** (порт mlflow
на хосте не опубликован, форвардим на IP контейнера в docker-сети):

```bash
MLFLOW_IP=$(ssh swezakk@213.171.29.124 \
  "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
   compose-quantify-haptic-sensor-fkssrm-mlflow-1")
ssh -L 5000:$MLFLOW_IP:5000 swezakk@213.171.29.124
# затем в браузере: http://localhost:5000  (Host localhost разрешён в --allowed-hosts mlflow)
```

**Переобучение** должно логировать в прод-MLflow (не в локальный sqlite!): one-off контейнер
на compose-сети VPS с `--mlflow-uri http://mlflow:5000` (полная команда — в памяти deploy,
раздел «Прод-переобучение волатильности»). После регистрации модель грузится при
старте/рестарте `api` (есть `depends_on: mlflow: service_healthy`).

Gotchas mlflow (НЕ откатывать): backend на **psycopg2**, а не psycopg3 (psycopg3 ломает
операции реестра на PostgreSQL); `--allowed-hosts` в команде; `mem_limit ≥ 1536m`. Детали —
в комментариях `docker-compose.prod.yml` и `services/mlflow/Dockerfile`.

## Восстановление из бэкапа

```bash
# Распаковать и залить дамп в работающий контейнер БД.
gunzip -c /opt/stocklens/backups/stocklens_YYYYMMDD_HHMMSS.sql.gz \
  | docker exec -i stocklens-db psql -U stocklens -d stocklens
```

## Откат деплоя

- В Dokploy: на приложении — Redeploy с предыдущим образом (тег `sha-<short>` нужного
  коммита через env `IMAGE_TAG`).
- Том `stocklens_pgdata` при этом не трогается (external) — данные сохраняются.

## Что появится позже

Все 7 сервисов (`db`, `redis`, `ingestor`, `api`, `dashboard`, `bot`, `mlflow`) развёрнуты;
ML-инференс волатильности активен. Остаётся: ротация docker-логов (`json-file` с `max-size`),
UFW, мониторинг; расписание алертов бота (B2) и дайджеста (B3); регулярная генерация прогнозов
(чтобы дашборд «Прогнозы» накапливал track record — сейчас прогнозы пишутся только on-demand
при `POST /predict/volatility` или оценке алерта); CI-smoke реестра (тикет a1c4f7e2).

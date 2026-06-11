#!/usr/bin/env bash
# Ежедневный бэкап БД StockLens: pg_dump из контейнера Postgres в каталог бэкапов.
#
# Назначение: новости из RSS невосстановимы (нет архива) — БД должна бэкапиться
# ДО того, как ingestor запишет первую новость (см. docs/deploy.md, фаза D).
# Запуск — host-cron'ом на VPS (не внутри Dokploy), пишет дампы вне тома данных.
#
# Переменные окружения (со значениями по умолчанию под прод-стек):
#   BACKUP_DIR       каталог дампов            (/opt/stocklens/backups)
#   DB_CONTAINER     имя контейнера Postgres   (stocklens-db)
#   DB_USER          пользователь БД           (stocklens)
#   DB_NAME          имя БД                    (stocklens)
#   RETENTION_DAYS   срок хранения дампов, сут (14)
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/opt/stocklens/backups}"
DB_CONTAINER="${DB_CONTAINER:-stocklens-db}"
DB_USER="${DB_USER:-stocklens}"
DB_NAME="${DB_NAME:-stocklens}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"

log() {
    printf '%s [pg_backup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

timestamp="$(date -u +%Y%m%d_%H%M%S)"
target="${BACKUP_DIR}/stocklens_${timestamp}.sql.gz"

# Контейнер БД обязан существовать и работать — иначе бэкап невозможен, это ошибка,
# а не «тихо ничего не делаем»: молчаливый пропуск означал бы потерю невосстановимых данных.
if ! docker inspect --format '{{.State.Running}}' "${DB_CONTAINER}" 2>/dev/null | grep -q '^true$'; then
    log "ОШИБКА: контейнер БД '${DB_CONTAINER}' не найден или не запущен — бэкап не выполнен"
    exit 1
fi

mkdir -p "${BACKUP_DIR}"

log "старт pg_dump базы '${DB_NAME}' из контейнера '${DB_CONTAINER}' -> ${target}"

# Пишем во временный .partial и переименовываем только при успехе — чтобы оборванный
# дамп не выглядел как валидный бэкап. pipefail ловит падение pg_dump в начале конвейера.
if docker exec "${DB_CONTAINER}" \
        pg_dump -U "${DB_USER}" -d "${DB_NAME}" --no-owner --no-privileges \
        | gzip -9 >"${target}.partial"; then
    mv "${target}.partial" "${target}"
    log "бэкап готов: ${target} ($(du -h "${target}" | cut -f1))"
else
    rm -f "${target}.partial"
    log "ОШИБКА: pg_dump завершился неуспешно — частичный файл удалён"
    exit 1
fi

# Ротация: удаляем дампы старше RETENTION_DAYS суток.
deleted="$(find "${BACKUP_DIR}" -name 'stocklens_*.sql.gz' -type f -mtime "+${RETENTION_DAYS}" -print -delete | wc -l | tr -d ' ')"
log "ротация: удалено старых дампов: ${deleted} (хранение ${RETENTION_DAYS} сут)"

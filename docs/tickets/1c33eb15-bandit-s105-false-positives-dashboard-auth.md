---
id: 1c33eb15
title: Bandit S105 false positives on constant names in dashboard auth.py
status: resolved
priority: low
component: dashboard
discovered: 2026-06-22
discovered-from: []
tags: ["security", "lint", "false-positive", "bandit", "ruff"]
---

# 1c33eb15: Bandit S105 false positives on constant names in dashboard auth.py

## What was observed
Стоп-гейт `python-security-stop-gate` (Ruff curated S-rules, Bandit-aligned) сообщает
7 находок S105 «Possible hardcoded password» в
[services/dashboard/src/dashboard/auth.py](../../services/dashboard/src/dashboard/auth.py):

| Строка | Константа | Фактическое назначение |
|--------|-----------|------------------------|
| 36 | `_TOKEN_PATH = "/auth/token"` | URL-путь эндпоинта выдачи токена |
| 45 | `_WRONG_PASSWORD_MESSAGE = "Неверный пароль"` | RU-текст ошибки UI |
| 48 | `_STATE_PASSWORD = "password"` | **ключ** `st.session_state`, не значение |
| 49 | `_STATE_TOKEN = "token"` | ключ `st.session_state` |
| 50 | `_STATE_TOKEN_EXPIRY = "token_expiry"` | ключ `st.session_state` |
| 55 | `_GATE_PASSWORD_LABEL = "Пароль"` | подпись поля формы |
| 57 | `_GATE_EMPTY_PASSWORD = "Введите пароль."` | RU-текст валидации |

Ни одно значение не является секретом. S105 триггерит на **имя** переменной, содержащее
`PASSWORD`/`TOKEN`, при присваивании строкового литерала — это документированный класс ложных
срабатываний Bandit. Дизайн дашборда прямо фиксирует отсутствие захардкоженного секрета
([auth.py:3-5](../../services/dashboard/src/dashboard/auth.py#L3-L5): «Отдельного
`DASHBOARD_PASSWORD` нет», пароль вводит пользователь и держится в `st.session_state`).

Находки внесены в прошлой сессии (коммит `06db689`), не текущим изменением.

## Why it is a problem
Сами по себе строки безопасны — реального риска утечки секрета нет. Проблема процессная:
гейт блокирует завершение хода на каждой сессии, где `auth.py` попадает в множество
модифицированных `.py`-файлов, пока находки не «разрешены». `# noqa` гейтом не уважается,
поэтому без явного решения шум воспроизводится постоянно и заглушает потенциально настоящие
S-находки в будущем (severity inflation наоборот — постоянный false-positive обесценивает гейт).

## Why it is not a duplicate
Первый тикет проекта; дублей в `docs/tickets/` нет (проверено `rg`).

## What probably needs to be done
Выбрать один из путей (требует решения владельца — это конфиг гейта, не бизнес-логика):

1. **Конфиг гейта/Ruff** — добавить `per-file-ignores` для `S105` на
   `services/dashboard/src/dashboard/auth.py` (или на `_STATE_*`/`_*_MESSAGE`/`_*_LABEL`
   паттерн), если гейт читает проектный `pyproject.toml`. Требует проверки: использует ли
   стоп-гейт проектный ruff-конфиг или собственный изолированный набор.
2. **Уточнить S106/S105-исключения через `flake8-bandit` конфиг** в `pyproject.toml`
   (`lint.flake8-bandit.*`), если применимо к curated-набору гейта.
3. **Принять как задокументированный FP** — оставить этот тикет как запись-обоснование,
   если гейт поддерживает allowlist по ticket-id.

Предпочтительно (1): целевой ruff-набор проекта по CLAUDE.md — `B`, `UP`, `SIM`, `PL`
(S-rules в проектный набор не входят, их добавляет сам гейт), поэтому точечный
`per-file-ignores` для одного файла не размывает security-покрытие остального кода.

Переименование констант ради обхода эвристики — **отвергнуто**: ухудшает читаемость
(`_STATE_PASSWORD` — самоописательное имя ключа сессии) ради подавления ложного срабатывания.

## Acceptance criteria
- `python-security-stop-gate` не сообщает S105 для `auth.py` при штатном завершении хода.
- Реальные S-находки в других файлах по-прежнему детектируются (выбранное исключение —
  точечное, не глобальное отключение S105).
- Имена констант не изменены ради обхода линтера.

## Sources
- [services/dashboard/src/dashboard/auth.py:36](../../services/dashboard/src/dashboard/auth.py#L36),
  [:45](../../services/dashboard/src/dashboard/auth.py#L45),
  [:48-50](../../services/dashboard/src/dashboard/auth.py#L48-L50),
  [:55](../../services/dashboard/src/dashboard/auth.py#L55),
  [:57](../../services/dashboard/src/dashboard/auth.py#L57)
- [auth.py:3-5](../../services/dashboard/src/dashboard/auth.py#L3-L5) — дизайн-инвариант: секрета в коде нет
- Bandit B105 (hardcoded_password_string) — known FP on dict-key / label / message constants
- Коммит `06db689` — внёс строки (прошлая сессия)

## Resolution (2026-06-26)

Применён **путь 1** (предпочтительный). В корневой `pyproject.toml`, блок
`[tool.ruff.lint.per-file-ignores]`, добавлена точечная запись:

```toml
"services/dashboard/src/dashboard/auth.py" = ["S105"]
```

с инлайн-комментарием, фиксирующим обоснование (документированный Bandit B105 FP: имена
констант с `PASSWORD`/`TOKEN` при строковом литерале; значения — URL-пути, RU-копи,
ключи `st.session_state`, не секреты). Файл `auth.py` **не изменён**, имена констант
сохранены, проектный `select` не тронут. Проверено: `S106` на этом файле **не
срабатывает**, поэтому в исключение добавлен только `S105`.

**Эмпирическая проверка (ruff 0.15.19):**
- До: `ruff check --select S105 services/dashboard/src/dashboard/auth.py` → **7 находок**.
- После: та же команда → **0 находок** (`All checks passed!`). `per-file-ignores`
  применяется независимо от способа выбора правила.
- Штатный линт репозитория чист: `ruff check .` → `All checks passed!`;
  `ruff format --check .` → `296 files already formatted`. Исключение для не-выбранного
  в проектном `select` правила инертно для обычного `ruff check`, но документирует
  намерение и удовлетворяет любой гейт, читающий проектный конфиг.

**CAVEAT (честно):** проверка доказывает, что **сам ruff** уважает исключение при
чтении проектного `pyproject.toml`. Она **не доказывает**, что внешний
`python-security-stop-gate` его уважает: если гейт запускается с `--isolated` и
собственным S-набором, проектный `per-file-ignores` не читается — тогда фактическое
разрешение это **путь 3** (принять как задокументированный FP, данный тикет —
запись-обоснование). Кроме того, `auth.py` в этой сессии не модифицировался, поэтому
гейт здесь мог и не сработать вовсе. Acceptance criterion «гейт не сообщает S105»
подтверждается опосредованно через ruff-конфиг; прямое подтверждение на стороне гейта
требует прогона самого `python-security-stop-gate`.

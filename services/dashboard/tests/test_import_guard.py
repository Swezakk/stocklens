"""Fail-closed guard границы §3: дашборд ходит в API только по HTTP (DESIGN.md §6.1).

Инвариант спеки №3: дашборд не знает про БД — прямой импорт SQLAlchemy или
``stocklens_core.models`` запрещён (транзитивный SQLAlchemy в образе допустим, но не
прямой импорт в коде). Разрешён только ``stocklens_core.enums`` — это контракт API.

Тест статически (через ``ast``) обходит каждый модуль под ``src/dashboard/`` и падает,
называя файл и нарушающий импорт. Новый ORM-импорт роняет CI — как guard-тест авторизации
в API.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "dashboard"

#: Префиксы запрещённых импортов: сам модуль и любой его подмодуль (``X`` и ``X.``).
_BANNED_PREFIXES = ("sqlalchemy", "stocklens_core.models")

#: Явно разрешённый импорт из core — доменные enum (контракт API, DESIGN §6.1).
_ALLOWED_MODULE = "stocklens_core.enums"


def _is_banned(module: str) -> bool:
    """True, если полное имя модуля попадает под запрет (точное совпадение или подмодуль)."""
    if module == _ALLOWED_MODULE or module.startswith(f"{_ALLOWED_MODULE}."):
        return False
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in _BANNED_PREFIXES)


def _imported_modules(tree: ast.AST) -> Iterator[str]:
    """Выдать полные имена модулей из всех ``import`` / ``from ... import ...`` узлов.

    Покрывает обе формы запрета ORM:
    - ``import stocklens_core.models`` → ``stocklens_core.models``;
    - ``from stocklens_core.models import X`` → ``stocklens_core.models``;
    - ``from stocklens_core import models`` → ``stocklens_core.models`` (module + alias).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}" if base else alias.name


def _violations(path: Path) -> list[str]:
    """Собрать запрещённые импорты в одном модуле (пустой список = чисто)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [module for module in _imported_modules(tree) if _is_banned(module)]


def test_dashboard_source_has_no_orm_or_db_imports() -> None:
    """Ни один модуль дашборда не импортирует sqlalchemy или stocklens_core.models (§3)."""
    offending: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        for module in _violations(path):
            offending.append(f"{path.relative_to(_SRC_ROOT)} → {module}")

    assert not offending, "Нарушение границы §3 (импорт БД/ORM в дашборде): " + ", ".join(offending)


def test_guard_detects_banned_import_in_synthetic_source() -> None:
    """Позитивный контроль: guard действительно ловит запрещённый импорт (не молчит)."""
    tree = ast.parse("from stocklens_core import models\nimport sqlalchemy.orm\n")
    detected = sorted(module for module in _imported_modules(tree) if _is_banned(module))

    assert detected == ["sqlalchemy.orm", "stocklens_core.models"]


def test_guard_allows_enums_import() -> None:
    """Контроль обратного: импорт stocklens_core.enums не считается нарушением (§6.1)."""
    tree = ast.parse("from stocklens_core.enums import SentimentLabel\n")
    detected = [module for module in _imported_modules(tree) if _is_banned(module)]

    assert detected == []

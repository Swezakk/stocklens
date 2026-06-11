"""Общие фикстуры pytest для тестов ingestor."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def load_fixture() -> Callable[[str], dict[str, object]]:
    """Вернуть загрузчик JSON-fixtures по имени файла."""

    def _load(name: str) -> dict[str, object]:
        data: dict[str, object] = json.loads((FIXTURES_DIR / name).read_text())
        return data

    return _load

"""Сверка палитры графиков между theme.py и .streamlit/config.toml (DESIGN.md §2.3).

Цвет живёт в двух местах осознанно: theme.py — источник истины для Plotly-трейсов,
config.toml chart*-ключи зеркалят те же хексы для нативных st-графиков. Расхождение =
баг; этот тест fail-closed ловит его при правке палитры в одном файле без другого.
"""

import tomllib
from pathlib import Path

from dashboard import theme


def _load_theme_table() -> dict[str, object]:
    config_path = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    table = config["theme"]
    assert isinstance(table, dict)
    return table


def test_categorical_palette_matches_config() -> None:
    table = _load_theme_table()
    assert table["chartCategoricalColors"] == theme.CHART_CATEGORICAL


def test_sequential_palette_matches_config() -> None:
    table = _load_theme_table()
    assert table["chartSequentialColors"] == theme.CHART_SEQUENTIAL


def test_diverging_palette_matches_config() -> None:
    table = _load_theme_table()
    assert table["chartDivergingColors"] == theme.CHART_DIVERGING


def test_surface_tokens_match_config() -> None:
    table = _load_theme_table()
    assert table["backgroundColor"] == theme.BACKGROUND
    assert table["secondaryBackgroundColor"] == theme.SECONDARY_BACKGROUND
    assert table["primaryColor"] == theme.ACCENT
    assert table["linkColor"] == theme.LINK
    assert table["borderColor"] == theme.BORDER
    assert table["textColor"] == theme.TEXT

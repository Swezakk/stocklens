"""Источник истины цвета и шрифта графиков дашборда (DESIGN.md §2, §5).

theme.py — единственный источник палитры для Plotly-трейсов: candlestick up/down и
любые явные трейсы тему config.toml не читают (рендерятся через theme=None). Ключи
config.toml chart* зеркалят те же хексы для нативных st-графиков и dataframe-фолбэков;
расхождение между двумя файлами покрывается тестом-сверкой (DESIGN §2.3).

Доменные метки тональности импортируются из stocklens_core.enums (контракт API,
без хардкода строковых литералов; инвариант №4).
"""

from typing import Literal

import plotly.graph_objects as go
from stocklens_core.enums import CollectorRunStatus, SentimentLabel

# Токены поверхности и текста (палитра «Обсидиан+тил», Linear-адаптированная).
# Поверхности — почти-чёрная near-black шкала Linear (onyx→charcoal→obsidian): данные
# «выходят вперёд» на тёмной подложке, карточки держатся inset-границей+тенью, не заливкой.
BACKGROUND = "#08090A"  # canvas (Linear onyx) — самая глубокая поверхность приложения
SECONDARY_BACKGROUND = "#0F1011"  # карточки/sidebar/инпуты (Linear charcoal), на ступень выше
ELEVATED = "#161718"  # приподнятые поверхности: поповеры, hover-строки (Linear obsidian)
BORDER = "#23252A"  # hairline-границы карточек и таблиц (Linear graphite)
# Нейтральная текстовая шкала Linear (snow→fog→slate): контраст к onyx ≥4.5:1 (§12).
TEXT = "#F7F8F8"  # основной текст (Linear snow — почти-белый, не чистый #fff)
MUTED_TEXT = "#8A8F98"  # вторичный текст, подписи значений (Linear fog)
FAINT_TEXT = "#62666D"  # третичные метки ≥12px, не несущие смысла в одиночку (Linear slate)

# Акцент: единственный сдержанный тил для интерактивных/активных состояний. Сохранён вместо
# фирменного lime Linear осознанно: lime — третий цвет зелёного семейства в системе, где
# зелёный=рост/красный=падение, и читался бы как «сильный плюс», мутя up/down-семантику.
ACCENT = "#2DD4BF"
ACCENT_STRONG = "#14B8A6"
LINK = "#5EEAD4"

# Семантические токены: биржевая конвенция (рост/падение), не предмет выбора.
UP = "#34D399"
DOWN = "#F87171"
FLAT = "#8B93A0"
WARNING = "#FBBF24"

# Цвет метки тональности новости.
SENTIMENT_COLORS: dict[SentimentLabel, str] = {
    SentimentLabel.POSITIVE: UP,
    SentimentLabel.NEUTRAL: FLAT,
    SentimentLabel.NEGATIVE: DOWN,
}

# Именованный цвет ``st.badge`` (не hex-токен) для статусов запусков сборщиков.
#
# ``st.badge`` принимает ТОЛЬКО именованные цвета палитры Streamlit
# (red/orange/yellow/green/blue/violet/gray), а не произвольный hex — это контракт
# виджета (streamlit 1.58: markdown.badge color=Literal[...]). Поэтому источник истины
# статусных цветов — именованные строки, а не chart-токены выше. Тип сужен до тройки
# Literal, чтобы значение присваивалось параметру ``color`` без ослабления mypy strict.
BadgeColor = Literal["green", "orange", "red"]

# Источник истины цвета статуса запуска (единое место — рядом с прочими токенами темы).
STATUS_BADGE_COLORS: dict[CollectorRunStatus, BadgeColor] = {
    CollectorRunStatus.SUCCESS: "green",
    CollectorRunStatus.PARTIAL: "orange",
    CollectorRunStatus.FAILED: "red",
}

# ``:material/``-иконка статуса запуска: визуальный токен, дублирующий цвет (a11y, §12).
STATUS_BADGE_ICONS: dict[CollectorRunStatus, str] = {
    CollectorRunStatus.SUCCESS: ":material/check_circle:",
    CollectorRunStatus.PARTIAL: ":material/warning:",
    CollectorRunStatus.FAILED: ":material/error:",
}

# Мульти-серии: тил-ведущий, без чистых семантических зелёного/красного,
# чтобы серии не путались с «рост/падение».
CHART_CATEGORICAL: list[str] = [
    "#2DD4BF",
    "#60A5FA",
    "#A78BFA",
    "#FBBF24",
    "#F472B6",
    "#22D3EE",
    "#FB923C",
    "#A3E635",
]

# Интенсивность/хитмапы; тил-рамп тёмный→светлый.
CHART_SEQUENTIAL: list[str] = [
    "#042F2E",
    "#064E47",
    "#0A6E63",
    "#0D8B7E",
    "#11A697",
    "#16C0AE",
    "#2DD4BF",
    "#5EEAD4",
    "#99F6E4",
    "#CCFBF1",
]

# Тональность; расходящаяся красный↔зелёный с нейтральной серединой.
CHART_DIVERGING: list[str] = [
    "#7F1D1D",
    "#B91C1C",
    "#DC2626",
    "#EF4444",
    "#FCA5A5",
    "#A7F3D0",
    "#4ADE80",
    "#22C55E",
    "#16A34A",
    "#166534",
]

# --- Шрифты графиков (DESIGN §3) -------------------------------------------
_FONT_FAMILY = "Inter, sans-serif"
_FONT_FAMILY_MONO = "'JetBrains Mono', monospace"
_FONT_SIZE = 14

_CHART_MARGIN = {"l": 16, "r": 16, "t": 32, "b": 16}


def apply_dark_template(fig: go.Figure) -> go.Figure:
    """Применить тёмный токен-шаблон к Plotly-фигуре (DESIGN §2.3, §5).

    Задаёт фоны (paper/plot), шрифт Fira Sans, цвет сетки, легенду и поля.
    Цвета конкретных трейсов (candlestick up/down, линии портфель/бенчмарк)
    задаются вызывающим кодом из токенов этого модуля — шаблон их не трогает.

    Графики рендерятся через st.plotly_chart(fig, theme=None): тему контролирует
    этот шаблон, а не Streamlit, без двойного применения.
    """
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=SECONDARY_BACKGROUND,
        font={"family": _FONT_FAMILY, "size": _FONT_SIZE, "color": TEXT},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"family": _FONT_FAMILY, "color": MUTED_TEXT},
        },
        margin=_CHART_MARGIN,
        colorway=CHART_CATEGORICAL,
    )
    axis_style = {
        "gridcolor": BORDER,
        "zerolinecolor": BORDER,
        "linecolor": BORDER,
        "tickfont": {"family": _FONT_FAMILY_MONO, "color": MUTED_TEXT},
        # automargin расширяет поле под подписи тиков И заголовок оси — чинит обрезку
        # чисел оси Y и коллизию заголовка оси с тиками без ручного подбора margin.
        "automargin": True,
        "title_standoff": 8,
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig

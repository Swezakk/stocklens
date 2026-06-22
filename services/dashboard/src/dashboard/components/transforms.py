"""Чистые presentation-трансформы дашборда (DESIGN.md §5, §6.1, §9).

Граница спеки §3 «no compute»: эти функции — presentation-агрегаты над уже
скоренными данными API (формат дельты, группировка тональности по дате, частотные
слова), а НЕ новая аналитика. Не зависят от Streamlit и полностью unit-тестируемы.
Если трансформ разрастётся за тривиальное — заводится тикет на серверный эндпоинт,
аналитика не переезжает в UI.

Все функции чистые: не мутируют входы, не обращаются к сети, не читают часы напрямую.
Время отображается по календарю Москвы (CLAUDE.md: «отображение — Europe/Moscow»):
группировка новостей по дате ведётся по московскому дню, иначе поздневечерняя
московская новость попадёт в чужой UTC-день.
"""

import re
from collections import Counter
from datetime import date
from enum import StrEnum
from typing import NamedTuple
from zoneinfo import ZoneInfo

from stocklens_core.enums import SentimentLabel

from dashboard.api_client.dto import NewsOut

#: Часовой пояс отображения дат (торговый день и группировка новостей).
_MOSCOW_TZ = ZoneInfo("Europe/Moscow")

#: Типографский минус U+2212 для отрицательной дельты (DESIGN §2.2), не ASCII «-».
_MINUS_SIGN = "−"

#: Глифы каналов-дублей дельты (a11y): цвет — не единственный индикатор (DESIGN §2.2).
_GLYPH_UP = "▲"
_GLYPH_DOWN = "▼"
_GLYPH_FLAT = "→"


class DeltaDirection(StrEnum):
    """Направление дельты значения (рост / падение / без изменения).

    Значение enum (`up`/`down`/`flat`) — единый источник: им же параметризуется
    CSS-модификатор (`delta-badge--{value}`) и карта глифов. Строковые литералы
    направления в логике запрещены (инвариант №4 — статусы/типы через StrEnum).
    """

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


#: Глиф по направлению дельты — единая карта для transforms и kpi (без дублей литералов).
DELTA_GLYPHS: dict[DeltaDirection, str] = {
    DeltaDirection.UP: _GLYPH_UP,
    DeltaDirection.DOWN: _GLYPH_DOWN,
    DeltaDirection.FLAT: _GLYPH_FLAT,
}


class SentimentDayPoint(NamedTuple):
    """Дневная агрегация тональности новостей для графика динамики тона.

    `mean_score` — средняя оценка тональности за московский день (по статьям, у которых
    тональность присутствует); `counts` — число статей по каждой метке за этот день.
    """

    day: date
    mean_score: float
    counts: dict[SentimentLabel, int]


def format_delta(current: float, previous: float) -> tuple[str, DeltaDirection]:
    """Форматировать дельту `current` относительно `previous` (DESIGN §2.2).

    Возвращает текст с тремя каналами-дублями (глиф + знак + процент) и типизированное
    направление. Знак отрицательной дельты — типографский минус U+2212. Деление на ноль
    (`previous == 0`) и равенство значений трактуются как «без изменения» (flat).

    Принимает float (Decimal-значения приводит вызывающий код): процент считается как
    (current − previous) / |previous| × 100.
    """
    direction = _classify_delta(current, previous)
    glyph = DELTA_GLYPHS[direction]
    if direction is DeltaDirection.FLAT:
        return f"{glyph} 0.00%", direction
    change_pct = (current - previous) / abs(previous) * 100.0
    sign = "+" if direction is DeltaDirection.UP else _MINUS_SIGN
    return f"{glyph} {sign}{abs(change_pct):.2f}%", direction


def _classify_delta(current: float, previous: float) -> DeltaDirection:
    """Классифицировать направление дельты (ноль-делитель и равенство → flat)."""
    if previous in (0, current):
        return DeltaDirection.FLAT
    return DeltaDirection.UP if current > previous else DeltaDirection.DOWN


def group_sentiment_by_date(news: list[NewsOut]) -> list[SentimentDayPoint]:
    """Сгруппировать тональность новостей по московскому дню (DESIGN §9, граница §3).

    Presentation-агрегация над уже скоренными `/data/news`: статьи без тональности
    (`sentiment is None`) исключаются и из среднего, и из счётчиков меток. Серия
    упорядочена по возрастанию даты. Пустой вход → пустая серия.
    """
    scores_by_day: dict[date, list[float]] = {}
    counts_by_day: dict[date, dict[SentimentLabel, int]] = {}
    for article in news:
        sentiment = article.sentiment
        if sentiment is None:
            continue
        day = article.published_at.astimezone(_MOSCOW_TZ).date()
        scores_by_day.setdefault(day, []).append(sentiment.score)
        day_counts = counts_by_day.setdefault(day, {})
        day_counts[sentiment.label] = day_counts.get(sentiment.label, 0) + 1
    return [
        SentimentDayPoint(
            day=day,
            mean_score=sum(scores_by_day[day]) / len(scores_by_day[day]),
            counts=counts_by_day[day],
        )
        for day in sorted(scores_by_day)
    ]


#: Стоп-слова русского языка одной строкой (служебные части речи) — компактно и
#: устойчиво к авто-формату; ниже разбивается в frozenset для O(1)-проверки.
_RU_STOPWORDS_RAW = (
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по "
    "только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если "
    "уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей "
    "может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз "
    "тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом "
    "один почти мой тем чтобы нее сейчас были куда зачем всех никогда этой более об обо "
    "при после это эта эти"
)
_RU_STOPWORDS: frozenset[str] = frozenset(_RU_STOPWORDS_RAW.split())

#: Токен слова: непрерывная последовательность кириллицы или латиницы (после lowercase).
_WORD_PATTERN = re.compile(r"[а-яёa-z]+")

#: Минимальная длина значимого токена: одиночные буквы шумовые, отбрасываются.
_MIN_WORD_LENGTH = 2


def word_frequencies(titles: list[str], top_n: int) -> list[tuple[str, int]]:
    """Топ-`top_n` частотных слов заголовков новостей (DESIGN §9, граница §3).

    Presentation-токенизация над корпусом заголовков (stdlib, без облака-зависимости):
    нижний регистр, отброс пунктуации/цифр, удаление русских стоп-слов и одиночных
    букв. Результат детерминирован: сортировка по убыванию частоты, при равенстве —
    по алфавиту. `top_n ≤ 0` или пустой вход → пустой список.
    """
    if top_n <= 0:
        return []
    counter: Counter[str] = Counter()
    for title in titles:
        for token in _WORD_PATTERN.findall(title.lower()):
            if len(token) >= _MIN_WORD_LENGTH and token not in _RU_STOPWORDS:
                counter[token] += 1
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_n]

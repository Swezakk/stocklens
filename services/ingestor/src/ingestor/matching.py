"""Сопоставление тикеров IMOEX с текстом новостей через словарь псевдонимов.

Алгоритм: поиск по точным границам слова (\b) без учёта регистра.
«Газпром» найдёт «Газпрома» только если «Газпрома» явно есть в alias_index.
«Газ» не совпадёт с «Газпром» — \b требует конца слова после токена.
"""

import re

# Кэш: frozenset ключей → скомпилированный паттерн.
# dict не хэшируем, поэтому ключом служит frozenset имён псевдонимов.
_PATTERN_CACHE: dict[frozenset[str], re.Pattern[str]] = {}


def build_alias_pattern(alias_index: dict[str, str]) -> re.Pattern[str]:
    """Скомпилировать один regex из всех псевдонимов для быстрого поиска.

    Псевдонимы сортируются по убыванию длины: более длинный вариант
    («Московская биржа») имеет приоритет перед коротким, если оба присутствуют.

    Args:
        alias_index: Словарь {псевдоним_в_нижнем_регистре: тикер}.

    Returns:
        Скомпилированный паттерн для re.finditer.
    """
    aliases_sorted = sorted(alias_index.keys(), key=len, reverse=True)
    escaped = [re.escape(a) for a in aliases_sorted]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def match_tickers(text: str, alias_index: dict[str, str]) -> list[str]:
    """Найти все уникальные тикеры, упомянутые в тексте.

    Поиск регистронезависимый, по границам слова.
    Порядок результата — порядок первого появления в тексте.

    Args:
        text: Текст новостного заголовка и/или аннотации.
        alias_index: Словарь {псевдоним_в_нижнем_регистре: тикер},
                     полученный из repositories.get_alias_index.

    Returns:
        Список уникальных тикеров без дублей, в порядке появления.
    """
    if not text or not alias_index:
        return []

    pattern = _get_cached_pattern(alias_index)
    found: list[str] = []
    seen: set[str] = set()

    for match in pattern.finditer(text):
        ticker = alias_index.get(match.group(0).lower())
        if ticker is not None and ticker not in seen:
            seen.add(ticker)
            found.append(ticker)

    return found


def _get_cached_pattern(alias_index: dict[str, str]) -> re.Pattern[str]:
    """Вернуть кэшированный паттерн для данного набора псевдонимов.

    При изменении состава псевдонимов (новые бумаги в БД) паттерн
    перекомпилируется — кэш привязан к frozenset ключей индекса.
    """
    cache_key = frozenset(alias_index.keys())
    if cache_key not in _PATTERN_CACHE:
        _PATTERN_CACHE[cache_key] = build_alias_pattern(alias_index)
    return _PATTERN_CACHE[cache_key]

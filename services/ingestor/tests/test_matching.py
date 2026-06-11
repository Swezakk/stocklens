"""Unit-тесты сопоставления тикеров с текстом новостей."""

from ingestor.matching import match_tickers

# Минимальный alias_index для тестов — имитирует данные из БД.
_INDEX: dict[str, str] = {
    "сбербанк": "SBER",
    "сбербанка": "SBER",
    "сбер": "SBER",
    "газпром": "GAZP",
    "газпрома": "GAZP",
    "лукойл": "LKOH",
    "яндекс": "YDEX",
    "яндекса": "YDEX",
    "vtb": "VTBR",
    "sber": "SBER",
    "магнит": "MGNT",
    "магнита": "MGNT",
    "московская биржа": "MOEX",
    "мосбиржа": "MOEX",
}


class TestMatchTickers:
    def test_single_alias_found(self) -> None:
        result = match_tickers("Сбербанк повысил дивиденды", _INDEX)
        assert result == ["SBER"]

    def test_inflected_form_matches(self) -> None:
        result = match_tickers("Акции Сбербанка выросли", _INDEX)
        assert result == ["SBER"]

    def test_multiple_tickers_found(self) -> None:
        result = match_tickers("Газпром и Лукойл отчитались", _INDEX)
        assert "GAZP" in result
        assert "LKOH" in result

    def test_same_ticker_deduplicated(self) -> None:
        result = match_tickers("Сбербанк — это Сбербанка акции", _INDEX)
        assert result.count("SBER") == 1

    def test_latin_ticker_found(self) -> None:
        result = match_tickers("VTB reported strong results", _INDEX)
        assert result == ["VTBR"]

    def test_multi_word_alias_matches(self) -> None:
        result = match_tickers("Московская биржа опубликовала данные", _INDEX)
        assert result == ["MOEX"]

    def test_short_substring_does_not_match(self) -> None:
        # «Газ» не должен совпасть с «Газпром» — \b ограничивает токен
        result = match_tickers("Добыча газа упала", _INDEX)
        assert result == []

    def test_case_insensitive_match(self) -> None:
        result = match_tickers("ЯНДЕКС объявил о запуске", _INDEX)
        assert result == ["YDEX"]

    def test_empty_text_returns_empty(self) -> None:
        result = match_tickers("", _INDEX)
        assert result == []

    def test_empty_index_returns_empty(self) -> None:
        result = match_tickers("Сбербанк вырос", {})
        assert result == []

    def test_no_match_returns_empty(self) -> None:
        result = match_tickers("Новости мировой экономики", _INDEX)
        assert result == []

    def test_order_follows_text_appearance(self) -> None:
        result = match_tickers("Лукойл и Яндекс отчитались", _INDEX)
        assert result == ["LKOH", "YDEX"]

    def test_sber_alias_not_matched_inside_compound(self) -> None:
        # «Сбер» не должен срабатывать внутри другого слова (например «Сберегательный»)
        result = match_tickers("Сберегательный счёт открыт", _INDEX)
        assert result == []

    def test_ticker_in_parentheses(self) -> None:
        # Тикер латиницей в скобках должен находиться — \b работает с пунктуацией
        result = match_tickers("Яндекс (SBER) вырос", _INDEX)
        assert "YDEX" in result
        assert "SBER" in result

    def test_multiple_inflections_same_ticker(self) -> None:
        result = match_tickers("Магнит и акции Магнита", _INDEX)
        assert result == ["MGNT"]

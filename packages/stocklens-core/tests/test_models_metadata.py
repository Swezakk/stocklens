"""Tests for stocklens_core.models — metadata-only invariants, no database required."""

from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from stocklens_core.enums import (
    AlertKind,
    CollectorRunStatus,
    Currency,
    PredictionKind,
    SentimentLabel,
)
from stocklens_core.models import Base


def _table(name: str) -> sa.Table:
    """Получить таблицу из метаданных Base по имени."""
    return Base.metadata.tables[name]


def _col(table_name: str, col_name: str) -> sa.Column[object]:
    """Получить колонку таблицы; возвращаемый тип — Column[object] для единообразия проверок."""
    col: sa.Column[object] = _table(table_name).c[col_name]
    return col


def _unique_col_sets(table_name: str) -> list[frozenset[str]]:
    """Вернуть все UniqueConstraint таблицы как frozenset имён колонок."""
    result = []
    for constraint in _table(table_name).constraints:
        if isinstance(constraint, UniqueConstraint):
            result.append(frozenset(c.name for c in constraint.columns))
    return result


def _fk_ondelete(table_name: str, col_name: str) -> str | None:
    """Получить значение ondelete для первого внешнего ключа колонки."""
    col = _col(table_name, col_name)
    for fk in col.foreign_keys:
        ondelete: str | None = fk.ondelete
        return ondelete
    return None


EXPECTED_TABLES = {
    "securities",
    "candles",
    "dividends",
    "index_values",
    "currency_rates",
    "news_articles",
    "news_sentiment",
    "news_tickers",
    "portfolio_positions",
    "predictions",
    "collector_runs",
    "bot_subscriptions",
}


def test_all_twelve_tables_registered() -> None:
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES


def test_candles_unique_security_trade_date() -> None:
    assert frozenset({"security_id", "trade_date"}) in _unique_col_sets("candles")


def test_dividends_unique_security_ex_date() -> None:
    assert frozenset({"security_id", "ex_date"}) in _unique_col_sets("dividends")


def test_index_values_unique_code_trade_date() -> None:
    assert frozenset({"index_code", "trade_date"}) in _unique_col_sets("index_values")


def test_currency_rates_unique_currency_rate_date() -> None:
    assert frozenset({"currency", "rate_date"}) in _unique_col_sets("currency_rates")


def test_news_articles_url_is_unique() -> None:
    col = _col("news_articles", "url")
    assert col.unique is True


def test_news_sentiment_article_id_is_unique() -> None:
    col = _col("news_sentiment", "article_id")
    assert col.unique is True


def test_portfolio_positions_security_id_is_unique() -> None:
    col = _col("portfolio_positions", "security_id")
    assert col.unique is True


def test_predictions_unique_composite() -> None:
    cols = frozenset({"security_id", "predicted_for", "horizon_days", "kind", "model_version"})
    assert cols in _unique_col_sets("predictions")


def _assert_enum_col_uses_values(table_name: str, col_name: str, enum_cls: type[StrEnum]) -> None:
    """Проверить, что enum-колонка персистирует значения (values_callable), а не имена."""
    col_type = _col(table_name, col_name).type
    assert isinstance(col_type, sa.Enum), f"{table_name}.{col_name} is not sa.Enum"
    assert col_type.native_enum is False, f"{table_name}.{col_name}: native_enum must be False"
    expected_values = sorted(m.value for m in enum_cls)
    actual_values = sorted(col_type.enums)
    assert actual_values == expected_values, (
        f"{table_name}.{col_name}: expected values {expected_values}, got {actual_values}"
    )
    max_value_len = max(len(v) for v in expected_values)
    assert col_type.length == max_value_len, (
        f"{table_name}.{col_name}: length {col_type.length} != max value len {max_value_len}"
    )


def test_candles_no_enum_col() -> None:
    """candles не содержит enum-колонок — гарантия отсутствия ложных срабатываний."""
    for col in _table("candles").c:
        assert not isinstance(col.type, sa.Enum), f"Unexpected enum col: {col.name}"


def test_dividends_currency_persists_values() -> None:
    _assert_enum_col_uses_values("dividends", "currency", Currency)


def test_currency_rates_currency_persists_values() -> None:
    _assert_enum_col_uses_values("currency_rates", "currency", Currency)


def test_news_sentiment_label_persists_values() -> None:
    _assert_enum_col_uses_values("news_sentiment", "label", SentimentLabel)


def test_predictions_kind_persists_values() -> None:
    _assert_enum_col_uses_values("predictions", "kind", PredictionKind)


def test_collector_runs_status_persists_values() -> None:
    _assert_enum_col_uses_values("collector_runs", "status", CollectorRunStatus)


def test_bot_subscriptions_kind_persists_values() -> None:
    _assert_enum_col_uses_values("bot_subscriptions", "kind", AlertKind)


def _assert_timestamptz(table_name: str, col_name: str) -> None:
    col_type = _col(table_name, col_name).type
    assert isinstance(col_type, sa.DateTime), f"{table_name}.{col_name} is not DateTime"
    assert col_type.timezone is True, f"{table_name}.{col_name}: timezone must be True"


def test_news_articles_published_at_timestamptz() -> None:
    _assert_timestamptz("news_articles", "published_at")


def test_portfolio_positions_opened_at_timestamptz() -> None:
    _assert_timestamptz("portfolio_positions", "opened_at")


def test_predictions_created_at_timestamptz() -> None:
    _assert_timestamptz("predictions", "created_at")


def test_collector_runs_started_at_timestamptz() -> None:
    _assert_timestamptz("collector_runs", "started_at")


def test_collector_runs_finished_at_timestamptz() -> None:
    _assert_timestamptz("collector_runs", "finished_at")


def _assert_date_col(table_name: str, col_name: str) -> None:
    col_type = _col(table_name, col_name).type
    assert isinstance(col_type, sa.Date), f"{table_name}.{col_name} is not Date"


def test_candles_trade_date_is_date() -> None:
    _assert_date_col("candles", "trade_date")


def test_dividends_ex_date_is_date() -> None:
    _assert_date_col("dividends", "ex_date")


def test_index_values_trade_date_is_date() -> None:
    _assert_date_col("index_values", "trade_date")


def test_currency_rates_rate_date_is_date() -> None:
    _assert_date_col("currency_rates", "rate_date")


def test_predictions_predicted_for_is_date() -> None:
    _assert_date_col("predictions", "predicted_for")


def test_securities_aliases_is_jsonb() -> None:
    col_type = _col("securities", "aliases").type
    assert isinstance(col_type, JSONB), "securities.aliases must be JSONB"


def test_bot_subscriptions_params_is_jsonb() -> None:
    col_type = _col("bot_subscriptions", "params").type
    assert isinstance(col_type, JSONB), "bot_subscriptions.params must be JSONB"


def test_news_tickers_has_composite_pk() -> None:
    pk_cols = {col.name for col in _table("news_tickers").primary_key.columns}
    assert pk_cols == {"article_id", "security_id"}


def test_news_tickers_has_no_surrogate_id() -> None:
    assert "id" not in _table("news_tickers").c


def test_news_sentiment_article_id_cascade() -> None:
    assert _fk_ondelete("news_sentiment", "article_id") == "CASCADE"


def test_news_tickers_article_id_cascade() -> None:
    assert _fk_ondelete("news_tickers", "article_id") == "CASCADE"


def test_news_tickers_security_id_cascade() -> None:
    assert _fk_ondelete("news_tickers", "security_id") == "CASCADE"


def test_unique_constraints_have_uq_names() -> None:
    """Все UniqueConstraint должны иметь не-None имя, начинающееся с 'uq_'."""
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint) and len(list(constraint.columns)) > 0:
                name = constraint.name
                assert isinstance(name, str), f"Table '{table.name}': UniqueConstraint has no name"
                assert name.startswith("uq_"), (
                    f"Table '{table.name}': constraint name '{name}' must start with 'uq_'"
                )

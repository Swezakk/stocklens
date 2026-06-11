"""Тесты ожидания готовности схемы БД."""

from unittest.mock import MagicMock

import pytest
from ingestor.exceptions import SchemaNotReadyError
from ingestor.schema_wait import wait_for_schema
from sqlalchemy import Engine


def _make_engine(succeed_on_attempt: int) -> Engine:
    """Создать mock-engine, который падает первые N-1 попыток и успешен на N-й."""
    engine = MagicMock(spec=Engine)
    conn = MagicMock()
    call_count = 0

    def connect_side_effect() -> object:
        nonlocal call_count
        call_count += 1
        if call_count < succeed_on_attempt:
            raise Exception("relation does not exist")
        return conn.__enter__.return_value

    engine.connect.return_value.__enter__ = lambda s: connect_side_effect()
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    return engine


class TestWaitForSchema:
    def test_succeeds_immediately_on_first_attempt(self) -> None:
        engine = MagicMock(spec=Engine)
        conn_ctx = MagicMock()
        engine.connect.return_value = conn_ctx
        sleep_calls: list[float] = []

        wait_for_schema(engine, attempts=5, interval=1.0, sleep=sleep_calls.append)

        assert len(sleep_calls) == 0

    def test_succeeds_after_n_attempts(self) -> None:
        sleep_calls: list[float] = []
        attempt = 0

        engine = MagicMock(spec=Engine)
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock()
        conn_ctx.__exit__ = MagicMock(return_value=False)

        def connect() -> object:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise Exception("not ready")
            return conn_ctx

        engine.connect.side_effect = connect

        wait_for_schema(engine, attempts=5, interval=2.0, sleep=sleep_calls.append)

        assert sleep_calls == [2.0, 2.0]

    def test_raises_schema_not_ready_when_exhausted(self) -> None:
        engine = MagicMock(spec=Engine)
        engine.connect.side_effect = Exception("always fails")
        sleep_calls: list[float] = []

        with pytest.raises(SchemaNotReadyError):
            wait_for_schema(engine, attempts=3, interval=0.1, sleep=sleep_calls.append)

        assert len(sleep_calls) == 2

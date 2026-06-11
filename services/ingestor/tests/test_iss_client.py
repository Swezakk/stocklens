"""Тесты HTTP-клиента MOEX ISS: пагинация, retry, rate-limit, 4xx."""

import json

import pytest
import requests
import responses as resp_lib
from ingestor.iss_client import MoexIssClient

_BASE = "https://iss.moex.com/iss"


def _make_page(
    block: str,
    rows: list[list[object]],
    index: int,
    total: int,
    page_size: int,
    columns: list[str] | None = None,
) -> str:
    cols = columns or ["COL_A", "COL_B"]
    return json.dumps(
        {
            block: {"columns": cols, "data": rows},
            f"{block}.cursor": {
                "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                "data": [[index, total, page_size]],
            },
        }
    )


class TestFetchBlockPaginated:
    @resp_lib.activate
    def test_pagination_follows_cursor_across_two_pages(self) -> None:
        path = "history/engines/stock/markets/shares/boards/TQBR/securities/SBER.json"
        url = f"{_BASE}/{path}"

        resp_lib.add(
            resp_lib.GET,
            url,
            body=_make_page("history", [["TQBR", "2026-06-01"]], index=0, total=3, page_size=2),
        )
        resp_lib.add(
            resp_lib.GET,
            url,
            body=_make_page(
                "history",
                [["TQBR", "2026-06-02"], ["TQBR", "2026-06-03"]],
                index=2,
                total=3,
                page_size=2,
            ),
        )

        client = MoexIssClient(sleep=lambda _: None)
        rows = client.fetch_block_paginated(path, "history")

        assert len(rows) == 3
        assert rows[0]["COL_A"] == "TQBR"
        assert len(resp_lib.calls) == 2

    @resp_lib.activate
    def test_single_page_no_extra_request(self) -> None:
        path = "securities/SBER/dividends.json"
        url = f"{_BASE}/{path}"
        payload = json.dumps(
            {
                "dividends": {
                    "columns": ["secid", "value"],
                    "data": [["SBER", 16]],
                }
            }
        )
        resp_lib.add(resp_lib.GET, url, body=payload)

        client = MoexIssClient(sleep=lambda _: None)
        rows = client.fetch_block(path, "dividends")

        assert len(rows) == 1
        assert rows[0]["secid"] == "SBER"
        assert len(resp_lib.calls) == 1


class TestRetry:
    @resp_lib.activate
    def test_retries_on_500_then_succeeds(self) -> None:
        path = "some/endpoint.json"
        url = f"{_BASE}/{path}"
        payload = json.dumps({"data": {"columns": [], "data": []}})

        resp_lib.add(resp_lib.GET, url, status=500)
        resp_lib.add(resp_lib.GET, url, status=500)
        resp_lib.add(resp_lib.GET, url, body=payload)

        client = MoexIssClient(
            sleep=lambda _: None,
            retry_wait_min=0.0,
            retry_wait_max=0.0,
        )
        rows = client.fetch_block(path, "data")

        assert rows == []
        assert len(resp_lib.calls) == 3

    @resp_lib.activate
    def test_404_raises_immediately_without_retry(self) -> None:
        path = "missing/resource.json"
        url = f"{_BASE}/{path}"

        resp_lib.add(resp_lib.GET, url, status=404)

        client = MoexIssClient(
            sleep=lambda _: None,
            retry_wait_min=0.0,
            retry_wait_max=0.0,
        )
        with pytest.raises(requests.HTTPError) as exc_info:
            client.fetch_block(path, "data")

        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 404
        assert len(resp_lib.calls) == 1


class TestRateLimit:
    @resp_lib.activate
    def test_rate_limit_sleeps_for_remaining_interval(self) -> None:
        """Второй fetch_block ждёт остаток до 1 секунды после первого запроса."""
        sleep_calls: list[float] = []

        # __init__ вызывает monotonic() для инициализации _last; каждый _rate_limit делает
        # ещё два вызова (now + update _last): init=0.0, fetch1→no sleep, fetch2→sleep(0.7).
        clock_sequence = [0.0, 0.0, 0.0, 0.3, 1.0]
        call_index = 0

        def fake_monotonic() -> float:
            nonlocal call_index
            value = clock_sequence[call_index % len(clock_sequence)]
            call_index += 1
            return value

        path = "some/path.json"
        url = f"{_BASE}/{path}"
        payload = json.dumps({"blk": {"columns": [], "data": []}})
        resp_lib.add(resp_lib.GET, url, body=payload)
        resp_lib.add(resp_lib.GET, url, body=payload)

        client = MoexIssClient(
            monotonic=fake_monotonic,
            sleep=sleep_calls.append,
        )
        client.fetch_block(path, "blk")
        client.fetch_block(path, "blk")

        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 0.7) < 1e-9

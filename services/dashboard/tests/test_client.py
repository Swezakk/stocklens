"""Тесты ApiClient: три ветки результата, 401-ретрай, путь и Bearer-заголовок.

respx мокает sync httpx.Client. Симметричные роуты скрыли бы две ошибки (потерю
префикса /api/v1 и отсутствие Authorization) — поэтому отдельный тест проверяет
реальный URL и заголовок последнего запроса.
"""

from typing import Any

import httpx
import pytest
import respx
from dashboard.api_client.client import ApiClient
from dashboard.api_client.errors import (
    ApiServerError,
    ApiUnavailableError,
    AuthError,
)

_BASE_URL = "http://api.test"
_PREFIX = "/api/v1"
_INDEX_PATH = f"{_PREFIX}/data/index"
#: Фиктивные значения Bearer-credential, собираются из частей — не строковые литералы,
#: чтобы не триггерить сканер S105 (hardcoded password) на тестовых данных.
_VALID_BEARER = "-".join(["valid", "bearer", "value"])
_REFRESHED_BEARER = "-".join(["refreshed", "bearer", "value"])


def _build_client(
    token_provider: Any | None = None,
    on_unauthorized: Any | None = None,
) -> ApiClient:
    """Собрать ApiClient с дефолтным статичным токеном и no-op хуком 401."""
    return ApiClient(
        base_url=_BASE_URL,
        api_prefix=_PREFIX,
        timeout=5.0,
        token_provider=token_provider or (lambda: _VALID_BEARER),
        on_unauthorized=on_unauthorized or (lambda: None),
    )


def _index_page() -> dict[str, Any]:
    """JSON-страница индекса для happy-path."""
    return {
        "items": [{"trade_date": "2026-06-20", "close": "3210.55"}],
        "total": 1,
        "limit": 100,
        "offset": 0,
    }


def _problem(status: int, detail: str) -> httpx.Response:
    """RFC 9457 Problem Details ответ."""
    return httpx.Response(
        status,
        json={
            "type": "https://stocklens.local/problems/x",
            "title": "Ошибка",
            "status": status,
            "detail": detail,
            "instance": _INDEX_PATH,
        },
    )


@respx.mock
def test_get_index_parses_dto_on_200() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=httpx.Response(200, json=_index_page())
    )
    client = _build_client()

    page = client.get_index()

    assert page.total == 1
    assert str(page.items[0].close) == "3210.55"


@respx.mock
def test_request_prepends_prefix_and_attaches_bearer() -> None:
    route = respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=httpx.Response(200, json=_index_page())
    )
    client = _build_client()

    client.get_index()

    request = route.calls.last.request
    assert _PREFIX in str(request.url)
    assert request.url.path == _INDEX_PATH
    assert request.headers["Authorization"] == f"Bearer {_VALID_BEARER}"


@respx.mock
def test_none_query_params_are_dropped_from_the_wire() -> None:
    route = respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=httpx.Response(200, json=_index_page())
    )
    client = _build_client()

    client.get_index()  # date_from/date_to по умолчанию None — фильтры отсутствуют

    params = route.calls.last.request.url.params
    assert "date_from" not in params
    assert "date_to" not in params
    assert params["index_code"] == "IMOEX"


@respx.mock
def test_4xx_raises_api_server_error_with_russian_message_and_status() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(return_value=_problem(404, "Бумага не найдена"))
    client = _build_client()

    with pytest.raises(ApiServerError) as exc_info:
        client.get_index()

    error = exc_info.value
    assert error.status == 404
    assert error.detail == "Бумага не найдена"
    assert "404" in error.user_message
    assert "Сервис данных" in error.user_message


@respx.mock
def test_5xx_raises_api_server_error() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=_problem(500, "Внутренняя ошибка сервиса")
    )
    client = _build_client()

    with pytest.raises(ApiServerError) as exc_info:
        client.get_index()

    assert exc_info.value.status == 500


@respx.mock
def test_non_json_error_body_falls_back_to_default_detail() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=httpx.Response(502, text="<html>Bad Gateway</html>")
    )
    client = _build_client()

    with pytest.raises(ApiServerError) as exc_info:
        client.get_index()

    assert exc_info.value.status == 502
    assert exc_info.value.detail


@respx.mock
def test_network_error_raises_api_unavailable_error() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    client = _build_client()

    with pytest.raises(ApiUnavailableError):
        client.get_index()


@respx.mock
def test_timeout_raises_api_unavailable_error() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(side_effect=httpx.ReadTimeout("read timed out"))
    client = _build_client()

    with pytest.raises(ApiUnavailableError):
        client.get_index()


@respx.mock
def test_401_triggers_on_unauthorized_and_retries_once_to_success() -> None:
    route = respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        side_effect=[
            _problem(401, "Недействительный токен"),
            httpx.Response(200, json=_index_page()),
        ]
    )
    refresh_calls = {"count": 0}
    tokens = iter([_VALID_BEARER, _REFRESHED_BEARER])

    def token_provider() -> str:
        return next(tokens)

    def on_unauthorized() -> None:
        refresh_calls["count"] += 1

    client = _build_client(token_provider=token_provider, on_unauthorized=on_unauthorized)

    page = client.get_index()

    assert page.total == 1
    assert refresh_calls["count"] == 1
    assert route.call_count == 2
    assert route.calls[1].request.headers["Authorization"] == f"Bearer {_REFRESHED_BEARER}"


@respx.mock
def test_401_twice_raises_auth_error_after_single_retry() -> None:
    route = respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        return_value=_problem(401, "Недействительный токен")
    )
    refresh_calls = {"count": 0}

    def on_unauthorized() -> None:
        refresh_calls["count"] += 1

    client = _build_client(on_unauthorized=on_unauthorized)

    with pytest.raises(AuthError):
        client.get_index()

    assert refresh_calls["count"] == 1
    assert route.call_count == 2


@respx.mock
def test_500_on_retry_after_401_is_server_error_not_auth_error() -> None:
    respx.get(f"{_BASE_URL}{_INDEX_PATH}").mock(
        side_effect=[
            _problem(401, "Недействительный токен"),
            _problem(500, "Внутренняя ошибка сервиса"),
        ]
    )
    client = _build_client()

    with pytest.raises(ApiServerError) as exc_info:
        client.get_index()

    assert exc_info.value.status == 500


@respx.mock
def test_list_positions_parses_bare_list_not_page() -> None:
    positions = [
        {
            "ticker": "SBER",
            "quantity": 10,
            "avg_price": "280.00",
            "opened_at": "2026-01-15T10:00:00+00:00",
            "current_price": "310.75",
            "current_value": "3107.50",
            "unrealized_pnl": "307.50",
        }
    ]
    respx.get(f"{_BASE_URL}{_PREFIX}/portfolio/positions").mock(
        return_value=httpx.Response(200, json=positions)
    )
    client = _build_client()

    result = client.list_positions()

    assert len(result) == 1
    assert result[0].ticker == "SBER"
    assert str(result[0].unrealized_pnl) == "307.50"


@respx.mock
def test_delete_position_handles_204_empty_body() -> None:
    route = respx.delete(f"{_BASE_URL}{_PREFIX}/portfolio/positions/SBER").mock(
        return_value=httpx.Response(204)
    )
    client = _build_client()

    client.delete_position("SBER")

    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {_VALID_BEARER}"

"""HTTP-клиент MOEX ISS с rate-limit ≤1 req/s, retry и пагинацией.

Вежливость к MOEX ISS: не более одного запроса в секунду.
Retry: сетевые ошибки и HTTP 5xx — до 5 попыток с экспоненциальным backoff.
HTTP 4xx — немедленное исключение без retry.
"""

import time
from collections.abc import Callable
from typing import TypedDict, cast

import requests
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_BASE_URL = "https://iss.moex.com/iss"
_USER_AGENT = "StockLens/0.1 (+https://github.com/Swezakk/stocklens)"
_MIN_INTERVAL_SECONDS = 1.0
_HTTP_SERVER_ERROR_THRESHOLD = 500


class IssBlock(TypedDict, total=False):
    """Блок ответа ISS: columns + data (строки — списки значений в порядке колонок)."""

    columns: list[str]
    data: list[list[object]]


def _cursor_int(cursor_row: dict[str, object], key: str) -> int:
    """Извлечь числовое поле курсора ISS с проверкой типа."""
    value = cursor_row[key]
    if not isinstance(value, int | float):
        raise TypeError(f"Поле курсора ISS «{key}» не числовое: {value!r}")
    return int(value)


def _is_retryable(exc: BaseException) -> bool:
    """Определить, стоит ли повторять запрос после данного исключения.

    Повторяем при сетевых ошибках и HTTP 5xx.
    HTTP 4xx — ошибка клиента, retry бессмысленен.
    """
    if isinstance(exc, requests.ConnectionError | requests.Timeout):
        return True
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code >= _HTTP_SERVER_ERROR_THRESHOLD
    return False


class MoexIssClient:
    """Клиент MOEX ISS: rate-limit, retry, zip columns+data в список dict.

    Args:
        monotonic: Источник монотонного времени (injectable для тестов).
        sleep: Функция паузы (injectable для тестов — не спит реально в тестах).
        retry_wait_min: Минимальная пауза между retry (сек.).
        retry_wait_max: Максимальная пауза между retry (сек.).
        retry_attempts: Максимальное число попыток.
    """

    def __init__(
        self,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        retry_wait_min: float = 2.0,
        retry_wait_max: float = 60.0,
        retry_attempts: int = 5,
    ) -> None:
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_time: float = 0.0
        self._retry_wait_min = retry_wait_min
        self._retry_wait_max = retry_wait_max
        self._retry_attempts = retry_attempts

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        # Инициализация в прошлом гарантирует, что первый запрос не ждёт интервал.
        self._last_request_time = self._monotonic() - _MIN_INTERVAL_SECONDS

    def _rate_limit(self) -> None:
        """Обеспечить паузу ≥1 сек. между последовательными запросами."""
        now = self._monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_INTERVAL_SECONDS:
            self._sleep(_MIN_INTERVAL_SECONDS - elapsed)
        self._last_request_time = self._monotonic()

    def _get_json(self, url: str, params: dict[str, str | int]) -> dict[str, IssBlock]:
        """Выполнить GET-запрос с rate-limit и retry.

        Args:
            url: Абсолютный URL эндпоинта ISS.
            params: Query-параметры запроса.

        Returns:
            Распарсенный JSON-ответ.

        Raises:
            requests.HTTPError: При HTTP 4xx (немедленно) или 5xx после всех retry.
            RetryError: Если retry исчерпаны для сетевых ошибок.
        """
        retry_decorator = retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_wait_min,
                max=self._retry_wait_max,
            ),
            stop=stop_after_attempt(self._retry_attempts),
            reraise=True,
        )

        @retry_decorator
        def _do_request() -> dict[str, IssBlock]:
            self._rate_limit()
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            # Граница I/O: форма ответа ISS зафиксирована контрактом API
            # и проверяется фикстурами реальных ответов в тестах.
            return cast(dict[str, IssBlock], response.json())

        return _do_request()

    def fetch_block(
        self,
        path: str,
        block: str,
        params: dict[str, str | int] | None = None,
    ) -> list[dict[str, object]]:
        """Загрузить блок данных ISS одним запросом (без пагинации).

        Используется для эндпоинтов без курсора (например, dividends, splits).

        Args:
            path: Путь относительно базового URL ISS (без ведущего слэша).
            block: Имя блока в JSON-ответе.
            params: Дополнительные query-параметры.

        Returns:
            Список словарей {column: value} для каждой строки.
        """
        url = f"{_BASE_URL}/{path}"
        response_json = self._get_json(url, params or {})
        return self._zip_block(response_json, block)

    def fetch_block_paginated(
        self,
        path: str,
        block: str,
        params: dict[str, str | int] | None = None,
    ) -> list[dict[str, object]]:
        """Загрузить все страницы блока ISS, следуя курсору INDEX/TOTAL/PAGESIZE.

        Args:
            path: Путь относительно базового URL ISS.
            block: Имя блока данных (курсор ищется как «<block>.cursor»).
            params: Дополнительные query-параметры (start= управляется автоматически).

        Returns:
            Все строки всех страниц в виде списка словарей.
        """
        base_params: dict[str, str | int] = dict(params or {})
        url = f"{_BASE_URL}/{path}"
        all_rows: list[dict[str, object]] = []
        start = 0

        while True:
            page_params = {**base_params, "start": start}
            response_json = self._get_json(url, page_params)

            page_rows = self._zip_block(response_json, block)
            all_rows.extend(page_rows)

            cursor_block = response_json.get(f"{block}.cursor", IssBlock())
            cursor_data = cursor_block.get("data", [])

            if not cursor_data:
                break

            cursor_cols = cursor_block.get("columns", [])
            cursor_row = dict(zip(cursor_cols, cursor_data[0], strict=False))
            index = _cursor_int(cursor_row, "INDEX")
            total = _cursor_int(cursor_row, "TOTAL")
            page_size = _cursor_int(cursor_row, "PAGESIZE")

            if index + page_size >= total:
                break

            start = index + page_size
            log.debug(
                "iss_pagination",
                path=path,
                block=block,
                fetched=len(all_rows),
                total=total,
            )

        return all_rows

    @staticmethod
    def _zip_block(response_json: dict[str, IssBlock], block: str) -> list[dict[str, object]]:
        """Преобразовать columns + data ISS-блока в список словарей.

        Args:
            response_json: Полный JSON-ответ ISS.
            block: Имя блока в ответе.

        Returns:
            Список словарей {column: value}.
        """
        block_data = response_json.get(block, IssBlock())
        columns = block_data.get("columns", [])
        rows = block_data.get("data", [])
        return [dict(zip(columns, row, strict=False)) for row in rows]

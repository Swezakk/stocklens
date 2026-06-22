"""Асинхронный HTTP-клиент бота к StockLens API (DESIGN.md §7, §11).

Зеркалит auth-agnostic-дизайн дашборда, но асинхронно (aiogram): текущий Bearer-токен
поставляет ``token_provider`` (``TokenManager.get_token``), форс-обновление при 401 —
``on_unauthorized`` (``TokenManager.force_refresh``). Сам клиент только прикрепляет токен,
делает запрос и раскладывает результат по трём веткам (DESIGN §7):

- сетевая ошибка / таймаут        → ApiUnavailableError;
- HTTP 401 (после одного ретрая)  → AuthError;
- прочие HTTP 4xx/5xx             → ApiServerError(status, detail из RFC 9457 body).

Базовый URL — только хост (``http://api:8000``); префикс версии (``/api/v1``) добавляется
внутри ``_request`` (ведущий слэш в path при base_url с путём затёр бы путь базы — RFC 3986).
"""

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from stocklens_core.enums import SentimentLabel

from bot.api_client.dto import (
    DividendPage,
    NewsPage,
    PortfolioSummaryOut,
    SubscriptionIn,
    SubscriptionOut,
)
from bot.api_client.errors import ApiServerError, ApiUnavailableError, AuthError

#: HTTP-методы, по которым ходит бот (без хардкода строк в логике).
_HTTP_GET = "GET"
_HTTP_POST = "POST"
_HTTP_DELETE = "DELETE"

#: Граница успешных статусов и код истёкшей сессии.
_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401

#: Фолбэк-текст, если тело ошибки не RFC 9457 JSON (например, HTML от прокси).
_FALLBACK_DETAIL = "Сервис данных вернул ответ без описания ошибки."

AsyncTokenProvider = Callable[[], Awaitable[str]]
AsyncOnUnauthorized = Callable[[], Awaitable[Any]]


def _parse_problem_detail(response: httpx.Response) -> str:
    """Достать detail из тела RFC 9457; при не-JSON / отсутствии — фолбэк."""
    try:
        body = response.json()
    except ValueError:
        return _FALLBACK_DETAIL
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return _FALLBACK_DETAIL


class ApiClient:
    """Тонкий async-клиент к API: токен, запрос, три ветки результата.

    Параметры конструктора:
    - ``base_url`` — хост API без префикса версии (``http://api:8000``);
    - ``api_prefix`` — префикс версии, добавляется к каждому path (``/api/v1``);
    - ``timeout`` — таймаут запроса в секундах;
    - ``token_provider`` — async-поставщик текущего Bearer-токена;
    - ``on_unauthorized`` — async-хук форс-обновления токена при 401.
    """

    def __init__(
        self,
        base_url: str,
        api_prefix: str,
        timeout: float,
        token_provider: AsyncTokenProvider,
        on_unauthorized: AsyncOnUnauthorized,
    ) -> None:
        self._prefix = api_prefix
        self._token_provider = token_provider
        self._on_unauthorized = on_unauthorized
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def aclose(self) -> None:
        """Закрыть нижележащий httpx.AsyncClient (graceful shutdown бота)."""
        await self._client.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json: Any | None,
        token: str,
    ) -> httpx.Response:
        """Отправить один запрос с Bearer-токеном; сеть → ApiUnavailableError.

        None-параметры отбрасываются: httpx сериализует None в пустую строку, а API
        объявляет такие фильтры как ``X | None`` — пустая строка валится в 422.
        """
        headers = {"Authorization": f"Bearer {token}"}
        clean_params = (
            {key: value for key, value in params.items() if value is not None}
            if params is not None
            else None
        )
        try:
            return await self._client.request(
                method,
                f"{self._prefix}{path}",
                params=clean_params,
                json=json,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise ApiUnavailableError() from exc

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> httpx.Response:
        """Выполнить запрос с обработкой 401 (форс-refresh + один ретрай) и ошибок.

        Порядок веток обязателен: 401 проверяется ДО общей 4xx/5xx, иначе истёкшая
        сессия маскируется под ApiServerError.
        """
        response = await self._send(method, path, params, json, await self._token_provider())

        if response.status_code == _HTTP_UNAUTHORIZED:
            await self._on_unauthorized()
            response = await self._send(method, path, params, json, await self._token_provider())
            if response.status_code == _HTTP_UNAUTHORIZED:
                raise AuthError()

        if response.status_code >= _HTTP_BAD_REQUEST:
            raise ApiServerError(
                status=response.status_code,
                detail=_parse_problem_detail(response),
            )
        return response

    async def list_subscriptions(self, chat_id: int) -> list[SubscriptionOut]:
        """GET /bot/subscriptions?chat_id= — подписки чата (ответ — bare list, не Page)."""
        response = await self._request(_HTTP_GET, "/bot/subscriptions", params={"chat_id": chat_id})
        body: list[Any] = response.json()
        return [SubscriptionOut.model_validate(item) for item in body]

    async def create_subscription(self, subscription: SubscriptionIn) -> SubscriptionOut:
        """POST /bot/subscriptions — создать подписку (201)."""
        response = await self._request(
            _HTTP_POST, "/bot/subscriptions", json=subscription.model_dump(mode="json")
        )
        return SubscriptionOut.model_validate(response.json())

    async def delete_subscription(self, sub_id: int) -> None:
        """DELETE /bot/subscriptions/{sub_id} — удалить подписку (204, тело пустое)."""
        await self._request(_HTTP_DELETE, f"/bot/subscriptions/{sub_id}")

    async def get_portfolio_summary(self, period_days: int = 365) -> PortfolioSummaryOut:
        """GET /portfolio/summary — сводка портфеля с риск-метриками."""
        response = await self._request(
            _HTTP_GET, "/portfolio/summary", params={"period_days": period_days}
        )
        return PortfolioSummaryOut.model_validate(response.json())

    async def get_dividends(self, ticker: str | None = None, limit: int = 100) -> DividendPage:
        """GET /data/dividends — дивидендные выплаты (Page)."""
        response = await self._request(
            _HTTP_GET, "/data/dividends", params={"ticker": ticker, "limit": limit}
        )
        return DividendPage.model_validate(response.json())

    async def get_news(
        self,
        ticker: str | None = None,
        sentiment: SentimentLabel | None = None,
        date_from: str | None = None,
        limit: int = 50,
    ) -> NewsPage:
        """GET /data/news — новости с тональностью (Page)."""
        response = await self._request(
            _HTTP_GET,
            "/data/news",
            params={
                "ticker": ticker,
                "sentiment": sentiment.value if sentiment is not None else None,
                "date_from": date_from,
                "limit": limit,
            },
        )
        return NewsPage.model_validate(response.json())

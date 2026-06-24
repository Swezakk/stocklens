"""Синхронный HTTP-клиент дашборда к StockLens API (DESIGN.md §6, §7).

Клиент auth-agnostic: он не знает, как добывается токен. Текущий Bearer-токен
поставляет ``token_provider`` (его реализует auth.py позже), а форс-обновление при
401 — ``on_unauthorized``. Сам клиент только прикрепляет токен, делает запрос и
раскладывает результат по трём веткам (DESIGN §7):

- сетевая ошибка / таймаут        → ApiUnavailableError;
- HTTP 401 (после одного ретрая)  → AuthError;
- прочие HTTP 4xx/5xx             → ApiServerError(status, detail из RFC 9457 body).

Базовый URL клиента — только хост (``http://api:8000``); префикс версии
(``/api/v1``) добавляется внутри ``_request``. Это обходит ловушку httpx: ведущий
слэш в path при ``base_url`` с путём затирает путь базы (RFC 3986).
"""

from collections.abc import Callable
from typing import Any

import httpx
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel

from dashboard.api_client.dto import (
    BacktestResultOut,
    CandlePage,
    CollectorRunPage,
    CurrencyRatePage,
    DividendPage,
    IndexValuePage,
    KeyRatePage,
    MoversOut,
    NewsPage,
    OptimizeResult,
    PortfolioSummaryOut,
    PositionOut,
    SecurityPage,
    VolatilityForecastHistoryOut,
)
from dashboard.api_client.errors import (
    ApiServerError,
    ApiUnavailableError,
    AuthError,
)

#: HTTP-методы, по которым ходит дашборд (без хардкода строк в логике).
_HTTP_GET = "GET"
_HTTP_POST = "POST"
_HTTP_DELETE = "DELETE"

#: Граница успешных статусов: всё ниже — успех, всё с 400 — ошибка.
_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401

#: Фолбэк-текст, если тело ошибки не RFC 9457 JSON (например, HTML от прокси).
_FALLBACK_DETAIL = "Сервис данных вернул ответ без описания ошибки."

TokenProvider = Callable[[], str]
OnUnauthorized = Callable[[], None]


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
    """Тонкий sync-клиент к API: токен, запрос, три ветки результата.

    Параметры конструктора:
    - ``base_url`` — хост API без префикса версии (``http://api:8000``);
    - ``api_prefix`` — префикс версии, добавляется к каждому path (``/api/v1``);
    - ``timeout`` — таймаут запроса в секундах;
    - ``token_provider`` — поставщик текущего Bearer-токена (auth.py);
    - ``on_unauthorized`` — хук форс-обновления токена при 401 (auth.py).
    """

    def __init__(
        self,
        base_url: str,
        api_prefix: str,
        timeout: float,
        token_provider: TokenProvider,
        on_unauthorized: OnUnauthorized,
    ) -> None:
        self._prefix = api_prefix
        self._token_provider = token_provider
        self._on_unauthorized = on_unauthorized
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        """Закрыть нижележащий httpx.Client (вызывается при пересоздании ресурса)."""
        self._client.close()

    def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json: Any | None,
        token: str,
    ) -> httpx.Response:
        """Отправить один HTTP-запрос с Bearer-токеном; сеть → ApiUnavailableError.

        None-параметры отбрасываются: httpx сериализует None в пустую строку
        (``date_from=``), а API объявляет такие фильтры как ``date | None`` —
        пустая строка не absent и валится в 422. Отсутствующий ключ = отсутствующий фильтр.
        """
        headers = {"Authorization": f"Bearer {token}"}
        clean_params = (
            {key: value for key, value in params.items() if value is not None}
            if params is not None
            else None
        )
        try:
            return self._client.request(
                method,
                f"{self._prefix}{path}",
                params=clean_params,
                json=json,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise ApiUnavailableError() from exc

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> httpx.Response:
        """Выполнить запрос с обработкой 401 (форс-refresh + один ретрай) и ошибок.

        Порядок веток обязателен: 401 проверяется ДО общей 4xx/5xx, иначе истёкшая
        сессия маскируется под ApiServerError. После ретрая финальный ответ
        оценивается один раз: 401 → AuthError, ≥400 → ApiServerError, иначе успех.
        """
        response = self._send(method, path, params, json, self._token_provider())

        if response.status_code == _HTTP_UNAUTHORIZED:
            self._on_unauthorized()
            response = self._send(method, path, params, json, self._token_provider())
            if response.status_code == _HTTP_UNAUTHORIZED:
                raise AuthError()

        if response.status_code >= _HTTP_BAD_REQUEST:
            raise ApiServerError(
                status=response.status_code,
                detail=_parse_problem_detail(response),
            )
        return response

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET, возвращающий сырой JSON-объект (для эндпоинтов вне типизированных хелперов)."""
        body: dict[str, Any] = self._request(_HTTP_GET, path, params=params).json()
        return body

    def get_index(
        self,
        index_code: str = "IMOEX",
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> IndexValuePage:
        """GET /data/index — значения биржевого индекса (Page)."""
        params = {
            "index_code": index_code,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "offset": offset,
        }
        return IndexValuePage.model_validate(self.get("/data/index", params))

    def get_currency_rates(
        self,
        currency: Currency | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> CurrencyRatePage:
        """GET /data/currency-rates — курсы валют ЦБ РФ (Page)."""
        params = {
            "currency": currency.value if currency is not None else None,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "offset": offset,
        }
        return CurrencyRatePage.model_validate(self.get("/data/currency-rates", params))

    def get_key_rate(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> KeyRatePage:
        """GET /data/key-rate — история ключевой ставки ЦБ РФ (Page)."""
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "offset": offset,
        }
        return KeyRatePage.model_validate(self.get("/data/key-rate", params))

    def get_movers(self, limit: int = 5) -> MoversOut:
        """GET /data/movers — лидеры роста и падения (не пагинированный ответ)."""
        return MoversOut.model_validate(self.get("/data/movers", {"limit": limit}))

    def get_volatility_forecast_history(
        self, ticker: str, lookback: int = 90
    ) -> VolatilityForecastHistoryOut:
        """GET /predict/volatility/history — прогноз vs реализованная за окно (не пагинирован)."""
        params = {"ticker": ticker, "lookback": lookback}
        return VolatilityForecastHistoryOut.model_validate(
            self.get("/predict/volatility/history", params)
        )

    def get_securities(
        self,
        is_active: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SecurityPage:
        """GET /data/securities — список бумаг (Page)."""
        params = {"is_active": is_active, "limit": limit, "offset": offset}
        return SecurityPage.model_validate(self.get("/data/securities", params))

    def get_candles(
        self,
        ticker: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> CandlePage:
        """GET /data/candles — свечи OHLCV по тикеру (Page). Тикер обязателен."""
        params = {
            "ticker": ticker,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "offset": offset,
        }
        return CandlePage.model_validate(self.get("/data/candles", params))

    def get_dividends(
        self,
        ticker: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> DividendPage:
        """GET /data/dividends — дивидендные выплаты (Page)."""
        params = {"ticker": ticker, "limit": limit, "offset": offset}
        return DividendPage.model_validate(self.get("/data/dividends", params))

    def get_news(
        self,
        ticker: str | None = None,
        sentiment: SentimentLabel | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> NewsPage:
        """GET /data/news — новости с тональностью (Page)."""
        params = {
            "ticker": ticker,
            "sentiment": sentiment.value if sentiment is not None else None,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "offset": offset,
        }
        return NewsPage.model_validate(self.get("/data/news", params))

    def get_collector_runs(
        self,
        source: str | None = None,
        status: CollectorRunStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> CollectorRunPage:
        """GET /monitoring/runs — журнал запусков сборщиков (Page)."""
        params = {
            "source": source,
            "status": status.value if status is not None else None,
            "limit": limit,
            "offset": offset,
        }
        return CollectorRunPage.model_validate(self.get("/monitoring/runs", params))

    def get_portfolio_summary(self, period_days: int = 365) -> PortfolioSummaryOut:
        """GET /portfolio/summary — сводка с риск-метриками."""
        params = {"period_days": period_days}
        return PortfolioSummaryOut.model_validate(self.get("/portfolio/summary", params))

    def list_positions(self) -> list[PositionOut]:
        """GET /portfolio/positions — позиции (бара list, НЕ Page-конверт)."""
        body: list[Any] = self._request(_HTTP_GET, "/portfolio/positions").json()
        return [PositionOut.model_validate(item) for item in body]

    def create_position(self, position: dict[str, Any]) -> PositionOut:
        """POST /portfolio/positions — upsert позиции (единственный write-путь дашборда)."""
        response = self._request(_HTTP_POST, "/portfolio/positions", json=position)
        return PositionOut.model_validate(response.json())

    def delete_position(self, ticker: str) -> None:
        """DELETE /portfolio/positions/{ticker} — удалить позицию (204, тело пустое)."""
        self._request(_HTTP_DELETE, f"/portfolio/positions/{ticker}")

    def optimize(self, request: dict[str, Any]) -> OptimizeResult:
        """POST /portfolio/optimize — оптимизация Марковица."""
        response = self._request(_HTTP_POST, "/portfolio/optimize", json=request)
        return OptimizeResult.model_validate(response.json())

    def get_backtest(self, months_back: int = 12) -> BacktestResultOut:
        """GET /portfolio/backtest — бэктест равновзвешенного портфеля vs IMOEX."""
        params = {"months_back": months_back}
        return BacktestResultOut.model_validate(self.get("/portfolio/backtest", params))

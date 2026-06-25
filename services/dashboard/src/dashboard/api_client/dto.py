"""Pydantic-DTO, зеркалящие JSON-ответы StockLens API (DESIGN.md §6, §9).

Mirror-DTO на HTTP-границе — слой клиента, а не второй источник истины: инвариант №4
запрещает дублировать schema-of-record (ORM-модели, доменные enum из stocklens-core),
но не клиентское зеркало JSON ответа (DESIGN §6.1).

Доменные enum импортируются из stocklens_core.enums (Currency, SentimentLabel,
CollectorRunStatus). OptimizationStrategy в core отсутствует — он определён локально
в схемах API, поэтому зеркалится здесь так же локально (тот же mirror-DTO-карв-аут).

DTO парсят JSON (не ORM), поэтому from_attributes не задаётся. Денежные значения —
Decimal без понижения точности; даты — date, метки времени — datetime.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator
from stocklens_core.enums import CollectorRunStatus, Currency, SentimentLabel


class Page[T](BaseModel):
    """Конверт постраничного ответа списковых эндпоинтов (items/total/limit/offset)."""

    items: list[T]
    total: int
    limit: int
    offset: int


class OptimizationStrategy(StrEnum):
    """Стратегия оптимизации портфеля по Марковицу (зеркало локального enum API).

    В stocklens-core отсутствует: это контракт схем API, не доменный enum уровня БД.
    """

    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"
    MAX_UTILITY = "max_utility"


class SecurityOut(BaseModel):
    """Ценная бумага."""

    id: int
    ticker: str
    name: str
    board: str
    aliases: list[str]
    is_active: bool


class CandleOut(BaseModel):
    """Дневная свеча OHLCV."""

    id: int
    security_id: int
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    value: Decimal
    is_weekend_session: bool


class DividendOut(BaseModel):
    """Дивидендная выплата."""

    id: int
    security_id: int
    ex_date: date
    value: Decimal
    currency: Currency


class IndexValueOut(BaseModel):
    """Значение биржевого индекса за торговый день."""

    trade_date: date
    close: Decimal


class CurrencyRateOut(BaseModel):
    """Курс валюты к рублю."""

    currency: Currency
    rate_date: date
    rate: Decimal


class KeyRateOut(BaseModel):
    """Ключевая ставка ЦБ РФ."""

    rate_date: date
    rate: Decimal


class MoverOut(BaseModel):
    """Бумага-лидер роста или падения дня."""

    ticker: str
    name: str
    close: Decimal
    prev_close: Decimal
    change_pct: float


class MoversOut(BaseModel):
    """Лидеры роста и падения (не пагинированный ответ)."""

    gainers: list[MoverOut]
    losers: list[MoverOut]


class SentimentOut(BaseModel):
    """Тональность новостной статьи."""

    label: SentimentLabel
    score: float
    model_version: str


class NewsOut(BaseModel):
    """Новостная статья с тональностью и связанными тикерами."""

    id: int
    source: str
    url: str
    title: str
    summary: str | None
    published_at: datetime
    sentiment: SentimentOut | None
    tickers: list[str]


class CollectorRunOut(BaseModel):
    """Запуск сборщика данных."""

    id: int
    source: str
    started_at: datetime
    finished_at: datetime | None
    status: CollectorRunStatus
    records_added: int
    error_message: str | None


class PositionIn(BaseModel):
    """Входные данные создания/обновления позиции (зеркало api PositionIn).

    Зеркалит имена/типы schemas.portfolio.PositionIn: `quantity > 0`, `avg_price > 0`,
    `opened_at` обязан быть timezone-aware. Клиент-side валидация ловит очевидные ошибки
    формы до сетевого вызова; источник истины правил остаётся на API.
    """

    ticker: str
    quantity: int = Field(gt=0, description="Количество лотов (> 0)")
    avg_price: Decimal = Field(gt=0, description="Средняя цена покупки (> 0)")
    opened_at: datetime = Field(description="Дата открытия позиции (timezone-aware)")

    @field_validator("opened_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Позиция обязана иметь временную зону (зеркало серверного правила)."""
        if value.tzinfo is None:
            raise ValueError("opened_at должен содержать временную зону (timezone-aware)")
        return value


class PositionOut(BaseModel):
    """Позиция портфеля с текущей рыночной оценкой."""

    ticker: str
    quantity: int
    avg_price: Decimal
    opened_at: datetime
    current_price: Decimal | None
    current_value: Decimal | None
    unrealized_pnl: Decimal | None


class PortfolioSummaryOut(BaseModel):
    """Сводка по портфелю с риск-метриками и сравнением с IMOEX."""

    positions: list[PositionOut]
    total_value: Decimal
    total_cost: Decimal
    total_unrealized_pnl: Decimal
    portfolio_return_pct: float
    imoex_return_pct: float
    sharpe: float
    max_drawdown: float
    imoex_sharpe: float
    imoex_max_drawdown: float
    period_from: date
    period_to: date


class OptimizeRequest(BaseModel):
    """Запрос оптимизации портфеля по Марковицу (зеркало api OptimizeRequest).

    Зеркалит имена/типы schemas.portfolio.OptimizeRequest. `tickers=None` — оптимизировать
    текущие позиции; `period_days ≥ 30`; параметры стратегии (`target_return` /
    `target_volatility` / `risk_aversion`) опциональны и задаются под выбранную стратегию.
    """

    tickers: list[str] | None = Field(
        default=None,
        description="Список тикеров для оптимизации. None — использовать текущие позиции.",
    )
    period_days: int = Field(
        default=365,
        ge=30,
        description="Глубина истории котировок в днях (не менее 30).",
    )
    strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.MAX_SHARPE,
        description="Стратегия оптимизации.",
    )
    target_return: float | None = Field(
        default=None,
        description="Целевая годовая доходность (для TARGET_RETURN).",
    )
    target_volatility: float | None = Field(
        default=None,
        description="Целевой уровень риска (для TARGET_RISK).",
    )
    risk_aversion: float | None = Field(
        default=None,
        description="Коэффициент неприятия риска λ (для MAX_UTILITY).",
    )


class FrontierPoint(BaseModel):
    """Точка на эффективной границе Марковица."""

    volatility: float
    expected_return: float


class OptimizeResult(BaseModel):
    """Результат оптимизации: веса стратегии, эффективная граница и бенчмарки."""

    strategy: OptimizationStrategy
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    frontier: list[FrontierPoint]
    equal_weight_sharpe: float
    imoex_sharpe: float


class EquityPointOut(BaseModel):
    """Точка кривой капитала бэктеста."""

    date: date
    portfolio: float
    imoex: float


class BacktestResultOut(BaseModel):
    """Результат бэктеста равновзвешенного портфеля vs IMOEX."""

    months_back: int
    period_from: date
    period_to: date
    portfolio_return_pct: float
    imoex_return_pct: float
    portfolio_sharpe: float
    imoex_sharpe: float
    portfolio_max_drawdown: float
    imoex_max_drawdown: float
    equity_curve: list[EquityPointOut]


class VolatilityMetricsOut(BaseModel):
    """Метрики модели волатильности vs naive baseline (walk-forward QLIKE/RMSE)."""

    qlike: float
    qlike_baseline: float
    rmse: float


class VolatilityForecastPointOut(BaseModel):
    """Точка ряда «прогноз vs реализованная волатильность» (ml-spec §10)."""

    date: date
    forecast: float | None = None
    realized: float | None = None


class VolatilityForecastHistoryOut(BaseModel):
    """История прогнозов волатильности с реализованными значениями (ml-spec §10).

    ``protected_namespaces=()`` — поле ``model`` иначе конфликтует с защищённым
    неймспейсом Pydantic v2 (``model_*``). ``model``/``metrics_vs_baseline`` — ``None``,
    если модель в API не загружена (degraded readiness): реализованная серия всё равно есть.

    ``live_metrics`` — live QLIKE модели и baseline по реальным созревшим прогнозам;
    ``None`` когда недостаточно созревших пар (порог определяет API). ``live_sample_size`` —
    число созревших пар (0 при отсутствии). Дефолты обеспечивают совместимость со старыми
    ответами API, не содержащими этих полей.
    """

    model_config = {"protected_namespaces": ()}

    ticker: str
    model: str | None
    model_version: str | None
    metrics_vs_baseline: VolatilityMetricsOut | None
    points: list[VolatilityForecastPointOut]
    live_metrics: VolatilityMetricsOut | None = None
    live_sample_size: int = 0


# Конкретные подклассы Page[T] для кэширования: параметризованный дженерик Page[NewsOut]
# имеет qualname «Page[NewsOut]» (со скобками), который не находится как атрибут модуля,
# поэтому pickle Streamlit (st.cache_data) падает на нём. Именованные подклассы получают
# обычный qualname и пиклятся по ссылке — кэш хранит DTO как требует DESIGN §8.
class SecurityPage(Page[SecurityOut]):
    """Страница ценных бумаг."""


class CandlePage(Page[CandleOut]):
    """Страница свечей OHLCV."""


class DividendPage(Page[DividendOut]):
    """Страница дивидендных выплат."""


class IndexValuePage(Page[IndexValueOut]):
    """Страница значений индекса."""


class CurrencyRatePage(Page[CurrencyRateOut]):
    """Страница курсов валют."""


class KeyRatePage(Page[KeyRateOut]):
    """Страница истории ключевой ставки."""


class NewsPage(Page[NewsOut]):
    """Страница новостей."""


class CollectorRunPage(Page[CollectorRunOut]):
    """Страница запусков сборщиков."""

"""DTO для операций с портфелем: позиции, сводка и оптимизация."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator
from stocklens_core.enums import AlertKind


class OptimizationStrategy(StrEnum):
    """Стратегия оптимизации портфеля по методу Марковица."""

    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"
    MAX_UTILITY = "max_utility"


class PositionIn(BaseModel):
    """Входные данные для создания или обновления позиции."""

    ticker: str
    quantity: int = Field(gt=0, description="Количество лотов (> 0)")
    avg_price: Decimal = Field(gt=0, description="Средняя цена покупки (> 0)")
    opened_at: datetime = Field(description="Дата открытия позиции (timezone-aware)")

    @field_validator("opened_at")
    @classmethod
    def require_timezone(cls, v: datetime) -> datetime:
        """Позиция обязана иметь временную зону."""
        if v.tzinfo is None:
            raise ValueError("opened_at должен содержать временную зону (timezone-aware)")
        return v


class PositionOut(BaseModel):
    """Выходные данные позиции с текущей рыночной оценкой."""

    model_config = {"from_attributes": True}

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
    """Запрос на оптимизацию портфеля по методу Марковица."""

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
    """Результат оптимизации: веса выбранной стратегии + эффективная граница и бенчмарки.

    Поля strategy/requested_strategy разделены для поддержки auto-fallback:
    - requested_strategy: стратегия, запрошенная клиентом.
    - strategy: фактически применённая стратегия (может отличаться при fallback).
    - fallback_reason: причина переключения стратегии на русском; None при отсутствии fallback.
    """

    strategy: OptimizationStrategy
    requested_strategy: OptimizationStrategy
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    frontier: list[FrontierPoint]
    equal_weight_sharpe: float
    imoex_sharpe: float
    fallback_reason: str | None = None


_PRICE_LEVEL_ALERT_KINDS = {AlertKind.PRICE_LEVEL}


class EquityPointOut(BaseModel):
    """DTO точки кривой капитала бэктеста."""

    date: date
    portfolio: float
    imoex: float


class BacktestResultOut(BaseModel):
    """DTO результата бэктеста равновзвешенного портфеля vs IMOEX."""

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

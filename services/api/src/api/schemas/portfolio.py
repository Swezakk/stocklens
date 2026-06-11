"""DTO для операций с портфелем: позиции, сводка и оптимизация."""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator
from stocklens_core.enums import AlertKind


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


class FrontierPoint(BaseModel):
    """Точка на эффективной границе Марковица."""

    volatility: float
    expected_return: float


class OptimizeResult(BaseModel):
    """Результат оптимизации: веса для max Sharpe и min vol + эффективная граница."""

    max_sharpe_weights: dict[str, float]
    min_volatility_weights: dict[str, float]
    frontier: list[FrontierPoint]
    equal_weight_sharpe: float
    imoex_sharpe: float


_PRICE_LEVEL_ALERT_KINDS = {AlertKind.PRICE_LEVEL}

"""Общие фикстуры тестов дашборда: образцы JSON-ответов API.

Полезные нагрузки повторяют форму ответов StockLens API (services/api/.../schemas),
чтобы тесты DTO и api_client валидировали зеркало контракта без поднятия сервера.
"""

from pathlib import Path
from typing import Any

import pytest

#: Корень проекта дашборда (для доступа к .streamlit/config.toml и src/ в тестах).
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dashboard_root() -> Path:
    """Абсолютный путь к корню сервиса dashboard."""
    return DASHBOARD_ROOT


@pytest.fixture
def security_payload() -> dict[str, Any]:
    """Образец JSON ценной бумаги (SecurityOut)."""
    return {
        "id": 1,
        "ticker": "SBER",
        "name": "Сбербанк",
        "board": "TQBR",
        "aliases": ["SBER", "Сбер"],
        "is_active": True,
    }


@pytest.fixture
def candle_payload() -> dict[str, Any]:
    """Образец JSON дневной свечи (CandleOut)."""
    return {
        "id": 10,
        "security_id": 1,
        "trade_date": "2026-06-19",
        "open": "305.50",
        "high": "312.00",
        "low": "304.10",
        "close": "310.75",
        "volume": 1542000,
        "value": "478123456.50",
        "is_weekend_session": False,
    }


@pytest.fixture
def page_payload(security_payload: dict[str, Any]) -> dict[str, Any]:
    """Образец конверта Page[T] (на примере списка бумаг)."""
    return {
        "items": [security_payload],
        "total": 1,
        "limit": 50,
        "offset": 0,
    }


@pytest.fixture
def movers_payload() -> dict[str, Any]:
    """Образец JSON лидеров роста/падения (MoversOut)."""
    return {
        "gainers": [
            {
                "ticker": "GAZP",
                "name": "Газпром",
                "close": "128.40",
                "prev_close": "120.10",
                "change_pct": 6.91,
            }
        ],
        "losers": [
            {
                "ticker": "LKOH",
                "name": "Лукойл",
                "close": "6500.00",
                "prev_close": "6800.00",
                "change_pct": -4.41,
            }
        ],
    }


@pytest.fixture
def news_payload() -> dict[str, Any]:
    """Образец JSON новостной статьи с тональностью (NewsOut)."""
    return {
        "id": 42,
        "source": "Интерфакс",
        "url": "https://example.com/news/42",
        "title": "Сбербанк отчитался о прибыли",
        "summary": "Чистая прибыль выросла за квартал.",
        "published_at": "2026-06-20T09:30:00+00:00",
        "sentiment": {
            "label": "positive",
            "score": 0.87,
            "model_version": "rubert-tiny2-v1",
        },
        "tickers": ["SBER"],
    }


@pytest.fixture
def collector_run_payload() -> dict[str, Any]:
    """Образец JSON запуска сборщика (CollectorRunOut)."""
    return {
        "id": 7,
        "source": "moex_candles",
        "started_at": "2026-06-21T03:00:00+00:00",
        "finished_at": "2026-06-21T03:02:15+00:00",
        "status": "success",
        "records_added": 312,
        "error_message": None,
    }


@pytest.fixture
def portfolio_summary_payload() -> dict[str, Any]:
    """Образец JSON сводки портфеля (PortfolioSummaryOut)."""
    return {
        "positions": [
            {
                "ticker": "SBER",
                "quantity": 10,
                "avg_price": "280.00",
                "opened_at": "2026-01-15T10:00:00+00:00",
                "current_price": "310.75",
                "current_value": "3107.50",
                "unrealized_pnl": "307.50",
            }
        ],
        "total_value": "3107.50",
        "total_cost": "2800.00",
        "total_unrealized_pnl": "307.50",
        "portfolio_return_pct": 10.98,
        "imoex_return_pct": 4.20,
        "sharpe": 1.35,
        "max_drawdown": -0.12,
        "imoex_sharpe": 0.80,
        "imoex_max_drawdown": -0.18,
        "period_from": "2026-01-15",
        "period_to": "2026-06-21",
    }

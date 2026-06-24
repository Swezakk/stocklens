"""Unit-тесты PredictionService.refresh_active_forecasts (без БД, без MLflow).

TDD: тесты написаны до реализации. Все четыре сценария покрывают:
- ошибка одного тикера не прерывает цикл (остальные обрабатываются);
- InsufficientHistoryError / SecurityNotFoundError → skipped, не failed;
- bundle.volatility is None → немедленный возврат, predict не вызывается;
- пустой список активных бумаг → нулевой итог.
"""

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from api.core.exceptions import InsufficientHistoryError, SecurityNotFoundError
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
from api.schemas.predict import ForecastRefreshSummary, VolatilityMetrics, VolatilityPredictionOut
from api.services.prediction import PredictionService
from stocklens_core.enums import PredictionKind
from stocklens_core.models.market import Security

from tests.unit.fakes import FakeSecurity

_METRICS = {"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025}
_STUB_METRICS = VolatilityMetrics(qlike=0.844, qlike_baseline=2.203, rmse=0.0025)


def _settings() -> ApiSettings:
    return ApiSettings.model_validate(
        {
            "database_url_async": "postgresql+asyncpg://u:p@h:5432/d",
            "redis_url": "redis://h:6379/0",
        }
    )


def _bundle() -> ModelBundle:
    class _StubPredictor:
        def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
            return np.array([0.0009], dtype=np.float64)

    return ModelBundle(
        volatility=LoadedVolatilityModel(
            predictor=_StubPredictor(),
            model_version="1",
            method="garch",
            metrics=_METRICS,
            horizon_days=5,
        )
    )


def _make_security(ticker: str) -> Security:
    return cast(
        Security,
        FakeSecurity(id=hash(ticker) & 0xFFFF, ticker=ticker, name=ticker, board="TQBR"),
    )


def _ok_result(ticker: str) -> VolatilityPredictionOut:
    """Синтетический успешный результат predict_volatility для заданного тикера."""
    return VolatilityPredictionOut(
        ticker=ticker,
        predicted_for=date(2024, 1, 2),
        horizon_days=5,
        volatility=0.03,
        model="garch",
        model_version="1",
        metrics_vs_baseline=_STUB_METRICS,
    )


@dataclass
class _FakeSecurityRepo:
    """Репозиторий, возвращающий заданный список активных бумаг."""

    actives: list[Security]

    async def get_by_ticker(self, ticker: str) -> Security | None:
        for s in self.actives:
            if s.ticker == ticker:
                return s
        return None

    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        page = self.actives[offset : offset + limit]
        return page, len(self.actives)


@dataclass
class _FakeFeatureRepo:
    async def load_candles(self, security_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["trade_date", "open", "high", "low", "close", "volume", "is_weekend_session"]
        )

    async def load_dividends(self, security_id: int) -> pd.DataFrame:
        return pd.DataFrame({"ex_date": [], "value": [], "currency": []})

    async def load_splits(self, security_id: int) -> pd.DataFrame:
        return pd.DataFrame({"split_date": [], "before": [], "after": []})


@dataclass
class _FakePredictionRepo:
    async def get_value(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        model_version: str,
    ) -> float | None:
        return None

    async def upsert(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        value: float,
        model_version: str,
    ) -> None:
        pass

    async def list_values(
        self,
        security_id: int,
        kind: PredictionKind,
        model_version: str,
        date_from: date,
        date_to: date,
    ) -> dict[date, float]:
        return {}


def _make_service_with_controlled_predict(
    actives: list[Security],
    side_effects: dict[str, Exception | VolatilityPredictionOut],
    bundle: ModelBundle,
) -> tuple[PredictionService, list[str]]:
    """Построить PredictionService, у которого predict_volatility управляется через side_effects.

    Возвращает сервис и разделяемый список вызванных тикеров (порядок сохранён).
    При отсутствии тикера в side_effects возвращает синтетический успешный результат.
    """
    called: list[str] = []

    class _ControllableService(PredictionService):
        async def predict_volatility(self, ticker: str) -> VolatilityPredictionOut:
            called.append(ticker)
            effect = side_effects.get(ticker)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, VolatilityPredictionOut):
                return effect
            return _ok_result(ticker)

    svc = _ControllableService(
        security_repo=_FakeSecurityRepo(actives),
        feature_repo=_FakeFeatureRepo(),
        prediction_repo=_FakePredictionRepo(),
        bundle=bundle,
        settings=_settings(),
    )
    return svc, called


async def test_refresh_isolates_failing_ticker() -> None:
    """Сбой 2-го тикера не прерывает цикл: 3-й обрабатывается, итог generated=2, failed=1."""
    tickers = ["SBER", "GAZP", "LKOH"]
    actives = [_make_security(t) for t in tickers]
    side_effects: dict[str, Exception | VolatilityPredictionOut] = {
        "GAZP": RuntimeError("simulated predict failure"),
    }

    svc, called = _make_service_with_controlled_predict(actives, side_effects, _bundle())
    summary: ForecastRefreshSummary = await svc.refresh_active_forecasts()

    assert summary.generated == 2
    assert summary.failed == 1
    assert summary.skipped == 0
    assert summary.total == 3
    assert "LKOH" in called, "Цикл должен продолжиться после сбоя GAZP"


async def test_refresh_counts_insufficient_history_as_skipped() -> None:
    """InsufficientHistoryError → skipped (ожидаемо для новых/тонких бумаг)."""
    actives = [_make_security("MOEX"), _make_security("VKCO")]
    side_effects: dict[str, Exception | VolatilityPredictionOut] = {
        "VKCO": InsufficientHistoryError("VKCO", 30, 100),
    }

    svc, _ = _make_service_with_controlled_predict(actives, side_effects, _bundle())
    summary = await svc.refresh_active_forecasts()

    assert summary.generated == 1
    assert summary.skipped == 1
    assert summary.failed == 0
    assert summary.total == 2


async def test_refresh_counts_security_not_found_as_skipped() -> None:
    """SecurityNotFoundError → skipped (бумага пропала из БД между запросами)."""
    actives = [_make_security("ALRS")]
    side_effects: dict[str, Exception | VolatilityPredictionOut] = {
        "ALRS": SecurityNotFoundError("ALRS"),
    }

    svc, _ = _make_service_with_controlled_predict(actives, side_effects, _bundle())
    summary = await svc.refresh_active_forecasts()

    assert summary.skipped == 1
    assert summary.failed == 0
    assert summary.generated == 0
    assert summary.total == 1


async def test_refresh_short_circuits_when_model_unavailable() -> None:
    """bundle.volatility is None → немедленный возврат, predict_volatility не вызывается."""
    actives = [_make_security("SBER"), _make_security("GAZP"), _make_security("LKOH")]

    svc, called = _make_service_with_controlled_predict(actives, {}, ModelBundle(volatility=None))

    summary = await svc.refresh_active_forecasts()

    assert summary.generated == 0
    assert summary.skipped == 0
    assert summary.failed == 0
    assert summary.total == 0
    assert called == [], "predict_volatility не должен вызываться при отсутствии модели"


async def test_refresh_empty_when_no_active_securities() -> None:
    """Нет активных бумаг → все счётчики нулевые, без ошибок."""
    svc, _ = _make_service_with_controlled_predict([], {}, _bundle())
    summary = await svc.refresh_active_forecasts()

    assert summary.generated == 0
    assert summary.skipped == 0
    assert summary.failed == 0
    assert summary.total == 0

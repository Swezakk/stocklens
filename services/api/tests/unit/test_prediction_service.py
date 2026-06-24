"""Unit-тесты PredictionService с фиктивными репозиториями и стаб-моделью (без БД/MLflow).

Покрывает контракт ml-spec §11.2: 404 для неизвестного тикера, 422 при нехватке истории,
503 при незагруженной модели, идемпотентность (read-through кэш), форма DTO.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from api.core.exceptions import (
    InsufficientHistoryError,
    ModelNotLoadedError,
    SecurityNotFoundError,
)
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
from api.services.prediction import MIN_REGIME_OBSERVATIONS, PredictionService, _Forecast
from stocklens_core.enums import PredictionKind
from stocklens_core.models.market import Security

from tests.unit.fakes import FakeSecurity

_METRICS = {"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025}
_VARIANCE = 0.0009  # доли² → волатильность sqrt = 0.03


def _settings() -> ApiSettings:
    return ApiSettings.model_validate(
        {
            "database_url_async": "postgresql+asyncpg://u:p@h:5432/d",
            "redis_url": "redis://h:6379/0",
        }
    )


def _candles(n: int, start: date = date(2022, 6, 1)) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = pd.bdate_range(start=start, periods=n).date
    close = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n),
            "is_weekend_session": [False] * n,
        }
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame({"ex_date": [], "value": [], "currency": []})


def _empty_splits() -> pd.DataFrame:
    return pd.DataFrame({"split_date": [], "before": [], "after": []})


class _StubPredictor:
    def __init__(self, variance: float = _VARIANCE) -> None:
        self.variance = variance
        self.calls = 0

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        self.calls += 1
        return np.array([self.variance], dtype=np.float64)


@dataclass
class _FakeSecurityRepo:
    security: Security | None

    async def get_by_ticker(self, ticker: str) -> Security | None:
        return self.security

    async def list_securities(
        self, is_active: bool | None, limit: int, offset: int
    ) -> tuple[list[Security], int]:
        return ([], 0)


@dataclass
class _FakeFeatureRepo:
    candles: pd.DataFrame

    async def load_candles(self, security_id: int) -> pd.DataFrame:
        return self.candles

    async def load_dividends(self, security_id: int) -> pd.DataFrame:
        return _empty_dividends()

    async def load_splits(self, security_id: int) -> pd.DataFrame:
        return _empty_splits()


@dataclass
class _FakePredictionRepo:
    cached: float | None = None
    upserts: list[tuple[int, date, int, PredictionKind, float, str]] = field(default_factory=list)

    async def get_value(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        model_version: str,
    ) -> float | None:
        return self.cached

    async def upsert(
        self,
        security_id: int,
        predicted_for: date,
        horizon_days: int,
        kind: PredictionKind,
        value: float,
        model_version: str,
    ) -> None:
        self.upserts.append((security_id, predicted_for, horizon_days, kind, value, model_version))


def _bundle(predictor: _StubPredictor) -> ModelBundle:
    return ModelBundle(
        volatility=LoadedVolatilityModel(
            predictor=predictor,
            model_version="3",
            method="garch",
            metrics=_METRICS,
            horizon_days=5,
        )
    )


def _security() -> Security:
    return cast(Security, FakeSecurity(id=1, ticker="SBER", name="Сбербанк", board="TQBR"))


def _service(
    *,
    security: Security | None = None,
    candles: pd.DataFrame | None = None,
    prediction_repo: _FakePredictionRepo | None = None,
    predictor: _StubPredictor | None = None,
    bundle: ModelBundle | None = None,
) -> tuple[PredictionService, _FakePredictionRepo, _StubPredictor]:
    predictor = predictor or _StubPredictor()
    repo = prediction_repo or _FakePredictionRepo()
    feature_candles = candles if candles is not None else _candles(160)
    service = PredictionService(
        security_repo=_FakeSecurityRepo(security),
        feature_repo=_FakeFeatureRepo(feature_candles),
        prediction_repo=repo,
        bundle=bundle if bundle is not None else _bundle(predictor),
        settings=_settings(),
    )
    return service, repo, predictor


async def test_predict_volatility_returns_404_for_unknown_ticker() -> None:
    service, _, _ = _service(security=None)

    with pytest.raises(SecurityNotFoundError):
        await service.predict_volatility("UNKNOWN")


async def test_predict_volatility_returns_422_for_insufficient_history() -> None:
    service, _, _ = _service(security=_security(), candles=_candles(40))

    with pytest.raises(InsufficientHistoryError):
        await service.predict_volatility("SBER")


async def test_predict_volatility_returns_503_when_model_not_loaded() -> None:
    service, _, _ = _service(security=_security(), bundle=ModelBundle(volatility=None))

    with pytest.raises(ModelNotLoadedError):
        await service.predict_volatility("SBER")


async def test_predict_volatility_returns_dto_with_sqrt_volatility() -> None:
    service, repo, predictor = _service(security=_security())

    result = await service.predict_volatility("sber")

    assert result.ticker == "SBER"
    assert result.volatility == pytest.approx(np.sqrt(_VARIANCE))
    assert result.model == "garch"
    assert result.model_version == "3"
    assert result.horizon_days == 5
    assert result.metrics_vs_baseline.qlike == pytest.approx(0.844)
    assert predictor.calls == 1
    assert len(repo.upserts) == 1


async def test_predict_volatility_uses_cache_and_skips_refit_on_repeat() -> None:
    cached = float(np.sqrt(_VARIANCE))
    repo = _FakePredictionRepo(cached=cached)
    service, repo_out, predictor = _service(security=_security(), prediction_repo=repo)

    result = await service.predict_volatility("SBER")

    assert result.volatility == pytest.approx(cached)
    assert predictor.calls == 0  # read-through кэш: модель не вызывается
    assert repo_out.upserts == []  # повторная запись не создаётся


async def test_assess_volatility_regime_returns_503_when_model_not_loaded() -> None:
    """assess_volatility_regime: bundle.volatility=None → ModelNotLoadedError."""
    service, _, _ = _service(security=_security(), bundle=ModelBundle(volatility=None))

    with pytest.raises(ModelNotLoadedError):
        await service.assess_volatility_regime("SBER", quantile=0.80, lookback=252)


async def test_assess_volatility_regime_is_elevated_true_when_forecast_exceeds_quantile() -> None:
    """assess_volatility_regime: прогноз выше 0.80-квантиля реализованных → is_elevated=True."""
    # sqrt(0.09) = 0.3 — высокая волатильность; исторические rv_target будут ≈0.01 (small),
    # поэтому 0.80-квантиль << 0.3.
    high_variance = 0.09
    predictor = _StubPredictor(variance=high_variance)
    service, _, _ = _service(security=_security(), candles=_candles(160), predictor=predictor)

    result = await service.assess_volatility_regime("SBER", quantile=0.80, lookback=252)

    assert result.is_elevated is True
    assert result.volatility == pytest.approx(np.sqrt(high_variance))
    assert result.quantile == pytest.approx(0.80)
    assert result.ticker == "SBER"


async def test_assess_volatility_regime_is_elevated_false_when_forecast_below_quantile() -> None:
    """assess_volatility_regime: прогноз ниже 0.01-квантиля реализованных → is_elevated=False."""
    # При quantile=0.01 порог будет очень маленьким, но мы используем очень маленькую дисперсию.
    # Безопаснее: quantile=0.99 — порог = 99%-квантиль реализованных, прогноз ниже.
    tiny_variance = 1e-12
    predictor = _StubPredictor(variance=tiny_variance)
    service, _, _ = _service(security=_security(), candles=_candles(160), predictor=predictor)

    result = await service.assess_volatility_regime("SBER", quantile=0.99, lookback=252)

    assert result.is_elevated is False
    assert result.volatility == pytest.approx(np.sqrt(tiny_variance))


async def test_assess_volatility_regime_raises_insufficient_history_when_few_rv_target() -> None:
    """assess_volatility_regime: менее MIN_REGIME_OBSERVATIONS rv_target → InsufficientHistoryError.

    rv_target forward-looking — при 160 свечах ≈156 ненулевых rv_target, хорошо выше 60.
    Чтобы изолированно проверить 60-gate (не 100-gate по r), подменяем _forecast через
    подкласс: возвращаем синтетический frame с достаточно r, но только 5 ненулевых rv_target.
    """

    class _StubPredictionService(PredictionService):
        async def _forecast(self, ticker: str) -> _Forecast:
            n = 160
            rng = np.random.default_rng(42)
            rv = np.full(n, np.nan)
            rv[-5:] = rng.uniform(0.001, 0.01, 5)
            frame = pd.DataFrame(
                {
                    "trade_date": pd.bdate_range(start=date(2022, 6, 1), periods=n).date,
                    "r": np.random.default_rng(1).normal(0, 0.01, n),
                    "rv_target": rv,
                }
            )
            loaded = _bundle(_StubPredictor()).volatility
            assert loaded is not None
            return _Forecast(
                security=_security(),
                frame=frame,
                predicted_for=date(2024, 1, 2),
                volatility=0.03,
                model=loaded,
            )

    service = _StubPredictionService(
        security_repo=_FakeSecurityRepo(_security()),
        feature_repo=_FakeFeatureRepo(_candles(160)),
        prediction_repo=_FakePredictionRepo(),
        bundle=_bundle(_StubPredictor()),
        settings=_settings(),
    )

    with pytest.raises(InsufficientHistoryError) as exc_info:
        await service.assess_volatility_regime("SBER", quantile=0.80, lookback=252)

    assert exc_info.value.available < MIN_REGIME_OBSERVATIONS

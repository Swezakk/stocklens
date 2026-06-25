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
from api.ml.features import build_serving_frame
from api.schemas.predict import VolatilityForecastHistoryOut, VolatilityRegime
from api.services.prediction import (
    MIN_REGIME_OBSERVATIONS,
    PredictionService,
    _Forecast,
)
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
    list_values_result: dict[date, float] = field(default_factory=dict)

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

    async def list_values(
        self,
        security_id: int,
        kind: PredictionKind,
        model_version: str,
        date_from: date,
        date_to: date,
    ) -> dict[date, float]:
        return {d: v for d, v in self.list_values_result.items() if date_from <= d <= date_to}


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


async def test_forecast_history_raises_404_for_unknown_ticker() -> None:
    """forecast_history: тикер не найден → SecurityNotFoundError (404)."""
    service, _, _ = _service(security=None)

    with pytest.raises(SecurityNotFoundError):
        await service.forecast_history("UNKNOWN", lookback=90)


async def test_forecast_history_returns_realized_and_forecast_points() -> None:
    """forecast_history: realized = sqrt(rv_target); forecast = значение из prediction_repo."""
    candles = _candles(160)
    frame = build_serving_frame(
        candles,
        _empty_dividends(),
        _empty_splits(),
        train_start=_settings().ml_train_start,
        horizon=5,
    )
    frame_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    # Берём дату внутри lookback-окна (последние 90 дат)
    window_dates = frame_dates[-90:]
    seed_date = window_dates[40]

    repo = _FakePredictionRepo(list_values_result={seed_date: 0.05})
    service, _, _ = _service(security=_security(), candles=candles, prediction_repo=repo)

    result: VolatilityForecastHistoryOut = await service.forecast_history("SBER", lookback=90)

    assert result.ticker == "SBER"
    assert result.model == "garch"
    assert result.model_version == "3"
    assert result.metrics_vs_baseline is not None

    realized_points = [p for p in result.points if p.realized is not None]
    assert len(realized_points) > 0, "Ожидались точки с realized"

    forecast_points = [p for p in result.points if p.forecast is not None]
    assert len(forecast_points) == 1
    assert forecast_points[0].date == seed_date
    assert forecast_points[0].forecast == pytest.approx(0.05)


async def test_forecast_history_no_model_returns_realized_without_forecast() -> None:
    """forecast_history: bundle без модели → metrics=None, model=None, forecast пустой."""
    service, _, _ = _service(
        security=_security(),
        candles=_candles(160),
        bundle=ModelBundle(volatility=None),
    )

    result = await service.forecast_history("SBER", lookback=90)

    assert result.model is None
    assert result.model_version is None
    assert result.metrics_vs_baseline is None
    realized_points = [p for p in result.points if p.realized is not None]
    assert len(realized_points) > 0, "Ожидались realized-точки даже без модели"
    assert all(p.forecast is None for p in result.points)


async def test_forecast_history_empty_candles_returns_empty_points() -> None:
    """forecast_history: пустые свечи → points=[], без ошибок."""
    service, _, _ = _service(
        security=_security(),
        candles=pd.DataFrame(
            columns=["trade_date", "open", "high", "low", "close", "volume", "is_weekend_session"]
        ),
    )

    result = await service.forecast_history("SBER", lookback=90)

    assert result.points == []
    assert result.ticker == "SBER"


async def test_forecast_history_forward_present() -> None:
    """forward присутствует и корректен, когда есть сохранённый прогноз и достаточно истории.

    Проверяет: volatility == последний прогноз (не первый), threshold == hand-computed quantile,
    is_elevated вычислен корректно.
    """
    candles = _candles(200)
    frame = build_serving_frame(
        candles,
        _empty_dividends(),
        _empty_splits(),
        train_start=_settings().ml_train_start,
        horizon=5,
    )
    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    window_dates = all_dates[-90:]

    realized_arr = np.sqrt(frame["rv_target"].dropna().to_numpy(dtype=float))
    trailing = realized_arr[-252:]
    expected_threshold = float(np.quantile(trailing, 0.80))

    early_date = window_dates[20]
    last_date = window_dates[60]
    last_forecast = expected_threshold * 1.5

    repo = _FakePredictionRepo(
        list_values_result={
            early_date: 0.0001,
            last_date: last_forecast,
        }
    )
    service, _, _ = _service(security=_security(), candles=candles, prediction_repo=repo)

    result = await service.forecast_history("SBER", lookback=90)

    assert result.forward is not None
    assert isinstance(result.forward, VolatilityRegime)
    assert result.forward.volatility == pytest.approx(last_forecast)
    assert result.forward.predicted_for == last_date
    assert result.forward.ticker == "SBER"
    assert result.forward.threshold == pytest.approx(expected_threshold)
    assert result.forward.is_elevated is True
    assert result.forward.quantile == pytest.approx(0.80)
    assert result.forward.lookback == 252


async def test_forecast_history_forward_none_when_no_forecast() -> None:
    """forward=None, когда ни одна точка не имеет сохранённого прогноза."""
    service, _, _ = _service(
        security=_security(),
        candles=_candles(200),
        prediction_repo=_FakePredictionRepo(list_values_result={}),
    )

    result = await service.forecast_history("SBER", lookback=90)

    assert result.forward is None
    # Additivity: points и live_metrics остаются корректными.
    assert len(result.points) > 0
    assert all(p.forecast is None for p in result.points)


async def test_forecast_history_forward_none_when_insufficient_history() -> None:
    """forecast_history не бросает InsufficientHistoryError при rv_target < MIN_REGIME_OBSERVATIONS.

    Прогноз присутствует (чтобы разграничить с test_forecast_history_forward_none_when_no_forecast),
    но история rv_target слишком мала для квантиля → forward=None gracefully.
    n=63 даёт 58 ненулевых rv_target < MIN_REGIME_OBSERVATIONS=60 (подтверждено assert-ом в теле).
    """
    candles = _candles(63)
    frame = build_serving_frame(
        candles,
        _empty_dividends(),
        _empty_splits(),
        train_start=_settings().ml_train_start,
        horizon=5,
    )
    rv_count = int(frame["rv_target"].dropna().shape[0])
    assert rv_count < MIN_REGIME_OBSERVATIONS, (
        f"Предусловие теста нарушено: rv_target={rv_count} >= {MIN_REGIME_OBSERVATIONS}"
    )

    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    window_dates = all_dates[-90:]
    seed_date = window_dates[min(20, len(window_dates) - 1)]

    repo = _FakePredictionRepo(list_values_result={seed_date: 0.03})
    service, _, _ = _service(security=_security(), candles=candles, prediction_repo=repo)

    result = await service.forecast_history("SBER", lookback=90)

    forecast_points = [p for p in result.points if p.forecast is not None]
    assert len(forecast_points) >= 1, "Прогноз должен присутствовать в points"
    assert result.forward is None


async def test_assess_regime_still_raises_on_insufficient_history() -> None:
    """assess_volatility_regime сохраняет прежнее поведение: бросает InsufficientHistoryError
    при нехватке rv_target — DRY-хелпер не ослабил его контракт.
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


async def test_forecast_history_additive_with_forward() -> None:
    """points, metrics_vs_baseline, live_metrics, live_sample_size идентичны прежнему контракту.

    Добавление forward не должно изменять ни один из существующих полей.
    """
    candles = _candles(200)
    frame = build_serving_frame(
        candles,
        _empty_dividends(),
        _empty_splits(),
        train_start=_settings().ml_train_start,
        horizon=5,
    )
    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    window_dates = all_dates[-90:]
    seed_date = window_dates[40]

    repo = _FakePredictionRepo(list_values_result={seed_date: 0.05})
    service, _, _ = _service(security=_security(), candles=candles, prediction_repo=repo)

    result = await service.forecast_history("SBER", lookback=90)

    # Существующие поля сохраняют прежний контракт.
    assert result.ticker == "SBER"
    assert result.model == "garch"
    assert result.model_version == "3"
    assert result.metrics_vs_baseline is not None
    assert len(result.points) > 0

    forecast_points = [p for p in result.points if p.forecast is not None]
    assert len(forecast_points) == 1
    assert forecast_points[0].date == seed_date
    assert forecast_points[0].forecast == pytest.approx(0.05)

    realized_points = [p for p in result.points if p.realized is not None]
    assert len(realized_points) > 0

    # forward — дополнительное поле, его наличие не ломает остальные.
    assert hasattr(result, "forward")

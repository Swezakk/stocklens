"""Unit-тесты живой (in-sample) метрики QLIKE для forecast_history.

Покрывает добавленные в PredictionService.forecast_history поля:
  live_metrics: VolatilityMetrics | None
  live_sample_size: int

Проверяет корректность единиц (baseline — дисперсия, НЕ квадрат дисперсии),
поведение при малой выборке и аддитивность (points и metrics_vs_baseline не меняются).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from api.core.settings import ApiSettings
from api.ml.bundle import LoadedVolatilityModel, ModelBundle
from api.ml.features import build_serving_frame
from api.schemas.predict import VolatilityForecastHistoryOut, VolatilityForecastPoint
from api.services.prediction import (
    MIN_LIVE_PAIRS,
    PredictionService,
    _build_baseline_map,
    _compute_live_metrics,
)
from stocklens_core.enums import PredictionKind
from stocklens_core.models.market import Security
from stocklens_ml.eval.metrics import qlike, rmse

from tests.unit.fakes import FakeSecurity

_METRICS = {"qlike": 0.844, "qlike_baseline": 2.203, "rmse": 0.0025}
_HORIZON = 5


def _settings() -> ApiSettings:
    return ApiSettings.model_validate(
        {
            "database_url_async": "postgresql+asyncpg://u:p@h:5432/d",
            "redis_url": "redis://h:6379/0",
        }
    )


def _security() -> Security:
    return cast(Security, FakeSecurity(id=1, ticker="SBER", name="Сбербанк", board="TQBR"))


def _candles(n: int, start: date = date(2022, 6, 1)) -> pd.DataFrame:
    rng = np.random.default_rng(42)
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
    def __init__(self, variance: float = 0.0009) -> None:
        self.variance = variance

    def forecast(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
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


def _bundle() -> ModelBundle:
    return ModelBundle(
        volatility=LoadedVolatilityModel(
            predictor=_StubPredictor(),
            model_version="3",
            method="garch",
            metrics=_METRICS,
            horizon_days=_HORIZON,
        )
    )


def _service(
    *,
    candles: pd.DataFrame,
    prediction_repo: _FakePredictionRepo | None = None,
) -> PredictionService:
    return PredictionService(
        security_repo=_FakeSecurityRepo(_security()),
        feature_repo=_FakeFeatureRepo(candles),
        prediction_repo=prediction_repo or _FakePredictionRepo(),
        bundle=_bundle(),
        settings=_settings(),
    )


def _make_matured_points(n_matured: int) -> tuple[list[VolatilityForecastPoint], pd.DataFrame]:
    """Список точек и frame с n_matured созревшими парами + 3 pending точки.

    frame содержит ['trade_date', 'r', 'rv_target'] с детерминированными значениями,
    чтобы hand-computed QLIKE был воспроизводим в тестах.
    """
    rng = np.random.default_rng(7)
    total = n_matured + 3

    r_vals = rng.uniform(0.001, 0.02, total)
    rv_target = r_vals**2 * _HORIZON

    trade_dates = list(pd.bdate_range(start=date(2024, 1, 2), periods=total).date)

    frame = pd.DataFrame(
        {
            "trade_date": trade_dates,
            "r": r_vals,
            "rv_target": np.where(np.arange(total) < n_matured, rv_target, np.nan),
        }
    )

    points: list[VolatilityForecastPoint] = []
    for i, d in enumerate(trade_dates):
        if i < n_matured:
            realized_vol = float(np.sqrt(rv_target[i]))
            forecast_vol = realized_vol * 1.1
            points.append(
                VolatilityForecastPoint(date=d, forecast=forecast_vol, realized=realized_vol)
            )
        else:
            points.append(VolatilityForecastPoint(date=d, forecast=0.05, realized=None))

    return points, frame


def test_live_qlike_paired_over_matured_pairs() -> None:
    """_compute_live_metrics: QLIKE и qlike_baseline равны hand-computed значениям."""
    n_matured = 15
    points, frame = _make_matured_points(n_matured)

    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    baseline_map = _build_baseline_map(frame, all_dates, _HORIZON)

    live_metrics, live_sample_size = _compute_live_metrics(points, baseline_map)

    assert live_metrics is not None, "Ожидался live_metrics при n_matured >= MIN_LIVE_PAIRS"

    matured = [
        (p.forecast, p.realized)
        for p in points
        if p.forecast is not None and p.realized is not None
    ]
    matured_dates = [p.date for p in points if p.forecast is not None and p.realized is not None]
    h_arr = np.array([f**2 for f, _ in matured], dtype=np.float64)
    rv_arr = np.array([r**2 for _, r in matured], dtype=np.float64)
    b_arr = np.array(
        [baseline_map[d] for d in matured_dates if d in baseline_map], dtype=np.float64
    )

    valid_in_baseline = [d in baseline_map for d in matured_dates]
    h_arr_f = h_arr[valid_in_baseline]
    rv_arr_f = rv_arr[valid_in_baseline]

    mask = (
        np.isfinite(h_arr_f)
        & (h_arr_f > 0)
        & np.isfinite(rv_arr_f)
        & (rv_arr_f > 0)
        & np.isfinite(b_arr)
        & (b_arr > 0)
    )

    expected_qlike = qlike(rv_arr_f[mask], h_arr_f[mask])
    expected_qlike_baseline = qlike(rv_arr_f[mask], b_arr[mask])
    expected_rmse = rmse(rv_arr_f[mask], h_arr_f[mask])
    expected_n = int(mask.sum())

    assert live_metrics.qlike == pytest.approx(expected_qlike, rel=1e-6)
    assert live_metrics.qlike_baseline == pytest.approx(expected_qlike_baseline, rel=1e-6)
    assert live_metrics.rmse == pytest.approx(expected_rmse, rel=1e-6)
    assert live_sample_size == expected_n


def test_live_baseline_uses_rw_rv_unsquared() -> None:
    """Baseline QLIKE вычисляется по b = rw_rv_forecast (уже дисперсия Σr²).

    rw_rv_forecast возвращает дисперсию, не волатильность — возводить в квадрат ошибочно.
    Тест подтверждает, что реализация использует b напрямую (не b**2), с тем же joint-mask
    что и _compute_live_metrics (h/RV/b одновременно конечны и положительны).
    """
    n_matured = 15
    points, frame = _make_matured_points(n_matured)

    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    baseline_map = _build_baseline_map(frame, all_dates, _HORIZON)

    live_metrics, _ = _compute_live_metrics(points, baseline_map)
    assert live_metrics is not None

    matured = [
        (p.date, p.forecast, p.realized)
        for p in points
        if p.forecast is not None and p.realized is not None and p.date in baseline_map
    ]
    h_arr = np.array([f**2 for _, f, _ in matured], dtype=np.float64)
    rv_arr = np.array([r**2 for _, _, r in matured], dtype=np.float64)
    b_correct = np.array([baseline_map[d] for d, _, _ in matured], dtype=np.float64)
    b_wrong = b_correct**2

    joint_mask_c = (
        np.isfinite(h_arr)
        & (h_arr > 0)
        & np.isfinite(rv_arr)
        & (rv_arr > 0)
        & np.isfinite(b_correct)
        & (b_correct > 0)
    )
    joint_mask_w = (
        np.isfinite(h_arr)
        & (h_arr > 0)
        & np.isfinite(rv_arr)
        & (rv_arr > 0)
        & np.isfinite(b_wrong)
        & (b_wrong > 0)
    )

    qlike_correct = qlike(rv_arr[joint_mask_c], b_correct[joint_mask_c])
    qlike_wrong = qlike(rv_arr[joint_mask_w], b_wrong[joint_mask_w])

    assert abs(qlike_correct - qlike_wrong) > 1e-6, (
        "QLIKE с корректным и ошибочным baseline должны отличаться"
    )
    assert live_metrics.qlike_baseline == pytest.approx(qlike_correct, rel=1e-6)
    assert live_metrics.qlike_baseline != pytest.approx(qlike_wrong, rel=1e-6)


def test_live_metrics_none_below_min_pairs() -> None:
    """_compute_live_metrics: меньше MIN_LIVE_PAIRS созревших пар → live_metrics is None,
    live_sample_size отражает реальное количество валидных пар (не 0, не обрезается порогом).

    При n_matured=9 и horizon=5 первые 5 строк имеют NaN baseline (rolling не заполнен),
    поэтому в joint-маске участвуют только пары с индексами 5–8 = 4 валидных пары.
    """
    n_matured = MIN_LIVE_PAIRS - 1  # = 9
    points, frame = _make_matured_points(n_matured)

    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    baseline_map = _build_baseline_map(frame, all_dates, _HORIZON)

    live_metrics, live_sample_size = _compute_live_metrics(points, baseline_map)

    assert live_metrics is None, f"Ожидался None при {n_matured} < {MIN_LIVE_PAIRS} парах"
    # Индексы 4–8 проходят joint-маску (0–3: baseline=NaN при min_periods=5).
    assert live_sample_size == 5


async def test_forecast_history_additive() -> None:
    """forecast_history: добавление live_metrics не меняет points и metrics_vs_baseline."""
    candles = _candles(200)
    frame = build_serving_frame(
        candles,
        _empty_dividends(),
        _empty_splits(),
        train_start=_settings().ml_train_start,
        horizon=_HORIZON,
    )
    frame_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    window_dates = frame_dates[-90:]
    seed_date = window_dates[40]

    repo = _FakePredictionRepo(list_values_result={seed_date: 0.05})
    svc = _service(candles=candles, prediction_repo=repo)

    result: VolatilityForecastHistoryOut = await svc.forecast_history("SBER", lookback=90)

    assert result.ticker == "SBER"
    assert result.model == "garch"
    assert result.model_version == "3"
    assert result.metrics_vs_baseline is not None
    assert result.metrics_vs_baseline.qlike == pytest.approx(0.844)
    assert result.metrics_vs_baseline.qlike_baseline == pytest.approx(2.203)
    assert result.metrics_vs_baseline.rmse == pytest.approx(0.0025)

    forecast_points = [p for p in result.points if p.forecast is not None]
    assert len(forecast_points) == 1
    assert forecast_points[0].date == seed_date
    assert forecast_points[0].forecast == pytest.approx(0.05)

    realized_points = [p for p in result.points if p.realized is not None]
    assert len(realized_points) > 0

    assert hasattr(result, "live_metrics")
    assert hasattr(result, "live_sample_size")
    assert isinstance(result.live_sample_size, int)


def test_live_excludes_pending_points() -> None:
    """_compute_live_metrics: точки с realized=None не учитываются в live_sample_size."""
    n_matured = 12
    points, frame = _make_matured_points(n_matured)

    pending = [p for p in points if p.realized is None]
    assert len(pending) == 3, "Ожидалось 3 pending точки в тестовой фикстуре"

    all_dates = [pd.Timestamp(d).date() for d in frame["trade_date"].tolist()]
    baseline_map = _build_baseline_map(frame, all_dates, _HORIZON)

    _, live_sample_size = _compute_live_metrics(points, baseline_map)

    assert live_sample_size <= n_matured

    extra_pending = [
        VolatilityForecastPoint(date=date(2099, 1, i), forecast=0.05, realized=None)
        for i in range(1, 6)
    ]
    _, live_sample_size_extended = _compute_live_metrics(points + extra_pending, baseline_map)
    assert live_sample_size_extended == live_sample_size

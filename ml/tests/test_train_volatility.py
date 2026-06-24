"""Tests for volatility training orchestration (ml-spec §11, §12): winner selection,
baseline gate, champion registration, MLflow logging."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stocklens_ml.config import MlSettings
from stocklens_ml.training import train_volatility

import mlflow
from mlflow import MlflowClient


def _settings() -> MlSettings:
    return MlSettings.model_validate(
        {"database_url": "postgresql+psycopg://user:pass@localhost:5432/db"}
    )


def _feature_frame(n: int = 360, seed: int = 11) -> pd.DataFrame:
    """Синтетический фрейм фич волатильности: r + HAR-регрессоры + forward-таргет."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, n)
    squared = pd.Series(returns**2)
    return pd.DataFrame(
        {
            "r": returns,
            "rv_d": squared,
            "rv_w": squared.rolling(5).mean(),
            "rv_m": squared.rolling(22).mean(),
            "rv_target": squared.shift(-5).rolling(5).sum(),
        }
    )


def test_beats_baseline_flags_models_below_baseline_qlike() -> None:
    metrics = {
        "baseline_rw_rv": {"qlike": 0.5, "rmse": 0.10},
        "har_rv": {"qlike": 0.3, "rmse": 0.08},
        "garch": {"qlike": 0.6, "rmse": 0.12},
    }

    result = train_volatility.beats_baseline(metrics)

    assert result == {"har_rv": True, "garch": False}


def test_log_run_records_params_metrics_and_gain(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    metrics = {
        "baseline_rw_rv": {"qlike": 0.5, "rmse": 0.10},
        "garch": {"qlike": 0.3, "rmse": 0.08},
    }

    train_volatility.log_run("TEST", metrics, _settings(), n_splits=5)

    runs = mlflow.search_runs(experiment_names=[train_volatility._EXPERIMENT])
    assert not isinstance(runs, list)  # output_format по умолчанию — pandas DataFrame
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.ticker"] == "TEST"
    assert row["metrics.qlike_garch"] == pytest.approx(0.3)
    assert row["metrics.qlike_gain_garch"] == pytest.approx(0.2)  # 0.5 − 0.3


def test_select_winner_picks_lowest_mean_qlike_beating_baseline() -> None:
    results = {
        "SBER": {
            "baseline_rw_rv": {"qlike": 2.20, "rmse": 0.0030},
            "garch": {"qlike": 0.84, "rmse": 0.0025},
            "har_rv": {"qlike": 0.86, "rmse": 0.0023},
        },
        "LKOH": {
            "baseline_rw_rv": {"qlike": 1.48, "rmse": 0.0022},
            "garch": {"qlike": 0.58, "rmse": 0.0019},
            "har_rv": {"qlike": 0.60, "rmse": 0.0018},
        },
    }

    winner = train_volatility.select_winner(results)

    assert winner is not None
    assert winner.method == "garch"
    assert winner.metrics["qlike"] == pytest.approx((0.84 + 0.58) / 2)
    assert winner.metrics["qlike_baseline"] == pytest.approx((2.20 + 1.48) / 2)
    assert winner.metrics["rmse"] == pytest.approx((0.0025 + 0.0019) / 2)


def test_select_winner_returns_none_when_no_method_beats_baseline() -> None:
    results = {
        "SBER": {
            "baseline_rw_rv": {"qlike": 0.50, "rmse": 0.0030},
            "garch": {"qlike": 0.61, "rmse": 0.0025},
            "har_rv": {"qlike": 0.70, "rmse": 0.0023},
        }
    }

    assert train_volatility.select_winner(results) is None


def test_select_winner_prefers_har_when_it_has_lower_qlike() -> None:
    results = {
        "SBER": {
            "baseline_rw_rv": {"qlike": 2.00, "rmse": 0.0030},
            "garch": {"qlike": 1.00, "rmse": 0.0025},
            "har_rv": {"qlike": 0.50, "rmse": 0.0023},
        }
    }

    winner = train_volatility.select_winner(results)

    assert winner is not None
    assert winner.method == "har_rv"


def test_register_champion_garch_logs_and_marks_alias(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    winner = train_volatility.WinnerSelection(
        "garch", {"qlike": 0.84, "qlike_baseline": 2.20, "rmse": 0.0025}
    )

    info = train_volatility.register_champion(_settings(), winner, [_feature_frame()])

    client = MlflowClient()
    champion = client.get_model_version_by_alias("stocklens-volatility", "champion")
    assert str(champion.version) == str(info.registered_model_version)
    loaded = mlflow.pyfunc.load_model("models:/stocklens-volatility@champion")
    assert loaded.unwrap_python_model().method == "garch"


def test_register_champion_har_fits_pooled_coefficients(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    winner = train_volatility.WinnerSelection(
        "har_rv", {"qlike": 0.86, "qlike_baseline": 2.20, "rmse": 0.0023}
    )

    info = train_volatility.register_champion(
        _settings(), winner, [_feature_frame(seed=1), _feature_frame(seed=2)]
    )

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    underlying = loaded.unwrap_python_model()
    assert underlying.method == "har_rv"
    assert underlying.har_coef is not None
    assert len(underlying.har_coef) == 3

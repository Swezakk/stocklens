"""Tests for volatility training orchestration (ml-spec §12): baseline gate + MLflow logging."""

from pathlib import Path

import mlflow
import pytest
from stocklens_ml.config import MlSettings
from stocklens_ml.training import train_volatility


def _settings() -> MlSettings:
    return MlSettings.model_validate(
        {"database_url": "postgresql+psycopg://user:pass@localhost:5432/db"}
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

"""Tests for ingestor scheduler job registration.

Verifies that build_scheduler registers all expected cron jobs with
correct trigger times. Job bodies are tested separately; here we only
assert job presence and trigger configuration.

OnnxSentimentScorer is patched to avoid loading ONNX model files from disk.
"""

from unittest.mock import MagicMock, patch

from apscheduler.schedulers.blocking import BlockingScheduler
from ingestor.scheduler import build_scheduler


def _build(scheduler_factory: type[BlockingScheduler] = BlockingScheduler) -> BlockingScheduler:
    """Build scheduler with all heavy dependencies replaced by fakes."""
    client = MagicMock()
    session_factory = MagicMock()
    settings = MagicMock()
    settings.heartbeat_path = "/tmp/test-heartbeat"

    with patch("ingestor.scheduler.OnnxSentimentScorer") as _mock_scorer:
        _mock_scorer.return_value = MagicMock()
        return build_scheduler(
            client=client,
            session_factory=session_factory,
            settings=settings,
        )


def _cron_hour_minute(scheduler: BlockingScheduler, job_id: str) -> tuple[str, str]:
    """Extract hour and minute strings from a cron job's trigger."""
    job = scheduler.get_job(job_id)
    assert job is not None, f"Job '{job_id}' not registered"
    trigger = job.trigger
    fields_by_name = {f.name: str(f) for f in trigger.fields}
    return fields_by_name["hour"], fields_by_name["minute"]


def test_build_scheduler_registers_evening_candle_sync() -> None:
    """Existing 23:55 candles_daily job remains registered (regression guard)."""
    scheduler = _build()
    assert scheduler.get_job("candles_daily") is not None


def test_build_scheduler_registers_evening_index_sync() -> None:
    """Existing 23:55 index_daily job remains registered (regression guard)."""
    scheduler = _build()
    assert scheduler.get_job("index_daily") is not None


def test_build_scheduler_registers_morning_candle_sync() -> None:
    """candles_morning job must be registered at 10:00 MSK.

    MOEX publishes the prior trading day's daily candle the next morning.
    The evening 23:55 job always trails by a day; this morning catch-up
    pulls the freshly-published candle so same-day forecasts are accurate.
    """
    scheduler = _build()
    assert scheduler.get_job("candles_morning") is not None
    hour, minute = _cron_hour_minute(scheduler, "candles_morning")
    assert hour == "10", f"Expected hour=10, got {hour!r}"
    assert minute == "0", f"Expected minute=0, got {minute!r}"


def test_build_scheduler_registers_morning_index_sync() -> None:
    """index_morning job must be registered at 10:00 MSK.

    Mirrors the candles morning catch-up for the IMOEX index values.
    """
    scheduler = _build()
    assert scheduler.get_job("index_morning") is not None
    hour, minute = _cron_hour_minute(scheduler, "index_morning")
    assert hour == "10", f"Expected hour=10, got {hour!r}"
    assert minute == "0", f"Expected minute=0, got {minute!r}"


def test_build_scheduler_evening_candle_trigger_unchanged() -> None:
    """candles_daily trigger stays at 23:55 MSK (backstop, must not shift)."""
    scheduler = _build()
    hour, minute = _cron_hour_minute(scheduler, "candles_daily")
    assert hour == "23"
    assert minute == "55"


def test_build_scheduler_evening_index_trigger_unchanged() -> None:
    """index_daily trigger stays at 23:55 MSK (backstop, must not shift)."""
    scheduler = _build()
    hour, minute = _cron_hour_minute(scheduler, "index_daily")
    assert hour == "23"
    assert minute == "55"

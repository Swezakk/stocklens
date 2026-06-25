"""Tests for BotSettings default values.

Asserts that forecast_refresh defaults are 11:00 MSK (after the morning
ingestor candle sync at 10:00) so forecasts anchor to the freshly-pulled
candle on the same day.
"""

from bot.settings import BotSettings

_MINIMAL_REQUIRED: dict[str, object] = {
    "telegram_bot_token": "1:abc",
    "auth_password": "owner-pw",
    "digest_chat_id": 111,
}


def test_forecast_refresh_hour_default_is_11() -> None:
    """forecast_refresh_hour_msk defaults to 11 (MSK), one hour after morning candle sync."""
    settings = BotSettings.model_validate(_MINIMAL_REQUIRED)
    assert settings.forecast_refresh_hour_msk == 11


def test_forecast_refresh_minute_default_is_0() -> None:
    """forecast_refresh_minute_msk defaults to 0 (top of the hour)."""
    settings = BotSettings.model_validate(_MINIMAL_REQUIRED)
    assert settings.forecast_refresh_minute_msk == 0


def test_forecast_refresh_hour_overrideable_via_env_alias() -> None:
    """Env alias FORECAST_REFRESH_HOUR_MSK must override the default."""
    settings = BotSettings.model_validate({**_MINIMAL_REQUIRED, "FORECAST_REFRESH_HOUR_MSK": "14"})
    assert settings.forecast_refresh_hour_msk == 14


def test_forecast_refresh_minute_overrideable_via_env_alias() -> None:
    """Env alias FORECAST_REFRESH_MINUTE_MSK must override the default."""
    settings = BotSettings.model_validate(
        {**_MINIMAL_REQUIRED, "FORECAST_REFRESH_MINUTE_MSK": "30"}
    )
    assert settings.forecast_refresh_minute_msk == 30

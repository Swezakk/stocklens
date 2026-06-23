"""Сборка зависимостей бота из настроек (DESIGN §7, §11).

Связывает ``BotSettings`` → ``AuthConfig`` → ``TokenManager`` → ``ApiClient``: бот логинится
в API учётными данными владельца из env (``SecretStr`` раскрывается здесь, на границе сборки).
Вынесено из точки входа ради тестируемости проводки (без рантайма Telegram).
"""

from bot.api_client.client import ApiClient
from bot.api_client.token_manager import AuthConfig, TokenManager
from bot.settings import BotSettings


def build_api_client(settings: BotSettings) -> ApiClient:
    """Собрать ApiClient с in-process JWT-менеджером из настроек бота."""
    config = AuthConfig(
        api_base_url=settings.api_base_url,
        api_prefix=settings.api_prefix,
        auth_username=settings.auth_username,
        auth_password=settings.auth_password.get_secret_value(),
        token_refresh_margin_seconds=settings.token_refresh_margin_seconds,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    manager = TokenManager(config)
    return ApiClient(
        base_url=settings.api_base_url,
        api_prefix=settings.api_prefix,
        timeout=settings.request_timeout_seconds,
        token_provider=manager.get_token,
        on_unauthorized=manager.force_refresh,
    )

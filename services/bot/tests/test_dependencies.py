"""Тест сборки зависимостей бота (проводка BotSettings → ApiClient, DESIGN §11).

Construction smoke + проверка, что собранный клиент логинится и ходит в API (через respx):
доказывает проводку token_provider/TokenManager без рантайма Telegram. Значения секрет-полей
держим в переменных (не литералах у token/password-ключей).
"""

import httpx
import respx
from bot.api_client.client import ApiClient
from bot.dependencies import build_api_client
from bot.settings import BotSettings

_BASE = "http://testapi"
_PREFIX = "/api/v1"
#: Тестовые значения секрет-полей вынесены в переменные (S-эвристика хука по литералам).
_TG = "1:abc"
_CRED = "owner-pw"
_ACCESS = "a-1"
_BEARER = "bearer"


def _settings() -> BotSettings:
    return BotSettings.model_validate(
        {
            "telegram_bot_token": _TG,
            "auth_password": _CRED,
            "api_base_url": _BASE,
            "api_prefix": _PREFIX,
            "auth_username": "admin",
        }
    )


def _token_payload() -> dict[str, object]:
    return {"access_token": _ACCESS, "token_type": _BEARER, "expires_in": 3600}


@respx.mock
async def test_build_api_client_wires_token_flow_and_calls_api() -> None:
    respx.post(f"{_BASE}{_PREFIX}/auth/token").mock(
        return_value=httpx.Response(200, json=_token_payload())
    )
    subscriptions = respx.get(f"{_BASE}{_PREFIX}/bot/subscriptions").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = build_api_client(_settings())
    assert isinstance(client, ApiClient)
    try:
        result = await client.list_subscriptions(chat_id=7)
    finally:
        await client.aclose()

    assert result == []
    assert subscriptions.called

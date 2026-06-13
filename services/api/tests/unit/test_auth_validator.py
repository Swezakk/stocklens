"""Unit-тесты валидатора JWT-токенов.

Тестируются: корректный токен, истёкший с leeway, неверный iss, неверный aud,
отсутствующий sub, попытка алгоритм-confusion (RS256-токен против HS256-конфига).
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from api.core.auth.settings import AuthSettings
from api.core.auth.validator import decode_token
from api.core.exceptions import UnauthorizedError
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, generate_private_key

_SIGNING_KEY = "unit-test-signing-key-hs256-padded-to-min-32-bytes"
_ISSUER = "https://stocklens.test"
_AUDIENCE = "stocklens-api-test"


def _make_settings(**overrides: object) -> AuthSettings:
    base: dict[str, object] = {
        "mode": "local",
        "secret": _SIGNING_KEY,
        "owner_username": "admin",
        "owner_password": "irrelevant-for-validator-tests",
        "issuer": _ISSUER,
        "audience": _AUDIENCE,
        "token_ttl_seconds": 3600,
        "leeway_seconds": 60,
    }
    base.update(overrides)
    return AuthSettings.model_validate(base)


def _encode_hs256(payload: dict[str, object], key: str = _SIGNING_KEY) -> str:
    return jwt.encode(payload, key, algorithm="HS256")


def _encode_rs256(payload: dict[str, object], key: RSAPrivateKey) -> str:
    return jwt.encode(payload, key, algorithm="RS256")


def _valid_payload(*, delta_seconds: int = 3600) -> dict[str, object]:
    now = datetime.now(tz=UTC)
    return {
        "sub": "admin",
        "scope": "read write",
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "iat": now,
        "exp": now + timedelta(seconds=delta_seconds),
    }


@pytest.mark.asyncio
async def test_decode_token_valid_returns_principal() -> None:
    token = _encode_hs256(_valid_payload())
    principal = await decode_token(token, _make_settings())

    assert principal.sub == "admin"
    assert "read" in principal.scopes
    assert "write" in principal.scopes


@pytest.mark.asyncio
async def test_decode_token_within_leeway_succeeds() -> None:
    """Токен, истёкший менее leeway секунд назад, должен быть принят."""
    token = _encode_hs256(_valid_payload(delta_seconds=-30))
    principal = await decode_token(token, _make_settings(leeway_seconds=60))

    assert principal.sub == "admin"


@pytest.mark.asyncio
async def test_decode_token_expired_beyond_leeway_raises() -> None:
    token = _encode_hs256(_valid_payload(delta_seconds=-120))

    with pytest.raises(UnauthorizedError):
        await decode_token(token, _make_settings(leeway_seconds=60))


@pytest.mark.asyncio
async def test_decode_token_wrong_issuer_raises() -> None:
    payload = {**_valid_payload(), "iss": "https://evil.example.com"}
    token = _encode_hs256(payload)

    with pytest.raises(UnauthorizedError):
        await decode_token(token, _make_settings())


@pytest.mark.asyncio
async def test_decode_token_wrong_audience_raises() -> None:
    payload = {**_valid_payload(), "aud": "wrong-audience"}
    token = _encode_hs256(payload)

    with pytest.raises(UnauthorizedError):
        await decode_token(token, _make_settings())


@pytest.mark.asyncio
async def test_decode_token_missing_sub_raises() -> None:
    now = datetime.now(tz=UTC)
    payload: dict[str, object] = {
        "scope": "read",
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    token = _encode_hs256(payload)

    with pytest.raises(UnauthorizedError):
        await decode_token(token, _make_settings())


@pytest.mark.asyncio
async def test_decode_token_malformed_raises() -> None:
    with pytest.raises(UnauthorizedError):
        await decode_token("not.a.jwt", _make_settings())


@pytest.mark.asyncio
async def test_decode_token_alg_confusion_rs256_rejected() -> None:
    """RS256-токен подписанный RSA-ключом должен быть отклонён HS256-конфигом.

    Конфиг пиннирует ["HS256"] — даже валидный RS256-токен не пройдёт верификацию,
    что блокирует confusion-атаку (RS256 pubkey как HMAC secret).
    """
    private_key = generate_private_key(public_exponent=65537, key_size=2048)
    token = _encode_rs256(_valid_payload(), private_key)

    with pytest.raises(UnauthorizedError):
        await decode_token(token, _make_settings())


@pytest.mark.asyncio
async def test_decode_token_tampered_signature_raises() -> None:
    token = _encode_hs256(_valid_payload())
    header, payload_b64, _ = token.split(".")
    tampered = f"{header}.{payload_b64}.invalidsignature"

    with pytest.raises(UnauthorizedError):
        await decode_token(tampered, _make_settings())


@pytest.mark.asyncio
async def test_decode_token_no_secret_configured_raises() -> None:
    settings = _make_settings(secret=None)

    with pytest.raises(UnauthorizedError, match="AUTH_SECRET"):
        await decode_token(_encode_hs256(_valid_payload()), settings)


@pytest.mark.asyncio
async def test_decode_token_empty_scope_gives_empty_list() -> None:
    payload = {**_valid_payload(), "scope": ""}
    token = _encode_hs256(payload)
    principal = await decode_token(token, _make_settings())

    assert principal.scopes == []

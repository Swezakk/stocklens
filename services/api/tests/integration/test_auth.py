"""Integration-тесты аутентификации: /auth/token и защищённые эндпоинты."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration

_ISSUER = "https://stocklens.test"
_AUDIENCE = "stocklens-api-test"
_SIGNING_KEY = "test-secret-for-integration-tests-only"
_OWNER_USERNAME = "testowner"
_OWNER_CREDENTIAL = "test-owner-credential-integration"


def _mint_token(
    *,
    sub: str = _OWNER_USERNAME,
    issuer: str = _ISSUER,
    audience: str = _AUDIENCE,
    delta_seconds: int = 3600,
    signing_key: str = _SIGNING_KEY,
) -> str:
    now = datetime.now(tz=UTC)
    payload: dict[str, object] = {
        "sub": sub,
        "scope": "",
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + timedelta(seconds=delta_seconds),
    }
    return jwt.encode(payload, signing_key, algorithm="HS256")


async def test_login_correct_credentials_returns_token(noauth_client: AsyncClient) -> None:
    response = await noauth_client.post(
        "/api/v1/auth/token",
        data={"username": _OWNER_USERNAME, "password": _OWNER_CREDENTIAL},
    )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body.get("token_type") == "bearer"
    assert body["expires_in"] > 0


async def test_login_wrong_credentials_returns_401(noauth_client: AsyncClient) -> None:
    response = await noauth_client.post(
        "/api/v1/auth/token",
        data={"username": _OWNER_USERNAME, "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_login_rate_limit_returns_429(noauth_client: AsyncClient) -> None:
    """После 5 попыток за окно должен вернуться 429 с Retry-After."""
    for _ in range(5):
        await noauth_client.post(
            "/api/v1/auth/token",
            data={"username": _OWNER_USERNAME, "password": "bad"},
        )

    response = await noauth_client.post(
        "/api/v1/auth/token",
        data={"username": _OWNER_USERNAME, "password": "bad"},
    )

    assert response.status_code == 429
    assert "Retry-After" in response.headers


async def test_protected_endpoint_without_token_returns_401(noauth_client: AsyncClient) -> None:
    response = await noauth_client.get("/api/v1/portfolio/positions")

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_protected_endpoint_with_valid_token_returns_200(noauth_client: AsyncClient) -> None:
    token = _mint_token()
    response = await noauth_client.get(
        "/api/v1/portfolio/positions",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


async def test_protected_endpoint_with_expired_token_returns_401(
    noauth_client: AsyncClient,
) -> None:
    token = _mint_token(delta_seconds=-3600)
    response = await noauth_client.get(
        "/api/v1/portfolio/positions",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


async def test_protected_endpoint_with_wrong_audience_returns_401(
    noauth_client: AsyncClient,
) -> None:
    token = _mint_token(audience="wrong-audience")
    response = await noauth_client.get(
        "/api/v1/portfolio/positions",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


async def test_protected_endpoint_with_wrong_issuer_returns_401(noauth_client: AsyncClient) -> None:
    token = _mint_token(issuer="https://evil.example.com")
    response = await noauth_client.get(
        "/api/v1/portfolio/positions",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


async def test_protected_endpoint_with_tampered_token_returns_401(
    noauth_client: AsyncClient,
) -> None:
    token = _mint_token()
    header, payload_b64, _ = token.split(".")
    tampered = f"{header}.{payload_b64}.invalidsignature"
    response = await noauth_client.get(
        "/api/v1/portfolio/positions",
        headers={"Authorization": f"Bearer {tampered}"},
    )

    assert response.status_code == 401


async def test_health_ready_without_token_returns_200(noauth_client: AsyncClient) -> None:
    response = await noauth_client.get("/api/v1/health/ready")

    assert response.status_code == 200

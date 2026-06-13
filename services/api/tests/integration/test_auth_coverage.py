"""Guard-тест fail-closed: каждый /api/v1 роут вне allowlist (health/auth) требует авторизацию.

Новый роутер, подключённый без require_auth, → этот тест падает, а не молчаливо
открывает данные наружу (fail-closed предпочтительнее fail-open для security-контроля).
"""

import pytest
from api.core.auth.deps import require_auth
from api.main import create_app
from fastapi.routing import APIRoute

pytestmark = pytest.mark.integration

_PUBLIC_PREFIXES = ("/api/v1/health", "/api/v1/auth")


def _requires_auth(route: APIRoute) -> bool:
    """Присутствует ли require_auth в дереве зависимостей роута."""
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is require_auth:
            return True
        stack.extend(dep.dependencies)
    return False


def test_all_api_v1_routes_except_public_require_auth() -> None:
    """Любой /api/v1 роут вне health/auth обязан требовать require_auth."""
    app = create_app()

    unprotected = [
        f"{sorted(route.methods)} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path.startswith("/api/v1/")
        and not route.path.startswith(_PUBLIC_PREFIXES)
        and not _requires_auth(route)
    ]

    assert not unprotected, f"Незащищённые роуты (fail-open): {unprotected}"

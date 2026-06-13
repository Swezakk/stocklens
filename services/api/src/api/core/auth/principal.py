"""Аутентифицированный субъект запроса."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """Верифицированный JWT-субъект, привязанный к запросу через DI."""

    sub: str
    scopes: list[str] = field(default_factory=list)
    claims: dict[str, object] = field(default_factory=dict)

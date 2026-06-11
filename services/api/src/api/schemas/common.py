"""Общие схемы: пагинация и Problem Details."""

from pydantic import BaseModel


class Page[T](BaseModel):
    """Постраничный ответ для списковых эндпоинтов."""

    items: list[T]
    total: int
    limit: int
    offset: int


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details — схема для документирования ошибок в OpenAPI."""

    type: str
    title: str
    status: int
    detail: str
    instance: str

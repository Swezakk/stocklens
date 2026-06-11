"""Переиспользуемые параметры пагинации для всех роутеров."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Query

from api.core.exceptions import InvalidSortFieldError

_MIN_LIMIT = 1
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50


@dataclass
class PageParams:
    """Параметры пагинации: limit зажат в [1, 200], offset >= 0.

    FastAPI читает Query-метаданные из аннотаций в сигнатуре функции-зависимости,
    а не из defaults dataclass-поля — поэтому defaults здесь простые int.
    Зажим limit/offset происходит в __post_init__ независимо от источника значения.
    """

    limit: int = _DEFAULT_LIMIT
    offset: int = 0

    def __post_init__(self) -> None:
        self.limit = max(_MIN_LIMIT, min(self.limit, _MAX_LIMIT))
        self.offset = max(0, self.offset)


def _page_params(
    limit: Annotated[int, Query(ge=_MIN_LIMIT, le=_MAX_LIMIT)] = _DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageParams:
    """FastAPI-зависимость для чтения и валидации параметров пагинации из query string."""
    return PageParams(limit=limit, offset=offset)


PageDep = Annotated[PageParams, Depends(_page_params)]


def validate_sort_field(field: str, allowed: list[str]) -> str:
    """Проверить, что запрошенное поле сортировки входит в белый список.

    Raises:
        InvalidSortFieldError: если поле не разрешено.
    """
    if field not in allowed:
        raise InvalidSortFieldError(field=field, allowed=allowed)
    return field

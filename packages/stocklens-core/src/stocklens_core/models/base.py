"""Базовый класс ORM и вспомогательные фабрики для SQLAlchemy 2.0.

values_callable в str_enum_type критичен: без него SQLAlchemy 2.0 персистирует
имена членов enum («POSITIVE»), а не значения («positive»), и вычисляет длину VARCHAR
по именам, а не по значениям.
"""

from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase

_NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Единый базовый класс для всех ORM-моделей StockLens."""

    metadata = sa.MetaData(naming_convention=_NAMING_CONVENTION)


def str_enum_type(enum_cls: type[StrEnum]) -> sa.Enum:
    """Создать sa.Enum, который хранит строковые значения enum, а не имена членов.

    Аргумент values_callable заставляет SQLAlchemy использовать .value каждого
    члена при создании CHECK-ограничения и определении длины VARCHAR.
    native_enum=False отключает создание PostgreSQL-типа ENUM в пользу VARCHAR.
    """
    return sa.Enum(
        enum_cls,
        native_enum=False,
        values_callable=lambda e: [m.value for m in e],
    )

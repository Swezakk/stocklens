"""Доменные исключения ingestor."""


class SchemaNotReadyError(RuntimeError):
    """Схема БД не готова после исчерпания всех попыток ожидания."""

"""Управление файлом-heartbeat для healthcheck контейнера.

Файл создаётся/обновляется при каждом вызове touch().
Docker healthcheck проверяет, что файл свежее N минут:
  find /tmp/ingestor-heartbeat -mmin -5
"""

from pathlib import Path


def touch(path: Path) -> None:
    """Создать файл heartbeat или обновить его mtime.

    Создаёт родительские директории при необходимости.
    Вызывается как из планировщика каждые 60 секунд,
    так и внутри длинных циклов обхода тикеров.

    Args:
        path: Путь к файлу-сигналу живости.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

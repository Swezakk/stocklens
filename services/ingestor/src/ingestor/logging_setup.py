"""Настройка structlog для ingestor.

JSONRenderer используется в проде (log_pretty=False).
ConsoleRenderer включается при log_pretty=True для локальной разработки.
"""

import structlog


def configure_logging(log_pretty: bool) -> None:
    """Инициализировать structlog с нужным рендерером.

    Args:
        log_pretty: True → ConsoleRenderer (цветной вывод), False → JSONRenderer.
    """
    renderer: structlog.types.Processor
    if log_pretty:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

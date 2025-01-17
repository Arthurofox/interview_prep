import structlog
import logging
from typing import Any
from .config import Settings, EnvironmentType

def setup_logging(settings: Settings) -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if settings.ENVIRONMENT == EnvironmentType.DEVELOPMENT:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO if settings.ENVIRONMENT == EnvironmentType.PRODUCTION else logging.DEBUG
        ),
        cache_logger_on_first_use=True,
    )
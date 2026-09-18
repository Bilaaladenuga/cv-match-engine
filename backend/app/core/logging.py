"""
Structured Logging — Phase 22.

Provides JSON-formatted structured logs with:
- Request ID tracking
- Performance metrics
- Error context
- Structured fields for log aggregation
"""

from __future__ import annotations

import logging
import json
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        req_id = request_id_var.get("")
        if req_id:
            log_entry["request_id"] = req_id

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add performance data if present
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        if hasattr(record, "method"):
            log_entry["method"] = record.method

        if hasattr(record, "path"):
            log_entry["path"] = record.path

        return json.dumps(log_entry, default=str)


class PerformanceFilter(logging.Filter):
    """Filter that adds performance context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        return True


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, use JSON format. If False, use human-readable.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))

    root_logger.addHandler(handler)

    # Add performance filter
    root_logger.addFilter(PerformanceFilter())


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex[:12]


class RequestTimer:
    """Context manager for timing operations."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        logger = logging.getLogger("performance")
        extra = {"duration_ms": round(self.duration_ms, 2), "operation": self.operation}
        if exc_type:
            logger.error(f"{self.operation} failed", extra={"extra_data": extra})
        else:
            logger.info(f"{self.operation} completed", extra={"extra_data": extra})
        return False


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str = "",
) -> None:
    """Log an HTTP request with performance metrics."""
    logger = logging.getLogger("http")
    extra = {
        "extra_data": {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": request_id or request_id_var.get(""),
        }
    }
    if status_code >= 500:
        logger.error(f"{method} {path} -> {status_code}", extra=extra)
    elif status_code >= 400:
        logger.warning(f"{method} {path} -> {status_code}", extra=extra)
    else:
        logger.info(f"{method} {path} -> {status_code}", extra=extra)


def log_ml_event(
    event: str,
    model_version: str = "",
    duration_ms: float = 0,
    success: bool = True,
) -> None:
    """Log an ML pipeline event."""
    logger = logging.getLogger("ml")
    extra = {
        "extra_data": {
            "event": event,
            "model_version": model_version,
            "duration_ms": round(duration_ms, 2),
            "success": success,
        }
    }
    if success:
        logger.info(f"ML: {event}", extra=extra)
    else:
        logger.error(f"ML: {event} failed", extra=extra)

"""
Central, non-blocking logger
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import (
    QueueHandler,
    QueueListener,
    TimedRotatingFileHandler,
)
from queue import Queue
from typing import Any, Dict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # …/app
LOG_DIR = os.path.join(BASE_DIR, "logs")  # …/app/logs
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Logging-tree policy
#   • Root -> WARNING (silence libraries)
#   • Project ("app.*") -> DEBUG (full detail)

ROOT_LOG_LEVEL = logging.WARNING
PROJECT_LOGGER_NAME = "app"


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[1;31m"  # Bold Red


LEVEL_COLORS = {
    logging.DEBUG: Colors.DEBUG,
    logging.INFO: Colors.INFO,
    logging.WARNING: Colors.WARNING,
    logging.ERROR: Colors.ERROR,
    logging.CRITICAL: Colors.CRITICAL,
}


def _rel(path: str) -> str:
    """Return *repo-relative* path or just the basename if outside repo."""
    try:
        p = os.path.relpath(path, BASE_DIR)
        return p if not p.startswith("..") else os.path.basename(path)
    except Exception:
        return os.path.basename(path)


_old_factory = logging.getLogRecordFactory()


def _record_factory(*a, **kw):
    r = _old_factory(*a, **kw)
    r.relativepath = _rel(r.pathname)
    return r


logging.setLogRecordFactory(_record_factory)


class UTCFormatter(logging.Formatter):
    """UTC timestamps - compact but grepable."""

    def __init__(self, fmt_src: str, fmt_no_src: str, colorize: bool = False, **kw):
        super().__init__(fmt_src, **kw)
        self.fmt_src, self.fmt_no_src = fmt_src, fmt_no_src
        self.colorize = colorize

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        # Compact: MM-DD HH:MM:SS.mmm (still grepable by date/hour)
        return dt.strftime("%m-%d %H:%M:%S.") + f"{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        hide = getattr(record, "hide_src", False) or record.name.startswith("uvicorn.")
        orig = self._style._fmt
        self._style._fmt = self.fmt_no_src if hide else self.fmt_src

        if self.colorize:
            color = LEVEL_COLORS.get(record.levelno, "")
            orig_levelname = record.levelname
            record.levelname = f"{color}{record.levelname:<5}{Colors.RESET}"
        try:
            return super().format(record)
        finally:
            self._style._fmt = orig
            if self.colorize:
                record.levelname = orig_levelname


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging - ideal for docker logs + Grafana/Loki."""

    def __init__(self, service: str, **kw):
        super().__init__(**kw)
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "service": self.service,
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "relativepath"):
            log_data["source"] = f"{record.relativepath}:{record.lineno}"
        return json.dumps(log_data, default=str)


# Templates - %-5s fits WARN/ERROR/INFO/DEBUG
FILE_FMT_SRC = "%(asctime)s | %(levelname)-5s | %(relativepath)s:%(lineno)d | %(message)s"
FILE_FMT_NOSRC = "%(asctime)s | %(levelname)-5s | %(message)s"
CON_FMT_SRC = "%(asctime)s | %(levelname)s | %(relativepath)s:%(lineno)d | %(message)s"
CON_FMT_NOSRC = "%(asctime)s | %(levelname)s | %(message)s"

file_formatter = UTCFormatter(FILE_FMT_SRC, FILE_FMT_NOSRC, colorize=False)
console_formatter = UTCFormatter(CON_FMT_SRC, CON_FMT_NOSRC, colorize=True)


file_handler = TimedRotatingFileHandler(
    filename=LOG_FILE,
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
    utc=True,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

log_queue: Queue = Queue(-1)

queue_handler = QueueHandler(log_queue)
queue_listener = QueueListener(
    log_queue,
    file_handler,
    console_handler,
    respect_handler_level=True,
)
queue_listener.start()
atexit.register(queue_listener.stop)

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(ROOT_LOG_LEVEL)

project_logger = logging.getLogger(PROJECT_LOGGER_NAME)
project_logger.handlers.clear()
project_logger.setLevel(logging.DEBUG)
project_logger.addHandler(queue_handler)
project_logger.propagate = False

for name in ("uvicorn", "uvicorn.error"):
    uv = logging.getLogger(name)
    uv.handlers.clear()
    uv.addHandler(queue_handler)
    uv.setLevel(logging.INFO)
    uv.propagate = False

logging.getLogger("uvicorn.access").handlers.clear()
logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.access").propagate = False

logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.CRITICAL)

logger = project_logger


# ─── Service Logger Factory ──────────────────────────────────────────────────
_service_loggers: Dict[str, logging.Logger] = {}


def get_service_logger(service_name: str) -> logging.Logger:
    """
    Factory for structured JSON service loggers.

    Usage:
        from app.logger import get_service_logger
        logger = get_service_logger("my_service")

    Creates:
        - Separate log file: logs/{service_name}.log
        - JSON formatted output for docker logs filtering
        - Non-blocking via QueueHandler
    """
    if service_name in _service_loggers:
        return _service_loggers[service_name]

    log_file = os.path.join(LOG_DIR, f"{service_name}.log")
    json_formatter = StructuredJSONFormatter(service=service_name)

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(json_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(json_formatter)

    queue: Queue = Queue(-1)
    queue_handler = QueueHandler(queue)
    listener = QueueListener(queue, file_handler, console_handler, respect_handler_level=True)
    listener.start()
    atexit.register(listener.stop)

    svc_logger = logging.getLogger(service_name)
    svc_logger.handlers.clear()
    svc_logger.setLevel(logging.DEBUG)
    svc_logger.addHandler(queue_handler)
    svc_logger.propagate = False

    _service_loggers[service_name] = svc_logger
    return svc_logger


# Pre-configured service loggers (for convenience)
interview_monitor_logger = get_service_logger("interview_monitor")

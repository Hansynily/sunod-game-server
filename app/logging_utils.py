from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


AUDIT_LOGGER_NAME = "sunod.audit"


def build_log_config(log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = (log_dir / "server.log").resolve()
    audit_log = (log_dir / "audit.log").resolve()

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "server_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(server_log),
                "maxBytes": 2_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(audit_log),
                "maxBytes": 2_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "server_file"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "server_file"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "server_file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "server_file"],
                "propagate": False,
            },
            AUDIT_LOGGER_NAME: {
                "level": "INFO",
                "handlers": ["console", "server_file", "audit_file"],
                "propagate": False,
            },
        },
    }


def get_audit_logger() -> logging.Logger:
    return logging.getLogger(AUDIT_LOGGER_NAME)


def audit_log(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        **fields,
    }
    get_audit_logger().info(json.dumps(payload, default=_json_default, sort_keys=True))


def _json_default(value: Any) -> str:
    return str(value)

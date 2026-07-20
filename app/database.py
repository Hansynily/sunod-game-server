import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from .repository import TelemetryRepository

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)

_client: MongoClient | None = None


@dataclass(frozen=True, slots=True)
class MongoSettings:
    uri: str
    database: str
    timeout_ms: int


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_mongo_settings() -> MongoSettings:
    timeout_raw = _require_env("MONGODB_TIMEOUT_MS")
    try:
        timeout_ms = int(timeout_raw)
    except ValueError as exc:
        raise RuntimeError("MONGODB_TIMEOUT_MS must be an integer.") from exc

    return MongoSettings(
        uri=_require_env("MONGODB_URI"),
        database=_require_env("MONGODB_DB"),
        timeout_ms=timeout_ms,
    )


def get_client() -> MongoClient:
    global _client

    if _client is None:
        settings = get_mongo_settings()
        _client = MongoClient(
            settings.uri,
            serverSelectionTimeoutMS=settings.timeout_ms,
            tz_aware=False,
        )

    return _client


def get_database() -> Database:
    return get_client()[get_mongo_settings().database]


def init_db() -> None:
    repository = TelemetryRepository(get_database())
    try:
        # Ping first: a cheap connectivity probe so an unreachable MongoDB fails
        # with one clear message instead of a traceback out of create_index.
        repository.ping()
        repository.ensure_indexes()
    except PyMongoError as exc:
        settings = get_mongo_settings()
        raise SystemExit(
            f"Cannot reach MongoDB at '{settings.uri}' (database '{settings.database}'): {exc}.\n"
            "Is MongoDB running? Start it first, then relaunch the server."
        ) from exc


def close_db() -> None:
    global _client

    if _client is not None:
        _client.close()
        _client = None


def get_db() -> Generator[TelemetryRepository, None, None]:
    yield TelemetryRepository(get_database())

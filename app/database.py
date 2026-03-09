import os
from typing import Generator

from pymongo import MongoClient
from pymongo.database import Database

from .repository import TelemetryRepository


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "telemetry_db")
MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "3000"))

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client

    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
            tz_aware=False,
        )

    return _client


def get_database() -> Database:
    return get_client()[MONGODB_DB]


def init_db() -> None:
    repository = TelemetryRepository(get_database())
    repository.ensure_indexes()
    repository.ping()


def close_db() -> None:
    global _client

    if _client is not None:
        _client.close()
        _client = None


def get_db() -> Generator[TelemetryRepository, None, None]:
    yield TelemetryRepository(get_database())

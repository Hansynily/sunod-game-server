from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import joblib
import numpy as np

from app.feature_pipeline import DIMENSIONS, FEATURE_SCHEMA


LOGGER = logging.getLogger(__name__)
ML_DIR = Path(__file__).resolve().parents[1] / "ml"
MODEL_PATH = ML_DIR / "model.joblib"
FEATURE_SCHEMA_PATH = ML_DIR / "feature_schema.json"
MODEL_VERSION_PATH = ML_DIR / "model_version.txt"


@dataclass(frozen=True, slots=True)
class RuntimeModelStatus:
    available: bool
    model: Any | None = None
    model_version: str | None = None
    reason: str | None = None


_status_lock = Lock()
_cached_status: RuntimeModelStatus | None = None


def warm_load_model(
    *,
    force_reload: bool = False,
    logger: logging.Logger | None = None,
) -> RuntimeModelStatus:
    global _cached_status

    active_logger = logger or LOGGER
    if _cached_status is not None and not force_reload:
        return _cached_status

    with _status_lock:
        if _cached_status is not None and not force_reload:
            return _cached_status

        status = _load_runtime_model()
        _cached_status = status

    if status.available:
        active_logger.info(
            "Runtime model ready: version=%s path=%s",
            status.model_version,
            MODEL_PATH,
        )
    else:
        active_logger.warning("Runtime model unavailable: %s", status.reason)

    return status


def get_model_status() -> RuntimeModelStatus:
    return warm_load_model()


def reset_runtime_model_cache() -> None:
    global _cached_status

    with _status_lock:
        _cached_status = None


def predict_scores(
    feature_vector: Sequence[float],
    status: RuntimeModelStatus | None = None,
) -> dict[str, float]:
    runtime_status = status or get_model_status()
    if not runtime_status.available or runtime_status.model is None:
        raise RuntimeError(runtime_status.reason or "Runtime model is unavailable.")

    values = [float(value) for value in feature_vector]
    if len(values) != len(FEATURE_SCHEMA):
        raise ValueError(
            f"Expected {len(FEATURE_SCHEMA)} runtime features, received {len(values)}."
        )

    features = np.asarray([values], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        raw_predictions = runtime_status.model.predict(features)

    predictions = np.asarray(raw_predictions, dtype=float)
    if predictions.shape != (1, len(DIMENSIONS)):
        raise RuntimeError(
            "Runtime model returned an unexpected prediction shape: "
            f"{predictions.shape!r}."
        )

    return {
        dimension: float(predictions[0, index])
        for index, dimension in enumerate(DIMENSIONS)
    }


def _load_runtime_model() -> RuntimeModelStatus:
    missing_paths = [
        str(path)
        for path in (MODEL_PATH, FEATURE_SCHEMA_PATH, MODEL_VERSION_PATH)
        if not path.exists()
    ]
    if missing_paths:
        return RuntimeModelStatus(
            available=False,
            reason="Missing runtime model artifacts: " + ", ".join(missing_paths),
        )

    try:
        saved_feature_schema = _load_feature_schema(FEATURE_SCHEMA_PATH)
    except Exception as exc:
        return RuntimeModelStatus(
            available=False,
            reason=f"Failed to read feature_schema.json: {exc}",
        )

    if saved_feature_schema != FEATURE_SCHEMA:
        return RuntimeModelStatus(
            available=False,
            reason=(
                "Saved feature_schema.json does not match "
                "app.feature_pipeline.FEATURE_SCHEMA."
            ),
        )

    try:
        model_version = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return RuntimeModelStatus(
            available=False,
            reason=f"Failed to read model_version.txt: {exc}",
        )

    if not model_version:
        return RuntimeModelStatus(
            available=False,
            reason="model_version.txt is empty.",
        )

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        return RuntimeModelStatus(
            available=False,
            reason=f"Failed to load model.joblib: {exc}",
        )

    if not hasattr(model, "predict"):
        return RuntimeModelStatus(
            available=False,
            reason="Loaded runtime model does not expose a predict() method.",
        )

    return RuntimeModelStatus(
        available=True,
        model=model,
        model_version=model_version,
    )


def _load_feature_schema(path: Path) -> tuple[str, ...]:
    raw_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, list) or any(
        not isinstance(feature_name, str) for feature_name in raw_value
    ):
        raise ValueError("feature_schema.json must contain a JSON array of strings.")

    return tuple(raw_value)

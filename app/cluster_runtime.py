from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ARTIFACTS_DIR = BACKEND_ROOT / "model_assets" / "runtime"
BUNDLE_ROOT = BACKEND_ROOT / "model_assets" / "bundles"
CLUSTER_MODEL_PATH = RUNTIME_ARTIFACTS_DIR / "riasec_cluster_model.pkl"
CAREER_MODEL_DIR = RUNTIME_ARTIFACTS_DIR
EXPECTED_CLUSTER_IDS = tuple(range(8))
EXPECTED_FEATURE_NAMES = tuple(
    f"{dimension}{slot}"
    for dimension in ("R", "I", "A", "S", "E", "C")
    for slot in range(1, 9)
)


@dataclass(frozen=True, slots=True)
class ClusterModelStatus:
    available: bool
    model: Any | None = None
    model_path: str | None = None
    classes: tuple[int, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CareerModelStatus:
    available: bool
    models: dict[int, Any] | None = None
    model_paths: dict[int, str] | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SavedBundleStatus:
    available: bool
    bundle_key: str | None = None
    bundle_path: str | None = None
    model_version: str | None = None
    cluster_model: Any | None = None
    career_models: dict[int, Any] | None = None
    cluster_model_path: str | None = None
    career_model_paths: dict[int, str] | None = None
    reason: str | None = None


_cluster_status_lock = Lock()
_cached_cluster_status: ClusterModelStatus | None = None
_career_status_lock = Lock()
_cached_career_status: CareerModelStatus | None = None


def warm_load_cluster_model(
    *,
    force_reload: bool = False,
    logger: logging.Logger | None = None,
) -> ClusterModelStatus:
    global _cached_cluster_status

    active_logger = logger or LOGGER
    if _cached_cluster_status is not None and not force_reload:
        return _cached_cluster_status

    with _cluster_status_lock:
        if _cached_cluster_status is not None and not force_reload:
            return _cached_cluster_status

        status = _load_cluster_model()
        _cached_cluster_status = status

    if status.available:
        active_logger.info("Cluster model ready: path=%s", status.model_path)
    else:
        active_logger.warning("Cluster model unavailable: %s", status.reason)

    return status


def get_cluster_model_status() -> ClusterModelStatus:
    return warm_load_cluster_model()


def reset_cluster_model_cache() -> None:
    global _cached_cluster_status

    with _cluster_status_lock:
        _cached_cluster_status = None


def warm_load_career_models(
    *,
    force_reload: bool = False,
    logger: logging.Logger | None = None,
    cluster_status: ClusterModelStatus | None = None,
) -> CareerModelStatus:
    global _cached_career_status

    active_logger = logger or LOGGER
    if _cached_career_status is not None and not force_reload:
        return _cached_career_status

    with _career_status_lock:
        if _cached_career_status is not None and not force_reload:
            return _cached_career_status

        status = _load_career_models(cluster_status=cluster_status)
        _cached_career_status = status

    if status.available:
        active_logger.info(
            "Career model bundle ready: clusters=%s",
            ", ".join(str(cluster_id) for cluster_id in sorted((status.models or {}).keys())),
        )
    else:
        active_logger.warning("Career model bundle unavailable: %s", status.reason)

    return status


def get_career_model_status() -> CareerModelStatus:
    return warm_load_career_models()


def reset_career_model_cache() -> None:
    global _cached_career_status

    with _career_status_lock:
        _cached_career_status = None


def predict_cluster(
    feature_vector: Sequence[float],
    status: ClusterModelStatus | None = None,
) -> int:
    cluster_status = status or get_cluster_model_status()
    if not cluster_status.available or cluster_status.model is None:
        raise RuntimeError(cluster_status.reason or "Cluster model is unavailable.")

    values = [float(value) for value in feature_vector]
    if len(values) != len(EXPECTED_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(EXPECTED_FEATURE_NAMES)} features, received {len(values)}."
        )

    feature_names = getattr(cluster_status.model, "feature_names_in_", None)
    features = _build_prediction_input(values, feature_names)
    raw_predictions = cluster_status.model.predict(features)
    predictions = np.asarray(raw_predictions)
    if predictions.shape[0] != 1:
        raise RuntimeError(
            "Cluster model returned an unexpected prediction shape: "
            f"{predictions.shape!r}."
        )

    return int(predictions[0])


def predict_career_cluster(
    feature_vector: Sequence[float],
    cluster_id: int,
    status: CareerModelStatus | None = None,
) -> int:
    career_status = status or get_career_model_status()
    if not career_status.available or not career_status.models:
        raise RuntimeError(career_status.reason or "Career models are unavailable.")

    model = career_status.models.get(int(cluster_id))
    if model is None:
        raise RuntimeError(f"Career model for cluster {cluster_id} is unavailable.")

    values = [float(value) for value in feature_vector]
    if len(values) != len(EXPECTED_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(EXPECTED_FEATURE_NAMES)} features, received {len(values)}."
        )

    feature_names = getattr(model, "feature_names_in_", None)
    features = _build_prediction_input(values, feature_names)
    raw_predictions = model.predict(features)
    predictions = np.asarray(raw_predictions)
    if predictions.shape[0] != 1:
        raise RuntimeError(
            "Career model returned an unexpected prediction shape: "
            f"{predictions.shape!r}."
        )

    return int(predictions[0])


# Holland code of each NAMED career cluster (outlier clusters 1 and 4 are excluded
# on purpose — a real player profile should never be routed into an outlier bucket).
CLUSTER_HOLLAND_CODES = {
    0: "IA",   # IA Research
    2: "RIC",  # RIC Engineering
    3: "AS",   # AS Art
    5: "SEC",  # SEC Business
    6: "S",    # S Social Services
    7: "SIA",  # SIA Healthcare
}
_RIASEC_ORDER = ("R", "I", "A", "S", "E", "C")


def holland_match_cluster(riasec_scores: dict[str, float]) -> int:
    """Pick the named cluster whose Holland code best matches the player's RIASEC
    profile (highest average score across the code's letters). Deterministic and
    traceable — this is what makes the career visibly follow the player's RIASEC."""
    scores = {str(k).upper(): float(v) for k, v in riasec_scores.items()}
    best_cluster = None
    best_score = float("-inf")
    for cluster_id, code in CLUSTER_HOLLAND_CODES.items():
        avg = sum(scores.get(letter, 0.0) for letter in code) / len(code)
        if avg > best_score:
            best_score = avg
            best_cluster = cluster_id
    return int(best_cluster)


def build_feature_vector_from_scores(
    riasec_scores: dict[str, float],
    peak: float = 5.0,
    base: float = 2.0,
) -> list[float]:
    """Map the 6-dim RIASEC profile onto the 48-feature questionnaire layout so the
    career sub-model sees an in-distribution vector. Each dimension's 8 slots are set
    to base + (peak-base) * (score / max_score)."""
    scores = {str(k).upper(): float(v) for k, v in riasec_scores.items()}
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0.0:
        max_score = 1.0
    vector: list[float] = []
    for letter in _RIASEC_ORDER:
        normalized = scores.get(letter, 0.0) / max_score
        vector.extend([base + (peak - base) * normalized] * 8)
    return vector


def _load_cluster_model() -> ClusterModelStatus:
    model_path = _get_cluster_model_path()
    if not model_path.exists():
        return ClusterModelStatus(
            available=False,
            reason="Cluster model artifact is missing.",
            model_path=str(model_path),
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        return ClusterModelStatus(
            available=False,
            reason="Cluster model could not be loaded.",
            model_path=str(model_path),
        )

    if not hasattr(model, "predict"):
        return ClusterModelStatus(
            available=False,
            reason="Loaded cluster model does not expose a predict() method.",
            model_path=str(model_path),
        )

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None or len(feature_names) != len(EXPECTED_FEATURE_NAMES):
        return ClusterModelStatus(
            available=False,
            reason=(
                "Loaded cluster model does not expose the expected 48 feature names."
            ),
            model_path=str(model_path),
        )

    classes_attr = getattr(model, "classes_", None)
    if classes_attr is None:
        classes = ()
    else:
        try:
            classes = tuple(sorted(int(value) for value in classes_attr))
        except Exception:
            classes = ()

    if classes != EXPECTED_CLUSTER_IDS:
        return ClusterModelStatus(
            available=False,
            reason=(
                "Loaded cluster model must predict cluster ids 0 through 7, "
                f"but found {classes!r}."
            ),
            model_path=str(model_path),
            classes=classes,
        )

    return ClusterModelStatus(
        available=True,
        model=model,
        model_path=str(model_path),
        classes=classes,
    )


def _load_career_models(
    *,
    cluster_status: ClusterModelStatus | None = None,
) -> CareerModelStatus:
    resolved_cluster_status = cluster_status or get_cluster_model_status()
    if not resolved_cluster_status.available or resolved_cluster_status.model is None:
        return CareerModelStatus(
            available=False,
            reason=resolved_cluster_status.reason or "Cluster model is unavailable.",
        )

    expected_cluster_ids = (
        resolved_cluster_status.classes
        if resolved_cluster_status.classes
        else EXPECTED_CLUSTER_IDS
    )

    models: dict[int, Any] = {}
    model_paths: dict[int, str] = {}
    missing_paths: list[str] = []

    for cluster_id in expected_cluster_ids:
        model_path = _get_career_model_path(cluster_id)
        if not model_path.exists():
            missing_paths.append(str(model_path))
            continue

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            return CareerModelStatus(
                available=False,
                reason=f"Career model for cluster {cluster_id} could not be loaded.",
                model_paths=model_paths,
            )

        if not hasattr(model, "predict"):
            return CareerModelStatus(
                available=False,
                reason=(
                    "Loaded career model does not expose a predict() method "
                    f"for cluster {cluster_id}."
                ),
                model_paths=model_paths,
            )

        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None or len(feature_names) != len(EXPECTED_FEATURE_NAMES):
            return CareerModelStatus(
                available=False,
                reason=(
                    "Loaded career model does not expose the expected 48 feature names "
                    f"for cluster {cluster_id}."
                ),
                model_paths=model_paths,
            )

        models[int(cluster_id)] = model
        model_paths[int(cluster_id)] = str(model_path)

    if missing_paths:
        return CareerModelStatus(
            available=False,
            reason="One or more career model artifacts are missing.",
            model_paths=model_paths,
        )

    return CareerModelStatus(
        available=True,
        models=models,
        model_paths=model_paths,
    )


def _build_prediction_input(
    values: Sequence[float],
    feature_names: Sequence[str] | None,
):
    if feature_names is None:
        return np.asarray([values], dtype=float)

    columns = [str(name) for name in feature_names]
    if len(columns) != len(values):
        return np.asarray([values], dtype=float)

    return pd.DataFrame([values], columns=columns, dtype=float)


def _get_cluster_model_path() -> Path:
    configured_path = (os.getenv("SUNOD_CLUSTER_MODEL_PATH") or "").strip()
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return CLUSTER_MODEL_PATH


def _get_career_model_path(cluster_id: int) -> Path:
    configured_dir = (os.getenv("SUNOD_CAREER_MODEL_DIR") or "").strip()
    if configured_dir:
        career_dir = Path(configured_dir).expanduser().resolve()
    else:
        career_dir = CAREER_MODEL_DIR

    return career_dir / f"career_predictor_model_cluster_{int(cluster_id)}.pkl"


def inspect_saved_bundle(bundle_key_or_path: str | Path) -> SavedBundleStatus:
    bundle_path = _resolve_bundle_path(bundle_key_or_path)
    if not bundle_path.exists():
        return SavedBundleStatus(
            available=False,
            bundle_key=bundle_path.name,
            bundle_path=str(bundle_path),
            reason="Saved bundle snapshot is missing.",
        )

    model_version_path = bundle_path / "model_version.txt"
    model_version = None
    if model_version_path.exists():
        try:
            model_version = model_version_path.read_text(encoding="utf-8").strip() or None
        except Exception:
            model_version = None

    cluster_model_path = bundle_path / "riasec_cluster_model.pkl"
    career_model_paths = {
        cluster_id: str(bundle_path / f"career_predictor_model_cluster_{cluster_id}.pkl")
        for cluster_id in EXPECTED_CLUSTER_IDS
    }
    missing_paths = [
        str(path)
        for path in [cluster_model_path, *(bundle_path / f"career_predictor_model_cluster_{cluster_id}.pkl" for cluster_id in EXPECTED_CLUSTER_IDS)]
        if not path.exists()
    ]
    if missing_paths:
        return SavedBundleStatus(
            available=False,
            bundle_key=bundle_path.name,
            bundle_path=str(bundle_path),
            model_version=model_version,
            cluster_model_path=str(cluster_model_path),
            career_model_paths=career_model_paths,
            reason="Saved bundle snapshot is incomplete.",
        )

    return SavedBundleStatus(
        available=True,
        bundle_key=bundle_path.name,
        bundle_path=str(bundle_path),
        model_version=model_version,
        cluster_model_path=str(cluster_model_path),
        career_model_paths=career_model_paths,
    )


def load_saved_bundle_models(bundle_key_or_path: str | Path) -> SavedBundleStatus:
    status = inspect_saved_bundle(bundle_key_or_path)
    if not status.available:
        return status

    assert status.cluster_model_path is not None
    assert status.career_model_paths is not None

    try:
        cluster_model = joblib.load(status.cluster_model_path)
    except Exception as exc:
        return SavedBundleStatus(
            available=False,
            bundle_key=status.bundle_key,
            bundle_path=status.bundle_path,
            model_version=status.model_version,
            cluster_model_path=status.cluster_model_path,
            career_model_paths=status.career_model_paths,
            reason=f"Saved bundle cluster model could not be loaded: {exc}",
        )

    career_models: dict[int, Any] = {}
    for cluster_id, model_path in status.career_model_paths.items():
        try:
            career_models[int(cluster_id)] = joblib.load(model_path)
        except Exception as exc:
            return SavedBundleStatus(
                available=False,
                bundle_key=status.bundle_key,
                bundle_path=status.bundle_path,
                model_version=status.model_version,
                cluster_model=cluster_model,
                career_models=career_models,
                cluster_model_path=status.cluster_model_path,
                career_model_paths=status.career_model_paths,
                reason=f"Saved bundle career model for cluster {cluster_id} could not be loaded: {exc}",
            )

    return SavedBundleStatus(
        available=True,
        bundle_key=status.bundle_key,
        bundle_path=status.bundle_path,
        model_version=status.model_version,
        cluster_model=cluster_model,
        career_models=career_models,
        cluster_model_path=status.cluster_model_path,
        career_model_paths=status.career_model_paths,
    )


def save_runtime_bundle_snapshot(
    bundle_key_or_path: str | Path,
    *,
    model_version: str | None = None,
    cluster_model_path: str | Path | None = None,
    career_model_dir: str | Path | None = None,
) -> SavedBundleStatus:
    target_path = _resolve_bundle_path(bundle_key_or_path)
    target_path.mkdir(parents=True, exist_ok=True)

    source_cluster_model_path = Path(cluster_model_path) if cluster_model_path else CLUSTER_MODEL_PATH
    source_career_model_dir = Path(career_model_dir) if career_model_dir else CAREER_MODEL_DIR

    if not source_cluster_model_path.exists():
        return SavedBundleStatus(
            available=False,
            bundle_key=target_path.name,
            bundle_path=str(target_path),
            reason="Cluster model artifact is missing from the active runtime.",
        )

    copied_cluster_model_path = target_path / "riasec_cluster_model.pkl"
    shutil.copy2(source_cluster_model_path, copied_cluster_model_path)

    career_model_paths: dict[int, str] = {}
    for cluster_id in EXPECTED_CLUSTER_IDS:
        source_model_path = source_career_model_dir / f"career_predictor_model_cluster_{cluster_id}.pkl"
        if not source_model_path.exists():
            return SavedBundleStatus(
                available=False,
                bundle_key=target_path.name,
                bundle_path=str(target_path),
                model_version=model_version,
                cluster_model_path=str(copied_cluster_model_path),
                career_model_paths=career_model_paths,
                reason="One or more career model artifacts are missing from the active runtime.",
            )

        target_model_path = target_path / source_model_path.name
        shutil.copy2(source_model_path, target_model_path)
        career_model_paths[int(cluster_id)] = str(target_model_path)

    source_feature_schema_path = source_career_model_dir / "feature_schema.json"
    if source_feature_schema_path.exists():
        shutil.copy2(source_feature_schema_path, target_path / source_feature_schema_path.name)

    if model_version is not None:
        (target_path / "model_version.txt").write_text(f"{model_version}\n", encoding="utf-8")

    return inspect_saved_bundle(target_path)


def _resolve_bundle_path(bundle_key_or_path: str | Path) -> Path:
    raw_value = Path(bundle_key_or_path)
    if raw_value.is_absolute():
        return raw_value.resolve()

    normalized = str(bundle_key_or_path).strip().replace("\\", "/")
    if normalized.startswith("model_assets/") or normalized.startswith("bundles/"):
        return (BACKEND_ROOT / normalized).resolve()

    if normalized:
        return (BUNDLE_ROOT / normalized).resolve()

    return BUNDLE_ROOT.resolve()

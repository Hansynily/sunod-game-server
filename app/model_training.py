from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app import cluster_runtime


DATASET_STORAGE_DIR = cluster_runtime.CAREER_MODEL_DIR / "datasets"
LIVE_FEATURE_NAMES = cluster_runtime.EXPECTED_FEATURE_NAMES
LIVE_CLUSTER_IDS = tuple(cluster_runtime.EXPECTED_CLUSTER_IDS)
CAREER_CLUSTER_COUNTS = (8, 9, 8, 1, 9, 1, 8, 8)
MINIMUM_TRAINING_ROWS = 8
MINIMUM_VALIDATION_ROWS = 10


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    dataset_name: str
    original_filename: str
    stored_filename: str
    storage_path: str
    file_size_bytes: int
    row_count: int
    feature_count: int
    label_count: int


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    model_version: str
    bundle_key: str
    bundle_path: str
    bundle_ready: bool
    row_count: int
    training_row_count: int
    validation_row_count: int
    cluster_accuracy: float | None
    career_model_count: int
    clusters_covered_count: int
    trained_at: datetime


def validate_dataset_bytes(
    raw_bytes: bytes,
    *,
    dataset_name: str,
    original_filename: str,
) -> DatasetValidationSummary:
    frame = _read_dataset_frame(raw_bytes)
    _validate_dataset_frame(frame)

    normalized_name = _normalize_dataset_name(dataset_name, original_filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    stored_filename = f"{timestamp}-{_slugify(normalized_name)}.csv"
    storage_path = DATASET_STORAGE_DIR / stored_filename

    return DatasetValidationSummary(
        dataset_name=normalized_name,
        original_filename=original_filename,
        stored_filename=stored_filename,
        storage_path=str(storage_path),
        file_size_bytes=len(raw_bytes),
        row_count=len(frame),
        feature_count=len(LIVE_FEATURE_NAMES),
        label_count=1,
    )


def save_uploaded_dataset(
    raw_bytes: bytes,
    *,
    dataset_name: str,
    original_filename: str,
) -> DatasetValidationSummary:
    summary = validate_dataset_bytes(
        raw_bytes,
        dataset_name=dataset_name,
        original_filename=original_filename,
    )
    storage_path = Path(summary.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(raw_bytes)
    return summary


def train_runtime_model_from_dataset(
    dataset_path: str | Path,
    *,
    dataset_id: int,
    dataset_name: str,
) -> TrainingSummary:
    source_path = Path(dataset_path)
    if not source_path.exists():
        raise ValueError("Dataset file was not found on the server.")

    frame = pd.read_csv(source_path)
    validated_frame = _validate_dataset_frame(frame)

    features = validated_frame[list(LIVE_FEATURE_NAMES)]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=len(LIVE_CLUSTER_IDS),
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(scaled_features)
    present_clusters = tuple(sorted({int(value) for value in cluster_labels}))
    if present_clusters != LIVE_CLUSTER_IDS:
        raise ValueError("Training data must cover all 8 clusters.")

    cluster_feature_frame = features.copy()
    cluster_feature_frame["cluster"] = cluster_labels

    cluster_accuracy: float | None = None
    training_row_count = len(cluster_feature_frame)
    validation_row_count = 0
    cluster_target = pd.Series(cluster_labels, index=features.index, name="cluster")

    if len(cluster_feature_frame) >= MINIMUM_VALIDATION_ROWS:
        features_train, features_val, labels_train, labels_val = train_test_split(
            features,
            cluster_target,
            test_size=0.2,
            random_state=42,
        )
        cluster_eval_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        cluster_eval_model.fit(features_train, labels_train)
        cluster_predictions = cluster_eval_model.predict(features_val)
        validation_row_count = len(features_val)
        training_row_count = len(features_train)
        cluster_accuracy = float(accuracy_score(labels_val, cluster_predictions))

    cluster_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    cluster_model.fit(features, cluster_target)

    trained_at = datetime.utcnow()
    model_version = _build_model_version(dataset_name, trained_at)
    bundle_key = _build_bundle_key(dataset_id, model_version)

    cluster_runtime.CLUSTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cluster_runtime.CAREER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cluster_model, cluster_runtime.CLUSTER_MODEL_PATH)

    career_model_count = 0
    for cluster_id, requested_career_clusters in enumerate(CAREER_CLUSTER_COUNTS):
        cluster_subset = cluster_feature_frame[cluster_feature_frame["cluster"] == cluster_id]
        if cluster_subset.empty:
            raise ValueError("Training data must cover all 8 clusters.")

        career_cluster_count = min(int(requested_career_clusters), len(cluster_subset))
        subset_features = cluster_subset[list(LIVE_FEATURE_NAMES)]
        career_scaler = StandardScaler()
        subset_scaled = career_scaler.fit_transform(subset_features)
        career_kmeans = KMeans(
            n_clusters=career_cluster_count,
            random_state=42,
            n_init=10,
        )
        career_labels = career_kmeans.fit_predict(subset_scaled)

        career_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        career_model.fit(subset_features, career_labels)
        model_path = cluster_runtime.CAREER_MODEL_DIR / f"career_predictor_model_cluster_{cluster_id}.pkl"
        joblib.dump(career_model, model_path)
        career_model_count += 1

    cluster_runtime.reset_cluster_model_cache()
    cluster_runtime.reset_career_model_cache()
    loaded_cluster_status = cluster_runtime.warm_load_cluster_model(force_reload=True)
    loaded_career_status = cluster_runtime.warm_load_career_models(force_reload=True)
    if not loaded_cluster_status.available:
        raise RuntimeError(loaded_cluster_status.reason or "Cluster model could not be reloaded.")
    if not loaded_career_status.available:
        raise RuntimeError(loaded_career_status.reason or "Career model bundle could not be reloaded.")

    bundle_status = cluster_runtime.save_runtime_bundle_snapshot(
        bundle_key,
        model_version=model_version,
        cluster_model_path=cluster_runtime.CLUSTER_MODEL_PATH,
        career_model_dir=cluster_runtime.CAREER_MODEL_DIR,
    )
    if not bundle_status.available:
        raise RuntimeError(bundle_status.reason or "Bundle snapshot could not be saved.")

    return TrainingSummary(
        model_version=model_version,
        bundle_key=bundle_status.bundle_key or bundle_key,
        bundle_path=bundle_status.bundle_path or str(cluster_runtime.BUNDLE_ROOT / bundle_key),
        bundle_ready=bool(bundle_status.available),
        row_count=len(validated_frame),
        training_row_count=training_row_count,
        validation_row_count=validation_row_count,
        cluster_accuracy=cluster_accuracy,
        career_model_count=career_model_count,
        clusters_covered_count=len(present_clusters),
        trained_at=trained_at,
    )


def _read_dataset_frame(raw_bytes: bytes) -> pd.DataFrame:
    if not raw_bytes:
        raise ValueError("Choose a CSV file to upload.")

    try:
        return pd.read_csv(BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError("Dataset file must be a valid CSV.") from exc


def _validate_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Dataset file is empty.")

    required_columns = list(LIVE_FEATURE_NAMES) + ["major"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    numeric_frame = frame[list(LIVE_FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    invalid_columns = [column for column in LIVE_FEATURE_NAMES if numeric_frame[column].isna().any()]
    if invalid_columns:
        raise ValueError(
            "Dataset columns must be numeric with no blank values: "
            + ", ".join(invalid_columns)
        )

    major_values = frame["major"].astype("string").fillna("").str.strip()
    if major_values.eq("").any():
        raise ValueError("major column must not be blank.")

    numeric_frame["major"] = major_values

    if len(numeric_frame) < MINIMUM_TRAINING_ROWS:
        raise ValueError(
            f"Dataset needs at least {MINIMUM_TRAINING_ROWS} rows to train a model."
        )

    return numeric_frame


def _normalize_dataset_name(dataset_name: str, original_filename: str) -> str:
    explicit_name = (dataset_name or "").strip()
    if explicit_name:
        return explicit_name

    fallback = Path(original_filename or "dataset").stem.strip()
    return fallback or "dataset"


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "dataset"


def _build_model_version(dataset_name: str, trained_at: datetime) -> str:
    return f"{_slugify(dataset_name)}-{trained_at.strftime('%Y%m%d%H%M%S')}"


def _build_bundle_key(dataset_id: int, model_version: str) -> str:
    return f"dataset-{int(dataset_id):04d}-{model_version}"

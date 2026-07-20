from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil

import joblib
import numpy as np
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
# Sub-cluster count per cluster id. Clusters 1 and 4 are the outlier buckets (one
# sub-cluster; no careers in career_map.json). This mirrors the corrected offline
# training array in sunod-model/kmeans/TRAININGkmeansclustering.py. The pre-2026-07
# value (8, 9, 8, 1, 9, 1, 8, 8) put the outliers at ids 3 and 5, producing the
# single-class career models that fix_degenerate_subclusters.py had to patch.
CAREER_CLUSTER_COUNTS = (8, 1, 8, 8, 1, 8, 8, 8)
NAMED_CLUSTER_IDS = tuple(sorted(cluster_runtime.CLUSTER_HOLLAND_CODES.keys()))
OUTLIER_CLUSTER_IDS = tuple(sorted(set(LIVE_CLUSTER_IDS) - set(NAMED_CLUSTER_IDS)))
# Minimum required gap (Euclidean distance, raw 1-5 feature space) between a new
# cluster's best centroid match and its second-best - see _identify_canonical_cluster_mapping.
# Empirically: every legitimate re-run (same data reshuffled, or a fresh KMeans refit with
# a different random_state) produced a worst-case margin of ~5.1-6.0; adversarial data
# (pure noise, or an artificially degenerate 3-group dataset) produced margins under 2,
# often negative. 2.0 sits well inside that gap on both sides.
MIN_CENTROID_MATCH_MARGIN = 2.0
# 8 rows can produce 8 singleton clusters that pass every check while yielding fully
# degenerate single-row career models. Require enough rows that each cluster can hold
# its sub-clusters with real support.
MINIMUM_TRAINING_ROWS = 400
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


def _load_canonical_centroids() -> dict[int, np.ndarray]:
    """The frozen reference centroids career_map.json/CLUSTER_HOLLAND_CODES were authored
    against - see cluster_runtime.CANONICAL_CENTROIDS_PATH. Regenerate this file only when
    the frozen partition is deliberately changed:

        import json, pandas as pd
        df = pd.read_csv("sunod-model/kmeansdata/riasec_clusters_big.csv")
        cols = [f"{l}{i}" for l in "RIASEC" for i in range(1, 9)]
        centroids = {str(c): df[df.cluster == c][cols].mean().round(6).tolist() for c in range(8)}
        json.dump({"feature_order": cols, "centroids": centroids}, open(path, "w"), indent=2)
    """
    path = cluster_runtime.CANONICAL_CENTROIDS_PATH
    if not path.exists():
        raise RuntimeError(
            f"Canonical cluster centroids reference is missing ({path}). "
            "Cannot safely re-identify cluster identity for a retrain without it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_order = payload["feature_order"]
    if tuple(feature_order) != LIVE_FEATURE_NAMES:
        raise RuntimeError(
            "Canonical cluster centroids reference has a different feature order than "
            "the live model expects - it is stale or corrupted. Regenerate it."
        )
    return {int(cid): np.asarray(vector, dtype=float) for cid, vector in payload["centroids"].items()}


def _compute_cluster_centroids(features: pd.DataFrame, labels: np.ndarray) -> dict[int, np.ndarray]:
    values = features.to_numpy()
    return {
        int(cid): values[labels == cid].mean(axis=0)
        for cid in sorted({int(v) for v in labels})
    }


def _identify_canonical_cluster_mapping(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, int]:
    """Maps this training run's arbitrary new cluster ids to the CANONICAL ids
    career_map.json/CLUSTER_HOLLAND_CODES were authored against, by nearest-centroid
    distance to the frozen reference (see _load_canonical_centroids).

    Raises ValueError if the match is not confidently a clean bijection - e.g. the
    uploaded dataset doesn't share the same underlying 8-group structure. That is the
    intended, safe failure mode: abort rather than silently deploy mislabeled careers.
    """
    frozen_centroids = _load_canonical_centroids()
    new_centroids = _compute_cluster_centroids(features, labels)

    if sorted(new_centroids.keys()) != list(LIVE_CLUSTER_IDS):
        raise ValueError("Training data must cover all 8 clusters.")

    distances = {
        (raw_id, canonical_id): float(np.linalg.norm(new_centroids[raw_id] - frozen_vector))
        for raw_id in new_centroids
        for canonical_id, frozen_vector in frozen_centroids.items()
    }

    # Greedy nearest-first bipartite match.
    ordered_pairs = sorted(distances.items(), key=lambda kv: kv[1])
    assigned_raw: set[int] = set()
    assigned_canonical: set[int] = set()
    mapping: dict[int, int] = {}
    for (raw_id, canonical_id), _distance in ordered_pairs:
        if raw_id in assigned_raw or canonical_id in assigned_canonical:
            continue
        mapping[raw_id] = canonical_id
        assigned_raw.add(raw_id)
        assigned_canonical.add(canonical_id)

    if len(mapping) != len(LIVE_CLUSTER_IDS):
        raise ValueError(
            "Could not uniquely match this dataset's clusters to the 8 canonical clusters "
            "career_map.json expects. Retrain offline instead and manually review career_map.json."
        )

    # Confidence gate: each match must clearly beat its runner-up, or cluster identity
    # is too ambiguous to trust automatically.
    for raw_id, canonical_id in mapping.items():
        own_distance = distances[(raw_id, canonical_id)]
        runner_up = min(
            distances[(raw_id, other_id)] for other_id in frozen_centroids if other_id != canonical_id
        )
        margin = runner_up - own_distance
        if margin < MIN_CENTROID_MATCH_MARGIN:
            raise ValueError(
                "This dataset's cluster structure is too ambiguous to safely match to the "
                f"career map (raw cluster {raw_id} is nearly equidistant between two canonical "
                "clusters). Retrain offline instead and manually review career_map.json."
            )

    return mapping


def _validate_staged_cluster_model(path: Path) -> None:
    model = joblib.load(path)
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    if feature_names != LIVE_FEATURE_NAMES:
        raise RuntimeError(f"Staged cluster model at {path} has an unexpected feature order.")
    classes = tuple(sorted(int(c) for c in getattr(model, "classes_", ())))
    if classes != LIVE_CLUSTER_IDS:
        raise RuntimeError(f"Staged cluster model at {path} has unexpected classes: {classes}.")


def _validate_staged_career_model(path: Path, *, cluster_id: int) -> None:
    model = joblib.load(path)
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    if feature_names != LIVE_FEATURE_NAMES:
        raise RuntimeError(f"Staged career model at {path} has an unexpected feature order.")
    # Outlier clusters (1, 4) are intentionally single-class - see the training loop.
    if cluster_id not in OUTLIER_CLUSTER_IDS and len(getattr(model, "classes_", ())) < 2:
        raise RuntimeError(f"Staged career model at {path} is degenerate (fewer than 2 classes).")


def train_runtime_model_from_dataset(
    dataset_path: str | Path,
    *,
    dataset_id: int,
    dataset_name: str,
) -> TrainingSummary:
    """Refit KMeans + career RFs from an uploaded dataset and deploy the result.

    Cluster identity: K-Means assigns cluster ids arbitrarily on every fit, but
    career_map.json/CLUSTER_HOLLAND_CODES are pinned to one frozen partition. This
    function re-identifies the new run's clusters by nearest-centroid match against
    that frozen partition (_identify_canonical_cluster_mapping) and relabels before
    training career sub-models, so the deployed labels stay correct. If the match isn't
    confidently a clean bijection, training ABORTS (ValueError) rather than deploying
    a dataset whose underlying cluster structure doesn't line up with the career map -
    retrain offline and re-author career_map.json in that case.

    Deployment is staged: every model is trained and validated in a temporary directory
    first; the live model_assets/runtime files are only touched (via atomic per-file
    replace) after all 9 models pass validation, so a failure at any point during
    training leaves the previously-deployed models untouched.
    """
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
    raw_cluster_labels = kmeans.fit_predict(scaled_features)
    present_clusters = tuple(sorted({int(value) for value in raw_cluster_labels}))
    if present_clusters != LIVE_CLUSTER_IDS:
        raise ValueError("Training data must cover all 8 clusters.")

    canonical_mapping = _identify_canonical_cluster_mapping(features, raw_cluster_labels)
    cluster_labels = np.array([canonical_mapping[int(raw_id)] for raw_id in raw_cluster_labels])

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

    # Stage every model in a scratch directory first. Nothing under model_assets/runtime
    # is touched until every model is trained AND validated - a failure anywhere above
    # this point, or during staging/validation below, leaves the live models untouched.
    staging_dir = cluster_runtime.RUNTIME_ARTIFACTS_DIR.parent / "_staging" / bundle_key
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        staged_cluster_path = staging_dir / cluster_runtime.CLUSTER_MODEL_PATH.name
        joblib.dump(cluster_model, staged_cluster_path)

        career_model_count = 0
        staged_career_paths: dict[int, Path] = {}
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

            # A single-class career model always predicts the same career for the whole
            # cluster - the exact degeneracy fix_degenerate_subclusters.py had to patch
            # offline. Catch it here before deployment, EXCEPT for the outlier clusters
            # (1, 4), which are intentionally single-sub-cluster (no careers in
            # career_map.json) - CAREER_CLUSTER_COUNTS requests only 1 for them.
            if cluster_id not in OUTLIER_CLUSTER_IDS and len(career_model.classes_) < 2:
                raise ValueError(
                    f"Career sub-model for cluster {cluster_id} would be degenerate "
                    f"(only {len(career_model.classes_)} class from {len(cluster_subset)} rows). "
                    "Upload a larger or more varied dataset."
                )

            staged_path = staging_dir / f"career_predictor_model_cluster_{cluster_id}.pkl"
            joblib.dump(career_model, staged_path)
            staged_career_paths[cluster_id] = staged_path
            career_model_count += 1

        # Validate every staged file the way the runtime loader will read it, before
        # any of them can reach the live directory.
        _validate_staged_cluster_model(staged_cluster_path)
        for cluster_id, staged_path in staged_career_paths.items():
            _validate_staged_career_model(staged_path, cluster_id=cluster_id)

        # All validated - deploy. Per-file os.replace is atomic on both POSIX and
        # Windows, so each individual file swap can't leave a half-written .pkl; this
        # loop as a whole is not a single transaction, but the window where a crash
        # could matter is now just this fast rename loop, not the training compute above.
        cluster_runtime.RUNTIME_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        os.replace(staged_cluster_path, cluster_runtime.CLUSTER_MODEL_PATH)
        for cluster_id, staged_path in staged_career_paths.items():
            live_path = cluster_runtime.CAREER_MODEL_DIR / f"career_predictor_model_cluster_{cluster_id}.pkl"
            os.replace(staged_path, live_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

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

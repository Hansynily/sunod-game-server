from datetime import date, datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr, TypeAdapter, ValidationError

from app import (
    account_lifecycle,
    career_mapping,
    cluster_mapping,
    cluster_runtime,
    feature_pipeline,
    mailer,
    ml_runtime,
    model_training,
    schemas,
)
from app.database import get_db
from app.logging_utils import audit_log
from app.repository import DuplicateUserError, TelemetryRepository
from app.security import (
    ADMIN_SESSION_COOKIE,
    create_access_token,
    describe_user_access,
    get_current_user,
    get_optional_user,
    parse_access_token,
    hash_password,
    require_admin,
    verify_password,
)


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
predict_router = APIRouter(prefix="/api", tags=["prediction"])
public_router = APIRouter(tags=["public"])
admin_router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
admin_ui_router = APIRouter(prefix="/admin", tags=["admin-ui"])

ROOT = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def _neutral_cluster_result(cluster_id: int, career_cluster_id: int | None) -> str:
    return "Specific career not yet mapped"


def _principal_fields(user, *, prefix: str) -> dict[str, Any]:
    if user is None:
        return {
            f"{prefix}_id": None,
            f"{prefix}_username": None,
            f"{prefix}_role": None,
        }

    return {
        f"{prefix}_id": user.id,
        f"{prefix}_username": user.username,
        f"{prefix}_role": user.role,
    }


def _build_auth_response(user) -> schemas.AuthResponse:
    access_state = describe_user_access(user)
    return schemas.AuthResponse(
        id=user.id,
        player_id=user.player_id,
        username=user.username,
        email=user.email,
        name=user.name,
        birthdate=user.birthdate,
        gender=user.gender,
        created_at=user.created_at,
        last_login=user.last_login,
        role=user.role,
        approval_state=user.approval_state,
        email_verification_state=access_state.email_verification_state,
        tutorial_completed=user.tutorial_completed,
        tutorial_completed_at=user.tutorial_completed_at,
        can_login=access_state.can_login,
        next_step=access_state.next_step,
        message=access_state.message,
        access_token=(
            create_access_token(
                user_id=user.id,
                username=user.username,
                role=user.role,
            )
            if access_state.can_login
            else None
        ),
        token_type="bearer" if access_state.can_login else None,
    )


def _build_runtime_status_response() -> schemas.RuntimeStatusOut:
    cluster_status = cluster_runtime.get_cluster_model_status()
    career_status = cluster_runtime.get_career_model_status()
    cluster_loaded = bool(cluster_status.available and cluster_status.model is not None)
    career_loaded = bool(career_status.available and career_status.models)
    active_source = "Cluster RF + Career RF"

    if cluster_loaded and career_loaded:
        career_model_count = len(career_status.models or {})
        active_version = (
            f"{Path(cluster_status.model_path).name if cluster_status.model_path else 'riasec_cluster_model.pkl'} "
            f"+ {career_model_count} career models"
        )
        return schemas.RuntimeStatusOut(
            cluster_model_loaded=True,
            career_model_bundle_loaded=True,
            active_source=active_source,
            active_version=active_version,
            detail="Live 48-feature prediction bundle loaded and ready.",
        )

    detail_source = cluster_status.reason if not cluster_loaded else career_status.reason
    return schemas.RuntimeStatusOut(
        cluster_model_loaded=cluster_loaded,
        career_model_bundle_loaded=career_loaded,
        active_source=active_source,
        active_version="untrained",
        detail=_sanitize_admin_error_message(
            detail_source,
            fallback="Live prediction bundle unavailable. Upload or train the runtime bundle on the server.",
        ),
    )


_WINDOWS_ABS_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s,;]+")
_POSIX_ABS_PATH_PATTERN = re.compile(r"(?:^|[\s(])/(?:[^/\s)]+/)+[^/\s),;]+")


def _sanitize_admin_error_message(message: str | None, *, fallback: str) -> str:
    normalized = (message or "").strip()
    if not normalized:
        return fallback

    normalized = _WINDOWS_ABS_PATH_PATTERN.sub("[server path]", normalized)
    normalized = _POSIX_ABS_PATH_PATTERN.sub(" [server path]", normalized)

    known_replacements = {
        "Runtime model artifacts are missing.": "Runtime model unavailable. Upload or train the required runtime files on the server.",
        "feature_schema.json could not be read.": "Runtime model configuration could not be read on the server.",
        "model_version.txt could not be read.": "Runtime model version information could not be read on the server.",
        "model.joblib could not be loaded.": "Runtime model could not be loaded on the server.",
        "Cluster model artifact is missing.": "Cluster model is unavailable on the server.",
        "Cluster model could not be loaded.": "Cluster model could not be loaded on the server.",
        "One or more career model artifacts are missing.": "Career model bundle is incomplete on the server.",
    }
    if normalized in known_replacements:
        return known_replacements[normalized]

    if "could not be loaded" in normalized.lower():
        return fallback
    if "missing" in normalized.lower() and "artifact" in normalized.lower():
        return fallback

    return normalized


def _resolve_cluster_model_version() -> str:
    cluster_status = cluster_runtime.get_cluster_model_status()
    if cluster_status.model_path:
        return Path(cluster_status.model_path).name
    return "cluster-model"


@dataclass(slots=True)
class PredictionRuntimeContext:
    cluster_status: cluster_runtime.ClusterModelStatus
    career_status: cluster_runtime.CareerModelStatus
    binding_source: str = "default"
    bundle_key: str | None = None
    dataset_id: int | None = None
    dataset_name: str | None = None
    binding_scope_type: str | None = None
    binding_scope_id: str | None = None
    challenge_id: str | None = None
    level_id: str | None = None
    model_version: str = "cluster-model"
    source: str = "Cluster RF + Career RF"


def _build_default_prediction_context() -> PredictionRuntimeContext:
    cluster_status = cluster_runtime.get_cluster_model_status()
    career_status = cluster_runtime.get_career_model_status()
    return PredictionRuntimeContext(
        cluster_status=cluster_status,
        career_status=career_status,
        model_version=_resolve_cluster_model_version(),
    )


def _resolve_prediction_runtime_context(
    db: TelemetryRepository,
    *,
    player_id: str,
    session_id: str,
) -> PredictionRuntimeContext:
    context = _build_default_prediction_context()
    session_run = db.find_session_run_for_player(player_id=player_id, session_id=session_id)
    if not session_run:
        return context

    rounds = session_run.get("rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return context

    last_round = next((entry for entry in reversed(rounds) if isinstance(entry, dict)), None)
    if not last_round:
        return context

    challenge_id = str(last_round.get("challenge_id") or "").strip()
    if not challenge_id:
        return context

    level_id = _extract_level_id(challenge_id)
    context.challenge_id = challenge_id
    context.level_id = level_id
    resolved_info = db.resolve_quest_model_binding(
        scope_type="quest",
        scope_id=challenge_id,
        level_id=level_id,
    )
    if not resolved_info:
        return context

    binding = resolved_info["binding"] or {}
    context.binding_scope_type = binding.get("scope_type")
    context.binding_scope_id = binding.get("scope_id")
    context.dataset_id = int(binding["dataset_id"]) if binding.get("dataset_id") is not None else None
    context.dataset_name = binding.get("dataset_name")
    bundle_key = str(binding.get("bundle_key") or "").strip()
    if not bundle_key or not binding.get("bundle_ready"):
        return context

    bundle_status = cluster_runtime.load_saved_bundle_models(bundle_key)
    if not bundle_status.available:
        LOGGER.warning(
            "Saved bundle %s could not be loaded for prediction context player_id=%s session_id=%s",
            bundle_key,
            player_id,
            session_id,
        )
        return context

    cluster_model = bundle_status.cluster_model
    career_models = bundle_status.career_models
    if cluster_model is None or not career_models:
        return context

    context.cluster_status = cluster_runtime.ClusterModelStatus(
        available=True,
        model=cluster_model,
        model_path=bundle_status.cluster_model_path,
    )
    context.career_status = cluster_runtime.CareerModelStatus(
        available=True,
        models=career_models,
        model_paths=bundle_status.career_model_paths,
    )
    context.binding_source = resolved_info.get("binding_source") or "default"
    context.bundle_key = bundle_status.bundle_key or bundle_key
    context.model_version = (
        bundle_status.model_version
        or str(binding.get("model_version") or _resolve_cluster_model_version())
    )
    return context


def _empty_admin_riasec_scores() -> dict[str, int]:
    return {
        "r": 0,
        "i": 0,
        "a": 0,
        "s": 0,
        "e": 0,
        "c": 0,
    }


def _normalize_admin_riasec_scores(raw_scores: dict[str, Any] | None) -> dict[str, int]:
    normalized = _empty_admin_riasec_scores()
    if not isinstance(raw_scores, dict):
        return normalized

    for key in normalized:
        try:
            normalized[key] = max(0, min(10, int(raw_scores.get(key, 0))))
        except (TypeError, ValueError):
            normalized[key] = 0

    return normalized


def _get_admin_timezone() -> timezone | ZoneInfo:
    timezone_name = (os.getenv("APP_TIMEZONE") or "Asia/Manila").strip()
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        LOGGER.warning("Unknown APP_TIMEZONE %s. Falling back to UTC.", timezone_name)
        return timezone.utc


def _to_admin_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(_get_admin_timezone())


def _summarize_session_run(document: dict[str, Any]) -> dict[str, Any]:
    rounds = document.get("rounds") or []
    rounds_attempted = len(rounds)
    rounds_cleared = sum(1 for round_entry in rounds if round_entry.get("solved"))
    total_stars = sum(int(round_entry.get("stars_earned", 0) or 0) for round_entry in rounds)

    return {
        "session_id": document.get("session_id", "unknown-session"),
        "created_at": _format_admin_datetime(document.get("created_at") or datetime.utcnow()),
        "source": document.get("source") or "Unknown",
        "model_version": document.get("model_version") or "n/a",
        "riasec_scores": _normalize_admin_riasec_scores(document.get("riasec_scores")),
        "holland_code": document.get("holland_code") or "N/A",
        "career_family": document.get("career_family") or "N/A",
        "career_result": document.get("career_result") or "N/A",
        "career_cluster": document.get("career_cluster"),
        "total_time_spent_seconds": float(document.get("total_time_spent_seconds", 0) or 0),
        "rounds_attempted": rounds_attempted,
        "rounds_cleared": rounds_cleared,
        "total_stars": total_stars,
        "predicted_cluster": document.get("predicted_cluster"),
        "cluster_label": document.get("cluster_label"),
        "cluster_holland_code": document.get("cluster_holland_code"),
        "cluster_source": document.get("cluster_source"),
        "cluster_model_version": document.get("cluster_model_version"),
        "cluster_example_careers": list(document.get("cluster_example_careers") or []),
    }


def _format_admin_datetime(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    if isinstance(value, datetime):
        return _to_admin_timezone(value).strftime("%Y-%m-%d %H:%M")

    return str(value)


def _format_admin_date(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    if isinstance(value, datetime):
        return _to_admin_timezone(value).strftime("%Y-%m-%d")

    return str(value)


def _format_file_size(value: Any) -> str:
    try:
        size = int(value or 0)
    except (TypeError, ValueError):
        size = 0

    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _normalize_admin_approval_state(value: str | None) -> str:
    if value == "grandfathered":
        return "approved"
    return value or "pending"


ADMIN_USER_SORT_DEFAULT_BY = "created_at"
ADMIN_USER_SORT_DEFAULT_DIR = "desc"
ADMIN_USER_SORT_LABELS = {
    "created_at": "Registered",
    "username": "Username",
    "email_verification_state": "Verification",
    "last_activity_at": "Last activity",
    "total_runs": "Runs",
}
ADMIN_USER_SORT_OPTIONS = (
    ("created_at", "Registered"),
    ("username", "Username"),
    ("email_verification_state", "Verification"),
    ("last_activity_at", "Last activity"),
    ("total_runs", "Runs"),
)


def _normalize_admin_user_sort(sort_by: str | None, sort_dir: str | None) -> tuple[str, str]:
    normalized_sort_by = (sort_by or ADMIN_USER_SORT_DEFAULT_BY).strip().lower()
    if normalized_sort_by not in ADMIN_USER_SORT_LABELS:
        normalized_sort_by = ADMIN_USER_SORT_DEFAULT_BY

    normalized_sort_dir = (sort_dir or ADMIN_USER_SORT_DEFAULT_DIR).strip().lower()
    if normalized_sort_dir not in {"asc", "desc"}:
        normalized_sort_dir = ADMIN_USER_SORT_DEFAULT_DIR

    return normalized_sort_by, normalized_sort_dir


def _latest_admin_datetime(*values: datetime | None) -> datetime | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return max(candidates)


def _admin_user_sort_value(row: dict[str, Any], sort_by: str) -> Any:
    if sort_by == "created_at":
        return row.get("created_at")
    if sort_by == "username":
        return (row.get("username") or "").casefold()
    if sort_by == "email_verification_state":
        return (row.get("email_verification_state") or "").casefold()
    if sort_by == "last_activity_at":
        return row.get("last_activity_at")
    if sort_by == "total_runs":
        return int(row.get("total_runs") or 0)
    return row.get("created_at")


def _build_admin_user_sort_controls(sort_by: str, sort_dir: str) -> list[dict[str, Any]]:
    controls = [
        ("created_at", "asc"),
        ("created_at", "desc"),
        ("username", "asc"),
        ("username", "desc"),
        ("email_verification_state", "asc"),
        ("email_verification_state", "desc"),
        ("last_activity_at", "asc"),
        ("last_activity_at", "desc"),
        ("total_runs", "asc"),
        ("total_runs", "desc"),
    ]

    sort_controls: list[dict[str, Any]] = []
    for control_sort_by, control_sort_dir in controls:
        label = ADMIN_USER_SORT_LABELS[control_sort_by]
        sort_controls.append(
            {
                "label": f"{label} {control_sort_dir.upper()}",
                "href": f"/admin/users?{urlencode({'sort_by': control_sort_by, 'sort_dir': control_sort_dir})}",
                "active": control_sort_by == sort_by and control_sort_dir == sort_dir,
            }
        )

    return sort_controls


def _build_admin_user_sort_options() -> list[dict[str, str]]:
    return [{"value": value, "label": label} for value, label in ADMIN_USER_SORT_OPTIONS]


def _sort_admin_user_rows(
    rows: list[dict[str, Any]],
    *,
    sort_by: str,
    sort_dir: str,
) -> list[dict[str, Any]]:
    decorated_rows: list[tuple[Any, dict[str, Any]]] = []
    missing_rows: list[dict[str, Any]] = []

    for row in rows:
        sort_value = _admin_user_sort_value(row, sort_by)
        if sort_value is None:
            missing_rows.append(row)
        else:
            decorated_rows.append((sort_value, row))

    decorated_rows.sort(key=lambda item: item[0], reverse=sort_dir == "desc")
    return [row for _, row in decorated_rows] + missing_rows


def _build_admin_user_rows(
    users: list,
    run_overview_by_player: dict[str, dict[str, Any]],
    *,
    sort_by: str = ADMIN_USER_SORT_DEFAULT_BY,
    sort_dir: str = ADMIN_USER_SORT_DEFAULT_DIR,
) -> list[dict[str, Any]]:
    sort_by, sort_dir = _normalize_admin_user_sort(sort_by, sort_dir)
    rows: list[dict[str, Any]] = []
    for user in users:
        overview = run_overview_by_player.get(user.player_id, {})
        access_state = describe_user_access(user)
        rows.append(
            {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "name": user.name,
                "birthdate": _format_admin_date(user.birthdate),
                "gender": user.gender,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "role": user.role,
                "approval_state": _normalize_admin_approval_state(user.approval_state),
                "email_verification_state": access_state.email_verification_state,
                "approved_at": user.approved_at,
                "approved_by_user_id": user.approved_by_user_id,
                "verification_sent_at": user.verification_sent_at,
                "verification_expires_at": user.verification_expires_at,
                "verified_at": user.verified_at,
                "rejection_reason": user.rejection_reason,
                "total_runs": int(overview.get("total_runs", 0) or 0),
                "last_run_at": overview.get("last_run_at"),
                "last_activity_at": _latest_admin_datetime(
                    user.last_login,
                    overview.get("last_run_at"),
                ),
                "last_source": overview.get("last_source"),
                "last_result": overview.get("last_result"),
                "last_holland_code": overview.get("last_holland_code"),
                "last_predicted_cluster": overview.get("last_predicted_cluster"),
                "last_cluster_label": overview.get("last_cluster_label"),
                "last_cluster_holland_code": overview.get("last_cluster_holland_code"),
            }
        )

    rows = _sort_admin_user_rows(rows, sort_by=sort_by, sort_dir=sort_dir)

    for row in rows:
        row["created_at"] = _format_admin_datetime(row["created_at"])
        row["last_login"] = _format_admin_datetime(row["last_login"])
        row["approved_at"] = _format_admin_datetime(row["approved_at"])
        row["verification_sent_at"] = _format_admin_datetime(row["verification_sent_at"])
        row["verification_expires_at"] = _format_admin_datetime(row["verification_expires_at"])
        row["verified_at"] = _format_admin_datetime(row["verified_at"])
        row["last_run_at"] = _format_admin_datetime(row["last_run_at"])
        row["last_activity_at"] = _format_admin_datetime(row["last_activity_at"])

    return rows


def _build_user_performance_payload(
    *,
    user,
    session_run_documents: list[dict[str, Any]],
    quest_attempt_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runs = [_summarize_session_run(document) for document in session_run_documents]
    quest_attempt_rows = list(quest_attempt_rows or [])
    total_runs = len(runs)
    total_rounds_attempted = sum(run["rounds_attempted"] for run in runs)
    total_rounds_cleared = sum(run["rounds_cleared"] for run in runs)
    total_time_spent = sum(run["total_time_spent_seconds"] for run in runs)
    total_stars = sum(run["total_stars"] for run in runs)
    latest_run = runs[0] if runs else None

    avg_clear_rate = (
        total_rounds_cleared / total_rounds_attempted * 100
        if total_rounds_attempted > 0
        else 0.0
    )
    avg_time_seconds = total_time_spent / total_runs if total_runs > 0 else 0.0
    avg_stars_per_run = total_stars / total_runs if total_runs > 0 else 0.0

    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "name": user.name,
        "birthdate": _format_admin_date(user.birthdate),
        "gender": user.gender,
        "created_at": _format_admin_datetime(user.created_at),
        "last_login": _format_admin_datetime(user.last_login),
        "role": user.role,
        "approval_state": _normalize_admin_approval_state(user.approval_state),
        "email_verification_state": describe_user_access(user).email_verification_state,
        "approved_at": _format_admin_datetime(user.approved_at),
        "verification_sent_at": _format_admin_datetime(user.verification_sent_at),
        "verification_expires_at": _format_admin_datetime(user.verification_expires_at),
        "verified_at": _format_admin_datetime(user.verified_at),
        "rejection_reason": user.rejection_reason,
        "total_runs": total_runs,
        "avg_clear_rate": avg_clear_rate,
        "avg_time_seconds": avg_time_seconds,
        "avg_stars_per_run": avg_stars_per_run,
        "latest_source": (latest_run["cluster_source"] or latest_run["source"]) if latest_run else None,
        "latest_result": latest_run["career_result"] if latest_run else None,
        "latest_holland_code": latest_run["cluster_holland_code"] if latest_run else None,
        "latest_career_family": latest_run["career_family"] if latest_run else None,
        "latest_model_version": latest_run["cluster_model_version"] if latest_run else None,
        "latest_riasec": latest_run["riasec_scores"] if latest_run else _empty_admin_riasec_scores(),
        "latest_predicted_cluster": latest_run["predicted_cluster"] if latest_run else None,
        "latest_career_cluster": latest_run["career_cluster"] if latest_run else None,
        "latest_career_result": latest_run["career_result"] if latest_run else None,
        "latest_cluster_label": (
            latest_run["career_result"]
            if latest_run and latest_run.get("career_result")
            else (latest_run["cluster_label"] if latest_run else None)
        ),
        "latest_cluster_holland_code": latest_run["cluster_holland_code"] if latest_run else None,
        "latest_cluster_source": latest_run["cluster_source"] if latest_run else None,
        "latest_cluster_model_version": latest_run["cluster_model_version"] if latest_run else None,
        "latest_cluster_example_careers": latest_run["cluster_example_careers"] if latest_run else [],
        "quest_attempts": quest_attempt_rows,
        "runs": runs,
    }


def _format_admin_quest_attempt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        display_row = dict(row)
        effective_at = row.get("completed_at") or row.get("started_at")
        display_row["started_at"] = _format_admin_datetime(row.get("started_at"))
        display_row["completed_at"] = _format_admin_datetime(row.get("completed_at"))
        display_row["effective_at"] = _format_admin_datetime(effective_at)
        display_row["outcome_label"] = "Success" if int(row.get("success", 0) or 0) == 1 else "Failure"
        formatted_rows.append(display_row)
    return formatted_rows


def _format_admin_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        display_row = dict(row)
        display_row["uploaded_at"] = _format_admin_datetime(row.get("uploaded_at"))
        display_row["trained_at"] = _format_admin_datetime(row.get("trained_at"))
        display_row["activated_at"] = _format_admin_datetime(row.get("activated_at"))
        display_row["file_size_label"] = _format_file_size(row.get("file_size_bytes"))
        display_row["mean_absolute_error_label"] = (
            f"{float(row['mean_absolute_error']):.3f}"
            if row.get("mean_absolute_error") is not None
            else "n/a"
        )
        display_row["r2_score_label"] = (
            f"{float(row['r2_score']):.3f}"
            if row.get("r2_score") is not None
            else "n/a"
        )
        display_row["cluster_accuracy_label"] = (
            f"{float(row['cluster_accuracy']):.3f}"
            if row.get("cluster_accuracy") is not None
            else "n/a"
        )
        display_row["career_model_count_label"] = (
            str(int(row["career_model_count"]))
            if row.get("career_model_count") is not None
            else "n/a"
        )
        display_row["clusters_covered_count_label"] = (
            f"{int(row['clusters_covered_count'])}/8"
            if row.get("clusters_covered_count") is not None
            else "n/a"
        )
        display_row["bundle_ready_label"] = (
            "Ready"
            if row.get("bundle_ready")
            else "Missing"
        )
        display_row["training_error"] = _sanitize_admin_error_message(
            row.get("training_error"),
            fallback="Training failed on the server.",
        ) if row.get("training_error") else None
        formatted_rows.append(display_row)
    return formatted_rows


def _extract_level_id(challenge_id: str) -> str | None:
    normalized = (challenge_id or "").strip().upper()
    if "_" not in normalized:
        return None
    prefix = normalized.split("_", 1)[0]
    if len(prefix) < 2 or not prefix.startswith("L"):
        return None
    if not prefix[1:].isdigit():
        return None
    return prefix


def _build_quest_binding_targets() -> list[dict[str, Any]]:
    level_ids: set[str] = set()
    quest_rows: list[dict[str, Any]] = []

    for challenge in feature_pipeline.CHALLENGE_DEFINITIONS:
        level_id = _extract_level_id(challenge.challenge_id)
        if level_id:
            level_ids.add(level_id)
        quest_rows.append(
            {
                "scope_type": "quest",
                "scope_id": challenge.challenge_id,
                "scope_label": challenge.challenge_id,
                "level_id": level_id,
                "primary_riasec": challenge.primary_riasec,
            }
        )

    level_rows = [
        {
            "scope_type": "level",
            "scope_id": level_id,
            "scope_label": f"Level {level_id}",
            "level_id": level_id,
            "primary_riasec": None,
        }
        for level_id in sorted(level_ids)
    ]

    return [*level_rows, *quest_rows]


def _format_admin_binding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        formatted_rows.append(_format_admin_binding_row(row))
    return formatted_rows


def _format_admin_binding_row(row: dict[str, Any]) -> dict[str, Any]:
    display_row = dict(row)
    display_row["updated_at"] = _format_admin_datetime(row.get("updated_at"))
    display_row["bundle_ready_label"] = "Ready" if row.get("bundle_ready") else "Missing"
    return display_row


def _format_admin_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted_rows: list[dict[str, Any]] = []
    for row in rows:
        display_row = dict(row)
        display_row["created_at"] = _format_admin_datetime(row.get("created_at"))
        formatted_rows.append(display_row)
    return formatted_rows


def _admin_row_for_user(db: TelemetryRepository, user) -> dict[str, Any]:
    return _build_admin_user_rows([user], db.list_session_run_overview_by_player())[0]


def _redirect_with_flash(
    base_url: str,
    *,
    notice: str | None = None,
    error: str | None = None,
    notice_link: str | None = None,
) -> RedirectResponse:
    params: dict[str, str] = {}
    if notice:
        params["notice"] = notice
    if error:
        params["error"] = error
    if notice_link:
        params["notice_link"] = notice_link

    url = base_url
    if params:
        url = f"{base_url}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


def _flash_context(request: Request) -> dict[str, str | None]:
    return {
        "notice_message": request.query_params.get("notice"),
        "error_message": request.query_params.get("error"),
        "notice_link": request.query_params.get("notice_link"),
    }


def _require_active_player_account(user) -> None:
    access_state = describe_user_access(user)
    if access_state.can_login:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=access_state.message,
    )


def _require_verification_ready_user(user) -> None:
    if user.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin accounts are exempt from email verification.",
        )

    if user.approval_state not in {
        account_lifecycle.APPROVAL_APPROVED,
        account_lifecycle.APPROVAL_GRANDFATHERED,
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Approve the account before sending a verification email.",
        )

    if not user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Add an email address before approval or verification.",
        )

    if user.email_verification_state == account_lifecycle.EMAIL_EXEMPT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account is verification-exempt.",
        )

    if user.email_verification_state == account_lifecycle.EMAIL_VERIFIED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account is already verified.",
        )


def _send_verification_email(db: TelemetryRepository, user) -> tuple[Any, mailer.DeliveryResult]:
    settings = mailer.get_mailer_settings()
    token, token_hash = account_lifecycle.issue_verification_token()
    expires_at = datetime.utcnow() + timedelta(hours=settings.verification_ttl_hours)
    updated_user = db.store_email_verification_token(
        user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    if updated_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    verification_link = mailer.build_verification_link(token)
    html_body, text_body = mailer.render_verification_email(
        username=updated_user.username,
        verification_link=verification_link,
        expires_in_hours=settings.verification_ttl_hours,
    )
    delivery_result = mailer.send_email(
        to_email=updated_user.email,
        subject="Verify your SUNOD account",
        html_body=html_body,
        text_body=text_body,
        verification_link=verification_link,
    )
    return updated_user, delivery_result


def _validate_email_form_input(email: str) -> str:
    return str(TypeAdapter(EmailStr).validate_python(email))


def _should_send_verification_on_approval(user) -> bool:
    return (
        user.role != "admin"
        and bool(user.email)
        and user.email_verification_state not in {
            account_lifecycle.EMAIL_VERIFIED,
            account_lifecycle.EMAIL_EXEMPT,
        }
    )


def _approve_and_maybe_send_verification(
    *,
    db: TelemetryRepository,
    user_id: int,
    actor_user_id: int | None,
    actor,
    via: str,
) -> tuple[Any, str, str | None]:
    user = db.approve_user(user_id, actor_user_id=actor_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_APPROVED",
        target_user_id=user.id,
        target_username=user.username,
        approval_state=user.approval_state,
        via=via,
        **_principal_fields(actor, prefix="actor"),
    )

    if not _should_send_verification_on_approval(user):
        return user, "Account approved.", None

    try:
        user, delivery_result = _send_verification_email(db, user)
    except Exception as exc:
        user = db.find_user_by_id(user.id) or user
        LOGGER.exception("Verification email send failed after approval for user_id=%s", user.id)
        audit_log(
            "USER_VERIFICATION_SEND_FAILED",
            target_user_id=user.id,
            target_username=user.username,
            reason=str(exc),
            via=via,
            **_principal_fields(actor, prefix="actor"),
        )
        return user, "Account approved. Verification email needs to be resent.", None

    audit_log(
        "USER_VERIFICATION_SENT",
        target_user_id=user.id,
        target_username=user.username,
        delivery_mode=delivery_result.mode,
        via=via,
        **_principal_fields(actor, prefix="actor"),
    )
    return user, f"Account approved. {delivery_result.message}", delivery_result.verification_link


def _dashboard_url_for_user(user) -> str:
    if user.role == "admin":
        return "/admin/users"
    return f"/admin/users/{user.id}"


def _current_ui_user(
    request: Request,
    db: TelemetryRepository,
):
    token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if not token:
        return None

    try:
        principal = parse_access_token(token)
    except ValueError:
        return None

    user = db.find_user_by_id(principal.user_id)
    if user is None or user.username != principal.username:
        return None

    if not describe_user_access(user).can_login:
        return None

    return user


def _require_ui_user(
    request: Request,
    db: TelemetryRepository,
):
    user = _current_ui_user(request, db)
    if user is not None:
        return user

    raise HTTPException(
        status_code=status.HTTP_303_SEE_OTHER,
        detail="Login required.",
        headers={"Location": "/admin/login"},
    )


def _require_admin_ui_user(
    request: Request,
    db: TelemetryRepository,
):
    user = _require_ui_user(request, db)
    if user.role == "admin":
        return user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required.",
    )


def _require_user_or_404(db: TelemetryRepository, user_id: int):
    user = db.find_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


@router.post("/users", response_model=schemas.AuthResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user_in: schemas.UserCreate,
    current_user = Depends(get_optional_user),
    db: TelemetryRepository = Depends(get_db),
):
    name = user_in.name.strip()
    username = user_in.username.strip()
    password_hash = hash_password(user_in.password)
    requested_role = user_in.role
    assigned_role = "user"
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name cannot be empty.",
        )
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty.",
        )

    if requested_role == "admin":
        if current_user is None or current_user.role != "admin":
            audit_log(
                "ADMIN_ACCOUNT_CREATE_DENIED",
                requested_username=username,
                requested_role=requested_role,
                **_principal_fields(current_user, prefix="actor"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to create an admin account.",
            )
        assigned_role = "admin"

    try:
        user = db.create_user(
            player_id=str(uuid.uuid4()),
            username=username,
            password_hash=password_hash,
            email=str(user_in.email) if user_in.email else None,
            name=name,
            birthdate=user_in.birthdate,
            gender=user_in.gender,
            role=assigned_role,
            riasec_profile=(
                user_in.riasec_profile.model_dump() if user_in.riasec_profile else None
            ),
        )
        audit_log(
            "USER_REGISTERED",
            user_id=user.id,
            username=user.username,
            role=user.role,
            mode="create",
            approval_state=user.approval_state,
            email_verification_state=user.email_verification_state,
            **_principal_fields(current_user, prefix="actor"),
        )
        return _build_auth_response(user)
    except DuplicateUserError as exc:
        upgraded_user = db.upgrade_legacy_user_password(
            username=username,
            password_hash=password_hash,
        )
        if upgraded_user:
            if user_in.email and upgraded_user.email != str(user_in.email):
                try:
                    upgraded_user = db.set_user_email(
                        upgraded_user.id,
                        str(user_in.email),
                    ) or upgraded_user
                except DuplicateUserError as email_exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="User with this username or email already exists.",
                    ) from email_exc
            upgraded_user = db.update_user_profile_fields(
                upgraded_user.id,
                name=name,
                birthdate=user_in.birthdate,
                gender=user_in.gender,
            ) or upgraded_user
            if upgraded_user.role != assigned_role:
                upgraded_user = db.set_user_role(upgraded_user.id, assigned_role) or upgraded_user
            audit_log(
                "USER_REGISTERED",
                user_id=upgraded_user.id,
                username=upgraded_user.username,
                role=upgraded_user.role,
                mode="upgrade_legacy_password",
                approval_state=upgraded_user.approval_state,
                email_verification_state=upgraded_user.email_verification_state,
                **_principal_fields(current_user, prefix="actor"),
            )
            return _build_auth_response(upgraded_user)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this username or email already exists.",
        ) from exc


@router.post("/auth/login", response_model=schemas.AuthResponse)
def login_user(
    user_in: schemas.UserLogin,
    db: TelemetryRepository = Depends(get_db),
):
    username = user_in.username.strip()
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty.",
        )

    user = db.find_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    if not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account has no password yet. Register the same username once to attach a password.",
        )

    if not verify_password(user_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    access_state = describe_user_access(user)
    if not access_state.can_login:
        if access_state.email_verification_state == account_lifecycle.EMAIL_EXPIRED:
            user = db.mark_user_email_expired(user.id) or user
            access_state = describe_user_access(user)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=access_state.message,
        )

    updated_user = db.touch_last_login(user.id)
    logged_in_user = updated_user or user
    audit_log(
        "USER_LOGGED_IN",
        user_id=logged_in_user.id,
        username=logged_in_user.username,
        role=logged_in_user.role,
        via="api",
    )
    return _build_auth_response(logged_in_user)


@router.get("/users/{user_id}", response_model=schemas.User)
def get_user(user_id: int, db: TelemetryRepository = Depends(get_db)):
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


@router.get("/users/{user_id}/profile", response_model=schemas.User)
def get_user_profile(
    user_id: int,
    current_user = Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own profile.",
        )

    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


@router.post(
    "/users/me/tutorial-complete",
    response_model=schemas.TutorialCompletionResponse,
)
def mark_tutorial_complete(
    current_user = Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    updated_user = db.mark_tutorial_completed(current_user.id) or current_user
    audit_log(
        "USER_TUTORIAL_COMPLETED",
        user_id=updated_user.id,
        username=updated_user.username,
        via="api",
    )
    return schemas.TutorialCompletionResponse(
        success=True,
        message="Tutorial completion saved.",
        tutorial_completed=updated_user.tutorial_completed,
        tutorial_completed_at=updated_user.tutorial_completed_at,
    )


@router.post(
    "/users/{user_id}/quest-attempts",
    response_model=schemas.QuestAttempt,
    status_code=status.HTTP_201_CREATED,
)
def create_quest_attempt(
    user_id: int,
    quest_in: schemas.QuestAttemptCreate,
    db: TelemetryRepository = Depends(get_db),
):
    user = _require_user_or_404(db, user_id)

    quest_attempt = db.add_quest_attempt(
        user_id=user_id,
        quest_id=quest_in.quest_id,
        quest_name=quest_in.quest_name,
        success=quest_in.success,
        completed_at=quest_in.completed_at,
        time_spent_seconds=quest_in.time_spent_seconds,
        quest_result=quest_in.quest_result,
        skills_used=[skill.model_dump() for skill in quest_in.skills_used],
    )
    if not quest_attempt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "QUEST_ATTEMPT_RECORDED",
        user_id=user.id,
        username=user.username,
        quest_id=quest_attempt.quest_id,
        quest_name=quest_attempt.quest_name,
        quest_result=quest_attempt.quest_result,
        success=quest_attempt.success,
        via="api",
    )
    return quest_attempt


@router.get(
    "/users/{user_id}/quest-attempts",
    response_model=list[schemas.QuestAttempt],
)
def list_quest_attempts(user_id: int, db: TelemetryRepository = Depends(get_db)):
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user.quest_attempts


@router.post(
    "/quest-attempt",
    response_model=schemas.QuestAttemptTelemetryOut,
    status_code=status.HTTP_201_CREATED,
)
def create_quest_attempt_telemetry(
    payload: schemas.QuestAttemptTelemetryIn,
    db: TelemetryRepository = Depends(get_db),
):
    user = db.find_user_by_player_id(payload.player_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found. Please log in again.",
        )
    _require_active_player_account(user)

    quest_attempt = db.add_quest_attempt(
        user_id=user.id,
        quest_id=payload.quest_id,
        quest_name=payload.quest_id,
        completed_at=datetime.utcnow(),
        time_spent_seconds=payload.time_spent_seconds,
        quest_result=payload.quest_result,
        success=1 if payload.quest_result.lower() == "success" else 0,
        skills_used=[
            {
                "skill_name": selected.skill_name,
                "riasec_code": selected.riasec_code,
                "usage_count": 1,
            }
            for selected in payload.selected_skills
        ],
        update_profile_from_skills=True,
    )
    if not quest_attempt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "QUEST_ATTEMPT_RECORDED",
        user_id=user.id,
        username=user.username,
        quest_id=payload.quest_id,
        quest_name=payload.quest_id,
        quest_result=payload.quest_result,
        success=1 if payload.quest_result.lower() == "success" else 0,
        via="telemetry",
    )
    return schemas.QuestAttemptTelemetryOut(
        success=True,
        message="Quest attempt telemetry recorded successfully.",
    )


@router.post(
    "/run-complete",
    response_model=schemas.RunSummaryTelemetryOut,
    status_code=status.HTTP_201_CREATED,
)
def create_run_complete_telemetry(
    payload: schemas.RunSummaryTelemetryIn,
    db: TelemetryRepository = Depends(get_db),
):
    user = db.find_user_by_player_id(payload.player_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found. Please log in again.",
        )
    _require_active_player_account(user)

    try:
        validated_payload = feature_pipeline.validate_run_summary(payload)
        aggregated_features = feature_pipeline.extract_feature_record(validated_payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    source = "Session Telemetry"
    model_version = "telemetry_v2"
    message = "Run-complete telemetry recorded successfully."
    riasec_scores = _empty_admin_riasec_scores()
    holland_code = ""
    career_family = "Pending Cluster Result"
    career_result = "Pending Cluster Result"

    runtime_status = ml_runtime.get_model_status()
    if runtime_status.available:
        try:
            feature_vector = feature_pipeline.build_feature_vector(aggregated_features)
            predicted_scores = ml_runtime.predict_scores(feature_vector, runtime_status)
            skill_use_totals = feature_pipeline.extract_skill_use_totals(aggregated_features)
            riasec_scores = {
                dimension: max(0, min(10, int(round(predicted_scores[dimension]))))
                for dimension in feature_pipeline.DIMENSIONS
            }
            holland_code = career_mapping.derive_holland_code(predicted_scores, skill_use_totals)
            career_family = career_mapping.derive_career_family(predicted_scores)
            career_result = career_family
            source = "Runtime ML Model"
            model_version = runtime_status.model_version or "runtime-model"
        except Exception as exc:
            LOGGER.exception("Runtime model prediction failed for session_id=%s", payload.session_id)
            audit_log(
                "RUN_COMPLETE_RUNTIME_MODEL_FAILED",
                player_id=user.player_id,
                username=user.username,
                session_id=payload.session_id,
                reason=str(exc),
            )

    db.add_session_run(
        player_id=user.player_id,
        username=user.username,
        session_id=payload.session_id,
        scene_version=payload.scene_version,
        total_time_spent_seconds=payload.total_time_spent_seconds,
        rounds=[round_entry.model_dump() for round_entry in payload.rounds],
        aggregated_features=aggregated_features,
        riasec_scores=riasec_scores,
        holland_code=holland_code,
        career_family=career_family,
        career_result=career_result,
        source=source,
        model_version=model_version,
    )
    audit_log(
        "RUN_COMPLETE_RECORDED",
        player_id=user.player_id,
        username=user.username,
        session_id=payload.session_id,
        source=source,
        model_version=model_version,
        total_time_spent_seconds=payload.total_time_spent_seconds,
        rounds_recorded=len(payload.rounds),
    )

    return schemas.RunSummaryTelemetryOut(
        success=True,
        message=message,
        source=source,
        riasec_scores=schemas.RiasecScoresOut(**riasec_scores),
        holland_code=holland_code,
        career_family=career_family,
        career_result=career_result,
        model_version=model_version,
    )


@router.get(
    "/runtime-status",
    response_model=schemas.RuntimeStatusOut,
)
def runtime_status():
    return _build_runtime_status_response()


@predict_router.post(
    "/predict",
    response_model=schemas.PredictionOut,
)
def predict_cluster(
    payload: schemas.PredictionIn,
    db: TelemetryRepository = Depends(get_db),
):
    if len(payload.features) != 48:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="features must contain exactly 48 floats.",
        )

    context = _resolve_prediction_runtime_context(
        db,
        player_id=payload.player_id,
        session_id=payload.session_id,
    )

    if not context.cluster_status.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_sanitize_admin_error_message(
                context.cluster_status.reason,
                fallback="Cluster model is unavailable.",
            ),
        )

    if not context.career_status.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_sanitize_admin_error_message(
                context.career_status.reason,
                fallback="Career models are unavailable.",
            ),
        )

    try:
        predicted_cluster = cluster_runtime.predict_cluster(payload.features, context.cluster_status)
        career_cluster = cluster_runtime.predict_career_cluster(
            payload.features,
            predicted_cluster,
            context.career_status,
        )
        cluster_profile = cluster_mapping.get_cluster_profile(predicted_cluster)
        if cluster_profile is None:
            raise RuntimeError(f"Unsupported predicted cluster: {predicted_cluster}")
        career_result = (
            career_mapping.resolve_career_result(predicted_cluster, career_cluster)
            or _neutral_cluster_result(predicted_cluster, career_cluster)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Cluster prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed on the server.",
        ) from exc

    return schemas.PredictionOut(
        predicted_cluster=predicted_cluster,
        career_cluster=career_cluster,
        career_result=career_result,
        cluster_label=cluster_profile.label,
        career_family=cluster_profile.label,
        cluster_holland_code=cluster_profile.holland_code,
        cluster_example_careers=list(cluster_profile.example_careers),
        source=context.source,
        model_version=context.model_version,
        binding_source=context.binding_source,
        bundle_key=context.bundle_key,
    )


@router.post(
    "/session-cluster",
    response_model=schemas.SessionClusterTelemetryOut,
)
def record_session_cluster(
    payload: schemas.SessionClusterTelemetryIn,
    db: TelemetryRepository = Depends(get_db),
):
    user = db.find_user_by_player_id(payload.player_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found. Please log in again.",
        )
    _require_active_player_account(user)

    context = _resolve_prediction_runtime_context(
        db,
        player_id=payload.player_id,
        session_id=payload.session_id,
    )

    profile = cluster_mapping.get_cluster_profile(payload.predicted_cluster)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="predicted_cluster must be between 0 and 7.",
        )

    resolved_career_result = (
        payload.career_result
        or (
            career_mapping.resolve_career_result(
                payload.predicted_cluster,
                payload.career_cluster if payload.career_cluster is not None else -1,
            )
            if payload.career_cluster is not None
            else None
        )
        or _neutral_cluster_result(payload.predicted_cluster, payload.career_cluster)
    )
    cluster_source = context.source
    cluster_model_version = context.model_version
    updated_run = db.attach_cluster_result(
        player_id=payload.player_id,
        session_id=payload.session_id,
        predicted_cluster=payload.predicted_cluster,
        career_cluster=payload.career_cluster,
        career_result=resolved_career_result,
        career_family=profile.label,
        cluster_holland_code=profile.holland_code,
        cluster_label=profile.label,
        cluster_example_careers=list(profile.example_careers),
        cluster_source=cluster_source,
        cluster_model_version=cluster_model_version,
        cluster_binding_source=context.binding_source,
        cluster_bundle_key=context.bundle_key,
        cluster_bundle_ready=bool(context.bundle_key),
        cluster_dataset_id=context.dataset_id,
        cluster_dataset_name=context.dataset_name,
        cluster_binding_scope_type=context.binding_scope_type,
        cluster_binding_scope_id=context.binding_scope_id,
        cluster_binding_level_id=context.level_id,
    )
    if updated_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recorded session run was found for that player/session pair.",
        )

    audit_log(
        "RUN_CLUSTER_RECORDED",
        player_id=user.player_id,
        session_id=payload.session_id,
        predicted_cluster=payload.predicted_cluster,
        career_cluster=payload.career_cluster,
        career_result=resolved_career_result,
        holland_code=profile.holland_code,
        career_family=profile.label,
        source=cluster_source,
        model_version=cluster_model_version,
        binding_source=context.binding_source,
        bundle_key=context.bundle_key,
    )

    return schemas.SessionClusterTelemetryOut(
        success=True,
        message="Cluster result recorded successfully.",
        predicted_cluster=payload.predicted_cluster,
        career_cluster=payload.career_cluster,
        career_result=resolved_career_result,
        holland_code=profile.holland_code,
        career_family=profile.label,
        cluster_label=profile.label,
        cluster_example_careers=list(profile.example_careers),
        source=cluster_source,
        model_version=cluster_model_version,
        binding_source=context.binding_source,
        bundle_key=context.bundle_key,
    )


@admin_router.get(
    "/runtime-status",
    response_model=schemas.RuntimeStatusOut,
)
def admin_runtime_status():
    return _build_runtime_status_response()


@admin_router.get(
    "/datasets",
    response_model=list[schemas.AdminDatasetRecord],
)
def admin_list_datasets(db: TelemetryRepository = Depends(get_db)):
    return [schemas.AdminDatasetRecord(**row) for row in db.list_ml_datasets()]


@admin_router.get(
    "/quest-binding-targets",
    response_model=list[schemas.AdminQuestBindingTarget],
)
def admin_list_quest_binding_targets():
    return [schemas.AdminQuestBindingTarget(**row) for row in _build_quest_binding_targets()]


@admin_router.get(
    "/quest-bindings",
    response_model=list[schemas.AdminQuestModelBinding],
)
def admin_list_quest_bindings(db: TelemetryRepository = Depends(get_db)):
    return [schemas.AdminQuestModelBinding(**row) for row in db.list_quest_model_bindings()]


@admin_router.get(
    "/reports",
    response_model=list[schemas.AdminReportRow],
)
def admin_list_reports(
    q: str | None = None,
    source: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: TelemetryRepository = Depends(get_db),
):
    return [
        schemas.AdminReportRow(**row)
        for row in db.list_admin_report_rows(
            q=q,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
    ]


@admin_router.post(
    "/quest-bindings",
    response_model=schemas.AdminQuestModelBinding,
)
def admin_upsert_quest_binding(
    payload: schemas.AdminQuestModelBindingUpdateRequest,
    current_admin=Depends(require_admin),
    db: TelemetryRepository = Depends(get_db),
):
    target_map = {
        (target["scope_type"], target["scope_id"]): target
        for target in _build_quest_binding_targets()
    }
    target = target_map.get((payload.scope_type, payload.scope_id))
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Binding target not found.",
        )

    if payload.dataset_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="dataset_id is required for API binding updates.",
        )

    dataset = db.find_ml_dataset_by_id(payload.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )
    if dataset.get("status") != "trained" or not dataset.get("model_version") or not dataset.get("bundle_ready"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Choose a trained dataset with a saved bundle snapshot.",
        )

    binding = db.upsert_quest_model_binding(
        scope_type=payload.scope_type,
        scope_id=payload.scope_id,
        scope_label=target["scope_label"],
        level_id=target.get("level_id"),
        dataset_id=dataset["id"],
        dataset_name=dataset["dataset_name"],
        model_version=dataset["model_version"],
        bundle_key=dataset.get("bundle_key"),
        bundle_path=dataset.get("bundle_path"),
        bundle_ready=bool(dataset.get("bundle_ready")),
        updated_by_user_id=current_admin.id,
        updated_by_username=current_admin.username,
    )
    return schemas.AdminQuestModelBinding(**binding)


@admin_router.get(
    "/users",
    response_model=list[schemas.AdminUser],
)
def admin_list_users(
    sort_by: str = ADMIN_USER_SORT_DEFAULT_BY,
    sort_dir: str = ADMIN_USER_SORT_DEFAULT_DIR,
    db: TelemetryRepository = Depends(get_db),
):
    users = db.list_users()
    rows = _build_admin_user_rows(
        users,
        db.list_session_run_overview_by_player(),
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return [schemas.AdminUser(**row) for row in rows]


@admin_router.get(
    "/quest-attempts",
    response_model=list[schemas.AdminQuestLogRow],
)
def admin_list_quest_attempts(
    q: str | None = None,
    user_id: int | None = None,
    quest_id: str | None = None,
    result: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: TelemetryRepository = Depends(get_db),
):
    rows = db.list_admin_quest_attempt_rows(
        user_id=user_id,
        q=q,
        quest_id=quest_id,
        result=result,
        date_from=date_from,
        date_to=date_to,
    )
    return [schemas.AdminQuestLogRow(**row) for row in rows]


@admin_router.get(
    "/users/{user_id}",
    response_model=schemas.AdminUser,
)
def admin_get_user(user_id: int, db: TelemetryRepository = Depends(get_db)):
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    row = _build_admin_user_rows([user], db.list_session_run_overview_by_player())[0]
    return schemas.AdminUser(**row)


@admin_router.get(
    "/users/{user_id}/performance",
    response_model=schemas.UserPerformance,
)
def admin_get_user_performance(
    user_id: int,
    db: TelemetryRepository = Depends(get_db),
):
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    payload = _build_user_performance_payload(
        user=user,
        session_run_documents=db.list_session_runs_for_player(user.player_id),
        quest_attempt_rows=db.list_admin_quest_attempt_rows(user_id=user.id),
    )
    return schemas.UserPerformance(**payload)


@admin_router.post(
    "/users/{user_id}/set-email",
    response_model=schemas.AdminUser,
)
def admin_set_user_email(
    user_id: int,
    payload: schemas.AdminEmailUpdateRequest,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    _require_user_or_404(db, user_id)
    try:
        user = db.set_user_email(user_id, str(payload.email))
    except DuplicateUserError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this username or email already exists.",
        ) from exc

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_EMAIL_UPDATED",
        target_user_id=user.id,
        target_username=user.username,
        email=user.email,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/approve",
    response_model=schemas.AdminUser,
)
def admin_approve_user(
    user_id: int,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    target_user = _require_user_or_404(db, user_id)
    if target_user.role != "admin" and not target_user.email and target_user.email_verification_state != account_lifecycle.EMAIL_EXEMPT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Add an email address before approving this account.",
        )

    user, _, _ = _approve_and_maybe_send_verification(
        db=db,
        user_id=user_id,
        actor_user_id=current_admin.id,
        actor=current_admin,
        via="admin_api",
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/reject",
    response_model=schemas.AdminUser,
)
def admin_reject_user(
    user_id: int,
    payload: schemas.AdminRejectRequest,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    _require_user_or_404(db, user_id)
    user = db.reject_user(
        user_id,
        rejection_reason=payload.rejection_reason,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_REJECTED",
        target_user_id=user.id,
        target_username=user.username,
        rejection_reason=user.rejection_reason,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/send-verification",
    response_model=schemas.AdminUser,
)
def admin_send_verification_email(
    user_id: int,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    target_user = _require_user_or_404(db, user_id)
    _require_verification_ready_user(target_user)
    user, delivery_result = _send_verification_email(db, target_user)
    audit_log(
        "USER_VERIFICATION_SENT",
        target_user_id=user.id,
        target_username=user.username,
        delivery_mode=delivery_result.mode,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/resend-verification",
    response_model=schemas.AdminUser,
)
def admin_resend_verification_email(
    user_id: int,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    target_user = _require_user_or_404(db, user_id)
    _require_verification_ready_user(target_user)
    user, delivery_result = _send_verification_email(db, target_user)
    audit_log(
        "USER_VERIFICATION_RESENT",
        target_user_id=user.id,
        target_username=user.username,
        delivery_mode=delivery_result.mode,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/mark-verified",
    response_model=schemas.AdminUser,
)
def admin_mark_user_verified(
    user_id: int,
    current_admin=Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    target_user = _require_user_or_404(db, user_id)
    _require_verification_ready_user(target_user)
    user = db.mark_user_email_verified(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_EMAIL_VERIFIED_MANUAL",
        target_user_id=user.id,
        target_username=user.username,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return schemas.AdminUser(**_admin_row_for_user(db, user))


@admin_router.post(
    "/users/{user_id}/make-admin",
    response_model=schemas.AdminUser,
)
def admin_make_user_admin(
    user_id: int,
    current_admin = Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    existing_user = _require_user_or_404(db, user_id)
    user = db.set_user_role(user_id, "admin")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "USER_ROLE_CHANGED",
        target_user_id=user.id,
        target_username=user.username,
        old_role=existing_user.role,
        new_role=user.role,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )

    row = _build_admin_user_rows([user], db.list_session_run_overview_by_player())[0]
    return schemas.AdminUser(**row)


@admin_router.post(
    "/users/{user_id}/make-user",
    response_model=schemas.AdminUser,
)
def admin_make_user_standard(
    user_id: int,
    current_admin = Depends(get_current_user),
    db: TelemetryRepository = Depends(get_db),
):
    existing_user = _require_user_or_404(db, user_id)
    user = db.set_user_role(user_id, "user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "USER_ROLE_CHANGED",
        target_user_id=user.id,
        target_username=user.username,
        old_role=existing_user.role,
        new_role=user.role,
        via="admin_api",
        **_principal_fields(current_admin, prefix="actor"),
    )

    row = _build_admin_user_rows([user], db.list_session_run_overview_by_player())[0]
    return schemas.AdminUser(**row)


@public_router.get(
    "/verify-email",
    response_class=HTMLResponse,
)
def verify_email(token: str, request: Request, db: TelemetryRepository = Depends(get_db)):
    normalized_token = token.strip()
    if not normalized_token:
        return templates.TemplateResponse(
            request,
            "verify_email_result.html",
            {
                "success": False,
                "title": "Verification link missing",
                "message": "This verification link is incomplete.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    token_hash = account_lifecycle.hash_verification_token(normalized_token)
    user = db.find_user_by_verification_token_hash(token_hash)
    if not user:
        return templates.TemplateResponse(
            request,
            "verify_email_result.html",
            {
                "success": False,
                "title": "Verification link invalid",
                "message": "This verification link is invalid, expired, or already used.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if user.verification_expires_at and user.verification_expires_at <= datetime.utcnow():
        user = db.mark_user_email_expired(user.id) or user
        audit_log(
            "USER_EMAIL_VERIFICATION_FAILED",
            user_id=user.id,
            username=user.username,
            reason="expired",
            via="public_link",
        )
        return templates.TemplateResponse(
            request,
            "verify_email_result.html",
            {
                "success": False,
                "title": "Verification link expired",
                "message": "This verification link expired. Ask an admin to resend it.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    updated_user = db.mark_user_email_verified(user.id) or user
    audit_log(
        "USER_EMAIL_VERIFIED",
        user_id=updated_user.id,
        username=updated_user.username,
        via="public_link",
    )
    return templates.TemplateResponse(
        request,
        "verify_email_result.html",
        {
            "success": True,
            "title": "Email verified",
            "message": "Your account is verified. You can return to the game and log in.",
        },
    )


@admin_ui_router.get(
    "",
    response_class=HTMLResponse,
)
@admin_ui_router.get(
    "/",
    response_class=HTMLResponse,
)
def admin_root(request: Request, db: TelemetryRepository = Depends(get_db)):
    user = _current_ui_user(request, db)
    if user is None:
        return RedirectResponse(url="/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=_dashboard_url_for_user(user), status_code=status.HTTP_303_SEE_OTHER)


@admin_ui_router.get(
    "/login",
    response_class=HTMLResponse,
)
def admin_login_page(
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _current_ui_user(request, db)
    if current_user is not None:
        return RedirectResponse(
            url=_dashboard_url_for_user(current_user),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return templates.TemplateResponse(
        request,
        "admin_login.html",
        _flash_context(request),
    )


@admin_ui_router.post("/login")
def admin_login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: TelemetryRepository = Depends(get_db),
):
    normalized_username = username.strip()
    if not normalized_username:
        return templates.TemplateResponse(
            request,
            "admin_login.html",
            {
                "error_message": "Username cannot be empty.",
                "notice_message": None,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    user = db.find_user_by_username(normalized_username)
    if user is None or not user.password_hash or not verify_password(password, user.password_hash):
        audit_log(
            "DASHBOARD_LOGIN_FAILED",
            username=normalized_username or None,
            reason="invalid_credentials",
            via="web",
        )
        return templates.TemplateResponse(
            request,
            "admin_login.html",
            {
                "error_message": "Invalid username or password.",
                "notice_message": None,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    access_state = describe_user_access(user)
    if not access_state.can_login:
        if access_state.email_verification_state == account_lifecycle.EMAIL_EXPIRED:
            user = db.mark_user_email_expired(user.id) or user
            access_state = describe_user_access(user)
        audit_log(
            "DASHBOARD_LOGIN_FAILED",
            username=normalized_username,
            reason=access_state.next_step,
            via="web",
        )
        return templates.TemplateResponse(
            request,
            "admin_login.html",
            {
                "error_message": access_state.message,
                "notice_message": None,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    updated_user = db.touch_last_login(user.id) or user
    audit_log(
        "USER_LOGGED_IN",
        user_id=updated_user.id,
        username=updated_user.username,
        role=updated_user.role,
        via="web",
    )
    response = RedirectResponse(
        url=_dashboard_url_for_user(updated_user),
        status_code=status.HTTP_303_SEE_OTHER,
    )
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=create_access_token(
            user_id=updated_user.id,
            username=updated_user.username,
            role=updated_user.role,
        ),
        httponly=True,
        samesite="lax",
    )
    return response


@admin_ui_router.post("/logout")
def admin_logout(request: Request, db: TelemetryRepository = Depends(get_db)):
    current_user = _current_ui_user(request, db)
    if current_user is not None:
        audit_log(
            "USER_LOGGED_OUT",
            user_id=current_user.id,
            username=current_user.username,
            role=current_user.role,
            via="web",
        )
    response = RedirectResponse(url="/admin/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(ADMIN_SESSION_COOKIE)
    return response


@admin_ui_router.get(
    "/users",
    response_class=HTMLResponse,
)
def admin_users_page(
    request: Request,
    sort_by: str = ADMIN_USER_SORT_DEFAULT_BY,
    sort_dir: str = ADMIN_USER_SORT_DEFAULT_DIR,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _require_ui_user(request, db)
    if current_user.role != "admin":
        return RedirectResponse(
            url=f"/admin/users/{current_user.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    normalized_sort_by, normalized_sort_dir = _normalize_admin_user_sort(sort_by, sort_dir)
    users = _build_admin_user_rows(
        db.list_users(),
        db.list_session_run_overview_by_player(),
        sort_by=normalized_sort_by,
        sort_dir=normalized_sort_dir,
    )
    return templates.TemplateResponse(
        request,
        "users.html",
        {
            "current_user": current_user,
            "users": users,
            "sort_by": normalized_sort_by,
            "sort_dir": normalized_sort_dir,
            "sort_options": _build_admin_user_sort_options(),
            "sort_summary": f"{ADMIN_USER_SORT_LABELS[normalized_sort_by]} {normalized_sort_dir.upper()}",
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/datasets",
    response_class=HTMLResponse,
)
def admin_datasets_page(request: Request, db: TelemetryRepository = Depends(get_db)):
    current_user = _require_admin_ui_user(request, db)
    raw_dataset_rows = db.list_ml_datasets()
    dataset_rows = _format_admin_dataset_rows(raw_dataset_rows)
    runtime_status = _build_runtime_status_response()
    trained_count = sum(
        1 for row in dataset_rows if row.get("status") == "trained" and row.get("bundle_ready")
    )
    latest_trained_row = max(
        (
            row
            for row in raw_dataset_rows
            if row.get("trained_at") and row.get("bundle_ready")
        ),
        default=None,
        key=lambda row: row["trained_at"],
    )
    return templates.TemplateResponse(
        request,
        "datasets.html",
        {
            "current_user": current_user,
            "datasets": dataset_rows,
            "runtime_status": runtime_status,
            "dataset_counts": {
                "total": len(dataset_rows),
                "trained": trained_count,
                "latest_trained_at": _format_admin_datetime(latest_trained_row.get("trained_at")) if latest_trained_row else None,
                "latest_model_version": latest_trained_row.get("model_version") if latest_trained_row else None,
            },
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/quest-bindings",
    response_class=HTMLResponse,
)
def admin_quest_bindings_page(request: Request, db: TelemetryRepository = Depends(get_db)):
    current_user = _require_admin_ui_user(request, db)
    target_rows = _build_quest_binding_targets()
    binding_rows = _format_admin_binding_rows(db.list_quest_model_bindings())
    binding_map = {
        (row["scope_type"], row["scope_id"]): row
        for row in binding_rows
    }
    dataset_rows = _format_admin_dataset_rows(db.list_ml_datasets())
    trained_datasets = [
        row
        for row in dataset_rows
        if row.get("status") == "trained" and row.get("model_version") and row.get("bundle_ready")
    ]

    rows: list[dict[str, Any]] = []
    bound_count = 0
    for target in target_rows:
        row = dict(target)
        exact_binding = binding_map.get((target["scope_type"], target["scope_id"]))
        resolved_info = db.resolve_quest_model_binding(
            scope_type=target["scope_type"],
            scope_id=target["scope_id"],
            level_id=target.get("level_id"),
        )
        resolved_binding = (
            _format_admin_binding_row(resolved_info["binding"])
            if resolved_info
            else None
        )
        binding_source = resolved_info["binding_source"] if resolved_info else None

        row["exact_binding"] = exact_binding
        row["binding"] = resolved_binding
        row["binding_source"] = binding_source
        row["binding_source_label"] = (
            "Exact binding"
            if exact_binding is not None
            else (
                f"Level fallback via {target['level_id']}"
                if resolved_binding is not None and target.get("level_id")
                else "No binding yet"
            )
        )
        if resolved_binding is not None:
            bound_count += 1
        rows.append(row)

    return templates.TemplateResponse(
        request,
        "quest_bindings.html",
        {
            "current_user": current_user,
            "bindings": rows,
            "trained_datasets": trained_datasets,
            "binding_counts": {
                "targets": len(rows),
                "bound": bound_count,
                "trained_datasets": len(trained_datasets),
            },
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/reports",
    response_class=HTMLResponse,
)
def admin_reports_page(
    request: Request,
    q: str = "",
    source: str = "",
    date_from: date | None = None,
    date_to: date | None = None,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _require_admin_ui_user(request, db)
    report_rows = _format_admin_report_rows(
        db.list_admin_report_rows(
            q=q,
            source=source,
            date_from=date_from,
            date_to=date_to,
        )
    )
    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            "current_user": current_user,
            "reports": report_rows,
            "filters": {
                "q": q,
                "source": source,
                "date_from": date_from.isoformat() if date_from else "",
                "date_to": date_to.isoformat() if date_to else "",
            },
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/reports/{player_id}/{session_id}",
    response_class=HTMLResponse,
)
def admin_report_viewer_page(
    player_id: str,
    session_id: str,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _require_admin_ui_user(request, db)
    document = db.find_session_run_for_player(player_id=player_id, session_id=session_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    user = db.find_user_by_player_id(player_id)
    run = _summarize_session_run(document)
    return templates.TemplateResponse(
        request,
        "report_viewer.html",
        {
            "current_user": current_user,
            "user": user,
            "run": run,
            "player_id": player_id,
            "session_id": session_id,
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/quest-logs",
    response_class=HTMLResponse,
)
def admin_quest_logs_page(
    request: Request,
    q: str = "",
    user_id: int | None = None,
    quest_id: str = "",
    result: str = "",
    date_from: date | None = None,
    date_to: date | None = None,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _require_admin_ui_user(request, db)
    quest_attempt_rows = db.list_admin_quest_attempt_rows(
        user_id=user_id,
        q=q,
        quest_id=quest_id,
        result=result,
        date_from=date_from,
        date_to=date_to,
    )
    formatted_rows = _format_admin_quest_attempt_rows(quest_attempt_rows)
    return templates.TemplateResponse(
        request,
        "quest_logs.html",
        {
            "current_user": current_user,
            "quest_logs": formatted_rows,
            "filters": {
                "q": q,
                "user_id": user_id,
                "quest_id": quest_id,
                "result": result,
                "date_from": date_from.isoformat() if date_from else "",
                "date_to": date_to.isoformat() if date_to else "",
            },
            **_flash_context(request),
        },
    )


@admin_ui_router.get(
    "/users/{user_id}",
    response_class=HTMLResponse,
)
def admin_user_performance_page(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_user = _require_ui_user(request, db)
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own dashboard.",
        )

    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    performance = _build_user_performance_payload(
        user=user,
        session_run_documents=db.list_session_runs_for_player(user.player_id),
        quest_attempt_rows=db.list_admin_quest_attempt_rows(user_id=user.id),
    )

    latest_source = performance["latest_source"] or "No runs yet"
    pending_cluster_label = "Pending cluster result" if performance["total_runs"] > 0 else "No cluster yet"
    summary = {
        "created_at": performance["created_at"],
        "last_login": performance["last_login"],
        "total_runs": performance["total_runs"],
        "avg_clear_rate": performance["avg_clear_rate"],
        "avg_time_seconds": performance["avg_time_seconds"],
        "avg_stars_per_run": performance["avg_stars_per_run"],
        "latest_source": latest_source,
        "latest_result": performance["latest_result"] or pending_cluster_label,
        "latest_holland_code": performance["latest_holland_code"] or "N/A",
        "latest_career_family": performance["latest_career_family"] or pending_cluster_label,
        "latest_model_version": performance["latest_model_version"] or "n/a",
        "latest_predicted_cluster": performance["latest_predicted_cluster"],
        "latest_career_cluster": performance["latest_career_cluster"],
        "latest_career_result": performance["latest_career_result"],
        "latest_cluster_label": performance["latest_cluster_label"] or pending_cluster_label,
        "latest_cluster_holland_code": performance["latest_cluster_holland_code"] or "N/A",
        "latest_cluster_source": performance["latest_cluster_source"] or latest_source,
        "latest_cluster_model_version": performance["latest_cluster_model_version"] or "n/a",
        "latest_cluster_example_careers": performance["latest_cluster_example_careers"] or [],
        "latest_riasec": performance["latest_riasec"],
        "approval_state": performance["approval_state"],
        "email_verification_state": performance["email_verification_state"],
        "approved_at": performance["approved_at"],
        "verification_sent_at": performance["verification_sent_at"],
        "verified_at": performance["verified_at"],
        "rejection_reason": performance["rejection_reason"],
    }

    return templates.TemplateResponse(
        request,
        "user_performance.html",
        {
            "current_user": current_user,
            "user": user,
            "can_manage_user": current_user.role == "admin",
            "runs": performance["runs"],
            "quest_attempts": _format_admin_quest_attempt_rows(performance["quest_attempts"]),
            "summary": summary,
            **_flash_context(request),
        },
    )


@admin_ui_router.post("/quest-bindings/assign")
def admin_assign_quest_binding(
    request: Request,
    scope_type: str = Form(...),
    scope_id: str = Form(...),
    dataset_id: int = Form(...),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    normalized_scope_type = (scope_type or "").strip().lower()
    normalized_scope_id = (scope_id or "").strip()

    target_map = {
        (target["scope_type"], target["scope_id"]): target
        for target in _build_quest_binding_targets()
    }
    target = target_map.get((normalized_scope_type, normalized_scope_id))
    if not target:
        return _redirect_with_flash("/admin/quest-bindings", error="Binding target not found.")

    dataset = db.find_ml_dataset_by_id(dataset_id)
    if not dataset:
        return _redirect_with_flash("/admin/quest-bindings", error="Dataset not found.")
    if dataset.get("status") != "trained" or not dataset.get("model_version") or not dataset.get("bundle_ready"):
        return _redirect_with_flash(
            "/admin/quest-bindings",
            error="Choose a trained dataset with a saved bundle snapshot.",
        )

    binding = db.upsert_quest_model_binding(
        scope_type=normalized_scope_type,
        scope_id=normalized_scope_id,
        scope_label=target["scope_label"],
        level_id=target.get("level_id"),
        dataset_id=dataset["id"],
        dataset_name=dataset["dataset_name"],
        model_version=dataset["model_version"],
        bundle_key=dataset.get("bundle_key"),
        bundle_path=dataset.get("bundle_path"),
        bundle_ready=bool(dataset.get("bundle_ready")),
        updated_by_user_id=current_admin.id,
        updated_by_username=current_admin.username,
    )
    audit_log(
        "QUEST_MODEL_BOUND",
        scope_type=binding["scope_type"],
        scope_id=binding["scope_id"],
        scope_label=binding["scope_label"],
        dataset_id=binding["dataset_id"],
        dataset_name=binding["dataset_name"],
        model_version=binding["model_version"],
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        "/admin/quest-bindings",
        notice=(
            f"Binding updated: {binding['scope_label']} -> "
            f"{binding['dataset_name']} ({binding['model_version']})"
        ),
    )


@admin_ui_router.post("/quest-bindings/clear")
def admin_clear_quest_binding(
    request: Request,
    scope_type: str = Form(...),
    scope_id: str = Form(...),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    normalized_scope_type = (scope_type or "").strip().lower()
    normalized_scope_id = (scope_id or "").strip()
    existing = db.find_quest_model_binding(
        scope_type=normalized_scope_type,
        scope_id=normalized_scope_id,
    )
    if not existing:
        return _redirect_with_flash("/admin/quest-bindings", error="Binding not found.")

    deleted = db.delete_quest_model_binding(
        scope_type=normalized_scope_type,
        scope_id=normalized_scope_id,
    )
    if not deleted:
        return _redirect_with_flash("/admin/quest-bindings", error="Binding not found.")

    audit_log(
        "QUEST_MODEL_BINDING_CLEARED",
        scope_type=existing["scope_type"],
        scope_id=existing["scope_id"],
        scope_label=existing.get("scope_label"),
        dataset_id=existing.get("dataset_id"),
        dataset_name=existing.get("dataset_name"),
        model_version=existing.get("model_version"),
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        "/admin/quest-bindings",
        notice=f"Binding cleared: {existing.get('scope_label') or existing['scope_id']}",
    )


@admin_ui_router.post("/datasets/upload")
def admin_upload_dataset_page(
    request: Request,
    dataset_name: str = Form(""),
    dataset_file: UploadFile = File(...),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)

    original_filename = (dataset_file.filename or "").strip()
    if not original_filename.lower().endswith(".csv"):
        return _redirect_with_flash("/admin/datasets", error="Upload a CSV dataset file.")

    raw_bytes = dataset_file.file.read()
    try:
        summary = model_training.save_uploaded_dataset(
            raw_bytes,
            dataset_name=dataset_name,
            original_filename=original_filename,
        )
    except ValueError as exc:
        return _redirect_with_flash(
            "/admin/datasets",
            error=_sanitize_admin_error_message(
                str(exc),
                fallback="Dataset upload failed.",
            ),
        )

    document = db.create_ml_dataset(
        dataset_name=summary.dataset_name,
        original_filename=summary.original_filename,
        stored_filename=summary.stored_filename,
        storage_path=summary.storage_path,
        file_size_bytes=summary.file_size_bytes,
        row_count=summary.row_count,
        feature_count=summary.feature_count,
        label_count=summary.label_count,
        uploaded_by_user_id=current_admin.id,
        uploaded_by_username=current_admin.username,
    )
    audit_log(
        "DATASET_UPLOADED",
        dataset_id=document["id"],
        dataset_name=document["dataset_name"],
        row_count=document["row_count"],
        filename=document["original_filename"],
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        "/admin/datasets",
        notice=f"Dataset uploaded: {document['dataset_name']}",
    )


@admin_ui_router.post("/datasets/{dataset_id}/train")
def admin_train_dataset_page(
    dataset_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    dataset = db.find_ml_dataset_by_id(dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    try:
        summary = model_training.train_runtime_model_from_dataset(
            dataset["storage_path"],
            dataset_id=dataset["id"],
            dataset_name=dataset["dataset_name"],
        )
        updated_dataset = db.update_ml_dataset_training(
            dataset_id,
            status="trained",
            trained_at=summary.trained_at,
            activated_at=summary.trained_at,
            model_version=summary.model_version,
            bundle_key=summary.bundle_key,
            bundle_path=summary.bundle_path,
            bundle_ready=summary.bundle_ready,
            mean_absolute_error=None,
            r2_score=None,
            cluster_accuracy=summary.cluster_accuracy,
            career_model_count=summary.career_model_count,
            clusters_covered_count=summary.clusters_covered_count,
            training_row_count=summary.training_row_count,
            validation_row_count=summary.validation_row_count,
        )
    except Exception as exc:
        safe_error = _sanitize_admin_error_message(
            str(exc),
            fallback="Training failed on the server.",
        )
        db.mark_ml_dataset_training_failed(dataset_id, error_message=safe_error)
        LOGGER.exception("Dataset training failed for dataset_id=%s", dataset_id)
        audit_log(
            "DATASET_TRAINING_FAILED",
            dataset_id=dataset["id"],
            dataset_name=dataset["dataset_name"],
            reason=safe_error,
            via="admin_web",
            **_principal_fields(current_admin, prefix="actor"),
        )
        return _redirect_with_flash("/admin/datasets", error=f"Training failed: {safe_error}")

    audit_log(
        "DATASET_TRAINED",
        dataset_id=dataset["id"],
        dataset_name=dataset["dataset_name"],
        model_version=summary.model_version,
        row_count=summary.row_count,
        validation_row_count=summary.validation_row_count,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        "/admin/datasets",
        notice=(
            f"Live bundle trained from {updated_dataset['dataset_name']}. "
            f"Active version: {updated_dataset['model_version']}"
        ),
    )


@admin_ui_router.post("/users/{user_id}/email")
def admin_set_user_email_page(
    user_id: int,
    request: Request,
    email: str = Form(...),
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    try:
        normalized_email = _validate_email_form_input(email)
    except ValidationError:
        return _redirect_with_flash(destination, error="Enter a valid email address.")

    try:
        user = db.set_user_email(user_id, normalized_email)
    except DuplicateUserError:
        return _redirect_with_flash(
            destination,
            error="A user with that email already exists.",
        )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_EMAIL_UPDATED",
        target_user_id=user.id,
        target_username=user.username,
        email=user.email,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(destination, notice="Email updated.")


@admin_ui_router.post("/users/{user_id}/approve")
def admin_approve_user_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    target_user = _require_user_or_404(db, user_id)
    if target_user.role != "admin" and not target_user.email and target_user.email_verification_state != account_lifecycle.EMAIL_EXEMPT:
        return _redirect_with_flash(
            destination,
            error="Add an email address before approving this account.",
        )

    _, notice, notice_link = _approve_and_maybe_send_verification(
        db=db,
        user_id=user_id,
        actor_user_id=current_admin.id,
        actor=current_admin,
        via="admin_web",
    )
    return _redirect_with_flash(destination, notice=notice, notice_link=notice_link)


@admin_ui_router.post("/users/{user_id}/reject")
def admin_reject_user_page(
    user_id: int,
    request: Request,
    rejection_reason: str = Form(""),
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    user = db.reject_user(
        user_id,
        rejection_reason=rejection_reason,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_REJECTED",
        target_user_id=user.id,
        target_username=user.username,
        rejection_reason=user.rejection_reason,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(destination, notice="Account rejected.")


@admin_ui_router.post("/users/{user_id}/send-verification")
def admin_send_verification_email_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    target_user = _require_user_or_404(db, user_id)
    try:
        _require_verification_ready_user(target_user)
        user, delivery_result = _send_verification_email(db, target_user)
    except HTTPException as exc:
        return _redirect_with_flash(destination, error=str(exc.detail))

    audit_log(
        "USER_VERIFICATION_SENT",
        target_user_id=user.id,
        target_username=user.username,
        delivery_mode=delivery_result.mode,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        destination,
        notice=delivery_result.message,
        notice_link=delivery_result.verification_link,
    )


@admin_ui_router.post("/users/{user_id}/resend-verification")
def admin_resend_verification_email_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    target_user = _require_user_or_404(db, user_id)
    try:
        _require_verification_ready_user(target_user)
        user, delivery_result = _send_verification_email(db, target_user)
    except HTTPException as exc:
        return _redirect_with_flash(destination, error=str(exc.detail))

    audit_log(
        "USER_VERIFICATION_RESENT",
        target_user_id=user.id,
        target_username=user.username,
        delivery_mode=delivery_result.mode,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(
        destination,
        notice=delivery_result.message,
        notice_link=delivery_result.verification_link,
    )


@admin_ui_router.post("/users/{user_id}/mark-verified")
def admin_mark_user_verified_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    target_user = _require_user_or_404(db, user_id)
    try:
        _require_verification_ready_user(target_user)
    except HTTPException as exc:
        return _redirect_with_flash(destination, error=str(exc.detail))

    user = db.mark_user_email_verified(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    audit_log(
        "USER_EMAIL_VERIFIED_MANUAL",
        target_user_id=user.id,
        target_username=user.username,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )
    return _redirect_with_flash(destination, notice="Email marked as verified.")


@admin_ui_router.post("/users/{user_id}/make-admin")
def admin_make_user_admin_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    existing_user = _require_user_or_404(db, user_id)
    user = db.set_user_role(user_id, "admin")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "USER_ROLE_CHANGED",
        target_user_id=user.id,
        target_username=user.username,
        old_role=existing_user.role,
        new_role=user.role,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )

    return _redirect_with_flash(destination, notice="Role updated to admin.")


@admin_ui_router.post("/users/{user_id}/make-user")
def admin_make_user_standard_page(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or f"/admin/users/{user_id}"
    existing_user = _require_user_or_404(db, user_id)
    user = db.set_user_role(user_id, "user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "USER_ROLE_CHANGED",
        target_user_id=user.id,
        target_username=user.username,
        old_role=existing_user.role,
        new_role=user.role,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )

    return _redirect_with_flash(destination, notice="Role updated to user.")


@admin_ui_router.post("/users/{user_id}/delete")
def admin_delete_user(
    user_id: int,
    request: Request,
    redirect_to: str = Form(""),
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
    destination = redirect_to or "/admin/users"
    target_user = _require_user_or_404(db, user_id)
    deleted = db.delete_user(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    audit_log(
        "USER_DELETED",
        target_user_id=target_user.id,
        target_username=target_user.username,
        target_role=target_user.role,
        via="admin_web",
        **_principal_fields(current_admin, prefix="actor"),
    )

    return _redirect_with_flash(destination, notice="User deleted.")

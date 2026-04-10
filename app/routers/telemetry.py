from datetime import datetime
import logging
from pathlib import Path
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import cluster_mapping, feature_pipeline, ml_runtime, schemas
from app.database import get_db
from app.logging_utils import audit_log
from app.repository import DuplicateUserError, TelemetryRepository
from app.security import (
    ADMIN_SESSION_COOKIE,
    create_access_token,
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
admin_router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
admin_ui_router = APIRouter(prefix="/admin", tags=["admin-ui"])

templates = Jinja2Templates(directory="templates")


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
    return schemas.AuthResponse(
        id=user.id,
        player_id=user.player_id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        last_login=user.last_login,
        role=user.role,
        access_token=create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role,
        ),
    )


def _build_runtime_status_response() -> schemas.RuntimeStatusOut:
    cluster_status = ml_runtime.get_cluster_model_status()
    if cluster_status.available:
        return schemas.RuntimeStatusOut(
            rf_model_loaded=True,
            active_source="Cluster Model",
            active_version=_resolve_cluster_model_version(),
            detail="Cluster model loaded and ready.",
        )

    return schemas.RuntimeStatusOut(
        rf_model_loaded=False,
        active_source="Cluster Model",
        active_version=_resolve_cluster_model_version(),
        detail=cluster_status.reason or "Cluster model unavailable.",
    )


def _resolve_cluster_model_version() -> str:
    cluster_status = ml_runtime.get_cluster_model_status()
    if cluster_status.model_path:
        return Path(cluster_status.model_path).name
    return "cluster-model"


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


def _summarize_session_run(document: dict[str, Any]) -> dict[str, Any]:
    rounds = document.get("rounds") or []
    rounds_attempted = len(rounds)
    rounds_cleared = sum(1 for round_entry in rounds if round_entry.get("solved"))
    total_stars = sum(int(round_entry.get("stars_earned", 0) or 0) for round_entry in rounds)

    return {
        "session_id": document.get("session_id", "unknown-session"),
        "created_at": document.get("created_at") or datetime.utcnow(),
        "source": document.get("source") or "Unknown",
        "model_version": document.get("model_version") or "n/a",
        "riasec_scores": _normalize_admin_riasec_scores(document.get("riasec_scores")),
        "holland_code": document.get("holland_code") or "N/A",
        "career_family": document.get("career_family") or "N/A",
        "career_result": document.get("career_result") or "N/A",
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


def _build_admin_user_rows(
    users: list,
    run_overview_by_player: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user in users:
        overview = run_overview_by_player.get(user.player_id, {})
        rows.append(
            {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "role": user.role,
                "total_runs": int(overview.get("total_runs", 0) or 0),
                "last_run_at": overview.get("last_run_at"),
                "last_source": overview.get("last_source"),
                "last_result": overview.get("last_result"),
                "last_holland_code": overview.get("last_holland_code"),
                "last_predicted_cluster": overview.get("last_predicted_cluster"),
                "last_cluster_label": overview.get("last_cluster_label"),
                "last_cluster_holland_code": overview.get("last_cluster_holland_code"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["total_runs"],
            row["last_run_at"] or datetime.min,
            row["last_login"] or datetime.min,
        ),
        reverse=True,
    )
    return rows


def _build_user_performance_payload(
    *,
    user,
    session_run_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    runs = [_summarize_session_run(document) for document in session_run_documents]
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
        "total_runs": total_runs,
        "avg_clear_rate": avg_clear_rate,
        "avg_time_seconds": avg_time_seconds,
        "avg_stars_per_run": avg_stars_per_run,
        "latest_source": (latest_run["cluster_source"] or latest_run["source"]) if latest_run else None,
        "latest_result": latest_run["cluster_label"] if latest_run else None,
        "latest_holland_code": latest_run["cluster_holland_code"] if latest_run else None,
        "latest_career_family": latest_run["cluster_label"] if latest_run else None,
        "latest_model_version": latest_run["cluster_model_version"] if latest_run else None,
        "latest_riasec": latest_run["riasec_scores"] if latest_run else _empty_admin_riasec_scores(),
        "latest_predicted_cluster": latest_run["predicted_cluster"] if latest_run else None,
        "latest_cluster_label": latest_run["cluster_label"] if latest_run else None,
        "latest_cluster_holland_code": latest_run["cluster_holland_code"] if latest_run else None,
        "latest_cluster_source": latest_run["cluster_source"] if latest_run else None,
        "latest_cluster_model_version": latest_run["cluster_model_version"] if latest_run else None,
        "latest_cluster_example_careers": latest_run["cluster_example_careers"] if latest_run else [],
        "runs": runs,
    }


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
    username = user_in.username.strip()
    password_hash = hash_password(user_in.password)
    requested_role = user_in.role
    assigned_role = "user"
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
            email=None,
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
            **_principal_fields(current_user, prefix="actor"),
        )
        return _build_auth_response(user)
    except DuplicateUserError as exc:
        upgraded_user = db.upgrade_legacy_user_password(
            username=username,
            password_hash=password_hash,
        )
        if upgraded_user:
            if upgraded_user.role != assigned_role:
                upgraded_user = db.set_user_role(upgraded_user.id, assigned_role) or upgraded_user
            audit_log(
                "USER_REGISTERED",
                user_id=upgraded_user.id,
                username=upgraded_user.username,
                role=upgraded_user.role,
                mode="upgrade_legacy_password",
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

    db.add_session_run(
        player_id=payload.player_id,
        username=payload.username,
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
        player_id=payload.player_id,
        username=payload.username,
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
def predict_cluster(payload: schemas.PredictionIn):
    if len(payload.features) != 48:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="features must contain exactly 48 floats.",
        )

    cluster_status = ml_runtime.get_cluster_model_status()
    if not cluster_status.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=cluster_status.reason or "Cluster model is unavailable.",
        )

    try:
        predicted_cluster = ml_runtime.predict_cluster(payload.features, cluster_status)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOGGER.exception("Cluster prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return schemas.PredictionOut(predicted_cluster=predicted_cluster)


@router.post(
    "/session-cluster",
    response_model=schemas.SessionClusterTelemetryOut,
)
def record_session_cluster(
    payload: schemas.SessionClusterTelemetryIn,
    db: TelemetryRepository = Depends(get_db),
):
    profile = cluster_mapping.get_cluster_profile(payload.predicted_cluster)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="predicted_cluster must be between 0 and 8.",
        )

    cluster_source = "Cluster Model"
    cluster_model_version = _resolve_cluster_model_version()
    updated_run = db.attach_cluster_result(
        player_id=payload.player_id,
        session_id=payload.session_id,
        predicted_cluster=payload.predicted_cluster,
        cluster_holland_code=profile.holland_code,
        cluster_label=profile.label,
        cluster_example_careers=list(profile.example_careers),
        cluster_source=cluster_source,
        cluster_model_version=cluster_model_version,
    )
    if updated_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recorded session run was found for that player/session pair.",
        )

    audit_log(
        "RUN_CLUSTER_RECORDED",
        player_id=payload.player_id,
        session_id=payload.session_id,
        predicted_cluster=payload.predicted_cluster,
        holland_code=profile.holland_code,
        career_family=profile.label,
        source=cluster_source,
        model_version=cluster_model_version,
    )

    return schemas.SessionClusterTelemetryOut(
        success=True,
        message="Cluster result recorded successfully.",
        predicted_cluster=payload.predicted_cluster,
        holland_code=profile.holland_code,
        career_family=profile.label,
        source=cluster_source,
        model_version=cluster_model_version,
    )


@admin_router.get(
    "/runtime-status",
    response_model=schemas.RuntimeStatusOut,
)
def admin_runtime_status():
    return _build_runtime_status_response()


@admin_router.get(
    "/users",
    response_model=list[schemas.AdminUser],
)
def admin_list_users(db: TelemetryRepository = Depends(get_db)):
    users = db.list_users()
    rows = _build_admin_user_rows(users, db.list_session_run_overview_by_player())
    return [schemas.AdminUser(**row) for row in rows]


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
    )
    return schemas.UserPerformance(**payload)


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
        "admin_login.html",
        {
            "request": request,
            "error_message": None,
        },
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
            "admin_login.html",
            {
                "request": request,
                "error_message": "Username cannot be empty.",
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
            "admin_login.html",
            {
                "request": request,
                "error_message": "Invalid username or password.",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
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
def admin_users_page(request: Request, db: TelemetryRepository = Depends(get_db)):
    current_user = _require_ui_user(request, db)
    if current_user.role != "admin":
        return RedirectResponse(
            url=f"/admin/users/{current_user.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    users = _build_admin_user_rows(
        db.list_users(),
        db.list_session_run_overview_by_player(),
    )
    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "current_user": current_user,
            "users": users,
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
    )

    latest_source = performance["latest_source"] or "No runs yet"
    pending_cluster_label = "Pending cluster result" if performance["total_runs"] > 0 else "No cluster yet"
    summary = {
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
        "latest_cluster_label": performance["latest_cluster_label"] or pending_cluster_label,
        "latest_cluster_holland_code": performance["latest_cluster_holland_code"] or "N/A",
        "latest_cluster_source": performance["latest_cluster_source"] or latest_source,
        "latest_cluster_model_version": performance["latest_cluster_model_version"] or "n/a",
        "latest_cluster_example_careers": performance["latest_cluster_example_careers"] or [],
    }

    return templates.TemplateResponse(
        "user_performance.html",
        {
            "request": request,
            "current_user": current_user,
            "user": user,
            "can_manage_user": current_user.role == "admin",
            "runs": performance["runs"],
            "summary": summary,
        },
    )


@admin_ui_router.post("/users/{user_id}/make-admin")
def admin_make_user_admin_page(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
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

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=status.HTTP_303_SEE_OTHER)


@admin_ui_router.post("/users/{user_id}/make-user")
def admin_make_user_standard_page(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
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

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=status.HTTP_303_SEE_OTHER)


@admin_ui_router.post("/users/{user_id}/delete")
def admin_delete_user(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    current_admin = _require_admin_ui_user(request, db)
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

    return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)

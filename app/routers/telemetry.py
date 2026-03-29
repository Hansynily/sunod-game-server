from datetime import datetime
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import career_mapping, feature_pipeline, ml_runtime, rubric, schemas
from app.database import get_db
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
admin_router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)
admin_ui_router = APIRouter(prefix="/admin", tags=["admin-ui"])

templates = Jinja2Templates(directory="templates")


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


def _calculate_rubric_integer_scores(
    payload: schemas.RunSummaryTelemetryIn,
) -> dict[str, int]:
    return rubric.calculate_riasec_scores(payload.rounds).integer_scores


def _build_run_complete_message(
    *,
    source: str,
    model_version: str,
    detail: str | None = None,
) -> str:
    if source == "RF Model":
        return (
            "Run-complete telemetry recorded successfully using the trained RF model "
            f"({model_version})."
        )

    message = (
        "Run-complete telemetry recorded successfully using backend rubric fallback "
        f"({model_version})."
    )
    if detail:
        message += f" {detail}"
    return message


def _build_runtime_status_response() -> schemas.RuntimeStatusOut:
    runtime_status = ml_runtime.get_model_status()
    if runtime_status.available:
        return schemas.RuntimeStatusOut(
            rf_model_loaded=True,
            active_source="RF Model",
            active_version=runtime_status.model_version or "rf_v1",
            detail="Runtime model loaded and ready.",
        )

    return schemas.RuntimeStatusOut(
        rf_model_loaded=False,
        active_source="Backend Rubric Fallback",
        active_version="rubric_v1",
        detail=runtime_status.reason or "Runtime model unavailable.",
    )


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
        "latest_source": latest_run["source"] if latest_run else None,
        "latest_result": latest_run["career_result"] if latest_run else None,
        "latest_holland_code": latest_run["holland_code"] if latest_run else None,
        "latest_career_family": latest_run["career_family"] if latest_run else None,
        "latest_model_version": latest_run["model_version"] if latest_run else None,
        "latest_riasec": latest_run["riasec_scores"] if latest_run else _empty_admin_riasec_scores(),
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
        return _build_auth_response(user)
    except DuplicateUserError as exc:
        upgraded_user = db.upgrade_legacy_user_password(
            username=username,
            password_hash=password_hash,
        )
        if upgraded_user:
            if upgraded_user.role != assigned_role:
                upgraded_user = db.set_user_role(upgraded_user.id, assigned_role) or upgraded_user
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
    return _build_auth_response(updated_user or user)


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
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

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
        feature_vector = feature_pipeline.build_feature_vector(aggregated_features)
        skill_use_totals = feature_pipeline.extract_skill_use_totals(aggregated_features)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    runtime_status = ml_runtime.get_model_status()
    if runtime_status.available:
        try:
            raw_model_scores = ml_runtime.predict_scores(feature_vector, runtime_status)
            riasec_scores = rubric.round_scores_to_integers(raw_model_scores)
            source = "RF Model"
            model_version = runtime_status.model_version or "rf_v1"
            message = _build_run_complete_message(
                source=source,
                model_version=model_version,
            )
        except Exception:
            LOGGER.exception(
                "RF model inference failed for session_id=%s. Using rubric fallback.",
                payload.session_id,
            )
            riasec_scores = _calculate_rubric_integer_scores(validated_payload)
            source = "Backend Rubric Fallback"
            model_version = "rubric_v1"
            message = _build_run_complete_message(
                source=source,
                model_version=model_version,
                detail="Trained model inference failed, so backend rubric fallback was used.",
            )
    else:
        LOGGER.info(
            "Runtime model unavailable for session_id=%s. Using rubric fallback.",
            payload.session_id,
        )
        riasec_scores = _calculate_rubric_integer_scores(validated_payload)
        source = "Backend Rubric Fallback"
        model_version = "rubric_v1"
        detail = "Trained model is unavailable, so backend rubric fallback was used."
        if runtime_status.reason:
            detail += f" {runtime_status.reason}"
        message = _build_run_complete_message(
            source=source,
            model_version=model_version,
            detail=detail,
        )

    holland_code = career_mapping.derive_holland_code(
        riasec_scores,
        skill_use_totals,
    )
    career_family = career_mapping.derive_career_family(riasec_scores)
    career_result = career_family

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
def admin_make_user_admin(user_id: int, db: TelemetryRepository = Depends(get_db)):
    user = db.set_user_role(user_id, "admin")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    row = _build_admin_user_rows([user], db.list_session_run_overview_by_player())[0]
    return schemas.AdminUser(**row)


@admin_router.post(
    "/users/{user_id}/make-user",
    response_model=schemas.AdminUser,
)
def admin_make_user_standard(user_id: int, db: TelemetryRepository = Depends(get_db)):
    user = db.set_user_role(user_id, "user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
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
        return templates.TemplateResponse(
            "admin_login.html",
            {
                "request": request,
                "error_message": "Invalid username or password.",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    updated_user = db.touch_last_login(user.id) or user
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
def admin_logout():
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
    latest_result = performance["latest_result"] or "N/A"
    summary = {
        "total_runs": performance["total_runs"],
        "avg_clear_rate": performance["avg_clear_rate"],
        "avg_time_seconds": performance["avg_time_seconds"],
        "avg_stars_per_run": performance["avg_stars_per_run"],
        "latest_source": latest_source,
        "latest_result": latest_result,
        "latest_holland_code": performance["latest_holland_code"] or "N/A",
        "latest_career_family": performance["latest_career_family"] or "N/A",
        "latest_model_version": performance["latest_model_version"] or "n/a",
    }

    return templates.TemplateResponse(
        "user_performance.html",
        {
            "request": request,
            "current_user": current_user,
            "user": user,
            "can_manage_user": current_user.role == "admin",
            "runs": performance["runs"],
            "riasec": performance["latest_riasec"],
            "summary": summary,
        },
    )


@admin_ui_router.post("/users/{user_id}/make-admin")
def admin_make_user_admin_page(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    _require_admin_ui_user(request, db)
    user = db.set_user_role(user_id, "admin")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=status.HTTP_303_SEE_OTHER)


@admin_ui_router.post("/users/{user_id}/make-user")
def admin_make_user_standard_page(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    _require_admin_ui_user(request, db)
    user = db.set_user_role(user_id, "user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=status.HTTP_303_SEE_OTHER)


@admin_ui_router.post("/users/{user_id}/delete")
def admin_delete_user(
    user_id: int,
    request: Request,
    db: TelemetryRepository = Depends(get_db),
):
    _require_admin_ui_user(request, db)
    deleted = db.delete_user(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)

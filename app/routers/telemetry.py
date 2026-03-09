from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app import schemas
from app.database import get_db
from app.repository import DuplicateUserError, TelemetryRepository
from app.security import hash_password, verify_password


router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
admin_router = APIRouter(prefix="/api/admin", tags=["admin"])
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
    )


@router.post("/users", response_model=schemas.AuthResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user_in: schemas.UserCreate,
    db: TelemetryRepository = Depends(get_db),
):
    username = user_in.username.strip()
    password_hash = hash_password(user_in.password)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty.",
        )

    try:
        user = db.create_user(
            player_id=str(uuid.uuid4()),
            username=username,
            password_hash=password_hash,
            email=None,
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


@admin_router.get(
    "/users",
    response_model=list[schemas.AdminUser],
)
def admin_list_users(db: TelemetryRepository = Depends(get_db)):
    users = db.list_users()
    return [
        schemas.AdminUser(
            user_id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_login=user.last_login,
            total_quest_attempts=len(user.quest_attempts),
        )
        for user in users
    ]


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

    return schemas.AdminUser(
        user_id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        last_login=user.last_login,
        total_quest_attempts=len(user.quest_attempts),
    )


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

    profile = user.riasec_profile
    if profile:
        aggregated = schemas.UserRIASECProfileBase(
            realistic=profile.realistic,
            investigative=profile.investigative,
            artistic=profile.artistic,
            social=profile.social,
            enterprising=profile.enterprising,
            conventional=profile.conventional,
        )
    else:
        aggregated = schemas.UserRIASECProfileBase(
            realistic=0.0,
            investigative=0.0,
            artistic=0.0,
            social=0.0,
            enterprising=0.0,
            conventional=0.0,
        )

    return schemas.UserPerformance(
        user_id=user.id,
        username=user.username,
        total_attempts=len(user.quest_attempts),
        attempts=user.quest_attempts,
        aggregated_riasec=aggregated,
    )


@admin_ui_router.get(
    "/users",
    response_class=HTMLResponse,
)
def admin_users_page(request: Request, db: TelemetryRepository = Depends(get_db)):
    users = db.list_users()
    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
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
    user = db.find_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    profile = user.riasec_profile
    riasec = {
        "realistic": profile.realistic if profile else 0.0,
        "investigative": profile.investigative if profile else 0.0,
        "artistic": profile.artistic if profile else 0.0,
        "social": profile.social if profile else 0.0,
        "enterprising": profile.enterprising if profile else 0.0,
        "conventional": profile.conventional if profile else 0.0,
    }

    total_attempts = len(user.quest_attempts)
    success_count = sum(1 for attempt in user.quest_attempts if attempt.success == 1)
    total_time = sum(attempt.time_spent_seconds or 0 for attempt in user.quest_attempts)

    success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0.0
    avg_time = (total_time / total_attempts) if total_attempts > 0 else 0.0
    last_result = user.quest_attempts[0].quest_result if user.quest_attempts else "N/A"

    summary = {
        "total_attempts": total_attempts,
        "success_rate": success_rate,
        "avg_time_seconds": avg_time,
        "last_result": last_result,
    }

    return templates.TemplateResponse(
        "user_performance.html",
        {
            "request": request,
            "user": user,
            "attempts": user.quest_attempts,
            "riasec": riasec,
            "summary": summary,
        },
    )


@admin_ui_router.post("/users/{user_id}/delete")
def admin_delete_user(user_id: int, db: TelemetryRepository = Depends(get_db)):
    deleted = db.delete_user(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return RedirectResponse(url="/admin/users", status_code=status.HTTP_303_SEE_OTHER)

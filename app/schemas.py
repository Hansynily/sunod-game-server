from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


UserRole = Literal["admin", "user"]


class SkillUsedBase(BaseModel):
    skill_name: str = Field(..., max_length=100)
    riasec_code: str = Field(..., max_length=10)
    usage_count: int = Field(1, ge=1)


class SkillUsedCreate(SkillUsedBase):
    pass


class SkillUsed(SkillUsedBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    quest_attempt_id: int


class QuestAttemptBase(BaseModel):
    quest_id: str = Field(..., max_length=100)
    quest_name: str = Field(..., max_length=100)
    success: int = Field(0, ge=0, le=1)
    completed_at: datetime | None = None
    time_spent_seconds: int = Field(0, ge=0)
    quest_result: str = Field("unknown", max_length=50)


class QuestAttemptCreate(QuestAttemptBase):
    skills_used: list[SkillUsedCreate] = Field(default_factory=list)


class QuestAttempt(QuestAttemptBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    started_at: datetime
    skills_used: list[SkillUsed] = Field(default_factory=list)


class UserRIASECProfileBase(BaseModel):
    realistic: float
    investigative: float
    artistic: float
    social: float
    enterprising: float
    conventional: float


class UserRIASECProfileCreate(UserRIASECProfileBase):
    pass


class UserRIASECProfile(UserRIASECProfileBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int


class UserBase(BaseModel):
    username: str = Field(..., max_length=50)
    email: EmailStr | None = None
    role: UserRole = "user"


class UserCreate(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    role: UserRole = "user"
    riasec_profile: UserRIASECProfileCreate | None = None


class User(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    player_id: str
    created_at: datetime
    last_login: datetime | None = None
    quest_attempts: list[QuestAttempt] = Field(default_factory=list)
    riasec_profile: UserRIASECProfile | None = None


class UserLogin(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class AuthResponse(BaseModel):
    id: int
    player_id: str
    username: str
    email: EmailStr | None = None
    created_at: datetime
    last_login: datetime | None = None
    role: UserRole
    access_token: str
    token_type: str = "bearer"


class SelectedSkill(BaseModel):
    riasec_code: str = Field(..., max_length=10)
    skill_name: str = Field(..., max_length=100)


class QuestAttemptTelemetryIn(BaseModel):
    player_id: str = Field(..., max_length=100)
    username: str = Field(..., max_length=50)
    email: EmailStr | None = None
    quest_id: str = Field(..., max_length=100)
    selected_skills: list[SelectedSkill]
    quest_result: str = Field(..., max_length=50)
    time_spent_seconds: int = Field(..., ge=0)


class QuestAttemptTelemetryOut(BaseModel):
    success: bool
    message: str


class ChallengeRoundTelemetryIn(BaseModel):
    challenge_id: str = Field(..., max_length=100)
    primary_riasec: str = Field(..., min_length=1, max_length=1)
    solved: bool
    stars_earned: int = Field(..., ge=0, le=3)
    retry_count: int = Field(..., ge=0)
    time_spent_seconds: float = Field(..., ge=0)
    skill_use_r: int = Field(..., ge=0)
    skill_use_i: int = Field(..., ge=0)
    skill_use_a: int = Field(..., ge=0)
    skill_use_s: int = Field(..., ge=0)
    skill_use_e: int = Field(..., ge=0)
    skill_use_c: int = Field(..., ge=0)

    @field_validator("challenge_id", mode="before")
    @classmethod
    def validate_challenge_id(cls, value: str) -> str:
        challenge_id = value.strip()
        if not challenge_id:
            raise ValueError("challenge_id cannot be empty.")
        return challenge_id

    @field_validator("primary_riasec", mode="before")
    @classmethod
    def normalize_primary_riasec(cls, value: str) -> str:
        riasec_code = value.strip().upper()
        if riasec_code not in {"R", "I", "A", "S", "E", "C"}:
            raise ValueError("primary_riasec must be one of R, I, A, S, E, or C.")
        return riasec_code


class RunSummaryTelemetryIn(BaseModel):
    player_id: str = Field(..., max_length=100)
    username: str = Field(..., max_length=50)
    session_id: str = Field(..., max_length=100)
    scene_version: str = Field("single_room_v1", max_length=100)
    total_time_spent_seconds: float = Field(..., ge=0)
    rounds: list[ChallengeRoundTelemetryIn] = Field(default_factory=list)

    @field_validator("player_id", "username", "session_id", "scene_version", mode="before")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field cannot be empty.")
        return normalized


class RiasecScoresOut(BaseModel):
    r: int = Field(..., ge=0, le=10)
    i: int = Field(..., ge=0, le=10)
    a: int = Field(..., ge=0, le=10)
    s: int = Field(..., ge=0, le=10)
    e: int = Field(..., ge=0, le=10)
    c: int = Field(..., ge=0, le=10)


class RunSummaryTelemetryOut(BaseModel):
    success: bool
    message: str
    source: str
    riasec_scores: RiasecScoresOut
    holland_code: str
    career_family: str
    career_result: str
    model_version: str


class RuntimeStatusOut(BaseModel):
    rf_model_loaded: bool
    active_source: str
    active_version: str
    detail: str


class PredictionIn(BaseModel):
    features: list[float] = Field(default_factory=list)


class PredictionOut(BaseModel):
    predicted_cluster: int


class SessionClusterTelemetryIn(BaseModel):
    player_id: str = Field(..., max_length=100)
    session_id: str = Field(..., max_length=100)
    predicted_cluster: int = Field(..., ge=0, le=8)

    @field_validator("player_id", "session_id", mode="before")
    @classmethod
    def validate_cluster_required_strings(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field cannot be empty.")
        return normalized


class SessionClusterTelemetryOut(BaseModel):
    success: bool
    message: str
    predicted_cluster: int
    holland_code: str
    career_family: str
    source: str
    model_version: str


class AdminUser(BaseModel):
    user_id: int
    username: str
    email: EmailStr | None = None
    created_at: datetime
    last_login: datetime | None = None
    role: UserRole
    total_runs: int
    last_run_at: datetime | None = None
    last_source: str | None = None
    last_result: str | None = None
    last_holland_code: str | None = None
    last_predicted_cluster: int | None = None
    last_cluster_label: str | None = None
    last_cluster_holland_code: str | None = None


class AdminSessionRun(BaseModel):
    session_id: str
    created_at: datetime
    source: str
    model_version: str
    riasec_scores: RiasecScoresOut
    holland_code: str
    career_family: str
    career_result: str
    total_time_spent_seconds: float
    rounds_attempted: int
    rounds_cleared: int
    total_stars: int
    predicted_cluster: int | None = None
    cluster_label: str | None = None
    cluster_holland_code: str | None = None
    cluster_source: str | None = None
    cluster_model_version: str | None = None
    cluster_example_careers: list[str] = Field(default_factory=list)


class UserPerformance(BaseModel):
    user_id: int
    username: str
    total_runs: int
    avg_clear_rate: float
    avg_time_seconds: float
    avg_stars_per_run: float
    latest_source: str | None = None
    latest_result: str | None = None
    latest_holland_code: str | None = None
    latest_career_family: str | None = None
    latest_model_version: str | None = None
    latest_riasec: RiasecScoresOut
    runs: list[AdminSessionRun]
    latest_predicted_cluster: int | None = None
    latest_cluster_label: str | None = None
    latest_cluster_holland_code: str | None = None
    latest_cluster_source: str | None = None
    latest_cluster_model_version: str | None = None
    latest_cluster_example_careers: list[str] = Field(default_factory=list)

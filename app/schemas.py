from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


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


class UserCreate(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
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


class AdminUser(BaseModel):
    user_id: int
    username: str
    email: EmailStr | None = None
    created_at: datetime
    last_login: datetime | None = None
    total_quest_attempts: int


class UserPerformance(BaseModel):
    user_id: int
    username: str
    total_attempts: int
    attempts: list[QuestAttempt]
    aggregated_riasec: UserRIASECProfileBase

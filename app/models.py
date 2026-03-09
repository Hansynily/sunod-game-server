from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SkillUsed:
    id: int
    quest_attempt_id: int
    skill_name: str
    riasec_code: str
    usage_count: int = 1


@dataclass(slots=True)
class QuestAttempt:
    id: int
    user_id: int
    quest_id: str
    quest_name: str
    started_at: datetime
    completed_at: datetime | None = None
    time_spent_seconds: int = 0
    quest_result: str = "unknown"
    success: int = 0
    skills_used: list[SkillUsed] = field(default_factory=list)


@dataclass(slots=True)
class UserRIASECProfile:
    id: int
    user_id: int
    realistic: float
    investigative: float
    artistic: float
    social: float
    enterprising: float
    conventional: float


@dataclass(slots=True)
class User:
    id: int
    player_id: str
    username: str
    email: str | None
    created_at: datetime
    password_hash: str | None = None
    last_login: datetime | None = None
    quest_attempts: list[QuestAttempt] = field(default_factory=list)
    riasec_profile: UserRIASECProfile | None = None

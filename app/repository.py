from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app import models


class DuplicateUserError(Exception):
    pass


class TelemetryRepository:
    def __init__(self, database: Database):
        self.database = database
        self.users = database["users"]
        self.counters = database["counters"]

    def ping(self) -> None:
        self.database.command("ping")

    def ensure_indexes(self) -> None:
        self.users.create_index([("id", ASCENDING)], unique=True, name="users_id_unique")
        self.users.create_index(
            [("player_id", ASCENDING)],
            unique=True,
            name="users_player_id_unique",
        )
        self.users.create_index(
            [("username", ASCENDING)],
            unique=True,
            name="users_username_unique",
        )
        self.users.create_index(
            [("email", ASCENDING)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}},
            name="users_email_unique",
        )
        self.users.create_index(
            [("created_at", DESCENDING)],
            name="users_created_at_idx",
        )

    def list_users(self) -> list[models.User]:
        documents = self.users.find().sort("created_at", DESCENDING)
        return [self._build_user(document) for document in documents]

    def find_user_by_id(self, user_id: int) -> models.User | None:
        document = self.users.find_one({"id": user_id})
        if not document:
            return None
        return self._build_user(document)

    def find_user_by_player_id(self, player_id: str) -> models.User | None:
        document = self.users.find_one({"player_id": player_id})
        if not document:
            return None
        return self._build_user(document)

    def find_user_by_username(self, username: str) -> models.User | None:
        document = self.users.find_one({"username": username})
        if not document:
            return None
        return self._build_user(document)

    def create_user(
        self,
        *,
        player_id: str,
        username: str,
        password_hash: str,
        email: str | None,
        riasec_profile: dict[str, float] | None = None,
    ) -> models.User:
        filters: list[dict[str, Any]] = [{"username": username}]
        if email:
            filters.append({"email": email})

        existing = self.users.find_one({"$or": filters}, {"_id": 1})
        if existing:
            raise DuplicateUserError

        user_id = self._next_sequence("users")
        now = datetime.utcnow()
        document: dict[str, Any] = {
            "id": user_id,
            "player_id": player_id,
            "username": username,
            "email": email,
            "created_at": now,
            "password_hash": password_hash,
            "last_login": now,
            "quest_attempts": [],
            "riasec_profile": None,
        }

        if riasec_profile:
            document["riasec_profile"] = self._profile_document(
                user_id=user_id,
                realistic=riasec_profile["realistic"],
                investigative=riasec_profile["investigative"],
                artistic=riasec_profile["artistic"],
                social=riasec_profile["social"],
                enterprising=riasec_profile["enterprising"],
                conventional=riasec_profile["conventional"],
            )

        try:
            self.users.insert_one(document)
        except DuplicateKeyError as exc:
            raise DuplicateUserError from exc

        return self._build_user(document)

    def touch_last_login(self, user_id: int) -> models.User | None:
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {"$set": {"last_login": datetime.utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def upgrade_legacy_user_password(
        self,
        *,
        username: str,
        password_hash: str,
    ) -> models.User | None:
        updated_document = self.users.find_one_and_update(
            {
                "username": username,
                "$or": [
                    {"password_hash": {"$exists": False}},
                    {"password_hash": None},
                    {"password_hash": ""},
                ],
            },
            {
                "$set": {
                    "password_hash": password_hash,
                    "last_login": datetime.utcnow(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def add_quest_attempt(
        self,
        *,
        user_id: int,
        quest_id: str,
        quest_name: str,
        success: int,
        completed_at: datetime | None,
        time_spent_seconds: int,
        quest_result: str,
        skills_used: list[dict[str, Any]] | None = None,
        update_profile_from_skills: bool = False,
    ) -> models.QuestAttempt | None:
        document = self.users.find_one({"id": user_id})
        if not document:
            return None

        skill_documents: list[dict[str, Any]] = []
        for skill in skills_used or []:
            skill_documents.append(
                {
                    "id": self._next_sequence("skills_used"),
                    "quest_attempt_id": 0,
                    "skill_name": skill["skill_name"],
                    "riasec_code": skill["riasec_code"],
                    "usage_count": skill.get("usage_count", 1),
                }
            )

        attempt_id = self._next_sequence("quest_attempts")
        for skill_document in skill_documents:
            skill_document["quest_attempt_id"] = attempt_id

        profile_document = document.get("riasec_profile")
        if update_profile_from_skills:
            profile_document = profile_document or self._empty_profile_document(user_id)
            for skill_document in skill_documents:
                self._apply_riasec_code(
                    profile_document,
                    skill_document["riasec_code"],
                    skill_document["usage_count"],
                )

        attempt_document = {
            "id": attempt_id,
            "user_id": user_id,
            "quest_id": quest_id,
            "quest_name": quest_name,
            "started_at": datetime.utcnow(),
            "completed_at": completed_at,
            "time_spent_seconds": time_spent_seconds,
            "quest_result": quest_result,
            "success": success,
            "skills_used": skill_documents,
        }

        update: dict[str, Any] = {"$push": {"quest_attempts": attempt_document}}
        if update_profile_from_skills:
            update["$set"] = {"riasec_profile": profile_document}

        result = self.users.update_one({"id": user_id}, update)
        if result.matched_count == 0:
            return None

        return self._build_quest_attempt(attempt_document)

    def delete_user(self, user_id: int) -> bool:
        result = self.users.delete_one({"id": user_id})
        return result.deleted_count == 1

    def _next_sequence(self, name: str) -> int:
        document = self.counters.find_one_and_update(
            {"_id": name},
            {"$inc": {"value": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(document["value"])

    def _build_user(self, document: dict[str, Any]) -> models.User:
        attempts = [
            self._build_quest_attempt(attempt)
            for attempt in document.get("quest_attempts", [])
        ]
        attempts.sort(key=lambda attempt: attempt.started_at, reverse=True)

        profile_document = document.get("riasec_profile")
        profile = None
        if profile_document:
            profile = self._build_profile(profile_document)

        return models.User(
            id=document["id"],
            player_id=document["player_id"],
            username=document["username"],
            email=document.get("email"),
            created_at=document["created_at"],
            password_hash=document.get("password_hash"),
            last_login=document.get("last_login"),
            quest_attempts=attempts,
            riasec_profile=profile,
        )

    def _build_quest_attempt(self, document: dict[str, Any]) -> models.QuestAttempt:
        skills = [
            self._build_skill_used(skill)
            for skill in document.get("skills_used", [])
        ]

        return models.QuestAttempt(
            id=document["id"],
            user_id=document["user_id"],
            quest_id=document["quest_id"],
            quest_name=document["quest_name"],
            started_at=document["started_at"],
            completed_at=document.get("completed_at"),
            time_spent_seconds=document.get("time_spent_seconds", 0),
            quest_result=document.get("quest_result", "unknown"),
            success=document.get("success", 0),
            skills_used=skills,
        )

    def _build_skill_used(self, document: dict[str, Any]) -> models.SkillUsed:
        return models.SkillUsed(
            id=document["id"],
            quest_attempt_id=document["quest_attempt_id"],
            skill_name=document["skill_name"],
            riasec_code=document["riasec_code"],
            usage_count=document.get("usage_count", 1),
        )

    def _build_profile(self, document: dict[str, Any]) -> models.UserRIASECProfile:
        return models.UserRIASECProfile(
            id=document["id"],
            user_id=document["user_id"],
            realistic=document["realistic"],
            investigative=document["investigative"],
            artistic=document["artistic"],
            social=document["social"],
            enterprising=document["enterprising"],
            conventional=document["conventional"],
        )

    def _empty_profile_document(self, user_id: int) -> dict[str, Any]:
        return self._profile_document(
            user_id=user_id,
            realistic=0.0,
            investigative=0.0,
            artistic=0.0,
            social=0.0,
            enterprising=0.0,
            conventional=0.0,
        )

    def _profile_document(
        self,
        *,
        user_id: int,
        realistic: float,
        investigative: float,
        artistic: float,
        social: float,
        enterprising: float,
        conventional: float,
    ) -> dict[str, Any]:
        return {
            "id": user_id,
            "user_id": user_id,
            "realistic": realistic,
            "investigative": investigative,
            "artistic": artistic,
            "social": social,
            "enterprising": enterprising,
            "conventional": conventional,
        }

    def _apply_riasec_code(
        self,
        profile_document: dict[str, Any],
        riasec_code: str,
        weight: int,
    ) -> None:
        code = riasec_code.upper()
        if "R" in code:
            profile_document["realistic"] += weight
        if "I" in code:
            profile_document["investigative"] += weight
        if "A" in code:
            profile_document["artistic"] += weight
        if "S" in code:
            profile_document["social"] += weight
        if "E" in code:
            profile_document["enterprising"] += weight
        if "C" in code:
            profile_document["conventional"] += weight

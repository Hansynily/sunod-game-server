from datetime import date, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app import account_lifecycle, models


class DuplicateUserError(Exception):
    pass


class TelemetryRepository:
    def __init__(self, database: Database):
        self.database = database
        self.users = database["users"]
        self.session_runs = database["session_runs"]
        self.ml_datasets = database["ml_datasets"]
        self.quest_model_bindings = database["quest_model_bindings"]
        self.counters = database["counters"]
        # One CURRENT save-state per player (upsert, not history) - backs Continue/Reset.
        # Distinct from session_runs, which is completed-run history for reporting.
        self.player_run_state = database["player_run_state"]

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
        self.users.create_index(
            [("approval_state", ASCENDING), ("created_at", DESCENDING)],
            name="users_approval_state_idx",
        )
        self.users.create_index(
            [("email_verification_state", ASCENDING), ("created_at", DESCENDING)],
            name="users_email_verification_state_idx",
        )
        self.users.create_index(
            [("verification_token_hash", ASCENDING)],
            partialFilterExpression={"verification_token_hash": {"$type": "string"}},
            name="users_verification_token_hash_idx",
        )
        self.session_runs.create_index(
            [("session_id", ASCENDING)],
            name="session_runs_session_id_idx",
        )
        self.session_runs.create_index(
            [("created_at", DESCENDING)],
            name="session_runs_created_at_idx",
        )
        self.player_run_state.create_index(
            [("player_id", ASCENDING)],
            unique=True,
            name="player_run_state_player_id_unique",
        )
        self.ml_datasets.create_index(
            [("id", ASCENDING)],
            unique=True,
            name="ml_datasets_id_unique",
        )
        self.ml_datasets.create_index(
            [("uploaded_at", DESCENDING)],
            name="ml_datasets_uploaded_at_idx",
        )
        self.ml_datasets.create_index(
            [("status", ASCENDING), ("trained_at", DESCENDING)],
            name="ml_datasets_status_trained_at_idx",
        )
        self.quest_model_bindings.create_index(
            [("scope_type", ASCENDING), ("scope_id", ASCENDING)],
            unique=True,
            name="quest_model_bindings_scope_unique",
        )
        self.quest_model_bindings.create_index(
            [("updated_at", DESCENDING)],
            name="quest_model_bindings_updated_at_idx",
        )
        self.backfill_user_lifecycle_fields()

    def backfill_user_lifecycle_fields(self) -> int:
        updated_count = 0
        filters = {
            "$or": [
                {"approval_state": {"$exists": False}},
                {"email_verification_state": {"$exists": False}},
                {"approved_at": {"$exists": False}},
                {"verified_at": {"$exists": False}},
                {"tutorial_completed": {"$exists": False}},
                {"tutorial_completed_at": {"$exists": False}},
            ]
        }

        for document in self.users.find(filters):
            lifecycle_fields = account_lifecycle.normalize_lifecycle_document(document)
            lifecycle_fields["tutorial_completed"] = bool(document.get("tutorial_completed", True))
            lifecycle_fields["tutorial_completed_at"] = document.get("tutorial_completed_at")
            result = self.users.update_one(
                {"_id": document["_id"]},
                {"$set": lifecycle_fields},
            )
            updated_count += result.modified_count

        return updated_count

    def list_users(self) -> list[models.User]:
        documents = self.users.find().sort("created_at", DESCENDING)
        return [self._build_user(document) for document in documents]

    def count_admins(self) -> int:
        return self.users.count_documents({"role": "admin"})

    def list_session_run_overview_by_player(self) -> dict[str, dict[str, Any]]:
        documents = self.session_runs.find(
            {},
            {
                "player_id": 1,
                "created_at": 1,
                "source": 1,
                "cluster_source": 1,
                "career_result": 1,
                "cluster_label": 1,
                "career_family": 1,
                "holland_code": 1,
                "cluster_holland_code": 1,
                "predicted_cluster": 1,
                "career_cluster": 1,
            },
        ).sort("created_at", DESCENDING)

        overview_by_player: dict[str, dict[str, Any]] = {}
        for document in documents:
            player_id = document.get("player_id")
            if not player_id:
                continue

            overview = overview_by_player.get(player_id)
            if overview is None:
                overview = {
                    "total_runs": 0,
                    "last_run_at": document.get("created_at"),
                    "last_source": document.get("cluster_source") or document.get("source"),
                    "last_result": (
                        document.get("career_result")
                        or document.get("cluster_label")
                        or document.get("career_family")
                    ),
                    "last_career_family": (
                        document.get("career_family")
                        or document.get("cluster_label")
                        or document.get("career_result")
                    ),
                    "last_holland_code": document.get("cluster_holland_code")
                    or document.get("holland_code"),
                    "last_predicted_cluster": document.get("predicted_cluster"),
                    "last_career_cluster": document.get("career_cluster"),
                    "last_cluster_label": (
                        document.get("career_result")
                        or document.get("cluster_label")
                        or document.get("career_family")
                    ),
                    "last_cluster_holland_code": document.get("cluster_holland_code")
                    or document.get("holland_code"),
                }
                overview_by_player[player_id] = overview

            overview["total_runs"] += 1

        return overview_by_player

    def list_session_runs_for_player(self, player_id: str) -> list[dict[str, Any]]:
        documents = self.session_runs.find({"player_id": player_id}).sort(
            "created_at",
            DESCENDING,
        )
        return list(documents)

    def list_admin_report_rows(
        self,
        *,
        q: str | None = None,
        source: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = (q or "").strip().lower()
        normalized_source = (source or "").strip().lower()
        users_by_player_id = {user.player_id: user for user in self.list_users()}

        rows: list[dict[str, Any]] = []
        documents = self.session_runs.find({}).sort("created_at", DESCENDING)
        for document in documents:
            player_id = document.get("player_id")
            if not player_id:
                continue

            user = users_by_player_id.get(player_id)
            created_at = document.get("created_at")
            if isinstance(created_at, datetime):
                created_date = created_at.date()
                if date_from and created_date < date_from:
                    continue
                if date_to and created_date > date_to:
                    continue

            effective_source = document.get("cluster_source") or document.get("source") or "Unknown"
            if normalized_source and effective_source.lower() != normalized_source:
                continue

            row = {
                "user_id": user.id if user else document.get("user_id"),
                "player_id": player_id,
                "username": document.get("username") or (user.username if user else "Unknown"),
                "session_id": document.get("session_id", "unknown-session"),
                "created_at": created_at,
                "source": document.get("source") or "Unknown",
                "model_version": document.get("model_version") or "n/a",
                "cluster_source": document.get("cluster_source"),
                "cluster_model_version": document.get("cluster_model_version"),
                "career_cluster": document.get("career_cluster"),
                "career_result": document.get("career_result"),
                "result": (
                    document.get("career_result")
                    or document.get("cluster_label")
                    or document.get("career_family")
                    or "Pending cluster result"
                ),
                "holland_code": (
                    document.get("cluster_holland_code")
                    or document.get("holland_code")
                    or "N/A"
                ),
                "career_family": (
                    document.get("career_family")
                    or document.get("cluster_label")
                    or "N/A"
                ),
                "total_time_spent_seconds": float(document.get("total_time_spent_seconds", 0) or 0),
            }

            if normalized_query:
                searchable = " ".join(
                    [
                        str(row["username"]),
                        str(row["session_id"]),
                        str(row["result"]),
                        str(row["holland_code"]),
                        str(row["career_family"]),
                    ]
                ).lower()
                if normalized_query not in searchable:
                    continue

            rows.append(row)

        return rows

    def find_session_run_for_player(
        self,
        *,
        player_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        return self.session_runs.find_one(
            {"player_id": player_id, "session_id": session_id},
            sort=[("created_at", DESCENDING)],
        )

    def find_latest_run_for_player(self, player_id: str) -> dict[str, Any] | None:
        return self.session_runs.find_one(
            {"player_id": player_id},
            sort=[("created_at", DESCENDING)],
        )

    # ── Player run-state (Continue/Reset save-state) ──────────────────────

    def get_run_state(self, player_id: str) -> dict[str, Any] | None:
        return self.player_run_state.find_one({"player_id": player_id})

    def upsert_run_state(
        self,
        *,
        player_id: str,
        session_id: str,
        completed_quest_ids: list[str],
        riasec_scores: dict[str, float],
        owned_skills: list[str],
        total_stars: int,
        floor_scene: str,
        tutorial_completed: bool,
        run_finished: bool = False,
        quest_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        document = {
            "player_id": player_id,
            "session_id": session_id,
            "completed_quest_ids": list(completed_quest_ids or []),
            "quest_records": list(quest_records or []),
            "riasec_scores": dict(riasec_scores or {}),
            "owned_skills": list(owned_skills or []),
            "total_stars": int(total_stars or 0),
            "floor_scene": floor_scene or "",
            "tutorial_completed": bool(tutorial_completed),
            "run_finished": bool(run_finished),
            "updated_at": datetime.utcnow(),
        }
        return self.player_run_state.find_one_and_update(
            {"player_id": player_id},
            {"$set": document},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def finish_run_state(self, player_id: str) -> None:
        """Marks the caller's run finished. Atomic $set on just the flag, so it cannot be
        lost to a racing checkpoint upsert and needs no round trip through the client.
        Upserts a minimal finished doc if none exists, so 'End Run ends the run' holds
        even when no checkpoint was ever pushed."""
        self.player_run_state.update_one(
            {"player_id": player_id},
            {
                "$set": {"run_finished": True, "updated_at": datetime.utcnow()},
                "$setOnInsert": {
                    "player_id": player_id,
                    "session_id": "",
                    "completed_quest_ids": [],
                    "quest_records": [],
                    "riasec_scores": {},
                    "owned_skills": [],
                    "total_stars": 0,
                    "floor_scene": "",
                    "tutorial_completed": False,
                },
            },
            upsert=True,
        )

    def clear_run_state(self, player_id: str) -> None:
        self.player_run_state.delete_one({"player_id": player_id})

    def stamp_prediction_source(
        self,
        *,
        player_id: str,
        session_id: str,
        prediction_source: str,
    ) -> None:
        # Same latest-run targeting as find_session_run_for_player.
        self.session_runs.find_one_and_update(
            {"player_id": player_id, "session_id": session_id},
            {"$set": {"prediction_source": prediction_source}},
            sort=[("created_at", DESCENDING)],
        )

    def list_ml_datasets(self) -> list[dict[str, Any]]:
        documents = self.ml_datasets.find({}).sort("uploaded_at", DESCENDING)
        return list(documents)

    def find_ml_dataset_by_id(self, dataset_id: int) -> dict[str, Any] | None:
        return self.ml_datasets.find_one({"id": dataset_id})

    def create_ml_dataset(
        self,
        *,
        dataset_name: str,
        original_filename: str,
        stored_filename: str,
        storage_path: str,
        file_size_bytes: int,
        row_count: int,
        feature_count: int,
        label_count: int,
        uploaded_by_user_id: int | None,
        uploaded_by_username: str | None,
    ) -> dict[str, Any]:
        dataset_id = self._next_sequence("ml_datasets")
        document = {
            "id": dataset_id,
            "dataset_name": dataset_name,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "storage_path": storage_path,
            "file_size_bytes": int(file_size_bytes),
            "row_count": int(row_count),
            "feature_count": int(feature_count),
            "label_count": int(label_count),
            "uploaded_at": datetime.utcnow(),
            "uploaded_by_user_id": uploaded_by_user_id,
            "uploaded_by_username": uploaded_by_username,
            "status": "uploaded",
            "trained_at": None,
            "activated_at": None,
            "model_version": None,
            "bundle_key": None,
            "bundle_path": None,
            "bundle_ready": False,
            "mean_absolute_error": None,
            "r2_score": None,
            "cluster_accuracy": None,
            "career_model_count": None,
            "clusters_covered_count": None,
            "training_row_count": None,
            "validation_row_count": None,
            "training_error": None,
        }
        self.ml_datasets.insert_one(document)
        return document

    def update_ml_dataset_training(
        self,
        dataset_id: int,
        *,
        status: str,
        trained_at: datetime,
        activated_at: datetime,
        model_version: str,
        bundle_key: str | None,
        bundle_path: str | None,
        bundle_ready: bool,
        mean_absolute_error: float | None,
        r2_score: float | None,
        cluster_accuracy: float | None,
        career_model_count: int | None,
        clusters_covered_count: int | None,
        training_row_count: int,
        validation_row_count: int,
    ) -> dict[str, Any] | None:
        return self.ml_datasets.find_one_and_update(
            {"id": dataset_id},
            {
                "$set": {
                    "status": status,
                    "trained_at": trained_at,
                    "activated_at": activated_at,
                    "model_version": model_version,
                    "bundle_key": bundle_key,
                    "bundle_path": bundle_path,
                    "bundle_ready": bool(bundle_ready),
                    "mean_absolute_error": mean_absolute_error,
                    "r2_score": r2_score,
                    "cluster_accuracy": cluster_accuracy,
                    "career_model_count": career_model_count,
                    "clusters_covered_count": clusters_covered_count,
                    "training_row_count": int(training_row_count),
                    "validation_row_count": int(validation_row_count),
                    "training_error": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    def mark_ml_dataset_training_failed(
        self,
        dataset_id: int,
        *,
        error_message: str,
    ) -> dict[str, Any] | None:
        return self.ml_datasets.find_one_and_update(
            {"id": dataset_id},
            {
                "$set": {
                    "status": "training_failed",
                    "training_error": error_message,
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    def list_quest_model_bindings(self) -> list[dict[str, Any]]:
        documents = self.quest_model_bindings.find({}).sort(
            [("scope_type", ASCENDING), ("scope_id", ASCENDING)]
        )
        return list(documents)

    def find_quest_model_binding(self, *, scope_type: str, scope_id: str) -> dict[str, Any] | None:
        return self.quest_model_bindings.find_one(
            {"scope_type": scope_type, "scope_id": scope_id}
        )

    def resolve_quest_model_binding(
        self,
        *,
        scope_type: str,
        scope_id: str,
        level_id: str | None = None,
    ) -> dict[str, Any] | None:
        exact_binding = self.find_quest_model_binding(scope_type=scope_type, scope_id=scope_id)
        if exact_binding:
            return {
                "binding": exact_binding,
                "binding_source": "exact",
            }

        if scope_type == "quest" and level_id:
            level_binding = self.find_quest_model_binding(scope_type="level", scope_id=level_id)
            if level_binding:
                return {
                    "binding": level_binding,
                    "binding_source": "level",
                }

        return None

    def upsert_quest_model_binding(
        self,
        *,
        scope_type: str,
        scope_id: str,
        scope_label: str,
        level_id: str | None,
        dataset_id: int,
        dataset_name: str,
        model_version: str,
        bundle_key: str | None,
        bundle_path: str | None,
        bundle_ready: bool,
        updated_by_user_id: int | None,
        updated_by_username: str | None,
    ) -> dict[str, Any]:
        now = datetime.utcnow()
        return self.quest_model_bindings.find_one_and_update(
            {"scope_type": scope_type, "scope_id": scope_id},
            {
                "$set": {
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "scope_label": scope_label,
                    "level_id": level_id,
                    "dataset_id": int(dataset_id),
                    "dataset_name": dataset_name,
                    "model_version": model_version,
                    "bundle_key": bundle_key,
                    "bundle_path": bundle_path,
                    "bundle_ready": bool(bundle_ready),
                    "updated_at": now,
                    "updated_by_user_id": updated_by_user_id,
                    "updated_by_username": updated_by_username,
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def delete_quest_model_binding(self, *, scope_type: str, scope_id: str) -> bool:
        result = self.quest_model_bindings.delete_one(
            {"scope_type": scope_type, "scope_id": scope_id}
        )
        return result.deleted_count == 1

    def list_admin_quest_attempt_rows(
        self,
        *,
        user_id: int | None = None,
        q: str | None = None,
        quest_id: str | None = None,
        result: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = (q or "").strip().lower()
        normalized_quest_id = (quest_id or "").strip().lower()
        normalized_result = (result or "").strip().lower()

        rows_by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
        for user in self.list_users():
            if user_id is not None and user.id != user_id:
                continue

            for session_run in self.list_session_runs_for_player(user.player_id):
                for row in self._build_admin_session_quest_attempt_rows(user, session_run):
                    if not self._admin_quest_attempt_row_matches_filters(
                        row,
                        normalized_query=normalized_query,
                        normalized_quest_id=normalized_quest_id,
                        normalized_result=normalized_result,
                        date_from=date_from,
                        date_to=date_to,
                    ):
                        continue

                    signature = self._admin_quest_attempt_row_signature(row)
                    rows_by_signature.setdefault(signature, row)

            for attempt in user.quest_attempts:
                row = self._build_admin_quest_attempt_row(user, attempt)
                if not self._admin_quest_attempt_row_matches_filters(
                    row,
                    normalized_query=normalized_query,
                    normalized_quest_id=normalized_quest_id,
                    normalized_result=normalized_result,
                    date_from=date_from,
                    date_to=date_to,
                ):
                    continue

                signature = self._admin_quest_attempt_row_signature(row)
                rows_by_signature.setdefault(signature, row)

        rows = list(rows_by_signature.values())
        rows.sort(
            key=lambda row: row["completed_at"] or row["started_at"],
            reverse=True,
        )
        return rows

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

    def resolve_user_id_by_player_id(self, player_id: str) -> int | None:
        document = self.users.find_one({"player_id": player_id}, {"id": 1})
        if not document:
            return None
        return int(document["id"])

    def find_user_by_username(self, username: str) -> models.User | None:
        document = self.users.find_one({"username": username})
        if not document:
            return None
        return self._build_user(document)

    def find_user_by_verification_token_hash(self, token_hash: str) -> models.User | None:
        document = self.users.find_one({"verification_token_hash": token_hash})
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
        name: str,
        birthdate: date,
        gender: str,
        role: str = "user",
        riasec_profile: dict[str, float] | None = None,
    ) -> models.User:
        filters: list[dict[str, Any]] = [{"username": username}]
        if email:
            filters.append({"email": email})

        existing = self.users.find_one({"$or": filters}, {"_id": 1})
        if existing:
            raise DuplicateUserError

        user_id = self._next_sequence("users")
        lifecycle_fields = account_lifecycle.build_new_user_lifecycle(role=role, email=email)
        document: dict[str, Any] = {
            "id": user_id,
            "player_id": player_id,
            "username": username,
            "email": lifecycle_fields["email"],
            "name": name,
            "birthdate": birthdate.isoformat(),
            "gender": gender,
            "created_at": datetime.utcnow(),
            "role": role,
            "password_hash": password_hash,
            "last_login": None,
            "tutorial_completed": False,
            "tutorial_completed_at": None,
            "quest_attempts": [],
            "riasec_profile": None,
            **lifecycle_fields,
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

    def set_user_role(self, user_id: int, role: str) -> models.User | None:
        current = self.users.find_one({"id": user_id})
        if not current:
            return None

        lifecycle_fields = account_lifecycle.normalize_lifecycle_document(current)
        update_fields: dict[str, Any] = {
            "role": role,
            "rejection_reason": None,
        }

        if role == "admin":
            now = datetime.utcnow()
            update_fields.update(
                {
                    "approval_state": account_lifecycle.APPROVAL_APPROVED,
                    "email_verification_state": account_lifecycle.EMAIL_EXEMPT,
                    "approved_at": lifecycle_fields["approved_at"] or now,
                    "approved_by_user_id": lifecycle_fields["approved_by_user_id"],
                    "verification_sent_at": None,
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                    "verified_at": lifecycle_fields["verified_at"] or now,
                }
            )
        else:
            stays_active = lifecycle_fields["approval_state"] in {
                account_lifecycle.APPROVAL_APPROVED,
                account_lifecycle.APPROVAL_GRANDFATHERED,
            }
            update_fields.update(
                {
                    "approval_state": (
                        account_lifecycle.APPROVAL_GRANDFATHERED
                        if stays_active
                        else account_lifecycle.APPROVAL_PENDING
                    ),
                    "email_verification_state": (
                        lifecycle_fields["email_verification_state"]
                        if stays_active
                        else (
                            account_lifecycle.EMAIL_QUEUED
                            if current.get("email")
                            else account_lifecycle.EMAIL_MISSING
                        )
                    ),
                    "approved_at": lifecycle_fields["approved_at"] if stays_active else None,
                    "approved_by_user_id": lifecycle_fields["approved_by_user_id"] if stays_active else None,
                    "verification_sent_at": None if lifecycle_fields["email_verification_state"] != account_lifecycle.EMAIL_VERIFIED else lifecycle_fields["verification_sent_at"],
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                    "verified_at": lifecycle_fields["verified_at"] if stays_active else None,
                }
            )

        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
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
                    "last_login": None,
                    "role": "user",
                    "tutorial_completed": True,
                    "tutorial_completed_at": None,
                    **account_lifecycle.build_legacy_user_lifecycle(
                        role="user",
                        created_at=datetime.utcnow(),
                    ),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def update_user_profile_fields(
        self,
        user_id: int,
        *,
        name: str,
        birthdate: date,
        gender: str,
    ) -> models.User | None:
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "name": name,
                    "birthdate": birthdate.isoformat(),
                    "gender": gender,
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

    def set_user_email(self, user_id: int, email: str | None) -> models.User | None:
        normalized_email = (email or "").strip() or None
        current = self.users.find_one({"id": user_id})
        if not current:
            return None

        lifecycle_fields = account_lifecycle.normalize_lifecycle_document(current)
        update_fields: dict[str, Any] = {"email": normalized_email}
        if lifecycle_fields["email_verification_state"] != account_lifecycle.EMAIL_EXEMPT:
            update_fields.update(
                {
                    "email_verification_state": (
                        account_lifecycle.EMAIL_QUEUED
                        if normalized_email
                        else account_lifecycle.EMAIL_MISSING
                    ),
                    "verification_sent_at": None,
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                    "verified_at": None,
                }
            )

        try:
            updated_document = self.users.find_one_and_update(
                {"id": user_id},
                {"$set": update_fields},
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            raise DuplicateUserError from exc

        if not updated_document:
            return None
        return self._build_user(updated_document)

    def approve_user(self, user_id: int, *, actor_user_id: int | None) -> models.User | None:
        current = self.users.find_one({"id": user_id})
        if not current:
            return None

        lifecycle_fields = account_lifecycle.normalize_lifecycle_document(current)
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "approval_state": account_lifecycle.APPROVAL_APPROVED,
                    "approved_at": datetime.utcnow(),
                    "approved_by_user_id": actor_user_id,
                    "rejection_reason": None,
                    "email_verification_state": (
                        lifecycle_fields["email_verification_state"]
                        if lifecycle_fields["email_verification_state"] != account_lifecycle.EMAIL_EXPIRED
                        else (
                            account_lifecycle.EMAIL_QUEUED
                            if current.get("email")
                            else account_lifecycle.EMAIL_MISSING
                        )
                    ),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def reject_user(
        self,
        user_id: int,
        *,
        rejection_reason: str | None,
    ) -> models.User | None:
        current = self.users.find_one({"id": user_id})
        if not current:
            return None

        lifecycle_fields = account_lifecycle.normalize_lifecycle_document(current)
        email_state = lifecycle_fields["email_verification_state"]
        if email_state in {
            account_lifecycle.EMAIL_SENT,
            account_lifecycle.EMAIL_EXPIRED,
        }:
            email_state = (
                account_lifecycle.EMAIL_QUEUED
                if current.get("email")
                else account_lifecycle.EMAIL_MISSING
            )

        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "approval_state": account_lifecycle.APPROVAL_REJECTED,
                    "approved_at": None,
                    "approved_by_user_id": None,
                    "email_verification_state": email_state,
                    "verification_sent_at": None,
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                    "rejection_reason": (rejection_reason or "").strip() or None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def store_email_verification_token(
        self,
        user_id: int,
        *,
        token_hash: str,
        expires_at: datetime,
    ) -> models.User | None:
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "email_verification_state": account_lifecycle.EMAIL_SENT,
                    "verification_sent_at": datetime.utcnow(),
                    "verification_expires_at": expires_at,
                    "verification_token_hash": token_hash,
                    "verified_at": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def mark_user_email_verified(self, user_id: int) -> models.User | None:
        now = datetime.utcnow()
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "email_verification_state": account_lifecycle.EMAIL_VERIFIED,
                    "verified_at": now,
                    "verification_sent_at": None,
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def mark_user_email_expired(self, user_id: int) -> models.User | None:
        current = self.users.find_one({"id": user_id})
        if not current:
            return None

        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "email_verification_state": (
                        account_lifecycle.EMAIL_QUEUED
                        if current.get("email")
                        else account_lifecycle.EMAIL_MISSING
                    ),
                    "verification_sent_at": None,
                    "verification_expires_at": None,
                    "verification_token_hash": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def mark_tutorial_completed(self, user_id: int) -> models.User | None:
        now = datetime.utcnow()
        updated_document = self.users.find_one_and_update(
            {"id": user_id},
            {
                "$set": {
                    "tutorial_completed": True,
                    "tutorial_completed_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not updated_document:
            return None
        return self._build_user(updated_document)

    def delete_user(self, user_id: int) -> bool:
        user_document = self.users.find_one({"id": user_id}, {"player_id": 1})
        if not user_document:
            return False

        result = self.users.delete_one({"id": user_id})
        if result.deleted_count != 1:
            return False

        player_id = user_document.get("player_id")
        delete_filter: dict[str, Any] = {"user_id": user_id}
        if player_id:
            delete_filter = {"$or": [{"user_id": user_id}, {"player_id": player_id}]}

        self.session_runs.delete_many(delete_filter)
        return result.deleted_count == 1

    def add_session_run(
        self,
        *,
        player_id: str,
        username: str,
        session_id: str,
        scene_version: str,
        total_time_spent_seconds: float,
        rounds: list[dict[str, Any]],
        aggregated_features: dict[str, float | int],
        riasec_scores: dict[str, int],
        holland_code: str,
        career_family: str,
        career_result: str,
        source: str,
        model_version: str,
    ) -> dict[str, Any]:
        document: dict[str, Any] = {
            "player_id": player_id,
            "username": username,
            "session_id": session_id,
            "scene_version": scene_version,
            "total_time_spent_seconds": total_time_spent_seconds,
            "rounds": rounds,
            "aggregated_features": aggregated_features,
            "riasec_scores": riasec_scores,
            "holland_code": holland_code,
            "career_family": career_family,
            "career_result": career_result,
            "source": source,
            "model_version": model_version,
            "created_at": datetime.utcnow(),
        }

        user_id = self.resolve_user_id_by_player_id(player_id)
        if user_id is not None:
            document["user_id"] = user_id

        self.session_runs.insert_one(document)
        return document

    def attach_cluster_result(
        self,
        *,
        player_id: str,
        session_id: str,
        predicted_cluster: int,
        career_cluster: int | None,
        career_result: str | None,
        career_family: str | None,
        cluster_holland_code: str,
        cluster_label: str,
        cluster_example_careers: list[str],
        cluster_source: str,
        cluster_model_version: str,
        cluster_binding_source: str | None = None,
        cluster_bundle_key: str | None = None,
        cluster_bundle_ready: bool | None = None,
        cluster_dataset_id: int | None = None,
        cluster_dataset_name: str | None = None,
        cluster_binding_scope_type: str | None = None,
        cluster_binding_scope_id: str | None = None,
        cluster_binding_level_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.session_runs.find_one_and_update(
            {
                "player_id": player_id,
                "session_id": session_id,
            },
            {
                "$set": {
                    "predicted_cluster": predicted_cluster,
                    "career_cluster": career_cluster,
                    "career_result": career_result,
                    "career_family": career_family,
                    "cluster_holland_code": cluster_holland_code,
                    "cluster_label": cluster_label,
                    "cluster_example_careers": list(cluster_example_careers),
                    "cluster_source": cluster_source,
                    "cluster_model_version": cluster_model_version,
                    "cluster_binding_source": cluster_binding_source,
                    "cluster_bundle_key": cluster_bundle_key,
                    "cluster_bundle_ready": bool(cluster_bundle_ready) if cluster_bundle_ready is not None else None,
                    "cluster_dataset_id": cluster_dataset_id,
                    "cluster_dataset_name": cluster_dataset_name,
                    "cluster_binding_scope_type": cluster_binding_scope_type,
                    "cluster_binding_scope_id": cluster_binding_scope_id,
                    "cluster_binding_level_id": cluster_binding_level_id,
                },
            },
            sort=[("created_at", DESCENDING)],
            return_document=ReturnDocument.AFTER,
        )

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

        lifecycle_fields = account_lifecycle.normalize_lifecycle_document(document)

        return models.User(
            id=document["id"],
            player_id=document["player_id"],
            username=document["username"],
            email=document.get("email"),
            name=document.get("name"),
            birthdate=self._normalize_birthdate(document.get("birthdate")),
            gender=document.get("gender"),
            created_at=document["created_at"],
            role=document.get("role", "user"),
            password_hash=document.get("password_hash"),
            last_login=document.get("last_login"),
            approval_state=lifecycle_fields["approval_state"],
            email_verification_state=lifecycle_fields["email_verification_state"],
            approved_at=lifecycle_fields["approved_at"],
            approved_by_user_id=lifecycle_fields["approved_by_user_id"],
            verification_sent_at=lifecycle_fields["verification_sent_at"],
            verification_expires_at=lifecycle_fields["verification_expires_at"],
            verification_token_hash=lifecycle_fields["verification_token_hash"],
            verified_at=lifecycle_fields["verified_at"],
            rejection_reason=lifecycle_fields["rejection_reason"],
            tutorial_completed=bool(document.get("tutorial_completed", True)),
            tutorial_completed_at=document.get("tutorial_completed_at"),
            quest_attempts=attempts,
            riasec_profile=profile,
        )

    def _normalize_birthdate(self, value: Any) -> date | None:
        if value is None:
            return None

        if isinstance(value, date):
            return value

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None

        return None

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

    def _build_admin_quest_attempt_row(
        self,
        user: models.User,
        attempt: models.QuestAttempt,
    ) -> dict[str, Any]:
        return {
            "user_id": user.id,
            "player_id": user.player_id,
            "username": user.username,
            "quest_id": attempt.quest_id,
            "quest_name": attempt.quest_name,
            "quest_result": attempt.quest_result,
            "success": attempt.success,
            "started_at": attempt.started_at,
            "completed_at": attempt.completed_at,
            "time_spent_seconds": attempt.time_spent_seconds,
            "skills_used_summary": self._format_skills_used_summary(attempt.skills_used),
        }

    def _build_admin_session_quest_attempt_rows(
        self,
        user: models.User,
        session_run: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rounds = session_run.get("rounds") or []
        if not isinstance(rounds, list) or not rounds:
            return []

        run_timestamp = self._coerce_datetime(session_run.get("created_at")) or datetime.utcnow()
        rows: list[dict[str, Any]] = []
        for index, round_entry in enumerate(rounds):
            if not isinstance(round_entry, dict):
                continue

            quest_id = str(round_entry.get("challenge_id") or "").strip()
            if not quest_id:
                quest_id = f"round-{index + 1}"

            time_spent_seconds = self._coerce_int(round_entry.get("time_spent_seconds"))
            solved = bool(round_entry.get("solved"))

            rows.append(
                {
                    "user_id": user.id,
                    "player_id": user.player_id,
                    "username": user.username,
                    "quest_id": quest_id,
                    "quest_name": quest_id,
                    "quest_result": "success" if solved else "failure",
                    "success": 1 if solved else 0,
                    "started_at": run_timestamp,
                    "completed_at": run_timestamp,
                    "time_spent_seconds": time_spent_seconds,
                    "skills_used_summary": self._format_session_round_skills(round_entry),
                }
            )

        return rows

    def _build_skill_used(self, document: dict[str, Any]) -> models.SkillUsed:
        return models.SkillUsed(
            id=document["id"],
            quest_attempt_id=document["quest_attempt_id"],
            skill_name=document["skill_name"],
            riasec_code=document["riasec_code"],
            usage_count=document.get("usage_count", 1),
        )

    def _format_skills_used_summary(self, skills: list[models.SkillUsed]) -> str:
        if not skills:
            return "No skills recorded"

        parts: list[str] = []
        for skill in skills:
            label = skill.skill_name.strip() if skill.skill_name else "Unknown skill"
            if skill.usage_count > 1:
                parts.append(f"{label} x{skill.usage_count}")
            else:
                parts.append(label)

        return ", ".join(parts)

    def _format_session_round_skills(self, round_entry: dict[str, Any]) -> str:
        parts: list[str] = []
        for letter in ("R", "I", "A", "S", "E", "C"):
            count = self._coerce_int(round_entry.get(f"skill_use_{letter.lower()}"))
            if count > 0:
                parts.append(f"{letter} x{count}")

        return ", ".join(parts) if parts else "No skills recorded"

    def _admin_quest_attempt_row_matches_filters(
        self,
        row: dict[str, Any],
        *,
        normalized_query: str,
        normalized_quest_id: str,
        normalized_result: str,
        date_from: date | None,
        date_to: date | None,
    ) -> bool:
        effective_at = row.get("completed_at") or row.get("started_at")
        if not isinstance(effective_at, datetime):
            return False

        effective_date = effective_at.date()

        if normalized_query:
            searchable = " ".join(
                [
                    str(row.get("username", "")),
                    str(row.get("quest_id", "")),
                    str(row.get("quest_name", "")),
                    str(row.get("quest_result", "")),
                    str(row.get("skills_used_summary", "")),
                ]
            ).lower()
            if normalized_query not in searchable:
                return False

        if normalized_quest_id:
            quest_id = str(row.get("quest_id", "")).lower()
            quest_name = str(row.get("quest_name", "")).lower()
            if normalized_quest_id not in quest_id and normalized_quest_id not in quest_name:
                return False

        if normalized_result and str(row.get("quest_result", "")).lower() != normalized_result:
            return False

        if date_from and effective_date < date_from:
            return False

        if date_to and effective_date > date_to:
            return False

        return True

    def _admin_quest_attempt_row_signature(self, row: dict[str, Any]) -> tuple[Any, ...]:
        effective_at = row.get("completed_at") or row.get("started_at")
        if isinstance(effective_at, datetime):
            effective_at = effective_at.replace(second=0, microsecond=0)
        else:
            effective_at = None

        return (
            row.get("user_id"),
            row.get("player_id"),
            str(row.get("quest_id", "")).strip().lower(),
            str(row.get("quest_name", "")).strip().lower(),
            str(row.get("quest_result", "")).strip().lower(),
            int(row.get("success", 0) or 0),
            self._coerce_int(row.get("time_spent_seconds")),
            effective_at,
        )

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None

        return None

    def _coerce_int(self, value: Any) -> int:
        try:
            return max(0, int(float(value or 0)))
        except (TypeError, ValueError):
            return 0

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

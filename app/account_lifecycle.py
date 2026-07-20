from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import secrets
from typing import Any


APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_GRANDFATHERED = "grandfathered"

EMAIL_MISSING = "missing"
EMAIL_QUEUED = "queued"
EMAIL_SENT = "sent"
EMAIL_VERIFIED = "verified"
EMAIL_EXPIRED = "expired"
EMAIL_EXEMPT = "exempt"

APPROVAL_STATES = {
    APPROVAL_PENDING,
    APPROVAL_APPROVED,
    APPROVAL_REJECTED,
    APPROVAL_GRANDFATHERED,
}

EMAIL_VERIFICATION_STATES = {
    EMAIL_MISSING,
    EMAIL_QUEUED,
    EMAIL_SENT,
    EMAIL_VERIFIED,
    EMAIL_EXPIRED,
    EMAIL_EXEMPT,
}

def _token_secret() -> bytes:
    # Lazy import: app.security imports this module at its top level, so importing
    # security here at import time would be circular. By the time a verification
    # token is hashed, security is loaded and both modules share the same resolved
    # secret (env var, or the persisted random secret) - no weak fallback constant.
    from app.security import get_token_secret

    return get_token_secret().encode("utf-8")


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    approval_state: str
    email_verification_state: str
    can_login: bool
    next_step: str
    message: str


def build_new_user_lifecycle(*, role: str, email: str | None) -> dict[str, Any]:
    now = datetime.utcnow()
    normalized_email = (email or "").strip() or None
    if role == "admin":
        return {
            "approval_state": APPROVAL_APPROVED,
            "email_verification_state": EMAIL_EXEMPT,
            "approved_at": now,
            "approved_by_user_id": None,
            "verification_sent_at": None,
            "verification_expires_at": None,
            "verification_token_hash": None,
            "verified_at": now,
            "rejection_reason": None,
            "email": normalized_email,
        }

    return {
        "approval_state": APPROVAL_PENDING,
        "email_verification_state": EMAIL_QUEUED if normalized_email else EMAIL_MISSING,
        "approved_at": None,
        "approved_by_user_id": None,
        "verification_sent_at": None,
        "verification_expires_at": None,
        "verification_token_hash": None,
        "verified_at": None,
        "rejection_reason": None,
        "email": normalized_email,
    }


def build_legacy_user_lifecycle(*, role: str, created_at: datetime | None) -> dict[str, Any]:
    approved_at = created_at or datetime.utcnow()
    approval_state = APPROVAL_APPROVED if role == "admin" else APPROVAL_GRANDFATHERED
    return {
        "approval_state": approval_state,
        "email_verification_state": EMAIL_EXEMPT,
        "approved_at": approved_at,
        "approved_by_user_id": None,
        "verification_sent_at": None,
        "verification_expires_at": None,
        "verification_token_hash": None,
        "verified_at": approved_at,
        "rejection_reason": None,
    }


def normalize_lifecycle_document(document: dict[str, Any]) -> dict[str, Any]:
    role = str(document.get("role") or "user")
    created_at = document.get("created_at")
    legacy_defaults = build_legacy_user_lifecycle(role=role, created_at=created_at)

    approval_state = str(document.get("approval_state") or legacy_defaults["approval_state"])
    if approval_state not in APPROVAL_STATES:
        approval_state = legacy_defaults["approval_state"]

    email_verification_state = str(
        document.get("email_verification_state") or legacy_defaults["email_verification_state"]
    )
    if email_verification_state not in EMAIL_VERIFICATION_STATES:
        email_verification_state = legacy_defaults["email_verification_state"]

    normalized = {
        "approval_state": approval_state,
        "email_verification_state": email_verification_state,
        "approved_at": document.get("approved_at") or legacy_defaults["approved_at"],
        "approved_by_user_id": document.get("approved_by_user_id"),
        "verification_sent_at": document.get("verification_sent_at"),
        "verification_expires_at": document.get("verification_expires_at"),
        "verification_token_hash": document.get("verification_token_hash"),
        "verified_at": document.get("verified_at") or (
            legacy_defaults["verified_at"] if email_verification_state == EMAIL_EXEMPT else None
        ),
        "rejection_reason": document.get("rejection_reason"),
    }

    effective_verification_state = resolve_effective_email_state(
        normalized["email_verification_state"],
        normalized["verification_expires_at"],
    )
    if effective_verification_state != normalized["email_verification_state"]:
        normalized["email_verification_state"] = effective_verification_state

    return normalized


def resolve_effective_email_state(state: str, expires_at: datetime | None) -> str:
    if state == EMAIL_SENT and expires_at and expires_at <= datetime.utcnow():
        return EMAIL_EXPIRED
    return state


def describe_login_state(*, role: str, approval_state: str, email_verification_state: str, rejection_reason: str | None) -> LifecycleSnapshot:
    if approval_state == APPROVAL_REJECTED:
        reason_text = rejection_reason.strip() if rejection_reason else None
        message = "This account was rejected."
        if reason_text:
            message = f"{message} Reason: {reason_text}"
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="contact_admin",
            message=message,
        )

    if approval_state == APPROVAL_PENDING:
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="await_admin_approval",
            message="Account created. Wait for admin approval before logging in.",
        )

    if approval_state not in {APPROVAL_APPROVED, APPROVAL_GRANDFATHERED}:
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="contact_admin",
            message="This account is not eligible to log in yet.",
        )

    if role == "admin":
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=True,
            next_step="none",
            message="Admin account is active.",
        )

    if email_verification_state == EMAIL_MISSING:
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="add_email",
            message="An email address must be added by an admin before verification can start.",
        )

    if email_verification_state in {EMAIL_QUEUED, EMAIL_EXPIRED}:
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="request_verification_email",
            message="Email verification is not complete yet.",
        )

    if email_verification_state == EMAIL_SENT:
        return LifecycleSnapshot(
            approval_state=approval_state,
            email_verification_state=email_verification_state,
            can_login=False,
            next_step="verify_email",
            message="Check your email and open the verification link before logging in.",
        )

    return LifecycleSnapshot(
        approval_state=approval_state,
        email_verification_state=email_verification_state,
        can_login=True,
        next_step="none",
        message="Account is active.",
    )


def issue_verification_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, hash_verification_token(token)


def hash_verification_token(token: str) -> str:
    digest = hmac.new(
        _token_secret(),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest

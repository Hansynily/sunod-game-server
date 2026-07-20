import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import os
import secrets
from pathlib import Path

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import account_lifecycle
from app.database import get_db
from app.repository import TelemetryRepository


_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 390000
_SALT_BYTES = 16
_logger = logging.getLogger("app.security")


def _resolve_token_secret() -> str:
    """Return the HMAC secret for signing access tokens.

    Priority:
      1. AUTH_TOKEN_SECRET environment variable (recommended for any real deployment).
      2. A random secret persisted to a local file, so tokens survive server restarts
         without shipping a guessable constant.
      3. A per-process random secret if the file cannot be written.

    The old behaviour fell back to a hardcoded string ("...change-me"), which let anyone
    forge login tokens. That constant has been removed.
    """
    env_secret = (os.getenv("AUTH_TOKEN_SECRET") or "").strip()
    if env_secret:
        return env_secret

    secret_path = Path(__file__).resolve().parent.parent / ".auth_token_secret"
    try:
        existing = secret_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except OSError:
        _logger.warning(
            "AUTH_TOKEN_SECRET not set and %s is unreadable; using a per-process secret "
            "(logins reset on restart).",
            secret_path,
        )
        return secrets.token_urlsafe(48)

    generated = secrets.token_urlsafe(48)
    try:
        secret_path.write_text(generated, encoding="utf-8")
        _logger.warning(
            "AUTH_TOKEN_SECRET not set; generated a persistent random secret at %s. "
            "Set AUTH_TOKEN_SECRET in the environment for production deployments, and keep "
            "this file out of shared/version-controlled folders.",
            secret_path,
        )
    except OSError:
        _logger.warning(
            "AUTH_TOKEN_SECRET not set and could not persist a secret; using a per-process "
            "secret (logins reset on restart).",
        )
    return generated


_TOKEN_SECRET = _resolve_token_secret()


def get_token_secret() -> str:
    """The resolved HMAC secret, shared by every module that signs or hashes tokens."""
    return _TOKEN_SECRET


_TOKEN_SCHEME = HTTPBearer(auto_error=False)
ADMIN_SESSION_COOKIE = "sunod_admin_session"


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    user_id: int
    username: str
    role: str


def describe_user_access(user) -> account_lifecycle.LifecycleSnapshot:
    effective_email_state = account_lifecycle.resolve_effective_email_state(
        user.email_verification_state,
        user.verification_expires_at,
    )
    return account_lifecycle.describe_login_state(
        role=user.role,
        approval_state=user.approval_state,
        email_verification_state=effective_email_state,
        rejection_reason=user.rejection_reason,
    )


def ensure_user_can_authenticate(user) -> account_lifecycle.LifecycleSnapshot:
    snapshot = describe_user_access(user)
    if snapshot.can_login:
        return snapshot

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=snapshot.message,
    )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _ITERATIONS,
    )
    return "$".join(
        [
            _ALGORITHM,
            str(_ITERATIONS),
            _encode_bytes(salt),
            _encode_bytes(digest),
        ]
    )


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False

    try:
        algorithm, iterations_text, salt_text, digest_text = stored_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != _ALGORITHM:
        return False

    try:
        iterations = int(iterations_text)
        salt = _decode_bytes(salt_text)
        expected_digest = _decode_bytes(digest_text)
    except (ValueError, TypeError):
        return False

    candidate_digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate_digest, expected_digest)


def create_access_token(*, user_id: int, username: str, role: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(
        _TOKEN_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).digest()
    return f"{_encode_bytes(payload_bytes)}.{_encode_bytes(signature)}"


def parse_access_token(token: str) -> AuthenticatedPrincipal:
    try:
        payload_text, signature_text = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("Invalid access token.") from exc

    try:
        payload_bytes = _decode_bytes(payload_text)
        signature = _decode_bytes(signature_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid access token.") from exc

    expected_signature = hmac.new(
        _TOKEN_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid access token.")

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
        user_id = int(payload["user_id"])
        username = str(payload["username"]).strip()
        role = str(payload["role"]).strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid access token.") from exc

    if not username or role not in {"admin", "user"}:
        raise ValueError("Invalid access token.")

    return AuthenticatedPrincipal(
        user_id=user_id,
        username=username,
        role=role,
    )


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_TOKEN_SCHEME),
) -> AuthenticatedPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    try:
        return parse_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def get_optional_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_TOKEN_SCHEME),
) -> AuthenticatedPrincipal | None:
    if credentials is None:
        return None

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    try:
        return parse_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def get_current_user(
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
    db: TelemetryRepository = Depends(get_db),
):
    user = db.find_user_by_id(principal.user_id)
    if user is None or user.username != principal.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )
    ensure_user_can_authenticate(user)
    return user


def get_optional_user(
    principal: AuthenticatedPrincipal | None = Depends(get_optional_principal),
    db: TelemetryRepository = Depends(get_db),
):
    if principal is None:
        return None

    user = db.find_user_by_id(principal.user_id)
    if user is None or user.username != principal.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )
    ensure_user_can_authenticate(user)
    return user


def require_admin(
    user = Depends(get_current_user),
):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_bytes(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))

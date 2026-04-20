from __future__ import annotations

import os
import uuid
from getpass import getpass

from app.database import close_db, get_database
from app.repository import DuplicateUserError, TelemetryRepository
from app.security import hash_password


def _prompt_username() -> str:
    env_username = os.getenv("ADMIN_USERNAME")
    if env_username and env_username.strip():
        return env_username.strip()

    while True:
        username = input("Admin username: ").strip()
        if username:
            return username
        print("Username cannot be empty.")


def _prompt_password() -> str:
    env_password = os.getenv("ADMIN_PASSWORD")
    if env_password is not None:
        env_password = env_password.strip()
        if len(env_password) < 6:
            raise ValueError("ADMIN_PASSWORD must be at least 6 characters.")
        return env_password

    while True:
        password = getpass("Admin password: ")
        if len(password) < 6:
            print("Password must be at least 6 characters.")
            continue

        confirmation = getpass("Confirm password: ")
        if password != confirmation:
            print("Passwords do not match.")
            continue

        return password


def main() -> None:
    repository = TelemetryRepository(get_database())
    username = ""

    try:
        username = _prompt_username()
        existing_user = repository.find_user_by_username(username)
        if existing_user:
            print(f"User '{username}' already exists. Skipping admin creation.")
            return

        password = _prompt_password()
        user = repository.create_user(
            player_id=str(uuid.uuid4()),
            username=username,
            password_hash=hash_password(password),
            email=None,
            role="admin",
        )
        print(f"Admin account created for '{user.username}' with user_id={user.id}.")
    except DuplicateUserError:
        print(f"User '{username}' already exists. Skipping admin creation.")
    finally:
        close_db()


if __name__ == "__main__":
    main()
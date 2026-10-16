"""One-off: register the survey participants from survey-records.csv.

Each player gets a random temporary password, is created already approved + email
verified, and is emailed their username and password (Brevo if BREVO_API_KEY is set,
otherwise the SMTP settings in .env).

Target database: IMPORT_MONGODB_URI / IMPORT_MONGODB_DB from `.env.import` (gitignored),
so the import can point at Atlas while .env keeps pointing at the local server.

Usage (from sunod-game-server/):
    venv\\Scripts\\python.exe import_survey_users.py            # dry run: validate only
    venv\\Scripts\\python.exe import_survey_users.py --commit   # create users + send emails

Safe to re-run: usernames/emails that already exist are skipped. Passwords are written
nowhere except the email, and import_results_PRIVATE.csv for any row whose email failed.
"""
from __future__ import annotations

import csv
import os
import secrets
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "survey-records.csv"
RESULTS_PATH = ROOT / "import_results_PRIVATE.csv"
GENDERS = {"male", "female", "prefer_not_to_say"}
# No look-alike characters (0/O, 1/l/I) - players type this on a phone.
PASSWORD_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.import", override=True)
if os.getenv("IMPORT_MONGODB_URI"):
    os.environ["MONGODB_URI"] = os.environ["IMPORT_MONGODB_URI"]
    os.environ["MONGODB_DB"] = os.getenv("IMPORT_MONGODB_DB") or "telemetry_db"

from pymongo import MongoClient  # noqa: E402

from app import account_lifecycle, mailer  # noqa: E402
from app.repository import DuplicateUserError, TelemetryRepository  # noqa: E402
from app.security import hash_password  # noqa: E402


def temp_password() -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(10))


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    problems = []
    for i, row in enumerate(rows, start=2):
        for key in ("username", "name", "email", "birthdate", "gender"):
            if not (row.get(key) or "").strip():
                problems.append(f"line {i}: missing {key}")
        if (row.get("gender") or "").strip() not in GENDERS:
            problems.append(f"line {i}: bad gender {row.get('gender')!r}")
        try:
            date.fromisoformat((row.get("birthdate") or "").strip())
        except ValueError:
            problems.append(f"line {i}: bad birthdate {row.get('birthdate')!r}")
    if problems:
        sys.exit("CSV problems:\n  " + "\n  ".join(problems))
    return rows


def credentials_email(name: str, username: str, password: str) -> tuple[str, str]:
    link = mailer.get_mailer_settings().app_public_url
    text = (
        f"Hello {name},\n\n"
        "Thank you for joining the SUNOD study. Your game account is ready.\n\n"
        f"Username: {username}\n"
        f"Password: {password}\n\n"
        "Open SUNOD and log in with these. Keep this password private.\n\n"
        "- SUNOD Studio\n"
    )
    html = (
        f"<p>Hello {name},</p>"
        "<p>Thank you for joining the SUNOD study. Your game account is ready.</p>"
        f"<p><b>Username:</b> {username}<br><b>Password:</b> <code>{password}</code></p>"
        "<p>Open SUNOD and log in with these. Keep this password private.</p>"
        "<p>- SUNOD Studio</p>"
    )
    return html, text


def main() -> None:
    commit = "--commit" in sys.argv
    rows = load_rows()

    uri = os.environ["MONGODB_URI"]
    target = "ATLAS" if uri.startswith("mongodb+srv") else "LOCAL"
    db_name = os.environ.get("MONGODB_DB", "telemetry_db")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    repo = TelemetryRepository(client[db_name])
    settings = mailer.get_mailer_settings()
    sender = "Brevo" if settings.brevo_api_key else ("SMTP" if settings.enabled else "NONE (log only)")
    print(f"Target: {target} db={db_name} | users in db now: {repo.users.count_documents({})}")
    print(f"Email via: {sender} from {settings.from_email} | CSV rows: {len(rows)}")

    new_rows, skipped = [], []
    for row in rows:
        username, email = row["username"].strip(), row["email"].strip().lower()
        if repo.users.find_one({"$or": [{"username": username}, {"email": email}]}, {"_id": 1}):
            skipped.append(username)
        else:
            new_rows.append(row)
    print(f"To create: {len(new_rows)} | already exist (skipped): {len(skipped)} {skipped[:10]}")

    if not commit:
        print("Dry run only. Re-run with --commit to create users and send emails.")
        return
    if sender.startswith("NONE"):
        sys.exit("No email sender configured - refusing to create accounts nobody can log in to.")

    results = []
    for row in new_rows:
        username, email = row["username"].strip(), row["email"].strip().lower()
        password = temp_password()
        try:
            user = repo.create_user(
                player_id=str(uuid.uuid4()),
                username=username,
                password_hash=hash_password(password),
                email=email,
                name=row["name"].strip(),
                birthdate=date.fromisoformat(row["birthdate"].strip()),
                gender=row["gender"].strip(),
            )
        except DuplicateUserError:
            results.append((username, email, "skipped_duplicate", "", ""))
            continue

        now = datetime.utcnow()
        created = None
        try:
            created = datetime.strptime(row.get("created", "").strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            pass
        repo.users.update_one({"id": user.id}, {"$set": {
            "approval_state": account_lifecycle.APPROVAL_APPROVED,
            "approved_at": now,
            "email_verification_state": account_lifecycle.EMAIL_VERIFIED,
            "verified_at": now,
            "imported_from": "survey-records.csv",
            **({"created_at": created} if created else {}),
        }})

        html, text = credentials_email(row["name"].strip(), username, password)
        try:
            mailer.send_email(to_email=email, subject="Your SUNOD account", html_body=html, text_body=text)
            results.append((username, email, "created", "sent", ""))
            print(f"  ok   {username}")
        except Exception as exc:  # keep going; password saved below so it can be handed out
            results.append((username, email, "created", f"FAILED: {exc}", password))
            print(f"  MAIL FAILED {username}: {exc}")

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["username", "email", "account", "email_status", "password_if_email_failed"])
        writer.writerows(results)
        writer.writerows((u, "", "already_existed", "", "") for u in skipped)

    sent = sum(1 for r in results if r[3] == "sent")
    print(f"Done: created {sum(1 for r in results if r[2] == 'created')}, emailed {sent}, "
          f"email failures {sum(1 for r in results if r[3].startswith('FAILED'))}. Results: {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()

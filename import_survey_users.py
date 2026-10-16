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
# v2 = same players plus the run result each produced on the earlier study server
# (48 item answers + the model's cluster/career). Those results are imported as one
# session run per player, so they show in the admin reports/player pages.
CSV_PATH = ROOT.parent / "survey-recordsv2.csv"
ITEM_CODES = [f"{letter}{n}" for letter in "RIASEC" for n in range(1, 9)]
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


def _csv_datetime(row: dict, key: str) -> datetime | None:
    try:
        return datetime.strptime((row.get(key) or "").strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _num(value, cast=float):
    try:
        return cast(str(value).strip()) if str(value).strip() != "" else None
    except ValueError:
        return None


def build_imported_run(row: dict, user, created: datetime | None) -> dict | None:
    """One session_runs doc from the v2 columns, or None if the row has no answers."""
    items = {code: _num(row.get(code)) for code in ITEM_CODES}
    if all(v is None for v in items.values()):
        return None
    # Per-letter 0-10 score from the 8 item answers (1-5 scale): mean 1 -> 0, mean 5 -> 10.
    raw = {}
    for letter in "RIASEC":
        vals = [items[f"{letter}{n}"] for n in range(1, 9) if items[f"{letter}{n}"] is not None]
        raw[letter] = sum(vals) / len(vals) if vals else 1.0
    riasec = {letter.lower(): max(0, min(10, int(round((raw[letter] - 1) / 4 * 10)))) for letter in "RIASEC"}
    holland = "".join(sorted("RIASEC", key=lambda letter: -raw[letter])[:3])

    run = {
        "player_id": user.player_id,
        "user_id": user.id,
        "username": user.username,
        "session_id": f"import-{(row.get('user_id') or user.username).strip()}",
        "scene_version": "imported_v1",
        "total_time_spent_seconds": 0.0,
        "rounds": [],
        "aggregated_features": {
            "item_scores": items,
            "quests_played": _num(row.get("played"), int),
            "real_slots": _num(row.get("real_slots"), int),
            "favorites": (row.get("favorites") or "").strip(),
            "prediction_latency_ms": _num(row.get("latency_ms")),
        },
        "riasec_scores": riasec,
        "holland_code": holland,
        "career_family": (row.get("cluster_label") or "").strip() or "Pending Cluster Result",
        "career_result": (row.get("career_result") or "").strip() or "Pending Cluster Result",
        "source": "model_rf",
        "model_version": "imported",
        "created_at": created or datetime.utcnow(),
        "imported_from": CSV_PATH.name,
    }
    cluster = _num(row.get("predicted_cluster"), int)
    if cluster is not None:
        run.update({
            "predicted_cluster": cluster,
            "career_cluster": _num(row.get("career_cluster"), int),
            "cluster_label": (row.get("cluster_label") or "").strip() or None,
            "cluster_holland_code": (row.get("cluster_holland_code") or "").strip() or None,
            "cluster_example_careers": [],
            "cluster_source": "model_rf",
            "cluster_model_version": "imported",
        })
    return run


def import_run(repo, row: dict, user, created: datetime | None) -> str:
    run = build_imported_run(row, user, created)
    if run is None:
        return "no_answers"
    if repo.session_runs.find_one({"player_id": user.player_id, "session_id": run["session_id"]}, {"_id": 1}):
        return "run_exists"
    repo.session_runs.insert_one(run)
    return "run_imported"


def main() -> None:
    commit = "--commit" in sys.argv
    # --no-email: skip sending; every password goes to import_results_PRIVATE.csv instead.
    no_email = "--no-email" in sys.argv
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

    # --runs-only: players already exist; just add each one's imported run (skips ones done).
    if "--runs-only" in sys.argv:
        statuses = []
        for row in rows:
            user = repo.find_user_by_username(row["username"].strip())
            if user is None:
                statuses.append("no_user")
                continue
            created = None
            try:
                created = datetime.strptime(row.get("created", "").strip(), "%Y-%m-%d %H:%M")
            except ValueError:
                pass
            if commit:
                statuses.append(import_run(repo, row, user, created))
            else:
                run = build_imported_run(row, user, created)
                exists = run and repo.session_runs.find_one(
                    {"player_id": user.player_id, "session_id": run["session_id"]}, {"_id": 1})
                statuses.append("no_answers" if run is None else ("run_exists" if exists else "would_import"))
        from collections import Counter
        print(f"Runs-only {'COMMIT' if commit else 'dry run'}: {dict(Counter(statuses))}")
        return

    if not commit:
        print("Dry run only. Re-run with --commit to create users and send emails.")
        return
    if sender.startswith("NONE") and not no_email:
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
            "imported_from": CSV_PATH.name,
            **({"created_at": created} if created else {}),
            **({"last_login": _csv_datetime(row, "last_login")} if _csv_datetime(row, "last_login") else {}),
        }})

        run_status = import_run(repo, row, repo.find_user_by_id(user.id) or user, created)

        if no_email:
            results.append((username, email, f"created+{run_status}", "not_sent", password))
            print(f"  ok   {username}")
            continue
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
        writer.writerow(["username", "email", "account", "email_status", "password_if_not_emailed"])
        writer.writerows(results)
        writer.writerows((u, "", "already_existed", "", "") for u in skipped)

    sent = sum(1 for r in results if r[3] == "sent")
    print(f"Runs: imported {sum(1 for r in results if 'run_imported' in r[2])}, "
          f"no answers {sum(1 for r in results if 'no_answers' in r[2])}")
    print(f"Done: created {sum(1 for r in results if r[2].startswith('created'))}, emailed {sent}, "
          f"email failures {sum(1 for r in results if r[3].startswith('FAILED'))}. Results: {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()

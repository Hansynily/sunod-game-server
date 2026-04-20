# sunod-game-server

Backend telemetry service for the Sunod game project.

This repository is the backend for:
https://github.com/Hansynily/sunod-game

## Tech Stack

- Python
- FastAPI
- MongoDB
- PyMongo
- Jinja2 (admin HTML pages)

## Features

- Collect quest attempt telemetry from the game client
- Store users, quest attempts, selected skills, and RIASEC profile aggregates
- Gate player login by admin approval and email verification status
- Send verification emails with token-based confirmation links
- Provide admin API endpoints for user and performance data
- Provide admin web pages to inspect users, account state, and player performance

## Database Choice

This project uses `MongoDB` as its database.

- `NoSQL` is the database category or model
- `MongoDB` is the specific NoSQL database used in this project

Application flow:

```text
Game client -> FastAPI backend -> MongoDB -> Admin API / Admin pages
```

Data is stored in MongoDB documents rather than SQL tables. In this project, each user
document contains embedded `quest_attempts` data and an embedded `riasec_profile`.

## Project Structure

```text
app/
  main.py                 # FastAPI app entrypoint
  database.py             # MongoDB connection and dependency wiring
  models.py               # In-memory domain models
  repository.py           # MongoDB persistence helpers
  schemas.py              # Pydantic schemas
  routers/telemetry.py    # Telemetry, admin API, admin UI routes
templates/
  users.html
  user_performance.html
requirements.txt
```

## Prerequisites

- Python 3.10+ (recommended)
- MongoDB running locally on the URI you put in `.env`

## Environment Variables

The app requires these variables. For local development, create `Project/sunod-game-server/.env`:

```env
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DB=telemetry_db
MONGODB_TIMEOUT_MS=5000
APP_PUBLIC_URL=http://127.0.0.1:8000
EMAIL_VERIFICATION_TTL_HOURS=24
SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=
```

`HOST` and `PORT` are optional. If omitted, the server listens on `0.0.0.0:8000`.

SMTP is optional for local development. If `SMTP_HOST` and `SMTP_FROM_EMAIL` are left empty, the server
will log the verification email content locally instead of sending it.

## Run Locally

On Windows, if MongoDB is already running, you can use:

```bat
launch_server.bat
```

Manual way:

### 1. Clone and enter the repo

```bash
git clone https://github.com/Hansynily/sunod-game-server.git
cd sunod-game-server
```

### 2. Create and activate virtual environment

Windows (PowerShell):

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows (CMD):

```cmd
py -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MongoDB

Make sure your local MongoDB service is running. The app will use the database from
`.env` and create its collections and indexes automatically.


### 5. Create `.env`

```env
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DB=telemetry_db
MONGODB_TIMEOUT_MS=5000
```

### 6. Start the server

```bash
py run_server.py
```

On startup, the app pings MongoDB and ensures the required indexes exist.
Legacy users without lifecycle fields are backfilled as grandfathered and verification-exempt.

## Local URLs

- API root docs (Swagger): `http://127.0.0.1:8000/docs`
- Admin users page: `http://127.0.0.1:8000/admin/users`

## Main Endpoints

Telemetry:
- `POST /api/telemetry/quest-attempt`
- `POST /api/telemetry/users`
- `GET /api/telemetry/users/{user_id}`
- `POST /api/telemetry/users/{user_id}/quest-attempts`
- `GET /api/telemetry/users/{user_id}/quest-attempts`

Admin API:
- `GET /api/admin/users`
- `GET /api/admin/users/{user_id}`
- `GET /api/admin/users/{user_id}/performance`
- `POST /api/admin/users/{user_id}/set-email`
- `POST /api/admin/users/{user_id}/approve`
- `POST /api/admin/users/{user_id}/reject`
- `POST /api/admin/users/{user_id}/send-verification`
- `POST /api/admin/users/{user_id}/resend-verification`
- `POST /api/admin/users/{user_id}/mark-verified`

Admin UI:
- `GET /admin/login`
- `GET /admin/users`
- `GET /admin/users/{user_id}`
- `POST /admin/users/{user_id}/email`
- `POST /admin/users/{user_id}/approve`
- `POST /admin/users/{user_id}/reject`
- `POST /admin/users/{user_id}/send-verification`
- `POST /admin/users/{user_id}/resend-verification`
- `POST /admin/users/{user_id}/mark-verified`
- `POST /admin/users/{user_id}/delete`

Public verification:
- `GET /verify-email?token=...`

## Tests

Run the route and admin UI smoke tests with:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

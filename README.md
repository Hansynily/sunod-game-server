# sunod-telemetry

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
- Provide admin API endpoints for user and performance data
- Provide admin web pages to inspect users and user performance

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
- MongoDB running locally (default: port 27017)

## Environment Variables

The app reads these variables from environment:

- `MONGODB_URI` (default: `mongodb://127.0.0.1:27017`)
- `MONGODB_DB` (default: `telemetry_db`)
- `MONGODB_TIMEOUT_MS` (default: `3000`)

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

Make sure your local MongoDB service is running. The app will use the `telemetry_db`
database by default and create its collections and indexes automatically.


### 5. Set environment variables

PowerShell:

```powershell
$env:MONGODB_URI="mongodb://127.0.0.1:27017"
$env:MONGODB_DB="telemetry_db"
```

CMD:

```cmd
set MONGODB_URI=mongodb://127.0.0.1:27017
set MONGODB_DB=telemetry_db
```

### 6. Start the server

```bash
py -m uvicorn app.main:app --reload
```

On startup, the app pings MongoDB and ensures the required indexes exist.

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

Admin UI:
- `GET /admin/users`
- `GET /admin/users/{user_id}`
- `POST /admin/users/{user_id}/delete`

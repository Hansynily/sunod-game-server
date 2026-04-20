import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.database import close_db, init_db
from app.cluster_runtime import warm_load_career_models, warm_load_cluster_model
from app.routers import telemetry


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    cluster_status = warm_load_cluster_model(logger=LOGGER)
    warm_load_career_models(logger=LOGGER, cluster_status=cluster_status)
    try:
        yield
    finally:
        close_db()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Game Telemetry API",
        version="0.1.0",
        description="Backend service for collecting and querying game telemetry.",
        lifespan=lifespan,
    )

    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    application.include_router(telemetry.router)
    application.include_router(telemetry.predict_router)
    application.include_router(telemetry.public_router)
    application.include_router(telemetry.admin_router)
    application.include_router(telemetry.admin_ui_router)

    return application


app = create_app()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import close_db, init_db
from app.ml_runtime import warm_load_model
from app.routers import telemetry


LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    warm_load_model(logger=LOGGER)
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

    application.include_router(telemetry.router)
    application.include_router(telemetry.admin_router)
    application.include_router(telemetry.admin_ui_router)

    return application


app = create_app()

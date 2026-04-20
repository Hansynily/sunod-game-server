import os
from importlib.util import find_spec
import logging.config
from pathlib import Path
import sys
if os.getenv("ADMIN_USERNAME") and os.getenv("ADMIN_PASSWORD"):
    try:
        print("Running admin bootstrap...")
        from create_admin_railway import main
        main()
    except Exception as e:
        print(f"Admin bootstrap failed: {e}")


from dotenv import load_dotenv
import uvicorn

from app.logging_utils import build_log_config


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
load_dotenv(ROOT / ".env")


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    spec = find_spec("app.main")
    resolved = spec.origin if spec else "missing"
    print(f"Resolved app.main -> {resolved}")
    print(f"Backend root -> {ROOT}")

    if "--check" in sys.argv:
        return

    # admin creation before server 
    try:
        if os.getenv("ADMIN_USERNAME") and os.getenv("ADMIN_PASSWORD"):
            print("Running admin bootstrap...")
            main()
        else:
            print("Skipping admin bootstrap (no env vars set)")
    except Exception as e:
        print(f"Admin bootstrap failed: {e}")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_config = build_log_config(LOG_DIR)
    logging.config.dictConfig(log_config)
    logging.getLogger(__name__).info("Server log file -> %s", LOG_DIR / "server.log")
    logging.getLogger(__name__).info("Audit log file -> %s", LOG_DIR / "audit.log")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=_env_flag("RELOAD", default=False),
        app_dir=str(ROOT),
        log_config=log_config,
        access_log=True,
    )


if __name__ == "__main__":
    main()

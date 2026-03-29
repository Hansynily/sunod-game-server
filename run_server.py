import os
from importlib.util import find_spec
from pathlib import Path
import sys

from dotenv import load_dotenv
import uvicorn


ROOT = Path(__file__).resolve().parent
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

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=_env_flag("RELOAD", default=False),
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()

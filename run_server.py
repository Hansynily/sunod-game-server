from importlib.util import find_spec
from pathlib import Path
import sys

import uvicorn


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = find_spec("app.main")
    resolved = spec.origin if spec else "missing"
    print(f"Resolved app.main -> {resolved}")
    print(f"Backend root -> {ROOT}")

    if "--check" in sys.argv:
        return

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()

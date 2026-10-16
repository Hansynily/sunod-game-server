"""Stress-test the career model with random player answers and save every result.

Each entry is a fake player: a 48-item answer vector built the same way the game builds
it (unplayed item = neutral 3, refused/unfinished quest = 1, finished quest = 2 + stars,
so 2..5). Only items that have a quest in the game get real answers; the rest stay 3.

Two targets:
  local  (default) - loads the model files in model_assets/runtime and calls them directly.
                     Fast; this is the raw model (cluster RF -> career RF), no server rules.
  <url>            - POSTs to <url>/api/predict on a running server (local or Render), so
                     the server's rules (coverage gate, outlier redirect, profile check)
                     apply too, and response times are measured.

Examples (from sunod-game-server/):
  venv\\Scripts\\python.exe stress_test_model.py                       # 500 entries, local
  venv\\Scripts\\python.exe stress_test_model.py -n 10000 --seed 1     # 10k, reproducible
  venv\\Scripts\\python.exe stress_test_model.py -n 300 --target http://127.0.0.1:8000 --workers 8
  venv\\Scripts\\python.exe stress_test_model.py --profile picky --play-rate 1.0
  venv\\Scripts\\python.exe stress_test_model.py --codes all           # pretend all 48 quests exist

Output: stress_results/stress_<target>_<time>.csv (one row per entry: profile, the 48
answers, prediction, source, latency) plus a summary printed at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAME_PREFABS = ROOT.parent / "sunod-game" / "Assets" / "Prefabs"
LETTERS = "RIASEC"
ALL_CODES = [f"{letter}{n}" for letter in LETTERS for n in range(1, 9)]  # model column order
# Codes with a built quest (scanned from the game's room prefabs on 2026-10-17). Used when
# the game folder isn't next to this one; otherwise the prefabs are scanned live.
KNOWN_GAME_CODES = [
    "A1", "A2", "A4", "A8", "C1", "C2", "C6", "E1", "E3", "E4", "E7",
    "I1", "I7", "I8", "R1", "R4", "R6", "R7", "R8", "S4", "S8",
]
PROFILES = ("uniform", "finisher", "picky", "refuser", "mixed")


def game_question_codes() -> list[str]:
    if not GAME_PREFABS.exists():
        return list(KNOWN_GAME_CODES)
    found = set()
    pattern = re.compile(r"^\s*questionCode:\s*([RIASEC][1-8])\s*$", re.M)
    for prefab in GAME_PREFABS.rglob("*.prefab"):
        try:
            found.update(pattern.findall(prefab.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return sorted(found) or list(KNOWN_GAME_CODES)


def finished_score(rng: random.Random) -> int:
    return 2 + rng.randint(0, 3)  # 2..5 (2 + stars)


def make_answers(profile: str, codes: list[str], play_rate: float, rng: random.Random) -> tuple[list[float], dict]:
    """Returns (48 values in R1..C8 order, info). Unplayed = 3.0."""
    if profile == "mixed":
        profile = rng.choice(PROFILES[:-1])
    played = [c for c in codes if rng.random() < play_rate]
    favorites = set(rng.sample(LETTERS, k=rng.randint(1, 2)))
    answers = {}
    for code in played:
        if profile == "uniform":
            answers[code] = rng.randint(1, 5)
        elif profile == "finisher":  # finishes nearly everything
            answers[code] = finished_score(rng) if rng.random() < 0.9 else 1
        elif profile == "refuser":  # turns most quests down
            answers[code] = 1 if rng.random() < 0.75 else finished_score(rng)
        elif profile == "picky":  # loves 1-2 letters, refuses most others
            if code[0] in favorites:
                answers[code] = rng.choice([4, 5, 5])
            else:
                answers[code] = 1 if rng.random() < 0.7 else finished_score(rng)
    vector = [float(answers.get(code, 3)) for code in ALL_CODES]
    info = {
        "profile": profile,
        "favorites": "".join(sorted(favorites)) if profile == "picky" else "",
        "played": len(played),
        "real_slots": sum(1 for v in vector if v != 3.0),
    }
    return vector, info


# ── targets ──────────────────────────────────────────────────────────────────────

class LocalModel:
    name = "local"

    def __init__(self) -> None:
        sys.path.insert(0, str(ROOT))
        from app import career_mapping, cluster_mapping, cluster_runtime

        self.cr, self.cm, self.cl = cluster_runtime, career_mapping, cluster_mapping
        cluster_status = cluster_runtime.warm_load_cluster_model()
        career_status = cluster_runtime.warm_load_career_models(cluster_status=cluster_status)
        if not (cluster_status.available and career_status.available):
            sys.exit(f"Model not loaded: {cluster_status.reason or career_status.reason}")
        # The server loads career models per request to save RAM; here keep all 8 loaded
        # so thousands of entries run fast.
        import joblib
        models = {cid: joblib.load(path) for cid, path in (career_status.model_paths or {}).items()}
        career_status = cluster_runtime.CareerModelStatus(
            available=True, models=models, model_paths=career_status.model_paths)
        self.cluster_status, self.career_status = cluster_status, career_status

    def predict(self, vector: list[float]) -> dict:
        cluster = self.cr.predict_cluster(vector, self.cluster_status)
        career_cluster = self.cr.predict_career_cluster(vector, cluster, self.career_status)
        profile = self.cl.get_cluster_profile(cluster)
        return {
            "predicted_cluster": cluster,
            "career_cluster": career_cluster,
            "career_result": self.cm.resolve_career_result(cluster, career_cluster) or "",
            "cluster_label": profile.label if profile else "",
            "cluster_holland_code": profile.holland_code if profile else "",
            "source": "raw_model" + ("_outlier" if profile and profile.is_outlier else ""),
        }


class HttpModel:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.url = base_url.rstrip("/") + "/api/predict"
        self.name = re.sub(r"[^A-Za-z0-9]+", "-", base_url.split("//")[-1]).strip("-")
        self.timeout = timeout

    def predict(self, vector: list[float]) -> dict:
        body = json.dumps({
            "player_id": f"stress-{uuid.uuid4().hex[:12]}",
            "session_id": f"stress-{uuid.uuid4().hex[:12]}",
            "features": vector,
        }).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, method="POST",
                                         headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        return {key: data.get(key, "") for key in (
            "predicted_cluster", "career_cluster", "career_result",
            "cluster_label", "cluster_holland_code", "source")}


# ── run ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--entries", type=int, default=500, help="how many fake players (default 500)")
    parser.add_argument("--target", default="local", help="'local' (default) or a server URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--profile", choices=PROFILES, default="mixed", help="answer style (default mixed)")
    parser.add_argument("--play-rate", type=float, default=0.8, help="share of coded quests each player plays, 0-1 (default 0.8)")
    parser.add_argument("--codes", default="game", help="'game' (quests that exist), 'all' (48), or a list like R1,I2,A3")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for repeatable runs")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests for URL targets (default 1)")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request for URL targets")
    parser.add_argument("--out", default=None, help="output CSV path (default stress_results/...)")
    args = parser.parse_args()

    if args.codes == "game":
        codes = game_question_codes()
    elif args.codes == "all":
        codes = list(ALL_CODES)
    else:
        codes = [c.strip().upper() for c in args.codes.split(",") if c.strip()]
        bad = [c for c in codes if c not in ALL_CODES]
        if bad:
            sys.exit(f"Unknown codes: {bad}")
    rng = random.Random(args.seed)
    target = LocalModel() if args.target == "local" else HttpModel(args.target, args.timeout)

    print(f"Target: {args.target} | entries: {args.entries} | profile: {args.profile} | "
          f"play rate: {args.play_rate} | coded items: {len(codes)} ({','.join(codes)})")

    entries = [make_answers(args.profile, codes, args.play_rate, rng) for _ in range(args.entries)]

    def run_one(index: int) -> dict:
        vector, info = entries[index]
        started = time.perf_counter()
        row = {"entry": index + 1, **info}
        try:
            row.update(target.predict(vector))
            row["error"] = ""
        except Exception as exc:  # recorded, the run continues
            row["error"] = str(exc)[:300]
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row.update({code: int(value) for code, value in zip(ALL_CODES, vector)})
        return row

    started = time.perf_counter()
    workers = max(1, args.workers if isinstance(target, HttpModel) else 1)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(run_one, range(len(entries))), start=1):
            rows.append(row)
            if done % max(1, len(entries) // 10) == 0:
                print(f"  {done}/{len(entries)}")
    elapsed = time.perf_counter() - started

    out = Path(args.out) if args.out else ROOT / "stress_results" / (
        f"stress_{target.name}_{datetime.now():%Y%m%d_%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["entry", "profile", "favorites", "played", "real_slots", "predicted_cluster",
               "career_cluster", "career_result", "cluster_label", "cluster_holland_code",
               "source", "latency_ms", "error", *ALL_CODES]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    latencies = sorted(r["latency_ms"] for r in ok)
    print(f"\nDone in {elapsed:.1f}s ({len(rows) / max(elapsed, 1e-9):.1f} entries/s) -> {out}")
    print(f"Errors: {len(rows) - len(ok)} / {len(rows)}")
    if latencies:
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"Latency ms: median {statistics.median(latencies):.1f}, p95 {p95:.1f}, max {latencies[-1]:.1f}")
    for title, key in (("Clusters", "cluster_label"), ("Careers (top 10)", "career_result"), ("Sources", "source")):
        counts = Counter(str(r.get(key) or "-") for r in ok)
        print(f"\n{title}:")
        for value, count in counts.most_common(10 if "top" in title else None):
            print(f"  {count:6d}  {count / max(len(ok), 1):6.1%}  {value}")
    errors = Counter(r["error"] for r in rows if r["error"])
    if errors:
        print("\nMost common errors:")
        for message, count in errors.most_common(3):
            print(f"  {count:6d}  {message}")


if __name__ == "__main__":
    main()

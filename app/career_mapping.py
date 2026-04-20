from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DIMENSIONS = ("R", "I", "A", "S", "E", "C")
BACKEND_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CAREER_MAP_PATH = BACKEND_ROOT / "model_assets" / "runtime" / "career_map.json"

UNDETERMINED_RESULT = "Undetermined"


def derive_holland_code(
    scores: dict[str, float],
    skill_use_totals: dict[str, int],
) -> str:
    if _all_scores_zero(scores):
        return UNDETERMINED_RESULT

    fixed_order = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
    ordered = sorted(
        DIMENSIONS,
        key=lambda dimension: (
            -scores.get(dimension.lower(), 0.0),
            -skill_use_totals.get(dimension.lower(), 0),
            fixed_order[dimension],
        ),
    )
    return "".join(ordered[:3])


def derive_career_family(scores: dict[str, float]) -> str:
    if _all_scores_zero(scores):
        return UNDETERMINED_RESULT

    family_scores = {
        "Technical & Operations": (0.60 * scores["r"]) + (0.40 * scores["c"]),
        "Research & Analysis": (0.75 * scores["i"]) + (0.25 * scores["c"]),
        "Creative & Media": (0.75 * scores["a"]) + (0.25 * scores["e"]),
        "People & Leadership": (0.65 * scores["s"]) + (0.35 * scores["e"]),
    }
    return max(family_scores, key=family_scores.get)


def _all_scores_zero(scores: dict[str, float]) -> bool:
    return all(value <= 0 for value in scores.values())


def resolve_career_result(cluster_id: int, career_cluster_id: int) -> str | None:
    runtime_map = _load_runtime_career_map()
    cluster_map = runtime_map.get(int(cluster_id))
    if not cluster_map:
        return None

    result = cluster_map.get(int(career_cluster_id))
    if result is None:
        return None

    result = str(result).strip()
    return result or None


@lru_cache(maxsize=1)
def _load_runtime_career_map() -> dict[int, dict[int, str]]:
    if not RUNTIME_CAREER_MAP_PATH.exists():
        return {}

    try:
        raw_value = json.loads(RUNTIME_CAREER_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw_value, dict):
        return {}

    runtime_map: dict[int, dict[int, str]] = {}
    for raw_cluster, careers in raw_value.items():
        try:
            cluster_id = int(raw_cluster)
        except (TypeError, ValueError):
            continue

        if not isinstance(careers, dict):
            continue

        cluster_map: dict[int, str] = {}
        for raw_career_cluster, career_label in careers.items():
            try:
                career_cluster_id = int(raw_career_cluster)
            except (TypeError, ValueError):
                continue

            cleaned_label = str(career_label or "").strip()
            if cleaned_label:
                cluster_map[career_cluster_id] = cleaned_label

        if cluster_map:
            runtime_map[cluster_id] = cluster_map

    return runtime_map

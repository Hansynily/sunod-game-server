from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app import schemas

LOGGER = logging.getLogger(__name__)


DIMENSIONS = ("r", "i", "a", "s", "e", "c")


@dataclass(frozen=True, slots=True)
class ChallengeDefinition:
    challenge_id: str
    primary_riasec: str


CHALLENGE_DEFINITIONS = (
    ChallengeDefinition("challenge_cat_quest", "R"),
    ChallengeDefinition("challenge_stub_02", "I"),
    ChallengeDefinition("challenge_stub_03", "A"),
    ChallengeDefinition("challenge_stub_04", "S"),
    ChallengeDefinition("challenge_stub_05", "E"),
    ChallengeDefinition("challenge_stub_06", "C"),
    ChallengeDefinition("L1_CatQuest", "S"),
    ChallengeDefinition("L1_LostFriend", "C"),
    ChallengeDefinition("L2_MissingKey", "R"),
    ChallengeDefinition("L2_CatMedicine", "A"),
    ChallengeDefinition("L3_FallenSparrow", "S"),
    ChallengeDefinition("L3_BlockedPath", "C"),
    ChallengeDefinition("L3_SlipperyWay", "E"),
    # New data-driven quest rooms - codes match what the game client sends.
    ChallengeDefinition("r_pump", "R"),
    ChallengeDefinition("a_mural", "A"),
    ChallengeDefinition("i_map", "I"),
)

CHALLENGE_TAG_BY_ID = {
    definition.challenge_id: definition.primary_riasec
    for definition in CHALLENGE_DEFINITIONS
}

FEATURE_SCHEMA = (
    "rounds_attempted",
    "rounds_cleared",
    "clear_rate",
    "total_stars",
    "avg_stars",
    "total_retries",
    "total_time_seconds",
    "avg_round_time_seconds",
    "skill_use_r",
    "skill_use_i",
    "skill_use_a",
    "skill_use_s",
    "skill_use_e",
    "skill_use_c",
    "skill_ratio_r",
    "skill_ratio_i",
    "skill_ratio_a",
    "skill_ratio_s",
    "skill_ratio_e",
    "skill_ratio_c",
    "solved_r",
    "solved_i",
    "solved_a",
    "solved_s",
    "solved_e",
    "solved_c",
    "stars_r",
    "stars_i",
    "stars_a",
    "stars_s",
    "stars_e",
    "stars_c",
)

LABEL_SCHEMA = tuple(f"label_{dimension}" for dimension in DIMENSIONS)


def validate_run_summary(
    payload: schemas.RunSummaryTelemetryIn | Mapping[str, Any],
) -> schemas.RunSummaryTelemetryIn:
    if isinstance(payload, schemas.RunSummaryTelemetryIn):
        validated = payload
    else:
        validated = schemas.RunSummaryTelemetryIn.model_validate(payload)

    _validate_round_catalog(validated)
    return validated


def extract_feature_vector(
    payload: schemas.RunSummaryTelemetryIn | Mapping[str, Any],
) -> list[float]:
    feature_record = extract_feature_record(payload)
    return build_feature_vector(feature_record)


def build_feature_vector(feature_record: Mapping[str, Any]) -> list[float]:
    missing_features = [
        feature_name for feature_name in FEATURE_SCHEMA if feature_name not in feature_record
    ]
    if missing_features:
        raise ValueError(
            f"Feature record is missing required fields: {', '.join(missing_features)}"
        )

    return [float(feature_record[name]) for name in FEATURE_SCHEMA]


def extract_feature_record(
    payload: schemas.RunSummaryTelemetryIn | Mapping[str, Any],
) -> dict[str, float]:
    validated = validate_run_summary(payload)
    rounds = validated.rounds

    rounds_attempted = len(rounds)
    rounds_cleared = sum(1 for round_entry in rounds if round_entry.solved)
    clear_rate = (rounds_cleared / rounds_attempted) if rounds_attempted else 0.0

    total_stars = sum(round_entry.stars_earned for round_entry in rounds)
    avg_stars = (total_stars / rounds_attempted) if rounds_attempted else 0.0

    total_retries = sum(round_entry.retry_count for round_entry in rounds)
    total_time_seconds = float(validated.total_time_spent_seconds)
    avg_round_time_seconds = (
        total_time_seconds / rounds_attempted if rounds_attempted else 0.0
    )

    skill_use_totals = {
        dimension: sum(
            int(getattr(round_entry, f"skill_use_{dimension}", 0))
            for round_entry in rounds
        )
        for dimension in DIMENSIONS
    }
    total_skill_uses = sum(skill_use_totals.values())
    skill_ratios = {
        dimension: (
            skill_use_totals[dimension] / total_skill_uses if total_skill_uses else 0.0
        )
        for dimension in DIMENSIONS
    }

    solved_by_dimension = {
        dimension: sum(
            1
            for round_entry in rounds
            if round_entry.primary_riasec.lower() == dimension and round_entry.solved
        )
        for dimension in DIMENSIONS
    }
    stars_by_dimension = {
        dimension: sum(
            round_entry.stars_earned
            for round_entry in rounds
            if round_entry.primary_riasec.lower() == dimension
        )
        for dimension in DIMENSIONS
    }

    return {
        "rounds_attempted": rounds_attempted,
        "rounds_cleared": rounds_cleared,
        "clear_rate": clear_rate,
        "total_stars": total_stars,
        "avg_stars": avg_stars,
        "total_retries": total_retries,
        "total_time_seconds": total_time_seconds,
        "avg_round_time_seconds": avg_round_time_seconds,
        "skill_use_r": skill_use_totals["r"],
        "skill_use_i": skill_use_totals["i"],
        "skill_use_a": skill_use_totals["a"],
        "skill_use_s": skill_use_totals["s"],
        "skill_use_e": skill_use_totals["e"],
        "skill_use_c": skill_use_totals["c"],
        "skill_ratio_r": skill_ratios["r"],
        "skill_ratio_i": skill_ratios["i"],
        "skill_ratio_a": skill_ratios["a"],
        "skill_ratio_s": skill_ratios["s"],
        "skill_ratio_e": skill_ratios["e"],
        "skill_ratio_c": skill_ratios["c"],
        "solved_r": solved_by_dimension["r"],
        "solved_i": solved_by_dimension["i"],
        "solved_a": solved_by_dimension["a"],
        "solved_s": solved_by_dimension["s"],
        "solved_e": solved_by_dimension["e"],
        "solved_c": solved_by_dimension["c"],
        "stars_r": stars_by_dimension["r"],
        "stars_i": stars_by_dimension["i"],
        "stars_a": stars_by_dimension["a"],
        "stars_s": stars_by_dimension["s"],
        "stars_e": stars_by_dimension["e"],
        "stars_c": stars_by_dimension["c"],
    }


def extract_skill_use_totals(feature_record: Mapping[str, Any]) -> dict[str, int]:
    return {
        dimension: int(feature_record[f"skill_use_{dimension}"])
        for dimension in DIMENSIONS
    }


def write_feature_schema(output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(list(FEATURE_SCHEMA), indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_round_catalog(payload: schemas.RunSummaryTelemetryIn) -> None:
    seen_challenge_ids: set[str] = set()
    for round_entry in payload.rounds:
        if round_entry.challenge_id in seen_challenge_ids:
            raise ValueError(
                f"Duplicate challenge_id '{round_entry.challenge_id}' found in run summary."
            )
        seen_challenge_ids.add(round_entry.challenge_id)

        expected_tag = CHALLENGE_TAG_BY_ID.get(round_entry.challenge_id)
        if expected_tag is None:
            LOGGER.warning(
                "Unknown challenge_id '%s' - not in catalog. "
                "Using client-provided primary_riasec '%s' for feature extraction.",
                round_entry.challenge_id,
                round_entry.primary_riasec,
            )
            continue

        if round_entry.primary_riasec.upper() != expected_tag:
            LOGGER.warning(
                "challenge_id '%s' expects primary_riasec '%s' but client sent '%s'. "
                "Catalog tag will be used for solved/stars dimension attribution.",
                round_entry.challenge_id,
                expected_tag,
                round_entry.primary_riasec,
            )

from collections.abc import Mapping
from dataclasses import dataclass


DIMENSIONS = ("r", "i", "a", "s", "e", "c")
INTEGER_SCORE_SCALE_MAX = 10


@dataclass(frozen=True, slots=True)
class RubricScores:
    scores: dict[str, float]
    skill_use_totals: dict[str, int]

    @property
    def is_undetermined(self) -> bool:
        return all(value <= 0 for value in self.scores.values())

    @property
    def integer_scores(self) -> dict[str, int]:
        return normalized_scores_to_integer_scale(self.scores)


def calculate_riasec_scores(rounds) -> RubricScores:
    skill_use_totals = {dimension: 0 for dimension in DIMENSIONS}
    tag_stars_totals = {dimension: 0 for dimension in DIMENSIONS}

    for round_entry in rounds:
        primary_riasec = str(round_entry.primary_riasec).strip().lower()
        for dimension in DIMENSIONS:
            skill_use_totals[dimension] += int(
                getattr(round_entry, f"skill_use_{dimension}", 0)
            )

        if primary_riasec in tag_stars_totals:
            tag_stars_totals[primary_riasec] += int(round_entry.stars_earned)

    raw_scores = {
        dimension: (2 * skill_use_totals[dimension]) + (3 * tag_stars_totals[dimension])
        for dimension in DIMENSIONS
    }
    total_raw = sum(raw_scores.values())

    if total_raw > 0:
        normalized_scores = {
            dimension: (100.0 * raw_scores[dimension]) / total_raw
            for dimension in DIMENSIONS
        }
    else:
        normalized_scores = {dimension: 0.0 for dimension in DIMENSIONS}

    return RubricScores(
        scores=normalized_scores,
        skill_use_totals=skill_use_totals,
    )


def normalized_scores_to_integer_scale(
    scores: Mapping[str, float],
    scale_max: int = INTEGER_SCORE_SCALE_MAX,
) -> dict[str, int]:
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")

    return {
        dimension: max(
            0,
            min(
                scale_max,
                _round_half_up(
                    (max(float(scores.get(dimension, 0.0)), 0.0) / 100.0) * scale_max
                ),
            ),
        )
        for dimension in DIMENSIONS
    }


def round_scores_to_integers(
    scores: Mapping[str, float],
    scale_max: int = INTEGER_SCORE_SCALE_MAX,
) -> dict[str, int]:
    if scale_max <= 0:
        raise ValueError("scale_max must be positive.")

    return {
        dimension: max(
            0,
            min(scale_max, _round_half_up(float(scores.get(dimension, 0.0)))),
        )
        for dimension in DIMENSIONS
    }


def _round_half_up(value: float) -> int:
    return int(value + 0.5)

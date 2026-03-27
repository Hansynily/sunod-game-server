DIMENSIONS = ("R", "I", "A", "S", "E", "C")

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

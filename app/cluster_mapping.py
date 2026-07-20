from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    cluster: int
    holland_code: str
    label: str
    example_careers: tuple[str, ...]
    is_outlier: bool = False

# Real cluster profiles for the frozen partition. Holland codes match
# cluster_runtime.CLUSTER_HOLLAND_CODES; labels/careers match career_map.json and the
# client's EndSceneUI.ClusterMap. Keep all three in sync. Clusters 1 and 4 are the
# outlier buckets the server never routes real players into.
_DEFAULT_CLUSTER_MAP: dict[int, ClusterProfile] = {
    0: ClusterProfile(0, "IA", "Research", ("Computer Scientist", "Biologist", "Chemist"), False),
    1: ClusterProfile(1, "N/A", "Varied Interests", tuple(), True),
    2: ClusterProfile(2, "RIC", "Engineering", ("Architect", "Civil Engineer", "Web Developer"), False),
    3: ClusterProfile(3, "AS", "Arts & Design", ("Graphic Artist", "Writer", "Musician"), False),
    4: ClusterProfile(4, "N/A", "Varied Interests", tuple(), True),
    5: ClusterProfile(5, "SEC", "Business & Finance", ("Accountant", "Financial Analyst", "Entrepreneur"), False),
    6: ClusterProfile(6, "S", "Social Services", ("Teacher", "Lawyer", "Counselor"), False),
    7: ClusterProfile(7, "SIA", "Healthcare", ("Doctor", "Nurse", "Pharmacist"), False),
}


def get_cluster_profile(cluster: int) -> ClusterProfile | None:
    return _DEFAULT_CLUSTER_MAP.get(cluster)

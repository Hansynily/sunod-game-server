from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    cluster: int
    holland_code: str
    label: str
    example_careers: tuple[str, ...]
    is_outlier: bool = False


_CLUSTER_MAP: dict[int, ClusterProfile] = {
    0: ClusterProfile(0, "RI", "Engineering", ("Civil Engineer", "Programmer", "Architect")),
    1: ClusterProfile(1, "IA", "Arts & Design", ("Fashion Designer", "Graphic Artist", "Writer")),
    2: ClusterProfile(2, "SEC", "Business & Finance", ("Accountant", "Financial Analyst", "Entrepreneur")),
    3: ClusterProfile(3, "AS", "Performing Arts", ("Musician", "Athlete", "Entertainer")),
    4: ClusterProfile(4, "-", "Varied Interests", tuple(), True),
    5: ClusterProfile(5, "IS", "Research", ("Computer Scientist", "Zoologist", "Epidemiologist")),
    6: ClusterProfile(6, "SC", "Social Services", ("Lawyer", "Teacher", "Counselor")),
    7: ClusterProfile(7, "-", "Varied Interests", tuple(), True),
    8: ClusterProfile(8, "IAS", "Healthcare", ("Doctor", "Nurse", "Pharmacist")),
}


def get_cluster_profile(cluster: int) -> ClusterProfile | None:
    return _CLUSTER_MAP.get(cluster)

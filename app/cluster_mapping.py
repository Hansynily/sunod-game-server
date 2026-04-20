from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterProfile:
    cluster: int
    holland_code: str
    label: str
    example_careers: tuple[str, ...]
    is_outlier: bool = False

_DEFAULT_CLUSTER_MAP: dict[int, ClusterProfile] = {
    cluster_id: ClusterProfile(
        cluster_id,
        "N/A",
        f"Cluster {cluster_id}",
        tuple(),
        False,
    )
    for cluster_id in range(8)
}


def get_cluster_profile(cluster: int) -> ClusterProfile | None:
    return _DEFAULT_CLUSTER_MAP.get(cluster)

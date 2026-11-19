"""Free-Space-Preserving Hierarchical Graph Coarsening (Phase B).

When the number of active hazards exceeds cognitive budget k, adjacent hazards
separated by an impassable gap (< min_navigable_gap, inspired by Hydra Layer-3 Places)
are merged into a single 'Super-Node' representing a unified no-go zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np


@dataclass
class CompressedEntity:
    """Unified output entity: either a single DSG object node or a coarsened Super-Node."""

    entity_id: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    score: float
    is_super_node: bool = False
    member_ids: Set[str] = field(default_factory=set)


def _xy_edge_gap(a: CompressedEntity, b: CompressedEntity) -> float:
    """Minimum horizontal surface-to-surface gap (meters) between two 3D bounding boxes."""
    dx = max(abs(float(a.pos[0] - b.pos[0])) - 0.5 * float(a.size[0] + b.size[0]), 0.0)
    dy = max(abs(float(a.pos[1] - b.pos[1])) - 0.5 * float(a.size[1] + b.size[1]), 0.0)
    return float(np.hypot(dx, dy))


def _merge_pair(a: CompressedEntity, b: CompressedEntity, merge_idx: int) -> CompressedEntity:
    """Merge two adjacent entities into an enclosing 3D bounding-box Super-Node."""
    a_min, a_max = a.pos - 0.5 * a.size, a.pos + 0.5 * a.size
    b_min, b_max = b.pos - 0.5 * b.size, b.pos + 0.5 * b.size
    box_min = np.minimum(a_min, b_min)
    box_max = np.maximum(a_max, b_max)

    members = set(a.member_ids) | set(b.member_ids)
    # Conservative velocity: keep faster member's velocity so DWA stays cautious
    vel = a.vel.copy() if float(np.linalg.norm(a.vel)) >= float(np.linalg.norm(b.vel)) else b.vel.copy()

    return CompressedEntity(
        entity_id=f"super_hazard_{merge_idx}",
        label=f"no_go_cluster({len(members)})",
        pos=0.5 * (box_min + box_max),
        size=box_max - box_min,
        vel=vel,
        score=max(a.score, b.score),
        is_super_node=True,
        member_ids=members,
    )


def coarsen_hazards_preserving_free_space(
    graph_obs: nx.DiGraph,
    scored_ids: List[Tuple[str, float]],
    hazard_ids: Set[str],
    k: int,
    min_navigable_gap: float = 0.70,
) -> List[CompressedEntity]:
    """Compress observed entities to <= k while merging impassable hazard clusters.

    Args:
        graph_obs: Observed 3D Scene Graph G_t^obs.
        scored_ids: All observed object IDs sorted descending by score (nid, score).
        hazard_ids: Subset of object IDs flagged as high-risk hazards (R_v > tau_haz).
        k: Cognitive bandwidth budget (max returned entities).
        min_navigable_gap: Minimum horizontal clearance (meters) required to treat a gap
            as a navigable Place passage (default 0.70m ~ human shoulder width + margin).

    Returns:
        List of <= k CompressedEntity items (single nodes and/or super-nodes).
    """
    score_map: Dict[str, float] = dict(scored_ids)

    # 1. Start with all flagged hazards as candidate entities
    haz_entities: List[CompressedEntity] = []
    for nid, sc in scored_ids:
        if nid in hazard_ids:
            d = graph_obs.nodes[nid]
            haz_entities.append(
                CompressedEntity(
                    entity_id=nid,
                    label=str(d.get("label", "obstacle")),
                    pos=np.array(d["pos"], dtype=float),
                    size=np.array(d["size"], dtype=float),
                    vel=np.array(d["vel"], dtype=float),
                    score=sc,
                    is_super_node=False,
                    member_ids={nid},
                )
            )

    # 2. If hazards overflow budget k, greedily merge closest IMPASSABLE pair (gap < min_navigable_gap)
    merge_idx = 0
    while len(haz_entities) > k:
        best_pair: Tuple[int, int] | None = None
        best_gap = float("inf")

        for i in range(len(haz_entities)):
            for j in range(i + 1, len(haz_entities)):
                gap = _xy_edge_gap(haz_entities[i], haz_entities[j])
                if gap < min_navigable_gap and gap < best_gap:
                    best_gap = gap
                    best_pair = (i, j)

        if best_pair is None:
            # No impassable pair left: stop merging to avoid blocking navigable gaps!
            break

        i, j = best_pair
        merged = _merge_pair(haz_entities[i], haz_entities[j], merge_idx)
        merge_idx += 1
        haz_entities = [e for idx, e in enumerate(haz_entities) if idx not in (i, j)]
        haz_entities.append(merged)

    # Sort hazards by score and cap at k if still > k after safe coarsening
    haz_entities.sort(key=lambda e: e.score, reverse=True)
    result = haz_entities[:k]

    # 3. Fill any remaining budget slots with highest-scoring non-hazard structural anchors
    if len(result) < k:
        used_members = set().union(*(e.member_ids for e in result)) if result else set()
        for nid, sc in scored_ids:
            if len(result) >= k:
                break
            if nid in used_members:
                continue
            d = graph_obs.nodes[nid]
            result.append(
                CompressedEntity(
                    entity_id=nid,
                    label=str(d.get("label", "object")),
                    pos=np.array(d["pos"], dtype=float),
                    size=np.array(d["size"], dtype=float),
                    vel=np.array(d["vel"], dtype=float),
                    score=score_map[nid],
                    is_super_node=False,
                    member_ids={nid},
                )
            )

    return result

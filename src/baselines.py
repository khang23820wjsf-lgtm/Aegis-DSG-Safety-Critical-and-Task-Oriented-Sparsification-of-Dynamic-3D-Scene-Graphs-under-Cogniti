"""Baseline DSG sparsification methods for Phase A benchmarking."""

from __future__ import annotations

from typing import Dict, List, Protocol, Set
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState


class Sparsifier(Protocol):
    """Common interface for all DSG sparsification algorithms."""

    name: str

    def reset(self) -> None:
        """Reset any temporal state before starting a new sequence."""

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        """Return up to k object node IDs from graph_obs."""


def _get_object_ids(graph_obs: nx.DiGraph) -> List[str]:
    return [nid for nid, data in graph_obs.nodes(data=True) if data.get("layer") == "object"]


class RandomSparsifier:
    """Uniformly samples up to k observed objects."""

    def __init__(self, seed: int = 0) -> None:
        self.name = "Random"
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._base_seed)

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = _get_object_ids(graph_obs)
        if len(obj_ids) <= k:
            return set(obj_ids)
        chosen = self._rng.choice(obj_ids, size=k, replace=False)
        return set(str(x) for x in chosen)


class DistanceSparsifier:
    """Selects k objects with smallest 3D Euclidean distance to the agent."""

    def __init__(self) -> None:
        self.name = "Distance"

    def reset(self) -> None:
        pass

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = _get_object_ids(graph_obs)
        if len(obj_ids) <= k:
            return set(obj_ids)
        scored = sorted(
            obj_ids,
            key=lambda nid: float(np.linalg.norm(graph_obs.nodes[nid]["pos"] - agent.pos)),
        )
        return set(scored[:k])


class SemanticSparsifier:
    """Prioritizes objects purely by semantic class priority, breaking ties by distance."""

    CLASS_PRIORITY: Dict[str, float] = {
        "pedestrian": 1.0,
        "stair_edge": 0.95,
        "low_barrier": 0.90,
        "chair": 0.70,
        "table": 0.70,
        "wall_pillar": 0.60,
        "door": 0.50,
        "shelf": 0.40,
        "unknown_ghost": 0.20,
        "wall_painting": 0.05,
        "ceiling_light": 0.05,
    }

    def __init__(self) -> None:
        self.name = "Semantic"

    def reset(self) -> None:
        pass

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = _get_object_ids(graph_obs)
        if len(obj_ids) <= k:
            return set(obj_ids)

        def score_fn(nid: str) -> tuple[float, float]:
            data = graph_obs.nodes[nid]
            prio = self.CLASS_PRIORITY.get(data.get("label", ""), 0.3)
            dist = float(np.linalg.norm(data["pos"] - agent.pos))
            return (-prio, dist)

        scored = sorted(obj_ids, key=score_fn)
        return set(scored[:k])

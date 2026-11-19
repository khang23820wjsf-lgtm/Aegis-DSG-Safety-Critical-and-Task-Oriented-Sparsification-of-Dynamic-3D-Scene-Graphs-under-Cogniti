"""Adapted Robotics Baselines for Phase B: D-Lite (Chang et al., 2023) & Clio (Maggio et al., 2024)."""

from __future__ import annotations

from typing import Dict, List, Set
import math
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState


class DLiteAdaptedSparsifier:
    """Adapted D-Lite (Chang et al., 2023) baseline under k-entity cognitive budget.

    Preserves global shortest-path topological anchors (doors, stairs, pillars,
    corridor-axis boundary obstacles) toward the forward navigation goal,
    representing global topology-preserving compression without fast local dynamic TTC.
    """

    def __init__(self) -> None:
        self.name = "D-Lite*"

    def reset(self) -> None:
        pass

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        if len(obj_ids) <= k:
            return set(obj_ids)

        heading = agent.vel[:2] / (float(np.linalg.norm(agent.vel[:2])) + 1e-8)

        def topo_score(nid: str) -> float:
            d = graph_obs.nodes[nid]
            pos = d["pos"]
            rel = pos[:2] - agent.pos[:2]
            dist = float(np.linalg.norm(rel))
            # Forward progress along global path + structural connectivity degree
            forward_proj = float(np.dot(rel, heading))
            lateral_off = float(abs(rel[0] * heading[1] - rel[1] * heading[0]))
            c_v = float(d.get("structural_importance", 0.1))
            deg = float(graph_obs.degree(nid)) / 6.0

            # D-Lite prioritizes global path anchors & corridor boundary clearance
            # (ignores relative velocity of fast moving obstacles)
            path_relevance = math.exp(-(lateral_off ** 2) / (2.0 * 1.1 ** 2)) if forward_proj > 0 else 0.1
            dist_term = 1.0 / (1.0 + 0.35 * dist)
            return 0.45 * c_v + 0.35 * (path_relevance * dist_term) + 0.20 * min(deg, 1.0)

        ranked = sorted(obj_ids, key=topo_score, reverse=True)
        return set(ranked[:k])


class ClioAdaptedSparsifier:
    """Adapted Clio (Maggio et al., 2024) baseline under k-entity cognitive budget.

    Approximates task-driven open-set Information Bottleneck for task prompt
    "avoid obstacles and navigate safely forward" using semantic task relevance
    weighted by observation confidence, without real-time geometric TTC or temporal hysteresis.
    """

    # Simulated CLIP/open-set cosine similarity to "avoid obstacles and navigate safely forward"
    TASK_SEMANTIC_SIM: Dict[str, float] = {
        "pedestrian": 0.92,
        "stair_edge": 0.90,
        "low_barrier": 0.88,
        "overhead_sign": 0.85,
        "table": 0.78,
        "chair": 0.76,
        "wall_pillar": 0.72,
        "door": 0.74,
        "shelf": 0.58,
        "unknown_ghost": 0.35,
        "floor_mat": 0.25,
        "wall_painting": 0.12,
        "ceiling_light": 0.10,
    }

    def __init__(self) -> None:
        self.name = "Clio*"

    def reset(self) -> None:
        pass

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        if len(obj_ids) <= k:
            return set(obj_ids)

        def ib_score(nid: str) -> float:
            d = graph_obs.nodes[nid]
            sim = self.TASK_SEMANTIC_SIM.get(str(d.get("label", "")), 0.30)
            q_v = float(d.get("confidence", 0.8))
            dist = float(np.linalg.norm(d["pos"][:2] - agent.pos[:2]))
            # Information Bottleneck relevance: semantic match * confidence * mild proximity
            return sim * q_v * (0.65 + 0.35 * math.exp(-dist / 8.0))

        ranked = sorted(obj_ids, key=ib_score, reverse=True)
        return set(ranked[:k])

"""Mũi 3: Faithful Mathematical Implementations of MIT SPARK Baselines.

1. DLiteTrueSparsifier (Chang et al., IEEE RA-L 2023):
   Iterative node pruning minimizing exact Dijkstra shortest-path distortion
   between the agent and the forward navigation goal on the local topological graph.

2. ClioTrueIBSparsifier (Maggio et al., IEEE RA-L 2024):
   Exact Agglomerative Information Bottleneck (aIB, Slonim & Tishby 1999)
   merging object clusters to minimize Jensen-Shannon mutual-information loss
   Delta I(X_tilde; Y) with respect to task-prompt semantic distributions p(Y | X).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import math
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState


class DLiteTrueSparsifier:
    """True D-Lite (Chang et al., 2023): Exact Dijkstra Shortest-Path Distortion Minimization."""

    def __init__(self, connect_radius: float = 4.5) -> None:
        self.name = "D-Lite(True)"
        self.connect_radius = connect_radius

    def reset(self) -> None:
        pass

    def _build_nav_graph(self, graph_obs: nx.DiGraph, agent: AgentState, obj_ids: List[str]) -> nx.Graph:
        """Build local navigable proximity graph linking __start__, objects, and __goal__."""
        g_nav = nx.Graph()
        spd = float(np.linalg.norm(agent.vel[:2]))
        heading = agent.vel[:2] / spd if spd > 0.1 else np.array([0.0, 1.0])
        goal_xy = agent.pos[:2] + heading * 10.0

        g_nav.add_node("__start__", pos=agent.pos[:2].copy())
        g_nav.add_node("__goal__", pos=goal_xy)

        coords: Dict[str, np.ndarray] = {"__start__": agent.pos[:2], "__goal__": goal_xy}
        for nid in obj_ids:
            p = graph_obs.nodes[nid]["pos"][:2]
            g_nav.add_node(nid, pos=p)
            coords[nid] = p

        all_nodes = list(coords.keys())
        for i in range(len(all_nodes)):
            u = all_nodes[i]
            pu = coords[u]
            for j in range(i + 1, len(all_nodes)):
                v = all_nodes[j]
                if {u, v} == {"__start__", "__goal__"}:
                    continue
                d = float(np.linalg.norm(pu - coords[v]))
                if d <= self.connect_radius:
                    # Slight weight bonus for structural anchors (doors/stairs) along global path
                    c_u = float(graph_obs.nodes[u].get("structural_importance", 0.2)) if u in graph_obs else 0.5
                    c_v = float(graph_obs.nodes[v].get("structural_importance", 0.2)) if v in graph_obs else 0.5
                    w = d / (1.0 + 0.25 * (c_u + c_v))
                    g_nav.add_edge(u, v, weight=w)
        return g_nav

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        if len(obj_ids) <= k:
            return set(obj_ids)

        g_nav = self._build_nav_graph(graph_obs, agent, obj_ids)
        try:
            base_dist = float(nx.shortest_path_length(g_nav, "__start__", "__goal__", weight="weight"))
        except nx.NetworkXNoPath:
            base_dist = 1e6

        active = set(obj_ids)
        # Iteratively prune the node causing MINIMUM shortest-path distortion until |active| <= k
        while len(active) > k:
            try:
                sp_nodes = set(nx.shortest_path(g_nav, "__start__", "__goal__", weight="weight"))
            except nx.NetworkXNoPath:
                sp_nodes = set()

            # Fast path: nodes NOT on current shortest path have 0 immediate path distortion;
            # break ties by removing farthest / lowest-degree off-path node first.
            off_path = [n for n in active if n not in sp_nodes]
            if off_path:
                prune_node = max(
                    off_path,
                    key=lambda n: (
                        float(np.linalg.norm(graph_obs.nodes[n]["pos"][:2] - agent.pos[:2]))
                        - 0.8 * float(graph_obs.nodes[n].get("structural_importance", 0.0))
                    ),
                )
            else:
                # All remaining nodes lie on the shortest path: compute exact Dijkstra distortion!
                best_node = None
                best_distortion = float("inf")
                for cand in active:
                    edges_backup = list(g_nav.edges(cand, data=True))
                    g_nav.remove_node(cand)
                    try:
                        new_dist = float(nx.shortest_path_length(g_nav, "__start__", "__goal__", weight="weight"))
                    except nx.NetworkXNoPath:
                        new_dist = 1e6
                    distortion = abs(new_dist - base_dist)
                    if distortion < best_distortion:
                        best_distortion = distortion
                        best_node = cand
                    g_nav.add_node(cand)
                    g_nav.add_edges_from(edges_backup)
                prune_node = best_node if best_node is not None else next(iter(active))

            g_nav.remove_node(prune_node)
            active.remove(prune_node)

        return active


class ClioTrueIBSparsifier:
    """True Clio (Maggio et al., 2024): Agglomerative Information Bottleneck (aIB) over Task Semantics.

    Task vector Y = [collision_obstacle, moving_agent, elevation_hazard, nav_landmark, harmless_decor].
    Uses exact Jensen-Shannon divergence merge cost Delta I(X_tilde; Y) until |X_tilde| <= k.
    """

    # Normalized conditional task-relevance distributions p(Y | class) approximating CLIP task embeddings
    # Columns: [collision_obstacle, moving_agent, elevation_hazard, nav_landmark, harmless_decor]
    CLASS_TASK_DIST: Dict[str, List[float]] = {
        "pedestrian":    [0.25, 0.68, 0.02, 0.03, 0.02],
        "stair_edge":    [0.15, 0.02, 0.68, 0.13, 0.02],
        "overhead_sign": [0.45, 0.02, 0.45, 0.05, 0.03],
        "low_barrier":   [0.68, 0.03, 0.22, 0.04, 0.03],
        "chair":         [0.72, 0.02, 0.06, 0.05, 0.15],
        "table":         [0.74, 0.02, 0.05, 0.08, 0.11],
        "wall_pillar":   [0.55, 0.02, 0.05, 0.30, 0.08],
        "door":          [0.15, 0.02, 0.05, 0.72, 0.06],
        "shelf":         [0.45, 0.02, 0.05, 0.28, 0.20],
        "unknown_ghost": [0.30, 0.10, 0.10, 0.10, 0.40],
        "floor_mat":     [0.08, 0.02, 0.05, 0.10, 0.75],
        "wall_painting": [0.03, 0.01, 0.02, 0.06, 0.88],
        "ceiling_light": [0.03, 0.01, 0.02, 0.04, 0.90],
    }
    # Task prompt relevance weights for "avoid obstacles and navigate safely forward"
    TASK_PROMPT_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.65, 0.05], dtype=float)

    def __init__(self) -> None:
        self.name = "Clio(aIB)"

    def reset(self) -> None:
        pass

    @staticmethod
    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 1e-12
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

    def _jsd_merge_cost(self, pi: float, qi: np.ndarray, pj: float, qj: np.ndarray) -> float:
        """Exact aIB mutual information loss: (p_i + p_j) * JSD(p(Y|i) || p(Y|j))."""
        p_sum = pi + pj
        wi, wj = pi / p_sum, pj / p_sum
        m = wi * qi + wj * qj
        jsd = wi * self._kl(qi, m) + wj * self._kl(qj, m)
        return p_sum * jsd

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        n = len(obj_ids)
        if n <= k:
            return set(obj_ids)

        # Initialize singleton clusters for aIB
        clusters: List[Dict] = []
        raw_weights = []
        for nid in obj_ids:
            d = graph_obs.nodes[nid]
            q_v = float(d.get("confidence", 0.8))
            dist = float(np.linalg.norm(d["pos"][:2] - agent.pos[:2]))
            w = q_v * (0.5 + 0.5 * math.exp(-dist / 8.0))
            raw_weights.append(w)

        w_sum = sum(raw_weights) + 1e-12
        for nid, w in zip(obj_ids, raw_weights):
            d = graph_obs.nodes[nid]
            lbl = str(d.get("label", "unknown_ghost")).lower()
            dist_y = np.array(self.CLASS_TASK_DIST.get(lbl, [0.35, 0.05, 0.10, 0.15, 0.35]), dtype=float)
            dist_y /= dist_y.sum()
            task_rel = float(np.dot(dist_y, self.TASK_PROMPT_WEIGHTS)) * float(d.get("confidence", 0.8))
            clusters.append({
                "p": w / w_sum,
                "q": dist_y,
                "rep_id": nid,
                "rep_score": task_rel,
            })

        # Agglomerative Information Bottleneck: merge pair with minimum Delta I(X_tilde; Y)
        while len(clusters) > k:
            best_i, best_j = 0, 1
            best_cost = float("inf")
            m = len(clusters)
            for i in range(m):
                ci = clusters[i]
                for j in range(i + 1, m):
                    cj = clusters[j]
                    cost = self._jsd_merge_cost(ci["p"], ci["q"], cj["p"], cj["q"])
                    if cost < best_cost:
                        best_cost = cost
                        best_i, best_j = i, j

            ci, cj = clusters[best_i], clusters[best_j]
            p_new = ci["p"] + cj["p"]
            q_new = (ci["p"] * ci["q"] + cj["p"] * cj["q"]) / p_new
            rep_id = ci["rep_id"] if ci["rep_score"] >= cj["rep_score"] else cj["rep_id"]
            rep_score = max(ci["rep_score"], cj["rep_score"])

            merged = {"p": p_new, "q": q_new, "rep_id": rep_id, "rep_score": rep_score}
            clusters = [c for idx, c in enumerate(clusters) if idx not in (best_i, best_j)]
            clusters.append(merged)

        return {c["rep_id"] for c in clusters}

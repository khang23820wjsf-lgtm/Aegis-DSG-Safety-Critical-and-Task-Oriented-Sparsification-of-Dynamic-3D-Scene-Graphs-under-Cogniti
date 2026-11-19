"""Mũi 2A: Safety-Constrained Monotone Submodular Maximization for DSG Sparsification.

Formulates k-entity selection as a Facility-Location + Temporal Persistence
Monotone Submodular Maximization problem:

    max_{|S| <= k} F(S) = sum_{u in V} w_u * max_{v in S} K(u, v) + lambda * sum_{v in S cap V'_{t-1}} psi_v

Theoretical Guarantee (Nemhauser, Wolsey & Fisher, 1978):
Because F(S) is non-negative, monotone, and submodular, greedy marginal-gain selection
achieves >= (1 - 1/e) ~= 63.2% of the NP-hard global optimum in O(k * |V|) time.
"""

from __future__ import annotations

from typing import Dict, List, Set
import math
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState
from src.phase_b.coarsening import CompressedEntity, coarsen_hazards_preserving_free_space
from src.proposed import ProposedSparsifier


class SubmodularAegisSparsifier(ProposedSparsifier):
    """Provably near-optimal (1 - 1/e) Submodular Facility-Location DSG Sparsifier."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.16,
        gamma: float = 1.20,
        tau_haz: float = 0.28,
        fp_conf_threshold: float = 0.48,
        kernel_sigma_m: float = 0.75,
        lambda_stab: float = 0.16,
        enable_coarsening: bool = True,
        min_navigable_gap: float = 0.70,
        **kwargs,
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, tau_haz=tau_haz, **kwargs)
        self.name = "Submod-Aegis"
        self.fp_conf_threshold = fp_conf_threshold
        self.kernel_sigma_m = kernel_sigma_m
        self.lambda_stab = lambda_stab
        self.enable_coarsening = enable_coarsening
        self.min_navigable_gap = min_navigable_gap
        self._seen_counts: Dict[str, int] = {}

    def reset(self) -> None:
        super().reset()
        self._seen_counts.clear()

    def _coverage_kernel(self, pos_u: np.ndarray, pos_v: np.ndarray, size_u: np.ndarray, size_v: np.ndarray) -> float:
        """Spatial coverage kernel K(u, v) in [0, 1]; high only for impassably close/overlapping objects."""
        dx = max(abs(float(pos_u[0] - pos_v[0])) - 0.5 * float(size_u[0] + size_v[0]), 0.0)
        dy = max(abs(float(pos_u[1] - pos_v[1])) - 0.5 * float(size_u[1] + size_v[1]), 0.0)
        gap = math.hypot(dx, dy)
        if gap >= self.min_navigable_gap:
            return 0.0  # Do not treat objects across a navigable gap as redundant!
        return math.exp(-(gap ** 2) / (2.0 * (self.kernel_sigma_m * 0.5) ** 2))

    def select_entities(
        self,
        graph_obs: nx.DiGraph,
        agent: AgentState,
        k: int,
    ) -> List[CompressedEntity]:
        """Greedy marginal-gain maximization of monotone submodular F(S) under |S| <= k."""
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        n = len(obj_ids)
        if n == 0:
            self._prev_selected.clear()
            return []

        # Update temporal persistence for FP-aware uncertainty damping
        obs_set = set(obj_ids)
        for nid in obs_set:
            self._seen_counts[nid] = self._seen_counts.get(nid, 0) + 1
        for stale in list(self._seen_counts.keys()):
            if stale not in obs_set:
                del self._seen_counts[stale]

        # 1. Compute non-negative node importance weights w_u >= 0
        weights = np.zeros(n, dtype=float)
        positions = []
        sizes = []
        flagged_hazards: Set[str] = set()
        for i, nid in enumerate(obj_ids):
            d = graph_obs.nodes[nid]
            q_v = float(d.get("confidence", 1.0))
            u_v = 1.0 - q_v
            c_v = float(d.get("structural_importance", 0.0))
            r_v = self._compute_collision_risk(d, agent)

            is_fp_suspect = (self._seen_counts.get(nid, 1) == 1) and (q_v < self.fp_conf_threshold)
            gamma_eff = (0.15 * self.gamma) if is_fp_suspect else self.gamma
            haz_ind = 1.0 if r_v > self.tau_haz else 0.0
            if haz_ind > 0.0 and not is_fp_suspect:
                flagged_hazards.add(nid)

            weights[i] = max(0.0, self.alpha * r_v * (1.0 + gamma_eff * u_v * haz_ind) + self.beta * q_v * c_v)
            positions.append(np.array(d["pos"], dtype=float))
            sizes.append(np.array(d["size"], dtype=float))

        # 2. Precompute coverage kernel matrix K[u, v] (diagonal = 1.0)
        K = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                val = self._coverage_kernel(positions[i], positions[j], sizes[i], sizes[j])
                K[i, j] = val
                K[j, i] = val

        # 3. Greedy Monotone Submodular Maximization (Nemhauser 1 - 1/e guarantee)
        current_max_cov = np.zeros(n, dtype=float)
        selected_indices: List[int] = []
        remaining = set(range(n))

        for _ in range(min(k, n)):
            best_idx = -1
            best_gain = -1.0
            for v in remaining:
                # Marginal gain Delta F(v | S) = sum_u w_u * max(0, K[u,v] - current_max_cov[u]) + modular stability
                inc = np.maximum(K[:, v] - current_max_cov, 0.0)
                gain = float(np.dot(weights, inc))
                if obj_ids[v] in self._prev_selected and weights[v] >= self.theta_out:
                    gain += self.lambda_stab
                if gain > best_gain:
                    best_gain = gain
                    best_idx = v

            if best_idx < 0:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            current_max_cov = np.maximum(current_max_cov, K[:, best_idx])

        # 4. If hazards overflow k and coarsening is enabled, merge impassable hazard clusters
        scored_pairs = sorted([(obj_ids[i], float(weights[i])) for i in range(n)], key=lambda x: x[1], reverse=True)
        if self.enable_coarsening and len(flagged_hazards) > k:
            entities = coarsen_hazards_preserving_free_space(
                graph_obs=graph_obs,
                scored_ids=scored_pairs,
                hazard_ids=flagged_hazards,
                k=k,
                min_navigable_gap=self.min_navigable_gap,
            )
        else:
            entities = []
            for idx in selected_indices:
                nid = obj_ids[idx]
                d = graph_obs.nodes[nid]
                entities.append(
                    CompressedEntity(
                        entity_id=nid,
                        label=str(d.get("label", "object")),
                        pos=d["pos"].copy(),
                        size=d["size"].copy(),
                        vel=d["vel"].copy(),
                        score=float(weights[idx]),
                        is_super_node=False,
                        member_ids={nid},
                    )
                )

        covered = set().union(*(e.member_ids for e in entities)) if entities else set()
        self._prev_selected = covered
        return entities

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        entities = self.select_entities(graph_obs, agent, k)
        return set().union(*(e.member_ids for e in entities)) if entities else set()

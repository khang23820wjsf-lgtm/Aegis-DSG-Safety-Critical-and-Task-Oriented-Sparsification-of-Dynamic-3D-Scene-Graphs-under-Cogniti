"""Aegis-DSG v2 (Phase B): FP-Aware Asymmetric Risk + Safety Bubble + Free-Space Coarsening."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import networkx as nx

from src.dsg_generator import AgentState
from src.phase_b.coarsening import CompressedEntity, coarsen_hazards_preserving_free_space
from src.proposed import ProposedSparsifier


class AegisV2Sparsifier(ProposedSparsifier):
    """Phase B upgraded sparsifier addressing Phase A EXP 2 (FP ghosts) & EXP 3 (overflow)."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.16,
        gamma: float = 1.20,
        tau_haz: float = 0.28,
        fp_conf_threshold: float = 0.48,
        min_navigable_gap: float = 0.70,
        enable_coarsening: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, tau_haz=tau_haz, **kwargs)
        self.name = "Aegis-v2"
        self.fp_conf_threshold = fp_conf_threshold
        self.min_navigable_gap = min_navigable_gap
        self.enable_coarsening = enable_coarsening
        self._seen_counts: Dict[str, int] = {}

    def reset(self) -> None:
        super().reset()
        self._seen_counts.clear()

    def select_entities(
        self,
        graph_obs: nx.DiGraph,
        agent: AgentState,
        k: int,
    ) -> List[CompressedEntity]:
        """Return <= k CompressedEntity objects (individual nodes and/or coarsened Super-Nodes)."""
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        if not obj_ids:
            self._prev_selected.clear()
            return []

        # Update temporal observation persistence (used to filter 1-frame FP ghosts)
        obs_set = set(obj_ids)
        for nid in obs_set:
            self._seen_counts[nid] = self._seen_counts.get(nid, 0) + 1
        for stale in list(self._seen_counts.keys()):
            if stale not in obs_set:
                del self._seen_counts[stale]

        scored: List[Tuple[str, float]] = []
        flagged_hazards: Set[str] = set()

        for nid in obj_ids:
            data = graph_obs.nodes[nid]
            q_v = float(data.get("confidence", 1.0))
            u_v = 1.0 - q_v
            c_v = float(data.get("structural_importance", 0.0))
            r_v = self._compute_collision_risk(data, agent)

            # FP-Aware Asymmetric Uncertainty Weighting:
            # 1-frame transient low-confidence detections get damped gamma so FP ghosts don't steal slots;
            # persistent occluded real hazards (seen >= 2 frames) receive full asymmetric safety boost.
            is_transient_fp_suspect = (self._seen_counts.get(nid, 1) == 1) and (q_v < self.fp_conf_threshold)
            gamma_eff = (0.15 * self.gamma) if is_transient_fp_suspect else self.gamma

            haz_indicator = 1.0 if r_v > self.tau_haz else 0.0
            if haz_indicator > 0.0 and not is_transient_fp_suspect:
                flagged_hazards.add(nid)

            s_v = self.alpha * r_v * (1.0 + gamma_eff * u_v * haz_indicator) + self.beta * q_v * c_v

            if nid in self._prev_selected and s_v >= self.theta_out:
                persisted = self._persistence.get(nid, 1)
                s_v += self.hysteresis_bonus * (1.5 if persisted < self.min_persistence_frames else 1.0)

            scored.append((nid, s_v))

        scored.sort(key=lambda item: item[1], reverse=True)

        if self.enable_coarsening and len(flagged_hazards) > k:
            entities = coarsen_hazards_preserving_free_space(
                graph_obs=graph_obs,
                scored_ids=scored,
                hazard_ids=flagged_hazards,
                k=k,
                min_navigable_gap=self.min_navigable_gap,
            )
        else:
            entities = []
            for nid, sc in scored[:k]:
                d = graph_obs.nodes[nid]
                entities.append(
                    CompressedEntity(
                        entity_id=nid,
                        label=str(d.get("label", "object")),
                        pos=d["pos"].copy(),
                        size=d["size"].copy(),
                        vel=d["vel"].copy(),
                        score=sc,
                        is_super_node=False,
                        member_ids={nid},
                    )
                )

        covered_ids = set().union(*(e.member_ids for e in entities)) if entities else set()
        self._persistence = {nid: self._persistence.get(nid, 0) + 1 for nid in covered_ids}
        self._prev_selected = covered_ids
        return entities

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        """Return all ground-truth object IDs covered by the <= k output entities."""
        entities = self.select_entities(graph_obs, agent, k)
        return set().union(*(e.member_ids for e in entities)) if entities else set()

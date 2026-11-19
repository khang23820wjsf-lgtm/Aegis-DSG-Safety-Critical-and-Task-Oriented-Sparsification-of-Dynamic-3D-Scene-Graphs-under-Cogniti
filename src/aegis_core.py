"""Aegis-DSG Unified Core Architecture (v1.0).

A single, cohesive 4-layer algorithm for Safety-Critical and Task-Oriented
Sparsification of Dynamic 3D Scene Graphs under Cognitive Bandwidth Constraints:

  Layer 1 — Temporal Belief Filter & Object Permanence:
      Maintains per-entity existence belief p_t(v) in [0, 1] via EMA/Bayes update,
      suppressing transient 1-frame false-positive ghosts while preserving briefly
      occluded real hazards via constant-velocity rollout.

  Layer 2 — Unified Continuous 3D Kinematic Risk Field + Asymmetric Cost:
      Combines immediate personal-space clearance (d_0) and Dynamic Collision Cone
      TTC clearance (d_TTC) in a single smooth 3D risk field R_v, weighted by
      belief-gated Asymmetric Safety Cost (penalizing uncertainty inside hazard zones).

  Layer 3 — Free-Space-Preserving Graph Coarsening:
      Merges adjacent same-class static clusters and (on budget overflow) closest
      impassable hazard pairs into Super-Nodes ONLY when horizontal clearance
      gap < min_navigable_gap_m (0.70m), strictly preserving navigable passages
      (inspired by Hydra Layer-3 Places).

  Layer 4 — Dynamic-Budget Monotone Submodular Selection + Hysteresis:
      Greedily maximizes a monotone submodular facility-location + temporal-persistence
      objective (Nemhauser 1 - 1/e optimality guarantee), stopping early when
      marginal gain < tau_min so |V'_t| <= k never wastes bandwidth on clutter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import math
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState


@dataclass(frozen=True)
class AegisConfig:
    """Single source of truth for all Aegis-DSG v1.0 physical & optimization parameters."""

    # Layer 1: Temporal Belief Filter & Object Permanence
    belief_init_scale: float = 0.75
    belief_ema_alpha: float = 0.65
    belief_occluded_decay: float = 0.65
    belief_min_keep: float = 0.28
    fp_belief_threshold: float = 0.45
    max_occluded_frames: int = 2

    # Layer 2: Continuous 3D Kinematic Risk Field & Asymmetric Cost
    horizon_s: float = 3.2
    cone_half_angle_deg: float = 38.0
    sigma_near_m: float = 0.50
    sigma_cone_m: float = 0.65
    alpha: float = 1.0
    beta: float = 0.16
    gamma: float = 1.10
    tau_haz: float = 0.28

    # Layer 3: Free-Space-Preserving Coarsening
    min_navigable_gap_m: float = 0.70
    proactive_same_class_merge: bool = True

    # Layer 4: Dynamic-Budget Monotone Submodular Selection & Hysteresis
    tau_min: float = 0.12
    hysteresis_bonus: float = 0.14
    theta_out: float = 0.12
    min_persistence_frames: int = 2
    kernel_sigma_m: float = 0.50


@dataclass
class AegisEntity:
    """Unified compressed entity output by AegisDSG (single node or coarsened Super-Node)."""

    entity_id: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    risk: float
    weight: float
    belief: float
    marginal_gain: float = 0.0
    is_super_node: bool = False
    is_occluded_memory: bool = False
    is_dynamic: bool = False
    member_ids: Set[str] = field(default_factory=set)

    @property
    def score(self) -> float:
        """Alias for backward compatibility with visualization/eval helpers."""
        return self.weight


@dataclass
class _TrackState:
    """Internal temporal belief state for Layer 1 Object Permanence."""

    obj_id: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    confidence: float
    structural_importance: float
    hazard_severity: float
    is_dynamic: bool
    belief: float
    seen_frames: int = 1
    missed_frames: int = 0
    last_risk: float = 0.0


def _xy_edge_gap(pos_a: np.ndarray, size_a: np.ndarray, pos_b: np.ndarray, size_b: np.ndarray) -> float:
    """Minimum horizontal surface-to-surface clearance (meters) between two 3D bounding boxes."""
    dx = max(abs(float(pos_a[0] - pos_b[0])) - 0.5 * float(size_a[0] + size_b[0]), 0.0)
    dy = max(abs(float(pos_a[1] - pos_b[1])) - 0.5 * float(size_a[1] + size_b[1]), 0.0)
    return math.hypot(dx, dy)


class AegisDSG:
    """Unified 4-Layer Aegis-DSG Sparsifier (v1.0)."""

    def __init__(self, config: AegisConfig | None = None) -> None:
        self.config = config or AegisConfig()
        self.name = "Aegis-Core"
        self._cone_half_rad = math.radians(self.config.cone_half_angle_deg)
        self._tracks: Dict[str, _TrackState] = {}
        self._prev_covered: Set[str] = set()
        self._persistence: Dict[str, int] = {}

    def reset(self) -> None:
        """Clear all temporal belief, permanence, and hysteresis memory."""
        self._tracks.clear()
        self._prev_covered.clear()
        self._persistence.clear()

    # ------------------------------------------------------------------
    # Layer 1: Temporal Belief Filter & Object Permanence
    # ------------------------------------------------------------------
    def _update_belief_state(self, graph_obs: nx.DiGraph, dt: float) -> List[_TrackState]:
        """Update existence belief p_t(v) and maintain occluded real tracks."""
        cfg = self.config
        observed_ids: Set[str] = set()

        for nid, d in graph_obs.nodes(data=True):
            if d.get("layer") != "object":
                continue
            observed_ids.add(nid)
            conf = float(np.clip(d.get("confidence", 1.0), 0.05, 1.0))
            pos = np.array(d["pos"], dtype=float)
            size = np.array(d["size"], dtype=float)
            vel = np.array(d.get("vel", np.zeros(3)), dtype=float)

            if nid in self._tracks:
                tr = self._tracks[nid]
                tr.pos = pos
                tr.size = size
                tr.vel = vel
                tr.confidence = conf
                tr.belief = float(np.clip((1.0 - cfg.belief_ema_alpha) * tr.belief + cfg.belief_ema_alpha * conf, 0.05, 0.99))
                tr.seen_frames += 1
                tr.missed_frames = 0
            else:
                # High-confidence real objects (e.g. conf=0.85 -> 0.49 >= 0.45) pass immediately;
                # low-confidence 1-frame FP ghosts (e.g. conf=0.35 -> 0.20 < 0.45) are damped.
                init_belief = float(np.clip(cfg.belief_init_scale * conf, 0.05, 0.95))
                self._tracks[nid] = _TrackState(
                    obj_id=nid,
                    label=str(d.get("label", "object")),
                    pos=pos,
                    size=size,
                    vel=vel,
                    confidence=conf,
                    structural_importance=float(d.get("structural_importance", 0.0)),
                    hazard_severity=float(d.get("hazard_severity", 0.7)),
                    is_dynamic=bool(d.get("is_dynamic", False)),
                    belief=init_belief,
                    seen_frames=1,
                    missed_frames=0,
                )

        # Object Permanence: propagate ONLY briefly occluded active hazards in front of the agent
        to_remove: List[str] = []
        for nid, tr in self._tracks.items():
            if nid not in observed_ids:
                tr.missed_frames += 1
                tr.belief *= cfg.belief_occluded_decay
                tr.pos = tr.pos + tr.vel * dt
                if (
                    tr.seen_frames < 2
                    or tr.last_risk <= cfg.tau_haz
                    or tr.missed_frames > cfg.max_occluded_frames
                    or tr.belief < cfg.belief_min_keep
                ):
                    to_remove.append(nid)

        for nid in to_remove:
            del self._tracks[nid]

        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Layer 2: Unified Continuous 3D Kinematic Risk Field + Asymmetric Cost
    # ------------------------------------------------------------------
    def _compute_unified_risk(self, tr: _TrackState, agent: AgentState) -> float:
        """Smoothly unify near-body safety bubble (d_0) and forward collision cone (d_TTC)."""
        cfg = self.config

        # 3D vertical relevance filter (ignore high overhead objects & flat floor mats)
        z_min = float(tr.pos[2] - tr.size[2] / 2.0)
        z_max = float(tr.pos[2] + tr.size[2] / 2.0)
        if z_min > agent.height + 0.15 or z_max < 0.05:
            return 0.0
        if tr.hazard_severity <= 0.0:
            return 0.0

        rel_pos = tr.pos[:2] - agent.pos[:2]
        dist = float(np.linalg.norm(rel_pos))
        if dist > agent.max_range:
            return 0.0

        half_ext = tr.size[:2] / 2.0 + agent.radius + 0.20

        agent_speed = float(np.linalg.norm(agent.vel[:2]))
        if agent_speed > 0.1 and dist > 1e-4:
            heading = agent.vel[:2] / agent_speed
            cos_ang = float(np.clip(np.dot(rel_pos / dist, heading), -1.0, 1.0))
            angle = math.acos(cos_ang)
        else:
            cos_ang = 1.0
            angle = 0.0

        # Component A: Smooth near-body clearance risk across front + side hemisphere (<= 100°)
        # Protects shoulders during sharp steering without wasting slots on objects already behind user.
        d_now = float(np.linalg.norm(np.maximum(np.abs(rel_pos) - half_ext, 0.0)))
        hemi_gate = 1.0 if cos_ang >= -0.18 else math.exp(-((cos_ang + 0.18) / 0.25) ** 2)
        r_near = math.exp(-(d_now ** 2) / (2.0 * cfg.sigma_near_m ** 2)) * hemi_gate

        # Component B: Smooth forward Dynamic Collision Cone & TTC risk
        rel_vel = tr.vel[:2] - agent.vel[:2]
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq < 1e-6:
            ttc = 0.0
        else:
            ttc = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, cfg.horizon_s))

        d_ttc = float(np.linalg.norm(np.maximum(np.abs(rel_pos + rel_vel * ttc) - half_ext, 0.0)))

        if angle <= self._cone_half_rad:
            cone_factor = 1.0
        else:
            cone_factor = math.exp(-((angle - self._cone_half_rad) / 0.35) ** 2)

        closing_speed = -float(np.dot(rel_pos / (dist + 1e-6), rel_vel))
        if closing_speed > 0.3:
            cone_factor = max(cone_factor, 0.78)

        spatial_cone = math.exp(-(d_ttc ** 2) / (2.0 * cfg.sigma_cone_m ** 2))
        time_urgency = 1.0 - 0.5 * (ttc / cfg.horizon_s)
        r_cone = spatial_cone * time_urgency * cone_factor

        # Smooth unified max field weighted by physical hazard severity
        r_v = max(r_near, r_cone) * max(tr.hazard_severity, 0.2)
        return float(np.clip(r_v, 0.0, 1.0))

    def _compute_asymmetric_weight(self, tr: _TrackState, r_v: float) -> float:
        """Belief-gated Asymmetric Safety Weight w_v >= 0."""
        cfg = self.config
        q_v = tr.confidence
        u_v = 1.0 - q_v
        c_v = tr.structural_importance

        # Gate uncertainty boost by temporal belief so 1-frame FP ghosts cannot abuse gamma
        is_reliable = tr.belief >= cfg.fp_belief_threshold
        gamma_eff = cfg.gamma if is_reliable else (0.15 * cfg.gamma)
        belief_gate = 1.0 if (is_reliable and tr.missed_frames == 0) else min(1.0, tr.belief)

        haz_ind = 1.0 if r_v > cfg.tau_haz else 0.0
        # Safety dominance: confirmed active hazards receive a priority offset (+0.5)
        # so non-hazard clutter/landmarks can never outrank a real collision hazard.
        safety_tier = 0.50 if (haz_ind > 0.0 and is_reliable and tr.missed_frames == 0) else 0.0
        w_v = belief_gate * (
            safety_tier + cfg.alpha * r_v * (1.0 + gamma_eff * u_v * haz_ind) + cfg.beta * q_v * c_v
        )
        return max(0.0, float(w_v))

    # ------------------------------------------------------------------
    # Layer 3: Free-Space-Preserving Graph Coarsening
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_two(a: AegisEntity, b: AegisEntity, merge_id: str, new_label: str) -> AegisEntity:
        a_min, a_max = a.pos - 0.5 * a.size, a.pos + 0.5 * a.size
        b_min, b_max = b.pos - 0.5 * b.size, b.pos + 0.5 * b.size
        box_min = np.minimum(a_min, b_min)
        box_max = np.maximum(a_max, b_max)
        vel = a.vel.copy() if float(np.linalg.norm(a.vel)) >= float(np.linalg.norm(b.vel)) else b.vel.copy()
        members = set(a.member_ids) | set(b.member_ids)

        return AegisEntity(
            entity_id=merge_id,
            label=new_label,
            pos=0.5 * (box_min + box_max),
            size=box_max - box_min,
            vel=vel,
            risk=max(a.risk, b.risk),
            weight=max(a.weight, b.weight),
            belief=max(a.belief, b.belief),
            is_super_node=True,
            is_occluded_memory=(a.is_occluded_memory and b.is_occluded_memory),
            is_dynamic=(a.is_dynamic or b.is_dynamic),
            member_ids=members,
        )

    def _coarsen_entities(self, entities: List[AegisEntity], k: int) -> List[AegisEntity]:
        """Merge impassable clusters (< min_navigable_gap_m) without blocking narrow passages."""
        cfg = self.config
        out = list(entities)
        merge_counter = 0

        # Stage 3A: Proactive same-class static clustering (e.g., adjacent dining chairs -> chair_group)
        if cfg.proactive_same_class_merge:
            changed = True
            while changed:
                changed = False
                for i in range(len(out)):
                    for j in range(i + 1, len(out)):
                        a, b = out[i], out[j]
                        if a.is_dynamic or b.is_dynamic:
                            continue
                        base_a = a.label.split("_group")[0]
                        base_b = b.label.split("_group")[0]
                        if base_a == base_b and _xy_edge_gap(a.pos, a.size, b.pos, b.size) < cfg.min_navigable_gap_m:
                            merged = self._merge_two(
                                a, b,
                                merge_id=f"super_{base_a}_{merge_counter}",
                                new_label=f"{base_a}_group(x{len(a.member_ids | b.member_ids)})",
                            )
                            merge_counter += 1
                            out = [e for idx, e in enumerate(out) if idx not in (i, j)]
                            out.append(merged)
                            changed = True
                            break
                    if changed:
                        break

        # Stage 3B: Overflow cross-class hazard coarsening (strictly gap < min_navigable_gap_m)
        haz_list = [e for e in out if e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold]
        non_haz = [e for e in out if not (e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold)]

        while len(haz_list) > k:
            best_pair: Tuple[int, int] | None = None
            best_gap = float("inf")
            for i in range(len(haz_list)):
                for j in range(i + 1, len(haz_list)):
                    gap = _xy_edge_gap(haz_list[i].pos, haz_list[i].size, haz_list[j].pos, haz_list[j].size)
                    if gap < cfg.min_navigable_gap_m and gap < best_gap:
                        best_gap = gap
                        best_pair = (i, j)
            if best_pair is None:
                break  # Never merge across a navigable free-space passage!
            i, j = best_pair
            members_n = len(haz_list[i].member_ids | haz_list[j].member_ids)
            merged = self._merge_two(
                haz_list[i], haz_list[j],
                merge_id=f"super_nogo_{merge_counter}",
                new_label=f"no_go_cluster(x{members_n})",
            )
            merge_counter += 1
            haz_list = [e for idx, e in enumerate(haz_list) if idx not in (i, j)]
            haz_list.append(merged)

        return haz_list + non_haz

    # ------------------------------------------------------------------
    # Layer 4: Dynamic-Budget Monotone Submodular Selection + Hysteresis
    # ------------------------------------------------------------------
    def _select_submodular_dynamic(self, entities: List[AegisEntity], k: int) -> List[AegisEntity]:
        """Greedy (1 - 1/e) Submodular Selection stopping early below tau_min."""
        cfg = self.config
        n = len(entities)
        if n == 0 or k <= 0:
            return []

        weights = np.array([e.weight for e in entities], dtype=float)

        # Build redundancy kernel K[u, v]:
        # Distinct active hazards stay non-redundant (K=0); non-hazard landmarks within
        # proximity share coverage (discounted by 0.4 so clutter never outranks hazards).
        K = np.eye(n, dtype=float)
        for i in range(n):
            if entities[i].risk > cfg.tau_haz:
                continue
            for j in range(i + 1, n):
                if entities[j].risk > cfg.tau_haz:
                    continue
                gap = _xy_edge_gap(entities[i].pos, entities[i].size, entities[j].pos, entities[j].size)
                if gap < cfg.min_navigable_gap_m:
                    val = 0.4 * math.exp(-(gap ** 2) / (2.0 * cfg.kernel_sigma_m ** 2))
                    K[i, j] = val
                    K[j, i] = val

        current_cov = np.zeros(n, dtype=float)
        selected: List[AegisEntity] = []
        remaining = set(range(n))

        for _ in range(min(k, n)):
            best_idx = -1
            best_gain = -1.0

            for v in remaining:
                ent = entities[v]
                inc = np.maximum(K[:, v] - current_cov, 0.0)
                gain = float(np.dot(weights, inc))

                # Modular temporal hysteresis bonus (preserves submodularity!)
                if (ent.member_ids & self._prev_covered) and ent.weight >= cfg.theta_out:
                    max_pers = max((self._persistence.get(m, 1) for m in ent.member_ids), default=1)
                    gain += cfg.hysteresis_bonus * (1.5 if max_pers < cfg.min_persistence_frames else 1.0)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = v

            if best_idx < 0:
                break

            chosen = entities[best_idx]
            # Dynamic budget floor: stop early if marginal gain < tau_min and not an active safety hazard
            if best_gain < cfg.tau_min and chosen.risk <= cfg.tau_haz:
                break

            chosen.marginal_gain = round(best_gain, 4)
            selected.append(chosen)
            remaining.remove(best_idx)
            current_cov = np.maximum(current_cov, K[:, best_idx])

        return selected

    # ------------------------------------------------------------------
    # Public Unified API
    # ------------------------------------------------------------------
    def compress(
        self,
        graph_obs: nx.DiGraph,
        agent: AgentState,
        k: int,
        dt: float = 0.1,
    ) -> List[AegisEntity]:
        """Run all 4 Aegis-DSG layers sequentially and return <= k compressed entities."""
        if k <= 0:
            self._prev_covered.clear()
            return []

        # Layer 1: Temporal Belief Filter & Object Permanence
        tracks = self._update_belief_state(graph_obs, dt)
        if not tracks:
            self._prev_covered.clear()
            return []

        # Layer 2: Unified 3D Kinematic Risk Field + Asymmetric Safety Cost
        candidates: List[AegisEntity] = []
        for tr in tracks:
            r_v = self._compute_unified_risk(tr, agent)
            tr.last_risk = r_v
            # Drop occluded memory tracks once they pass behind the agent (risk drops below tau_haz)
            if tr.missed_frames > 0 and r_v <= self.config.tau_haz:
                continue
            w_v = self._compute_asymmetric_weight(tr, r_v)
            candidates.append(
                AegisEntity(
                    entity_id=tr.obj_id,
                    label=tr.label,
                    pos=tr.pos.copy(),
                    size=tr.size.copy(),
                    vel=tr.vel.copy(),
                    risk=r_v,
                    weight=w_v,
                    belief=tr.belief,
                    is_super_node=False,
                    is_occluded_memory=(tr.missed_frames > 0),
                    is_dynamic=tr.is_dynamic,
                    member_ids={tr.obj_id},
                )
            )

        # Layer 3: Free-Space-Preserving Graph Coarsening
        coarsened = self._coarsen_entities(candidates, k)

        # Layer 4: Dynamic-Budget Monotone Submodular Selection + Hysteresis
        selected = self._select_submodular_dynamic(coarsened, k)

        # Update persistence memory for next timestep
        covered = set().union(*(e.member_ids for e in selected)) if selected else set()
        self._persistence = {nid: self._persistence.get(nid, 0) + 1 for nid in covered}
        self._prev_covered = covered
        return selected

    def select_entities(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> List[AegisEntity]:
        """Alias matching Phase B closed-loop MuJoCo interface."""
        return self.compress(graph_obs, agent, k)

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        """Return set of covered object IDs (compatible with Phase A/B metrics)."""
        entities = self.compress(graph_obs, agent, k)
        return set().union(*(e.member_ids for e in entities)) if entities else set()

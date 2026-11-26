"""Aegis-V4 Unified Core Architecture: Counterfactual Control-Barrier Sparsification (CCBS) + Online ACI.

Unifies 4 cohesive layers into a single real-time (<2 ms/frame) DSG sparsifier and
closed-loop CBF-QP safety controller:

  Layer 1 — Temporal Belief Filter & Object Permanence:
      Per-entity existence belief p_t(v) via EMA update, suppressing 1-frame ghosts
      while propagating briefly occluded active hazards via constant-velocity rollout.

  Layer 2 — CBF-QP Counterfactual Safety Regret + Epistemic 3D Eye-Gaze + Affordance Risk:
      Solves the discrete-time CBF-QP safety filter to compute exact counterfactual
      control distortion `delta_cf(v) = ||u*(G) - u*(G \\ {v})||_2` and barrier margin
      urgency, modulated by `EpistemicGazeModulator` and `AffordanceRiskMapper`.

  Layer 3 — Free-Space-Preserving OBB Coarsening + Non-Coarsenable Invariants:
      Merges adjacent same-class static geometric clusters and overflow geometric
      hazards into Rotating-Calipers Minimum-Area OBB Super-Nodes ONLY when clearance
      < 0.70 m, while STRICTLY protecting Atomic Hazards (`is_atomic_hazard=True`) and
      strictly CBF-active barriers (`delta_cf > 0`) from distortion or semantic erasure.

  Layer 4 — Online ACI-Calibrated Monotone Submodular Selection + Hysteresis:
      Greedily maximizes a monotone submodular objective with temporal hysteresis,
      dynamically calibrated online by `OnlineACISafetyCalibrator` (Gibbs & Candès,
      NeurIPS 2021) with zero exchangeability or i.i.d. assumptions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np

from src.aegis_core import (
    AegisEntity,
    AffordanceRiskMapper,
    _DEFAULT_AFFORDANCE_MAPPER,
    _xy_edge_gap,
)
from src.aegis_v3_modules import (
    ZeroShotCLIPCache,
    compute_min_area_obb_2d,
    entity_corners_2d,
)
from src.aegis_v4.adaptive_conformal import (
    ACIStepResult,
    EpistemicGazeModulator,
    OnlineACISafetyCalibrator,
)
from src.aegis_v4.cbf_sparsifier import (
    CBFSparsificationResult,
    build_cbf_constraints,
    compute_counterfactual_regrets,
    solve_cbf_qp,
)
from src.dsg_generator import AgentState


def _fast_convex_hull_2d(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Fast pure-Python Andrew's monotone chain 2D convex hull (zero np.unique overhead)."""
    pts_sorted = sorted(set(pts))
    n = len(pts_sorted)
    if n <= 2:
        return pts_sorted

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _fast_entity_corners_2d(pos_xy: np.ndarray, size_xy: np.ndarray, yaw: float) -> List[Tuple[float, float]]:
    """Return 4 corner (x, y) tuples of a 2D box without NumPy allocation overhead."""
    cx, cy = float(pos_xy[0]), float(pos_xy[1])
    hx, hy = 0.5 * float(size_xy[0]), 0.5 * float(size_xy[1])
    if abs(yaw) <= 1e-6:
        return [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
        ]
    c = math.cos(yaw)
    s = math.sin(yaw)
    chx, shx = c * hx, s * hx
    chy, shy = c * hy, s * hy
    return [
        (cx - chx + shy, cy - shx - chy),
        (cx + chx + shy, cy + shx - chy),
        (cx + chx - shy, cy + shx + chy),
        (cx - chx - shy, cy - shx + chy),
    ]


def _fast_min_area_obb_2d(pts: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fast rotating-calipers minimum-area 2D OBB from list of (x, y) tuples."""
    hull = _fast_convex_hull_2d(pts)
    m = len(hull)
    if m == 0:
        return np.zeros(2, dtype=float), np.array([0.05, 0.05], dtype=float), 0.0, 0.0025
    if m <= 2:
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(max_x - min_x, 0.05)
        h = max(max_y - min_y, 0.05)
        return np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y)], dtype=float), np.array([w, h], dtype=float), 0.0, w * h

    best_area = float("inf")
    best_yaw = 0.0
    best_w = 0.05
    best_h = 0.05
    best_cx = 0.0
    best_cy = 0.0

    for i in range(m):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % m]
        ex = x1 - x0
        ey = y1 - y0
        elen = math.hypot(ex, ey)
        if elen < 1e-9:
            continue
        c = ex / elen
        s = ey / elen
        min_u = float("inf")
        max_u = float("-inf")
        min_v = float("inf")
        max_v = float("-inf")
        for px, py in hull:
            u = c * px + s * py
            v = -s * px + c * py
            if u < min_u:
                min_u = u
            if u > max_u:
                max_u = u
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v
        w = max(max_u - min_u, 0.05)
        h = max(max_v - min_v, 0.05)
        area = w * h
        if area < best_area:
            best_area = area
            best_w = w
            best_h = h
            best_yaw = math.atan2(s, c)
            cu = 0.5 * (min_u + max_u)
            cv = 0.5 * (min_v + max_v)
            best_cx = c * cu - s * cv
            best_cy = s * cu + c * cv

    return np.array([best_cx, best_cy], dtype=float), np.array([best_w, best_h], dtype=float), best_yaw, best_area


@dataclass(frozen=True)
class AegisV4Config:
    """Configuration for Aegis-V4 (Counterfactual Control-Barrier Sparsifier + Online ACI)."""

    # Layer 1: Temporal Belief Filter & Object Permanence
    belief_init_scale: float = 0.75
    belief_ema_alpha: float = 0.65
    belief_occluded_decay: float = 0.65
    belief_min_keep: float = 0.28
    fp_belief_threshold: float = 0.45
    max_occluded_frames: int = 2

    # Layer 2: CBF-QP Counterfactual Safety Regret + Kinematic Risk + Epistemic Gaze
    cbf_dt: float = 0.10
    gamma_cbf: float = 0.10
    cbf_safety_margin: float = 0.15
    u_max: float = 2.0
    cbf_slack_penalty: float = 100.0
    cbf_regret_weight: float = 1.80
    cbf_active_tier: float = 1.25
    margin_urgency_weight: float = 0.45
    horizon_s: float = 3.2
    cone_half_angle_deg: float = 38.0
    sigma_near_m: float = 0.50
    sigma_cone_m: float = 0.65
    atomic_reach_m: float = 2.20
    alpha: float = 1.0
    beta: float = 0.16
    gamma: float = 1.10
    tau_haz: float = 0.25
    enable_epistemic_gaze: bool = True
    lambda_epistemic: float = 0.60
    gaze_fovea_deg: float = 15.0
    gaze_ema_decay: float = 0.65
    clip_task_weight: float = 0.35

    # Layer 3: OBB Free-Space-Preserving Coarsening + Non-Coarsenable Invariants
    min_navigable_gap_m: float = 0.70
    max_aabb_bloat_ratio: float = 1.85
    enable_obb_supernodes: bool = True
    proactive_same_class_merge: bool = True
    enable_hierarchical_zoom: bool = True
    hierarchical_zoom_dist_m: float = 3.2
    protect_cbf_active_from_merge: bool = True
    enforce_kkt_support: bool = True

    # Layer 4: Online ACI + Monotone Submodular Selection + Hysteresis
    enable_online_aci: bool = True
    aci_alpha_target: float = 0.05
    aci_eta: float = 0.02
    aci_window_size: int = 200
    aci_tau_floor: float = 0.015
    tau_min: float = 0.12
    hysteresis_bonus: float = 0.14
    theta_out: float = 0.12
    min_persistence_frames: int = 2
    kernel_sigma_m: float = 0.50


@dataclass
class AegisV4Entity(AegisEntity):
    """Compressed entity output by Aegis-V4, extending AegisEntity with CCBS metrics."""

    delta_cf: float = 0.0
    margin_urgency: float = 0.0
    epistemic_priority: float = 0.0
    dual_multiplier: float = 0.0
    is_kkt_support: bool = False


@dataclass
class _V4TrackState:
    """Internal Layer 1 temporal belief & permanence state."""

    obj_id: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    confidence: float
    structural_importance: float
    hazard_severity: float
    is_dynamic: bool
    is_atomic_hazard: bool
    belief: float
    seen_frames: int = 1
    missed_frames: int = 0
    last_risk: float = 0.0
    gaze_attention: float = 0.0
    parent_place_id: Optional[str] = None


class AegisV4CCBS:
    """Aegis-V4: Counterfactual Control-Barrier Sparsifier with Online ACI.

    Exposes:
      - `compress(graph_obs, agent, k, dt=0.1, u_nom=None) -> List[AegisV4Entity]`
      - `select(graph_obs, agent, k) -> Set[str]`
      - `select_entities(graph_obs, agent, k) -> List[AegisV4Entity]`
      - `compute_safe_velocity(entities, agent, u_nom, dt=None, u_max=None, return_info=False)`
      - `update_aci_feedback(min_retained_hazard_score=None, err_t=None) -> ACIStepResult`
      - `reset() -> None`
    """

    def __init__(
        self,
        config: Optional[AegisV4Config] = None,
        risk_mapper: Optional[AffordanceRiskMapper] = None,
        clip_cache: Optional[ZeroShotCLIPCache] = None,
    ) -> None:
        self.config = config or AegisV4Config()
        self.clip_cache = clip_cache
        if risk_mapper is not None:
            self.risk_mapper = risk_mapper
        elif clip_cache is not None and clip_cache.is_real_clip_loaded:
            self.risk_mapper = AffordanceRiskMapper(embedding_fn=clip_cache.encode_text)
        else:
            self.risk_mapper = _DEFAULT_AFFORDANCE_MAPPER

        self.name = "Aegis-V4 (CCBS+ACI)"
        self._cone_half_rad = math.radians(self.config.cone_half_angle_deg)

        self.aci = OnlineACISafetyCalibrator(
            alpha_target=self.config.aci_alpha_target,
            eta=self.config.aci_eta,
            window_size=self.config.aci_window_size,
            default_tau=self.config.tau_min,
        )
        self.gaze_modulator = EpistemicGazeModulator(
            lambda_epistemic=self.config.lambda_epistemic if self.config.enable_epistemic_gaze else 0.0,
            fovea_half_angle_deg=self.config.gaze_fovea_deg,
            ema_decay=self.config.gaze_ema_decay,
        )

        self._tracks: Dict[str, _V4TrackState] = {}
        self._prev_covered: Set[str] = set()
        self._persistence: Dict[str, int] = {}
        self.last_cbf_result: Optional[CBFSparsificationResult] = None
        self.last_aci_step: Optional[ACIStepResult] = None

    def reset(self) -> None:
        """Clear all temporal belief, gaze EMA, ACI, and hysteresis state."""
        self._tracks.clear()
        self._prev_covered.clear()
        self._persistence.clear()
        self.aci.reset()
        self.gaze_modulator.reset()
        self.last_cbf_result = None
        self.last_aci_step = None

    def set_task_prompt(self, task_prompt: str, vocabulary: Optional[Iterable[str]] = None) -> None:
        """Bind an open-vocabulary CLIP task prompt (encoded once offline)."""
        if self.clip_cache is None:
            self.clip_cache = ZeroShotCLIPCache()
            self.clip_cache.load()
        self.clip_cache.set_task_prompt(task_prompt, vocabulary=vocabulary)

    # ------------------------------------------------------------------
    # Layer 1: Temporal Belief Filter & Object Permanence
    # ------------------------------------------------------------------
    def _update_belief_state(self, graph_obs: nx.DiGraph, dt: float) -> List[_V4TrackState]:
        """Update existence belief p_t(v) and maintain briefly occluded active hazards."""
        cfg = self.config
        observed_ids: Set[str] = set()

        for nid, d in graph_obs.nodes(data=True):
            if d.get("layer") != "object":
                continue
            observed_ids.add(nid)
            conf_raw = float(d.get("confidence", 1.0))
            conf = 0.05 if conf_raw < 0.05 else (1.0 if conf_raw > 1.0 else conf_raw)
            pos = np.asarray(d["pos"], dtype=float)
            size = np.asarray(d["size"], dtype=float)
            vel_raw = d.get("vel")
            vel = np.zeros(3, dtype=float) if vel_raw is None else np.asarray(vel_raw, dtype=float)
            lbl = str(d.get("label", "object"))
            intrinsic_sev, is_atomic, _ = self.risk_mapper.assess(
                lbl,
                explicit_flag=d.get("is_atomic_hazard"),
                affordances=d.get("affordances"),
            )

            parent_place: Optional[str] = None
            for succ in graph_obs.successors(nid):
                slayer = graph_obs.nodes[succ].get("layer")
                if slayer in ("place", "room"):
                    parent_place = succ
                    if slayer == "place":
                        break

            if nid in self._tracks:
                tr = self._tracks[nid]
                tr.pos = pos
                tr.size = size
                tr.vel = vel
                tr.confidence = conf
                tr.is_atomic_hazard = is_atomic
                tr.parent_place_id = parent_place
                b_new = (1.0 - cfg.belief_ema_alpha) * tr.belief + cfg.belief_ema_alpha * conf
                tr.belief = 0.05 if b_new < 0.05 else (0.99 if b_new > 0.99 else b_new)
                tr.seen_frames += 1
                tr.missed_frames = 0
            else:
                b_init = cfg.belief_init_scale * conf
                init_belief = 0.05 if b_init < 0.05 else (0.95 if b_init > 0.95 else b_init)
                self._tracks[nid] = _V4TrackState(
                    obj_id=nid,
                    label=lbl,
                    pos=pos,
                    size=size,
                    vel=vel,
                    confidence=conf,
                    structural_importance=float(d.get("structural_importance", 0.0)),
                    hazard_severity=1.0 if is_atomic else max(float(d.get("hazard_severity", 0.7)), intrinsic_sev),
                    is_dynamic=bool(d.get("is_dynamic", False)),
                    is_atomic_hazard=is_atomic,
                    belief=init_belief,
                    seen_frames=1,
                    missed_frames=0,
                    parent_place_id=parent_place,
                )

        # Object Permanence: propagate briefly occluded active hazards
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

        self.gaze_modulator.prune_tracks(self._tracks.keys())
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Layer 2: 3D Kinematic Risk + CBF-QP Counterfactual Regret + Epistemic Gaze
    # ------------------------------------------------------------------
    def _compute_kinematic_risk_and_ttc(self, tr: _V4TrackState, agent: AgentState) -> Tuple[float, float]:
        """Compute smooth 3D kinematic risk r_v in [0, 1] and estimated TTC (s)."""
        cfg = self.config
        pz = float(tr.pos[2])
        hz = 0.5 * float(tr.size[2])
        if not tr.is_atomic_hazard:
            if (pz - hz) > agent.height + 0.15 or (pz + hz) < 0.05:
                return 0.0, float("inf")
        if tr.hazard_severity <= 0.0:
            return 0.0, float("inf")

        rx = float(tr.pos[0]) - float(agent.pos[0])
        ry = float(tr.pos[1]) - float(agent.pos[1])
        dist = math.hypot(rx, ry)
        if dist > agent.max_range:
            return 0.0, float("inf")

        pad = agent.radius + 0.20
        hx = 0.5 * float(tr.size[0]) + pad
        hy = 0.5 * float(tr.size[1]) + pad

        avx = float(agent.vel[0])
        avy = float(agent.vel[1])
        agent_speed = math.hypot(avx, avy)
        if agent_speed > 0.1 and dist > 1e-4:
            cos_ang = (rx * avx + ry * avy) / (dist * agent_speed)
            if cos_ang > 1.0:
                cos_ang = 1.0
            elif cos_ang < -1.0:
                cos_ang = -1.0
            angle = math.acos(cos_ang)
        else:
            cos_ang = 1.0
            angle = 0.0

        # Component A: Near-body clearance
        dx0 = abs(rx) - hx
        dy0 = abs(ry) - hy
        d_now = math.hypot(dx0 if dx0 > 0.0 else 0.0, dy0 if dy0 > 0.0 else 0.0)
        hemi_gate = 1.0 if cos_ang >= -0.18 else math.exp(-((cos_ang + 0.18) / 0.25) ** 2)
        sigma_near = (cfg.sigma_near_m * 1.45) if tr.is_atomic_hazard else cfg.sigma_near_m
        r_near = math.exp(-(d_now * d_now) / (2.0 * sigma_near * sigma_near)) * hemi_gate

        # Component B: Forward Dynamic Collision Cone & TTC
        rvx = float(tr.vel[0]) - avx
        rvy = float(tr.vel[1]) - avy
        rel_speed_sq = rvx * rvx + rvy * rvy
        if rel_speed_sq < 1e-6:
            ttc = 0.0
        else:
            ttc_raw = -(rx * rvx + ry * rvy) / rel_speed_sq
            ttc = 0.0 if ttc_raw < 0.0 else (cfg.horizon_s if ttc_raw > cfg.horizon_s else ttc_raw)
        dx_t = abs(rx + rvx * ttc) - hx
        dy_t = abs(ry + rvy * ttc) - hy
        d_ttc = math.hypot(dx_t if dx_t > 0.0 else 0.0, dy_t if dy_t > 0.0 else 0.0)

        cone_factor = (
            1.0 if angle <= self._cone_half_rad else math.exp(-((angle - self._cone_half_rad) / 0.35) ** 2)
        )
        closing_speed = -(rx * rvx + ry * rvy) / (dist + 1e-6)
        if closing_speed > 0.3 and cone_factor < 0.78:
            cone_factor = 0.78

        spatial_cone = math.exp(-(d_ttc * d_ttc) / (2.0 * cfg.sigma_cone_m * cfg.sigma_cone_m))
        time_urgency = 1.0 - 0.5 * (ttc / cfg.horizon_s)
        r_cone = spatial_cone * time_urgency * cone_factor

        # Component C: Close-range personal-reach risk for intrinsic atomic hazards
        r_atomic = 0.0
        if tr.is_atomic_hazard and dist <= cfg.atomic_reach_m:
            s_at = cfg.atomic_reach_m * 0.55
            r_atomic = math.exp(-(dist * dist) / (2.0 * s_at * s_at))

        r_raw = max(r_near, r_cone, r_atomic) * max(tr.hazard_severity, 0.2)
        r_kin = 0.0 if r_raw < 0.0 else (1.0 if r_raw > 1.0 else r_raw)
        eff_ttc = ttc if (closing_speed > 0.15 and d_ttc < 0.35) else float("inf")
        return r_kin, eff_ttc

    def _compute_ccbs_weight(
        self,
        tr: _V4TrackState,
        r_v: float,
        delta_cf: float,
        is_cbf_binding: bool,
        margin_urgency: float,
        epistemic_prio: float,
    ) -> Tuple[float, float]:
        """Compute belief-gated CCBS + Epistemic Gaze + Asymmetric Safety weight w_v >= 0."""
        cfg = self.config
        q_v = tr.confidence
        u_v = 1.0 - q_v
        c_struct = tr.structural_importance

        clip_rel = 0.0
        if self.clip_cache is not None and self.clip_cache._task_vec is not None:
            clip_rel = self.clip_cache.task_relevance(tr.label)
            c_v = max(c_struct, clip_rel)
            beta_eff = max(cfg.beta, cfg.clip_task_weight)
        else:
            c_v = c_struct
            beta_eff = cfg.beta

        is_reliable = tr.belief >= cfg.fp_belief_threshold
        gamma_eff = cfg.gamma if is_reliable else (0.15 * cfg.gamma)
        belief_gate = 1.0 if (is_reliable and tr.missed_frames == 0) else min(1.0, tr.belief)

        is_cbf_active = (delta_cf > 1e-4) or is_cbf_binding
        haz_ind = 1.0 if (r_v > cfg.tau_haz or is_cbf_active) else 0.0

        safety_tier = 0.0
        if is_cbf_active and tr.missed_frames == 0:
            # Active CBF barrier directly alters safe control u* -> highest safety tier
            safety_tier = cfg.cbf_active_tier + (0.35 if tr.is_atomic_hazard else 0.0)
        elif haz_ind > 0.0 and is_reliable and tr.missed_frames == 0:
            safety_tier = 0.90 if tr.is_atomic_hazard else 0.50

        # Epistemic Counterfactual Priority combines (delta_cf + r_v) * (1 + lambda * (1 - A_gaze))
        core_score = (
            cfg.alpha * epistemic_prio
            + cfg.cbf_regret_weight * delta_cf
            + cfg.margin_urgency_weight * margin_urgency
        )
        unc_boost = 1.0 + gamma_eff * u_v * haz_ind
        w_v = belief_gate * (safety_tier + core_score * unc_boost + beta_eff * q_v * c_v)
        return max(0.0, float(w_v)), clip_rel

    # ------------------------------------------------------------------
    # Layer 3: OBB Free-Space-Preserving Coarsening + Non-Coarsenable Invariants
    # ------------------------------------------------------------------
    @staticmethod
    def _obb_bloat_ok(group: List[AegisV4Entity], max_ratio: float, use_obb: bool = True) -> bool:
        if len(group) <= 1:
            return True
        sum_area = 0.0
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for e in group:
            sx = float(e.size[0])
            sy = float(e.size[1])
            sum_area += (sx if sx > 0.05 else 0.05) * (sy if sy > 0.05 else 0.05)
            px, py = float(e.pos[0]), float(e.pos[1])
            hx, hy = 0.5 * sx, 0.5 * sy
            if px - hx < min_x:
                min_x = px - hx
            if px + hx > max_x:
                max_x = px + hx
            if py - hy < min_y:
                min_y = py - hy
            if py + hy > max_y:
                max_y = py + hy

        limit = max_ratio * (sum_area if sum_area > 1e-4 else 1e-4)
        aabb_area = max(max_x - min_x, 0.05) * max(max_y - min_y, 0.05)
        # Short-circuit: OBB area <= AABB area always!
        if aabb_area <= limit:
            return True
        if not use_obb:
            return False

        pts: List[Tuple[float, float]] = []
        for e in group:
            sz = e.obb_size if e.obb_size is not None else e.size
            pts.extend(_fast_entity_corners_2d(e.pos, sz, e.obb_yaw))
        _, _, _, obb_area = _fast_min_area_obb_2d(pts)
        return obb_area <= limit

    @staticmethod
    def _merge_group(
        group: List[AegisV4Entity],
        merge_id: str,
        new_label: str,
        use_obb: bool = True,
        dsg_layer: str = "object",
    ) -> AegisV4Entity:
        mins = np.min([e.pos - 0.5 * e.size for e in group], axis=0)
        maxs = np.max([e.pos + 0.5 * e.size for e in group], axis=0)
        aabb_center = 0.5 * (mins + maxs)
        aabb_size = maxs - mins

        obb_yaw = 0.0
        obb_size_3d: Optional[np.ndarray] = None
        if use_obb and len(group) >= 2:
            pts: List[Tuple[float, float]] = []
            for e in group:
                sz = e.obb_size if e.obb_size is not None else e.size
                pts.extend(_fast_entity_corners_2d(e.pos, sz, e.obb_yaw))
            obb_c_xy, obb_wh, obb_yaw, obb_area = _fast_min_area_obb_2d(pts)
            aabb_area = max(float(aabb_size[0]), 0.05) * max(float(aabb_size[1]), 0.05)
            if obb_area < 0.96 * aabb_area:
                pos_out = np.array([obb_c_xy[0], obb_c_xy[1], aabb_center[2]], dtype=float)
                obb_size_3d = np.array([obb_wh[0], obb_wh[1], aabb_size[2]], dtype=float)
                cx0, cy0 = float(obb_c_xy[0]), float(obb_c_xy[1])
                hx_span = max(abs(px - cx0) for px, _ in pts)
                hy_span = max(abs(py - cy0) for _, py in pts)
                aabb_size = np.array([2.0 * hx_span, 2.0 * hy_span, aabb_size[2]], dtype=float)
            else:
                obb_yaw = 0.0
                pos_out = aabb_center
                obb_size_3d = aabb_size.copy()
        else:
            pos_out = aabb_center

        fastest = max(group, key=lambda e: math.hypot(float(e.vel[0]), float(e.vel[1])))
        members: Set[str] = set()
        for e in group:
            members |= e.member_ids

        return AegisV4Entity(
            entity_id=merge_id,
            label=new_label,
            pos=pos_out,
            size=aabb_size,
            vel=fastest.vel.copy(),
            risk=max(e.risk for e in group),
            weight=max(e.weight for e in group),
            belief=max(e.belief for e in group),
            is_super_node=True,
            is_atomic_hazard=False,
            is_occluded_memory=all(e.is_occluded_memory for e in group),
            is_dynamic=any(e.is_dynamic for e in group),
            dsg_layer=dsg_layer,
            obb_yaw=obb_yaw,
            obb_size=obb_size_3d,
            gaze_attention=max(e.gaze_attention for e in group),
            task_relevance=max(e.task_relevance for e in group),
            member_ids=members,
            delta_cf=max(e.delta_cf for e in group),
            margin_urgency=max(e.margin_urgency for e in group),
            epistemic_priority=max(e.epistemic_priority for e in group),
            dual_multiplier=max(e.dual_multiplier for e in group),
            is_kkt_support=False,
        )

    def _coarsen_entities(
        self,
        entities: List[AegisV4Entity],
        k: int,
        graph_obs: Optional[nx.DiGraph] = None,
        agent: Optional[AgentState] = None,
        entity_parent_place: Optional[Dict[str, str]] = None,
    ) -> List[AegisV4Entity]:
        """OBB Union-Find coarsening strictly protecting Atomic Hazards and KKT-support barriers."""
        cfg = self.config
        merge_counter = 0

        # HARD NON-COARSENABLE INVARIANT:
        # 1. Atomic Hazards NEVER merge (zero semantic erasure).
        # 2. Minimal Caratheodory KKT support barriers (|S_KKT| <= 3) stay exact so u*(G') == u*(G),
        #    while all non-support obstacles remain free to merge into tight OBB super-nodes.
        protected: List[AegisV4Entity] = []
        coarsenable: List[AegisV4Entity] = []
        for e in entities:
            if e.is_atomic_hazard or (cfg.protect_cbf_active_from_merge and e.is_kkt_support):
                protected.append(e)
            else:
                coarsenable.append(e)

        # Stage 3A: Fast Union-Find same-class static geometric clustering
        if cfg.proactive_same_class_merge and len(coarsenable) > 1:
            by_class: Dict[str, List[AegisV4Entity]] = defaultdict(list)
            dyn_non_atomic: List[AegisV4Entity] = []
            for e in coarsenable:
                if e.is_dynamic:
                    dyn_non_atomic.append(e)
                else:
                    by_class[e.label].append(e)

            clustered: List[AegisV4Entity] = list(dyn_non_atomic)
            for lbl, grp in by_class.items():
                m = len(grp)
                if m == 1:
                    clustered.append(grp[0])
                    continue
                parent = list(range(m))
                comp_members: Dict[int, List[AegisV4Entity]] = {idx: [grp[idx]] for idx in range(m)}
                comp_bbox: Dict[int, Tuple[float, float, float, float, float]] = {}
                geom_xy: List[Tuple[float, float, float, float]] = []
                for idx in range(m):
                    px, py = float(grp[idx].pos[0]), float(grp[idx].pos[1])
                    hx, hy = 0.5 * float(grp[idx].size[0]), 0.5 * float(grp[idx].size[1])
                    geom_xy.append((px, py, hx, hy))
                    area = max(2.0 * hx, 0.05) * max(2.0 * hy, 0.05)
                    comp_bbox[idx] = (px - hx, py - hy, px + hx, py + hy, area)

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                min_gap = cfg.min_navigable_gap_m
                min_gap_sq = min_gap * min_gap
                for i in range(m):
                    pxi, pyi, hxi, hyi = geom_xy[i]
                    for j in range(i + 1, m):
                        pxj, pyj, hxj, hyj = geom_xy[j]
                        dx = abs(pxi - pxj) - (hxi + hxj)
                        if dx >= min_gap:
                            continue
                        dy = abs(pyi - pyj) - (hyi + hyj)
                        if dy >= min_gap:
                            continue
                        dx_p = dx if dx > 0.0 else 0.0
                        dy_p = dy if dy > 0.0 else 0.0
                        if dx_p * dx_p + dy_p * dy_p < min_gap_sq:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                b1, b2 = comp_bbox[ri], comp_bbox[rj]
                                nx0, ny0 = min(b1[0], b2[0]), min(b1[1], b2[1])
                                nx1, ny1 = max(b1[2], b2[2]), max(b1[3], b2[3])
                                n_sum = b1[4] + b2[4]
                                m_area = max(nx1 - nx0, 0.05) * max(ny1 - ny0, 0.05)
                                ok_merge = m_area <= cfg.max_aabb_bloat_ratio * n_sum
                                if (
                                    not ok_merge
                                    and cfg.enable_obb_supernodes
                                    and (len(comp_members[ri]) + len(comp_members[rj])) <= 4
                                ):
                                    ok_merge = self._obb_bloat_ok(
                                        comp_members[ri] + comp_members[rj],
                                        cfg.max_aabb_bloat_ratio,
                                        use_obb=True,
                                    )
                                if ok_merge:
                                    parent[rj] = ri
                                    comp_members[ri].extend(comp_members[rj])
                                    comp_bbox[ri] = (nx0, ny0, nx1, ny1, n_sum)
                                    del comp_members[rj]
                                    del comp_bbox[rj]

                for comp in comp_members.values():
                    if len(comp) == 1:
                        clustered.append(comp[0])
                    else:
                        total_m = sum(len(e.member_ids) for e in comp)
                        clustered.append(
                            self._merge_group(
                                comp,
                                merge_id=f"super_{lbl}_{merge_counter}",
                                new_label=f"{lbl}_group(x{total_m})",
                                use_obb=cfg.enable_obb_supernodes,
                            )
                        )
                        merge_counter += 1
            coarsenable = clustered

        # Stage 3B: Overflow cross-class geometric hazard coarsening
        geom_haz = [e for e in coarsenable if e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold]
        geom_non_haz = [e for e in coarsenable if not (e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold)]

        active_protected_count = sum(1 for e in protected if e.risk > cfg.tau_haz or e.delta_cf > 1e-4)
        target_geom_budget = max(1, k - active_protected_count)

        while len(geom_haz) > target_geom_budget:
            best_pair: Optional[Tuple[int, int]] = None
            best_gap = float("inf")
            for i in range(len(geom_haz)):
                for j in range(i + 1, len(geom_haz)):
                    gap = _xy_edge_gap(geom_haz[i].pos, geom_haz[i].size, geom_haz[j].pos, geom_haz[j].size)
                    if (
                        gap < cfg.min_navigable_gap_m
                        and gap < best_gap
                        and self._obb_bloat_ok(
                            [geom_haz[i], geom_haz[j]], cfg.max_aabb_bloat_ratio, use_obb=cfg.enable_obb_supernodes
                        )
                    ):
                        best_gap = gap
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            members_n = len(geom_haz[i].member_ids | geom_haz[j].member_ids)
            merged = self._merge_group(
                [geom_haz[i], geom_haz[j]],
                merge_id=f"super_nogo_{merge_counter}",
                new_label=f"no_go_cluster(x{members_n})",
                use_obb=cfg.enable_obb_supernodes,
            )
            merge_counter += 1
            geom_haz = [e for idx, e in enumerate(geom_haz) if idx not in (i, j)]
            geom_haz.append(merged)

        # Stage 3C: Cross-Layer Hierarchical Semantic Zoom for distant background clutter
        if (
            cfg.enable_hierarchical_zoom
            and graph_obs is not None
            and agent is not None
            and entity_parent_place
            and len(geom_non_haz) > 1
        ):
            by_place: Dict[str, List[AegisV4Entity]] = defaultdict(list)
            kept_near: List[AegisV4Entity] = []
            for e in geom_non_haz:
                dist_xy = float(np.linalg.norm(e.pos[:2] - agent.pos[:2]))
                p_id = None
                for mid in e.member_ids:
                    if mid in entity_parent_place:
                        p_id = entity_parent_place[mid]
                        break
                if p_id is not None and p_id in graph_obs and dist_xy >= cfg.hierarchical_zoom_dist_m:
                    by_place[p_id].append(e)
                else:
                    kept_near.append(e)

            for p_id, grp in by_place.items():
                if len(grp) == 1:
                    kept_near.append(grp[0])
                else:
                    p_node = graph_obs.nodes[p_id]
                    p_layer = str(p_node.get("layer", "place"))
                    p_lbl = str(p_node.get("label", p_id))
                    total_m = sum(len(e.member_ids) for e in grp)
                    zoomed = self._merge_group(
                        grp,
                        merge_id=f"hier_{p_id}_{merge_counter}",
                        new_label=f"{p_layer}:{p_lbl}(x{total_m})",
                        use_obb=cfg.enable_obb_supernodes,
                        dsg_layer=p_layer,
                    )
                    p_cv = float(p_node.get("structural_importance", 0.5))
                    zoomed.weight = max(zoomed.weight, cfg.beta * p_cv)
                    kept_near.append(zoomed)
                    merge_counter += 1
            geom_non_haz = kept_near

        return protected + geom_haz + geom_non_haz

    # ------------------------------------------------------------------
    # Layer 4: Online ACI-Calibrated Monotone Submodular Selection + Hysteresis
    # ------------------------------------------------------------------
    def _select_submodular_aci(
        self,
        entities: List[AegisV4Entity],
        k: int,
        tau_min_eff: float,
    ) -> List[AegisV4Entity]:
        """Greedy (1 - 1/e) Submodular Selection with Caratheodory KKT Support & Online ACI."""
        cfg = self.config
        if not entities or k <= 0:
            return []

        pool = [
            e
            for e in entities
            if e.weight > 1e-4 or e.risk > cfg.tau_haz or e.delta_cf > 1e-4 or e.is_kkt_support
        ]
        n = len(pool)
        if n == 0:
            return []

        weights = np.array([e.weight for e in pool], dtype=float)

        # Redundancy kernel K[u, v]: active hazards, atomic hazards, and KKT/CBF barriers are non-redundant
        K = np.eye(n, dtype=float)
        non_prot_idx = [
            i for i in range(n)
            if not (pool[i].risk > cfg.tau_haz or pool[i].is_atomic_hazard or pool[i].delta_cf > 1e-4 or pool[i].is_kkt_support)
        ]
        if len(non_prot_idx) > 1:
            min_gap = cfg.min_navigable_gap_m
            min_gap_sq = min_gap * min_gap
            inv_2sig2 = 1.0 / (2.0 * cfg.kernel_sigma_m * cfg.kernel_sigma_m)
            np_geom = [
                (i, float(pool[i].pos[0]), float(pool[i].pos[1]), 0.5 * float(pool[i].size[0]), 0.5 * float(pool[i].size[1]))
                for i in non_prot_idx
            ]
            m_np = len(np_geom)
            for a in range(m_np):
                i, pxi, pyi, hxi, hyi = np_geom[a]
                for b in range(a + 1, m_np):
                    j, pxj, pyj, hxj, hyj = np_geom[b]
                    dx = abs(pxi - pxj) - (hxi + hxj)
                    if dx >= min_gap:
                        continue
                    dy = abs(pyi - pyj) - (hyi + hyj)
                    if dy >= min_gap:
                        continue
                    dx_p = dx if dx > 0.0 else 0.0
                    dy_p = dy if dy > 0.0 else 0.0
                    gap_sq = dx_p * dx_p + dy_p * dy_p
                    if gap_sq < min_gap_sq:
                        val = 0.4 * math.exp(-gap_sq * inv_2sig2)
                        K[i, j] = val
                        K[j, i] = val

        current_cov = np.zeros(n, dtype=float)
        selected: List[AegisV4Entity] = []
        remaining = set(range(n))

        # Caratheodory Zero-Regret Support Reservation (|S_KKT| <= 3):
        # Guarantees ||u*(G) - u*(G')||_2 == 0 whenever |S_KKT| + |active_atomic| <= k,
        # while never displacing active non-coarsenable atomic hazards under tight k.
        if cfg.enforce_kkt_support:
            kkt_indices = [i for i in range(n) if pool[i].is_kkt_support]
            if kkt_indices:
                kkt_indices.sort(
                    key=lambda idx: (
                        1 if pool[idx].is_atomic_hazard else 0,
                        pool[idx].dual_multiplier + pool[idx].delta_cf,
                        pool[idx].weight,
                    ),
                    reverse=True,
                )
                non_kkt_active_atomics = sum(
                    1
                    for i in range(n)
                    if pool[i].is_atomic_hazard and pool[i].risk > cfg.tau_haz and not pool[i].is_kkt_support
                )
                kkt_budget = min(len(kkt_indices), max(0, k - non_kkt_active_atomics))
                for idx in kkt_indices[:kkt_budget]:
                    ent = pool[idx]
                    inc = np.maximum(K[:, idx] - current_cov, 0.0)
                    gain = float(np.dot(weights, inc))
                    ent.marginal_gain = round(gain, 4)
                    selected.append(ent)
                    remaining.remove(idx)
                    current_cov = np.maximum(current_cov, K[:, idx])

        while len(selected) < k and remaining:
            best_idx = -1
            best_gain = -1.0

            for v in remaining:
                ent = pool[v]
                inc = np.maximum(K[:, v] - current_cov, 0.0)
                gain = float(np.dot(weights, inc))

                if (ent.member_ids & self._prev_covered) and ent.weight >= cfg.theta_out:
                    max_pers = max((self._persistence.get(m, 1) for m in ent.member_ids), default=1)
                    gain += cfg.hysteresis_bonus * (1.5 if max_pers < cfg.min_persistence_frames else 1.0)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = v

            if best_idx < 0:
                break

            chosen = pool[best_idx]
            if best_gain < tau_min_eff and chosen.risk <= cfg.tau_haz and chosen.delta_cf <= 1e-4 and not chosen.is_kkt_support:
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
        u_nom: Optional[np.ndarray] = None,
    ) -> List[AegisV4Entity]:
        """Run all 4 Aegis-V4 CCBS+ACI layers and return <= k compressed entities."""
        if k <= 0:
            self._prev_covered.clear()
            return []

        tracks = self._update_belief_state(graph_obs, dt)
        if not tracks:
            self._prev_covered.clear()
            return []

        cfg = self.config
        dt_eff = float(dt) if dt > 0 else cfg.cbf_dt

        # Layer 2A: Exact CBF-QP Counterfactual Safety Regrets + Minimal KKT Support
        cbf_res = compute_counterfactual_regrets(
            entities=tracks,
            agent=agent,
            u_nom=u_nom,
            dt=dt_eff,
            gamma_cbf=cfg.gamma_cbf,
            safety_margin=cfg.cbf_safety_margin,
            u_max=cfg.u_max,
            slack_penalty=cfg.cbf_slack_penalty,
            risk_mapper=self.risk_mapper,
        )
        self.last_cbf_result = cbf_res

        # Layer 2B: Kinematic Risk + Epistemic 3D Eye-Gaze Modulation + CCBS Weight
        candidates: List[AegisV4Entity] = []
        entity_parent_place: Dict[str, str] = {}

        for tr in tracks:
            r_v, _ = self._compute_kinematic_risk_and_ttc(tr, agent)
            tr.last_risk = r_v

            d_cf = float(cbf_res.regrets.get(tr.obj_id, 0.0))
            is_kkt_supp = tr.obj_id in cbf_res.support_ids
            dual_lam = float(cbf_res.dual_multipliers.get(tr.obj_id, 0.0))
            is_cbf_binding = is_kkt_supp or (tr.obj_id in cbf_res.active_ids)
            m_urg = float(cbf_res.margin_urgencies.get(tr.obj_id, 0.0))

            if tr.missed_frames > 0 and r_v <= cfg.tau_haz and d_cf <= 1e-4 and not is_cbf_binding:
                continue

            if cfg.enable_epistemic_gaze and agent.gaze_dir is not None:
                ep_res = self.gaze_modulator.compute_priority(
                    entity_id=tr.obj_id,
                    entity_pos=tr.pos,
                    agent=agent,
                    delta_cf=d_cf,
                    r_v=r_v,
                    update_ema=True,
                )
                gaze_att = ep_res.gaze_attention
                ep_prio = ep_res.priority
            else:
                gaze_att = 0.0
                ep_prio = d_cf + r_v
            tr.gaze_attention = gaze_att

            w_v, clip_rel = self._compute_ccbs_weight(
                tr=tr,
                r_v=r_v,
                delta_cf=d_cf,
                is_cbf_binding=is_cbf_binding,
                margin_urgency=m_urg,
                epistemic_prio=ep_prio,
            )

            if tr.parent_place_id is not None:
                entity_parent_place[tr.obj_id] = tr.parent_place_id

            candidates.append(
                AegisV4Entity(
                    entity_id=tr.obj_id,
                    label=tr.label,
                    pos=tr.pos.copy(),
                    size=tr.size.copy(),
                    vel=tr.vel.copy(),
                    risk=r_v,
                    weight=w_v,
                    belief=tr.belief,
                    is_super_node=False,
                    is_atomic_hazard=tr.is_atomic_hazard,
                    is_occluded_memory=(tr.missed_frames > 0),
                    is_dynamic=tr.is_dynamic,
                    dsg_layer="object",
                    gaze_attention=gaze_att,
                    task_relevance=clip_rel,
                    member_ids={tr.obj_id},
                    delta_cf=d_cf,
                    margin_urgency=m_urg,
                    epistemic_priority=ep_prio,
                    dual_multiplier=dual_lam,
                    is_kkt_support=is_kkt_supp,
                )
            )

        # Layer 3: Free-Space-Preserving OBB Coarsening (preserving Atomic & KKT Support)
        coarsened = self._coarsen_entities(
            candidates,
            k,
            graph_obs=graph_obs,
            agent=agent,
            entity_parent_place=entity_parent_place,
        )

        # Layer 4: Online ACI-calibrated Monotone Submodular Selection + Hysteresis
        tau_min_eff = (
            self.aci.get_calibrated_tau_min(default_tau_min=cfg.tau_min, floor=cfg.aci_tau_floor)
            if cfg.enable_online_aci
            else cfg.tau_min
        )
        selected = self._select_submodular_aci(coarsened, k, tau_min_eff=tau_min_eff)

        covered = set().union(*(e.member_ids for e in selected)) if selected else set()
        self._persistence = {nid: self._persistence.get(nid, 0) + 1 for nid in covered}
        self._prev_covered = covered

        # Self-supervised online ACI update (zero GT labels needed at deployment!):
        # Measures actual sparsification distortion ||u*(G_obs) - u*(G')||_2 / u_max
        # and maximum dropped kinematic risk, updated causally via Gibbs & Candes ACI.
        if cfg.enable_online_aci:
            cached_c = {c.entity_id: c for c in cbf_res.constraints}
            c_sparse = []
            super_nodes = []
            for ent in selected:
                if not ent.is_super_node:
                    c_hit = cached_c.get(ent.entity_id)
                    if c_hit is not None:
                        c_sparse.append(c_hit)
                else:
                    super_nodes.append(ent)
            if super_nodes:
                c_sparse.extend(
                    build_cbf_constraints(
                        entities=super_nodes,
                        agent=agent,
                        dt=dt_eff,
                        gamma_cbf=cfg.gamma_cbf,
                        safety_margin=cfg.cbf_safety_margin,
                        risk_mapper=self.risk_mapper,
                    )
                )
            u_sparse = solve_cbf_qp(
                u_nom=cbf_res.u_nom[:2],
                constraints=c_sparse,
                u_max=cfg.u_max,
                slack_penalty=cfg.cbf_slack_penalty,
            )
            ctrl_dist = math.hypot(
                float(cbf_res.u_safe[0] - u_sparse[0]),
                float(cbf_res.u_safe[1] - u_sparse[1]),
            )
            e_ctrl = ctrl_dist / max(cfg.u_max, 1e-3)
            e_drop = max(
                (e.risk for e in coarsened if not (e.member_ids & covered)),
                default=0.0,
            )
            e_self = max(e_ctrl, e_drop)
            e_self = 0.0 if e_self < 0.0 else (1.0 if e_self > 1.0 else e_self)
            self.last_aci_step = self.aci.update(nonconformity_score=e_self)

        return selected

    def update_aci_feedback(
        self,
        min_retained_hazard_score: Optional[float] = None,
        err_t: Optional[int] = None,
    ) -> ACIStepResult:
        """Explicit online ACI feedback step when external/delayed safety labels arrive."""
        step_res = self.aci.update(
            min_retained_hazard_score=min_retained_hazard_score,
            err_t=err_t,
        )
        self.last_aci_step = step_res
        return step_res

    def select_entities(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> List[AegisV4Entity]:
        """Return <= k compressed entities (MuJoCo closed-loop interface)."""
        return self.compress(graph_obs, agent, k)

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        """Return set of covered object IDs (Phase A/B metrics interface)."""
        entities = self.compress(graph_obs, agent, k)
        return set().union(*(e.member_ids for e in entities)) if entities else set()

    def compute_safe_velocity(
        self,
        entities: Sequence[AegisEntity],
        agent: AgentState,
        u_nom: np.ndarray,
        dt: Optional[float] = None,
        u_max: Optional[float] = None,
        gamma_cbf: Optional[float] = None,
        safety_margin: Optional[float] = None,
        enable_deadlock_escape: bool = False,
        return_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, List[int]]]:
        """CBF-QP safety filter operating directly on the <= k compressed graph.

        When `enable_deadlock_escape=True`, collinear CBF-QP equilibrium stalls
        (||u_safe|| < 0.35 * ||u_nom|| under active barriers) are broken by
        testing tangential perturbations +/-t_hat orthogonal to the dominant
        active barrier normal and re-solving the EXACT same CBF-QP constraints,
        preserving 100% barrier forward invariance.
        """
        cfg = self.config
        dt_eff = cfg.cbf_dt if dt is None else float(dt)
        u_max_eff = cfg.u_max if u_max is None else float(u_max)
        gamma_eff = cfg.gamma_cbf if gamma_cbf is None else float(gamma_cbf)
        margin_eff = cfg.cbf_safety_margin if safety_margin is None else float(safety_margin)

        constraints = build_cbf_constraints(
            entities=entities,
            agent=agent,
            dt=dt_eff,
            gamma_cbf=gamma_eff,
            safety_margin=margin_eff,
            risk_mapper=self.risk_mapper,
        )
        u_safe, slack, active = solve_cbf_qp(
            u_nom=u_nom,
            constraints=constraints,
            u_max=u_max_eff,
            slack_penalty=cfg.cbf_slack_penalty,
            return_info=True,
        )

        if enable_deadlock_escape and active:
            u_arr = np.asarray(u_nom, dtype=float)
            nom_spd = math.hypot(float(u_arr[0]), float(u_arr[1]))
            safe_spd = math.hypot(float(u_safe[0]), float(u_safe[1]))
            if nom_spd > 0.25 and safe_spd < 0.35 * nom_spd:
                # Pick active barrier most directly opposing u_nom
                dom_idx = min(
                    active,
                    key=lambda idx: float(constraints[idx].a[0] * u_arr[0] + constraints[idx].a[1] * u_arr[1]),
                )
                ax, ay = float(constraints[dom_idx].a[0]), float(constraints[dom_idx].a[1])
                tx, ty = -ay, ax
                best_u, best_sl, best_act = u_safe, slack, active
                best_metric = safe_spd
                for sgn in (1.0, -1.0):
                    u_try = u_arr.copy()
                    u_try[0] = float(u_arr[0]) + sgn * 0.90 * tx
                    u_try[1] = float(u_arr[1]) + sgn * 0.90 * ty
                    u_c, sl_c, act_c = solve_cbf_qp(
                        u_nom=u_try,
                        constraints=constraints,
                        u_max=u_max_eff,
                        slack_penalty=cfg.cbf_slack_penalty,
                        return_info=True,
                    )
                    if sl_c <= slack + 1e-4:
                        c_spd = math.hypot(float(u_c[0]), float(u_c[1]))
                        if c_spd > best_metric + 0.05:
                            best_metric = c_spd
                            best_u, best_sl, best_act = u_c, sl_c, act_c
                u_safe, slack, active = best_u, best_sl, best_act

        return (u_safe, slack, active) if return_info else u_safe


__all__ = [
    "AegisV4Config",
    "AegisV4Entity",
    "AegisV4CCBS",
]

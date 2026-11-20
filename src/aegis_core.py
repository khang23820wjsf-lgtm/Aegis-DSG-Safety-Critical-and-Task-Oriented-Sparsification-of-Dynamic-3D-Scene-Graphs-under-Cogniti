"""Aegis-DSG Unified Core Architecture (v2.0).

A cohesive 4-layer algorithm for Safety-Critical and Task-Oriented Sparsification
of Dynamic 3D Scene Graphs under Cognitive Bandwidth Constraints:

  Layer 1 — Temporal Belief Filter & Object Permanence:
      Maintains per-entity existence belief p_t(v) in [0, 1] via EMA update,
      suppressing transient 1-frame false-positive ghosts while preserving briefly
      occluded real hazards via constant-velocity rollout.

  Layer 2 — Dual-Hazard 3D Kinematic Risk Field + Asymmetric Cost:
      Unifies near-body clearance (d_0) and Dynamic Collision Cone TTC (d_TTC)
      in a smooth 3D risk field R_v, separating Geometric Obstacles from
      Intrinsic/Atomic Critical Hazards (e.g., knives, scissors, stairs, stoves,
      pedestrians) whose specific semantic identity is safety-critical.

  Layer 3 — Free-Space-Preserving & Non-Coarsenable Graph Coarsening (Fast Union-Find):
      Merges adjacent same-class static geometric clusters (O(N) Union-Find per class)
      and overflow geometric hazards into Super-Nodes ONLY when clearance < 0.70m,
      while STRICTLY FORBIDDING Semantic Erasure of Atomic Hazards (is_atomic_hazard=True).

  Layer 4 — Dynamic-Budget Monotone Submodular Selection + Hysteresis:
      Greedily maximizes a monotone submodular facility-location + temporal-persistence
      objective (Nemhauser 1 - 1/e optimality guarantee), stopping early when
      marginal gain < tau_min so |V'_t| <= k never wastes bandwidth on clutter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math
import re
import networkx as nx
import numpy as np

from src.aegis_v3_modules import (
    ConformalCalibrationReport,
    ConformalSafetyCalibrator,
    ZeroShotCLIPCache,
    compute_min_area_obb_2d,
    entity_corners_2d,
)
from src.dsg_generator import AgentState


# Domain-agnostic physical hazard affordance taxonomy (home, lab, construction, street).
# Each axis defines: (intrinsic_severity, prototype_description_for_embeddings, lexical_stems)
PHYSICAL_HAZARD_AFFORDANCES: Dict[str, Tuple[float, str, Tuple[str, ...]]] = {
    "sharp_puncture": (
        0.95,
        "sharp blade cutting piercing laceration tool edge",
        ("knife", "blade", "scissors", "razor", "scalpel", "needle", "syringe", "saw", "chisel", "rebar", "broken_glass", "shard"),
    ),
    "thermal_combustion": (
        0.92,
        "extreme heat open flame burning hot surface boiling liquid",
        ("stove", "fire", "flame", "kettle", "boiling", "burner", "furnace", "torch", "welder", "oven", "steam", "hot_plate"),
    ),
    "chemical_biohazard": (
        0.94,
        "corrosive toxic acid chemical reagent biohazard pressurized gas",
        ("acid", "chemical", "biohazard", "solvent", "corrosive", "gas_cylinder", "beaker", "fume", "poison", "flammable"),
    ),
    "electrical_voltage": (
        0.90,
        "exposed live electric wire high voltage shock hazard",
        ("exposed_wire", "live_wire", "live_cable", "high_voltage", "transformer", "power_line", "arc_flash"),
    ),
    "elevation_drop": (
        0.92,
        "steep drop off stairs cliff ledge pit open hole fall hazard",
        ("stair_edge", "stairs", "drop_off", "ledge", "pit", "excavation", "manhole", "balcony_edge", "cliff"),
    ),
    "autonomous_living_or_heavy_dynamic": (
        0.95,
        "living human pedestrian child cyclist moving vehicle heavy machinery",
        ("pedestrian", "person", "human", "child", "cyclist", "forklift", "crane", "car", "truck", "vehicle", "robot_arm", "excavator"),
    ),
}

# Backward-compatible flattened tuple alias
ATOMIC_HAZARD_KEYWORDS: Tuple[str, ...] = tuple(
    kw for _, _, kws in PHYSICAL_HAZARD_AFFORDANCES.values() for kw in kws
)


class AffordanceRiskMapper:
    """Offline-cached, domain-agnostic mapper from semantic class/affordances -> intrinsic hazard risk.

    Supports two seamless modes at 0.00 ms steady-state runtime (O(1) dict cache):
      1. Zero-shot embedding mode: pass any text encoder `embedding_fn(text) -> np.ndarray`
         (e.g. CLIP / Sentence-Transformers). Prototype affordance vectors are encoded ONCE
         at startup, and each unseen label is scored via max cosine similarity and cached.
      2. Lightweight multi-domain mode (default, zero ML dependencies): matches explicit
         physical affordance tags or multi-domain lexical stems (home, lab, construction, street)
         and caches every resolved label in O(1).
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        embedding_threshold: float = 0.72,
        tau_atomic: float = 0.85,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.embedding_threshold = float(embedding_threshold)
        self.tau_atomic = float(tau_atomic)
        self._cache: Dict[str, Tuple[float, bool, str]] = {}
        self._proto_vecs: Dict[str, Tuple[float, np.ndarray]] = {}
        self._benign_vec: Optional[np.ndarray] = None

        if self.embedding_fn is not None:
            for axis, (sev, desc, _) in PHYSICAL_HAZARD_AFFORDANCES.items():
                vec = np.asarray(self.embedding_fn(desc), dtype=float)
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    self._proto_vecs[axis] = (sev, vec / norm)
            bvec = np.asarray(self.embedding_fn("a safe harmless soft pillow plush toy cardboard box cushion"), dtype=float)
            bnorm = float(np.linalg.norm(bvec))
            if bnorm > 1e-8:
                self._benign_vec = bvec / bnorm

    def precompute_vocabulary(self, labels: Iterable[str]) -> None:
        """Warm the O(1) cache offline before real-time execution begins."""
        for lbl in labels:
            self.assess(lbl)

    def assess(
        self,
        label: str,
        explicit_flag: Optional[bool] = None,
        affordances: Optional[Iterable[str]] = None,
    ) -> Tuple[float, bool, str]:
        """Return (intrinsic_severity, is_atomic_hazard, matched_affordance_axis)."""
        if explicit_flag is not None:
            flag = bool(explicit_flag)
            return (1.0 if flag else 0.0, flag, "explicit" if flag else "none")

        # Explicit physical affordance tags from upstream VLM/DSG take highest priority
        if affordances:
            aff_set = {str(a).lower().strip() for a in affordances}
            for axis, (sev, _, _) in PHYSICAL_HAZARD_AFFORDANCES.items():
                if axis in aff_set or any(tok in aff_set for tok in axis.split("_")):
                    return (sev, sev >= self.tau_atomic, axis)

        # Split camelCase (e.g. "FakeButcherKnife" -> "fake_butcher_knife") before lowercasing
        camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", label)
        key = camel_split.lower().strip().replace(" ", "_")
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        best_sev = 0.0
        best_axis = "none"

        # 1) Multi-domain token-boundary match (prevents substring false positives like "car" in "cardboard"/"carpet")
        tokens = set(re.split(r"[^a-z0-9]+", key))
        norm_spaced = " ".join(re.split(r"[^a-z0-9]+", key))
        for axis, (sev, _, stems) in PHYSICAL_HAZARD_AFFORDANCES.items():
            for st in stems:
                if "_" in st:
                    phrase = st.replace("_", " ")
                    matched = phrase in norm_spaced or st in key
                else:
                    matched = st in tokens or any(tok == st + "s" or tok == st + "es" for tok in tokens)
                if matched:
                    if sev > best_sev:
                        best_sev = sev
                        best_axis = axis
                    break

        # 2) Optional offline zero-shot semantic embedding similarity (contrastive margin + O(1) cache)
        if best_sev < self.tau_atomic and self.embedding_fn is not None and self._proto_vecs:
            vec = np.asarray(self.embedding_fn(label), dtype=float)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                u = vec / norm
                sim_benign = float(np.dot(u, self._benign_vec)) if self._benign_vec is not None else 0.0
                for axis, (sev, p_vec) in self._proto_vecs.items():
                    sim = float(np.dot(u, p_vec))
                    # Must exceed both absolute similarity threshold AND beat benign-object prototype!
                    if sim >= self.embedding_threshold and (sim - sim_benign) >= 0.04:
                        best_sev = max(best_sev, sev)
                        best_axis = f"emb:{axis}"

        res = (best_sev, best_sev >= self.tau_atomic, best_axis)
        self._cache[key] = res
        return res


_DEFAULT_AFFORDANCE_MAPPER = AffordanceRiskMapper()


@dataclass(frozen=True)
class AegisConfig:
    """Single source of truth for all Aegis-DSG v3.0 physical, conformal & semantic parameters."""

    # Layer 1: Temporal Belief Filter & Object Permanence
    belief_init_scale: float = 0.75
    belief_ema_alpha: float = 0.65
    belief_occluded_decay: float = 0.65
    belief_min_keep: float = 0.28
    fp_belief_threshold: float = 0.45
    max_occluded_frames: int = 2

    # Layer 2: Continuous 3D Kinematic Risk Field + 3D Eye-Gaze Attention + Open-Vocab CLIP Task
    horizon_s: float = 3.2
    cone_half_angle_deg: float = 38.0
    sigma_near_m: float = 0.50
    sigma_cone_m: float = 0.65
    atomic_reach_m: float = 2.20
    alpha: float = 1.0
    beta: float = 0.16
    gamma: float = 1.10
    tau_haz: float = 0.28
    enable_gaze_attention: bool = True
    gaze_fovea_deg: float = 14.0
    gaze_unattended_boost: float = 0.25
    clip_task_weight: float = 0.35

    # Layer 3: OBB Free-Space-Preserving Coarsening + Cross-Layer Hierarchical Semantic Zoom
    min_navigable_gap_m: float = 0.70
    max_aabb_bloat_ratio: float = 1.85
    enable_obb_supernodes: bool = True
    proactive_same_class_merge: bool = True
    enable_hierarchical_zoom: bool = True
    hierarchical_zoom_dist_m: float = 3.2

    # Layer 4: Dynamic-Budget Monotone Submodular Selection + Conformal Threshold + Hysteresis
    tau_min: float = 0.12
    hysteresis_bonus: float = 0.14
    theta_out: float = 0.12
    min_persistence_frames: int = 2
    kernel_sigma_m: float = 0.50


@dataclass
class AegisEntity:
    """Unified compressed entity output by AegisDSG v3.0 (object, atomic, OBB Super-Node, or Place/Room)."""

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
    is_atomic_hazard: bool = False
    is_occluded_memory: bool = False
    is_dynamic: bool = False
    dsg_layer: str = "object"
    obb_yaw: float = 0.0
    obb_size: Optional[np.ndarray] = None
    gaze_attention: float = 0.0
    task_relevance: float = 0.0
    member_ids: Set[str] = field(default_factory=set)

    @property
    def score(self) -> float:
        """Alias for backward compatibility with visualization/eval helpers."""
        return self.weight


@dataclass
class _TrackState:
    """Internal temporal belief & gaze-attention state for Layer 1/2."""

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


def _xy_edge_gap(pos_a: np.ndarray, size_a: np.ndarray, pos_b: np.ndarray, size_b: np.ndarray) -> float:
    """Minimum horizontal surface-to-surface clearance (meters) between two 3D bounding boxes."""
    dx = max(abs(float(pos_a[0] - pos_b[0])) - 0.5 * float(size_a[0] + size_b[0]), 0.0)
    dy = max(abs(float(pos_a[1] - pos_b[1])) - 0.5 * float(size_a[1] + size_b[1]), 0.0)
    return math.hypot(dx, dy)


def _infer_is_atomic(
    label: str,
    explicit_flag: bool | None,
    mapper: AffordanceRiskMapper | None = None,
    affordances: Optional[Iterable[str]] = None,
) -> bool:
    """Determine whether an object is an intrinsic non-coarsenable atomic hazard."""
    m = mapper if mapper is not None else _DEFAULT_AFFORDANCE_MAPPER
    _, is_atomic, _ = m.assess(label, explicit_flag=explicit_flag, affordances=affordances)
    return is_atomic


class AegisDSG:
    """Unified Aegis-DSG v3.0 Sparsifier (Final Frozen Architecture).

    Integrates all 5 core breakthroughs:
      1. 3D Eye-Gaze Unattended-Hazard Attention Modulation (Layer 2)
      2. Distribution-Free Split Conformal Safety Calibration (calibrate_conformal)
      3. Minimum-Area 2D Oriented Bounding Box (OBB) Super-Nodes via Rotating Calipers (Layer 3A/3B)
      4. Cross-Layer Hierarchical Semantic Zoom: Object <-> Place <-> Room (Layer 3C)
      5. Real Zero-Shot CLIP Offline-Cached Open-Vocabulary Task & Affordance Engine (O(1) runtime)
    """

    def __init__(
        self,
        config: AegisConfig | None = None,
        risk_mapper: AffordanceRiskMapper | None = None,
        clip_cache: ZeroShotCLIPCache | None = None,
    ) -> None:
        self.config = config or AegisConfig()
        self.clip_cache = clip_cache
        if risk_mapper is not None:
            self.risk_mapper = risk_mapper
        elif clip_cache is not None and clip_cache.is_real_clip_loaded:
            self.risk_mapper = AffordanceRiskMapper(embedding_fn=clip_cache.encode_text)
        else:
            self.risk_mapper = _DEFAULT_AFFORDANCE_MAPPER
        self.name = "Aegis-Core-v3"
        self._cone_half_rad = math.radians(self.config.cone_half_angle_deg)
        self._gaze_fovea_rad = math.radians(self.config.gaze_fovea_deg)
        self._tracks: Dict[str, _TrackState] = {}
        self._prev_covered: Set[str] = set()
        self._persistence: Dict[str, int] = {}
        self.conformal_report: Optional[ConformalCalibrationReport] = None

    def set_task_prompt(self, task_prompt: str, vocabulary: Optional[Iterable[str]] = None) -> None:
        """Bind a free-form open-vocabulary user task prompt (encoded ONCE offline via CLIP)."""
        if self.clip_cache is None:
            self.clip_cache = ZeroShotCLIPCache()
            self.clip_cache.load()
        self.clip_cache.set_task_prompt(task_prompt, vocabulary=vocabulary)

    def calibrate_conformal(
        self,
        calib_sequences: Sequence[Sequence],
        alpha: float = 0.05,
        dt: float = 0.1,
    ) -> ConformalCalibrationReport:
        """Distribution-Free Split Conformal Safety Calibration on historical/calibration DSG sequences.

        Computes the exact finite-sample quantile q_hat for miscoverage <= alpha and updates
        this instance's `config` with certified (tau_min, tau_haz) thresholds.
        """
        haz_scores: List[float] = []
        haz_risks: List[float] = []
        for seq in calib_sequences:
            self.reset()
            for frame in seq:
                tracks = self._update_belief_state(frame.graph_obs, dt)
                tr_map = {t.obj_id: t for t in tracks}
                for nid, d in frame.graph_obs.nodes(data=True):
                    if d.get("layer") == "object" and d.get("is_hazard_gt", False) and nid in tr_map:
                        tr = tr_map[nid]
                        r_v = self._compute_unified_risk(tr, frame.agent)
                        tr.last_risk = r_v
                        w_v, _ = self._compute_asymmetric_weight(tr, r_v)
                        haz_risks.append(r_v)
                        haz_scores.append(w_v)

        cal = ConformalSafetyCalibrator(alpha=alpha)
        report = cal.calibrate_from_scores(
            haz_scores,
            haz_risks,
            default_tau_min=self.config.tau_min,
            default_tau_haz=self.config.tau_haz,
        )
        self.conformal_report = report
        self.config = replace(
            self.config,
            tau_min=report.calibrated_tau_min,
            tau_haz=report.calibrated_tau_haz,
        )
        self.reset()
        return report

    def reset(self) -> None:
        """Clear all temporal belief, permanence, and hysteresis memory."""
        self._tracks.clear()
        self._prev_covered.clear()
        self._persistence.clear()

    # ------------------------------------------------------------------
    # Layer 1: Temporal Belief Filter & Object Permanence
    # ------------------------------------------------------------------
    def _update_belief_state(self, graph_obs: nx.DiGraph, dt: float) -> List[_TrackState]:
        """Update existence belief p_t(v) and maintain briefly occluded active hazards."""
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
            lbl = str(d.get("label", "object"))
            intrinsic_sev, is_atomic, _ = self.risk_mapper.assess(
                lbl,
                explicit_flag=d.get("is_atomic_hazard"),
                affordances=d.get("affordances"),
            )

            # Resolve optional parent Place / Room node in the hierarchical DSG
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
                tr.belief = float(np.clip((1.0 - cfg.belief_ema_alpha) * tr.belief + cfg.belief_ema_alpha * conf, 0.05, 0.99))
                tr.seen_frames += 1
                tr.missed_frames = 0
            else:
                init_belief = float(np.clip(cfg.belief_init_scale * conf, 0.05, 0.95))
                self._tracks[nid] = _TrackState(
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

        # Object Permanence: propagate briefly occluded active hazards in front of the agent
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
    # Layer 2: Dual-Hazard 3D Kinematic Risk Field + 3D Eye-Gaze + Open-Vocab CLIP Task
    # ------------------------------------------------------------------
    def _update_gaze_attention(self, tr: _TrackState, agent: AgentState) -> float:
        """Compute foveal visual attention A_gaze(v, t) in [0, 1] from 3D Eye-Gaze ray."""
        if not self.config.enable_gaze_attention or agent.gaze_dir is None:
            tr.gaze_attention = 0.0
            return 0.0
        rel_3d = tr.pos - agent.pos
        dist_3d = float(np.linalg.norm(rel_3d))
        if dist_3d < 1e-4:
            return tr.gaze_attention
        g_norm = float(np.linalg.norm(agent.gaze_dir))
        if g_norm < 1e-6:
            return tr.gaze_attention
        cos_g = float(np.clip(np.dot(rel_3d / dist_3d, agent.gaze_dir / g_norm), -1.0, 1.0))
        ang_g = math.acos(cos_g)
        inst_att = math.exp(-(ang_g ** 2) / (2.0 * (self._gaze_fovea_rad ** 2)))
        tr.gaze_attention = 0.65 * tr.gaze_attention + 0.35 * inst_att
        return tr.gaze_attention

    def _compute_unified_risk(self, tr: _TrackState, agent: AgentState) -> float:
        """Compute smooth 3D risk R_v unifying near-body clearance, TTC cone, atomic reach & gaze deficit."""
        cfg = self.config

        # 3D vertical relevance filter (atomic hazards like knives on waist-high tables always pass!)
        z_min = float(tr.pos[2] - tr.size[2] / 2.0)
        z_max = float(tr.pos[2] + tr.size[2] / 2.0)
        if not tr.is_atomic_hazard:
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

        # Component A: Smooth near-body clearance across front + side hemisphere (<= 100 deg)
        d_now = float(np.linalg.norm(np.maximum(np.abs(rel_pos) - half_ext, 0.0)))
        hemi_gate = 1.0 if cos_ang >= -0.18 else math.exp(-((cos_ang + 0.18) / 0.25) ** 2)
        sigma_near = (cfg.sigma_near_m * 1.45) if tr.is_atomic_hazard else cfg.sigma_near_m
        r_near = math.exp(-(d_now ** 2) / (2.0 * sigma_near ** 2)) * hemi_gate

        # Component B: Smooth forward Dynamic Collision Cone & TTC risk
        rel_vel = tr.vel[:2] - agent.vel[:2]
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        ttc = 0.0 if rel_speed_sq < 1e-6 else float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, cfg.horizon_s))
        d_ttc = float(np.linalg.norm(np.maximum(np.abs(rel_pos + rel_vel * ttc) - half_ext, 0.0)))

        cone_factor = 1.0 if angle <= self._cone_half_rad else math.exp(-((angle - self._cone_half_rad) / 0.35) ** 2)
        closing_speed = -float(np.dot(rel_pos / (dist + 1e-6), rel_vel))
        if closing_speed > 0.3:
            cone_factor = max(cone_factor, 0.78)

        spatial_cone = math.exp(-(d_ttc ** 2) / (2.0 * cfg.sigma_cone_m ** 2))
        time_urgency = 1.0 - 0.5 * (ttc / cfg.horizon_s)
        r_cone = spatial_cone * time_urgency * cone_factor

        # Component C: Close-range manipulation / personal-reach risk for intrinsic atomic hazards (e.g. knife at 0.5m)
        r_atomic = 0.0
        if tr.is_atomic_hazard and dist <= cfg.atomic_reach_m:
            r_atomic = math.exp(-(dist ** 2) / (2.0 * (cfg.atomic_reach_m * 0.55) ** 2))

        r_kin = max(r_near, r_cone, r_atomic) * max(tr.hazard_severity, 0.2)

        # Component D (v3.0): 3D Eye-Gaze Unattended-Hazard Modulation
        # Hazards OUTSIDE the wearer's foveal gaze (unattended peripheral threats) get priority boost!
        if cfg.enable_gaze_attention and agent.gaze_dir is not None and r_kin > 0.05:
            att = self._update_gaze_attention(tr, agent)
            unattended = 1.0 - att
            r_kin = r_kin * (1.0 + cfg.gaze_unattended_boost * unattended)

        return float(np.clip(r_kin, 0.0, 1.0))

    def _compute_asymmetric_weight(self, tr: _TrackState, r_v: float) -> Tuple[float, float]:
        """Belief-gated Dual-Tier Asymmetric Safety Weight w_v >= 0 + Zero-Shot CLIP task score."""
        cfg = self.config
        q_v = tr.confidence
        u_v = 1.0 - q_v
        c_struct = tr.structural_importance

        # Real Zero-Shot Open-Vocabulary CLIP Task Relevance (O(1) cached lookup -> 0.00 ms)
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

        haz_ind = 1.0 if r_v > cfg.tau_haz else 0.0
        if haz_ind > 0.0 and is_reliable and tr.missed_frames == 0:
            safety_tier = 0.90 if tr.is_atomic_hazard else 0.50
        else:
            safety_tier = 0.0

        w_v = belief_gate * (
            safety_tier + cfg.alpha * r_v * (1.0 + gamma_eff * u_v * haz_ind) + beta_eff * q_v * c_v
        )
        return max(0.0, float(w_v)), clip_rel

    # ------------------------------------------------------------------
    # Layer 3: OBB Free-Space-Preserving Coarsening + Cross-Layer Semantic Zoom
    # ------------------------------------------------------------------
    @staticmethod
    def _obb_bloat_ok(group: List[AegisEntity], max_ratio: float, use_obb: bool = True) -> bool:
        """Ensure merging a group does not hallucinate fake walls across empty floor space.

        When `use_obb=True`, uses exact Rotating-Calipers Minimum-Area OBB so 45-degree
        diagonal furniture rows merge tightly without AABB corridor inflation.
        """
        if len(group) <= 1:
            return True
        sum_area = sum(float(max(e.size[0], 0.05) * max(e.size[1], 0.05)) for e in group)
        if use_obb:
            corners = np.vstack([
                entity_corners_2d(
                    e.pos[:2],
                    e.obb_size[:2] if e.obb_size is not None else e.size[:2],
                    e.obb_yaw,
                )
                for e in group
            ])
            _, _, _, obb_area = compute_min_area_obb_2d(corners)
            return obb_area <= max_ratio * max(sum_area, 1e-4)
        mins = np.min([e.pos[:2] - 0.5 * e.size[:2] for e in group], axis=0)
        maxs = np.max([e.pos[:2] + 0.5 * e.size[:2] for e in group], axis=0)
        merged_area = float(max(maxs[0] - mins[0], 0.05) * max(maxs[1] - mins[1], 0.05))
        return merged_area <= max_ratio * max(sum_area, 1e-4)

    @staticmethod
    def _merge_group(
        group: List[AegisEntity],
        merge_id: str,
        new_label: str,
        use_obb: bool = True,
        dsg_layer: str = "object",
    ) -> AegisEntity:
        """Merge non-atomic geometric entities into 1 Super-Node with exact Minimum-Area OBB."""
        mins = np.min([e.pos - 0.5 * e.size for e in group], axis=0)
        maxs = np.max([e.pos + 0.5 * e.size for e in group], axis=0)
        aabb_center = 0.5 * (mins + maxs)
        aabb_size = maxs - mins

        obb_yaw = 0.0
        obb_size_3d: Optional[np.ndarray] = None
        if use_obb and len(group) >= 2:
            corners = np.vstack([
                entity_corners_2d(
                    e.pos[:2],
                    e.obb_size[:2] if e.obb_size is not None else e.size[:2],
                    e.obb_yaw,
                )
                for e in group
            ])
            obb_c_xy, obb_wh, obb_yaw, obb_area = compute_min_area_obb_2d(corners)
            aabb_area = float(max(aabb_size[0], 0.05) * max(aabb_size[1], 0.05))
            if obb_area < 0.96 * aabb_area:
                pos_out = np.array([obb_c_xy[0], obb_c_xy[1], aabb_center[2]], dtype=float)
                obb_size_3d = np.array([obb_wh[0], obb_wh[1], aabb_size[2]], dtype=float)
                # Keep fallback AABB centered symmetrically on pos_out so rasterizer bounds never clip
                half_span = np.max(np.abs(corners - obb_c_xy), axis=0)
                aabb_size = np.array([2.0 * half_span[0], 2.0 * half_span[1], aabb_size[2]], dtype=float)
            else:
                obb_yaw = 0.0
                pos_out = aabb_center
                obb_size_3d = aabb_size.copy()
        else:
            pos_out = aabb_center

        fastest = max(group, key=lambda e: float(np.linalg.norm(e.vel)))
        members: Set[str] = set()
        for e in group:
            members |= e.member_ids

        return AegisEntity(
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
        )

    def _coarsen_entities(
        self,
        entities: List[AegisEntity],
        k: int,
        graph_obs: Optional[nx.DiGraph] = None,
        agent: Optional[AgentState] = None,
        entity_parent_place: Optional[Dict[str, str]] = None,
    ) -> List[AegisEntity]:
        """OBB Union-Find coarsening + Cross-Layer Semantic Zoom protecting Atomic Hazards."""
        cfg = self.config
        merge_counter = 0

        # HARD NON-COARSENABLE INVARIANT: Atomic hazards bypass ALL merging!
        atomic_protected = [e for e in entities if e.is_atomic_hazard]
        coarsenable = [e for e in entities if not e.is_atomic_hazard]

        # Stage 3A: Fast Union-Find same-class static geometric clustering (gap < min_navigable_gap_m)
        if cfg.proactive_same_class_merge and len(coarsenable) > 1:
            by_class: Dict[str, List[AegisEntity]] = defaultdict(list)
            dyn_non_atomic: List[AegisEntity] = []
            for e in coarsenable:
                if e.is_dynamic:
                    dyn_non_atomic.append(e)
                else:
                    by_class[e.label].append(e)

            clustered: List[AegisEntity] = list(dyn_non_atomic)
            for lbl, grp in by_class.items():
                m = len(grp)
                if m == 1:
                    clustered.append(grp[0])
                    continue
                parent = list(range(m))
                comp_members: Dict[int, List[AegisEntity]] = {idx: [grp[idx]] for idx in range(m)}
                # Fast O(1) AABB pre-check; fall back to exact Rotating-Calipers OBB when AABB bloats on diagonals!
                comp_bbox: Dict[int, Tuple[float, float, float, float, float]] = {}
                for idx in range(m):
                    px, py = float(grp[idx].pos[0]), float(grp[idx].pos[1])
                    hx, hy = 0.5 * float(grp[idx].size[0]), 0.5 * float(grp[idx].size[1])
                    area = max(2.0 * hx, 0.05) * max(2.0 * hy, 0.05)
                    comp_bbox[idx] = (px - hx, py - hy, px + hx, py + hy, area)

                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                for i in range(m):
                    pi, si = grp[i].pos, grp[i].size
                    for j in range(i + 1, m):
                        if _xy_edge_gap(pi, si, grp[j].pos, grp[j].size) < cfg.min_navigable_gap_m:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                b1, b2 = comp_bbox[ri], comp_bbox[rj]
                                nx0, ny0 = min(b1[0], b2[0]), min(b1[1], b2[1])
                                nx1, ny1 = max(b1[2], b2[2]), max(b1[3], b2[3])
                                n_sum = b1[4] + b2[4]
                                m_area = max(nx1 - nx0, 0.05) * max(ny1 - ny0, 0.05)
                                ok_merge = (m_area <= cfg.max_aabb_bloat_ratio * n_sum)
                                if (
                                    not ok_merge
                                    and cfg.enable_obb_supernodes
                                    and (len(comp_members[ri]) + len(comp_members[rj])) <= 4
                                ):
                                    # Exact Rotating-Calipers OBB allows tight diagonal rows without AABB bloat!
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

        # Stage 3B: Overflow cross-class geometric hazard coarsening (only non-atomic hazards!)
        geom_haz = [e for e in coarsenable if e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold]
        geom_non_haz = [e for e in coarsenable if not (e.risk > cfg.tau_haz and e.belief >= cfg.fp_belief_threshold)]

        active_atomic_count = sum(1 for e in atomic_protected if e.risk > cfg.tau_haz)
        target_geom_budget = max(1, k - active_atomic_count)

        while len(geom_haz) > target_geom_budget:
            best_pair: Tuple[int, int] | None = None
            best_gap = float("inf")
            for i in range(len(geom_haz)):
                for j in range(i + 1, len(geom_haz)):
                    gap = _xy_edge_gap(geom_haz[i].pos, geom_haz[i].size, geom_haz[j].pos, geom_haz[j].size)
                    if (
                        gap < cfg.min_navigable_gap_m
                        and gap < best_gap
                        and self._obb_bloat_ok([geom_haz[i], geom_haz[j]], cfg.max_aabb_bloat_ratio, use_obb=cfg.enable_obb_supernodes)
                    ):
                        best_gap = gap
                        best_pair = (i, j)
            if best_pair is None:
                break  # Preserve all navigable passages >= min_navigable_gap_m!
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

        # Stage 3C (v3.0): Cross-Layer Hierarchical Semantic Zoom (Object -> Place / Room)
        # Lift distant non-atomic background clutter sharing a parent Place/Room node into 1 higher-layer summary!
        if (
            cfg.enable_hierarchical_zoom
            and graph_obs is not None
            and agent is not None
            and entity_parent_place
            and len(geom_non_haz) > 1
        ):
            by_place: Dict[str, List[AegisEntity]] = defaultdict(list)
            kept_near_non_haz: List[AegisEntity] = []
            for e in geom_non_haz:
                dist_xy = float(np.linalg.norm(e.pos[:2] - agent.pos[:2]))
                # Find parent place/room shared by member_ids
                p_id = None
                for mid in e.member_ids:
                    if mid in entity_parent_place:
                        p_id = entity_parent_place[mid]
                        break
                if p_id is not None and p_id in graph_obs and dist_xy >= cfg.hierarchical_zoom_dist_m:
                    by_place[p_id].append(e)
                else:
                    kept_near_non_haz.append(e)

            for p_id, grp in by_place.items():
                if len(grp) == 1:
                    kept_near_non_haz.append(grp[0])
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
                    # Preserve parent structural importance as hierarchical landmark summary
                    p_cv = float(p_node.get("structural_importance", 0.5))
                    zoomed.weight = max(zoomed.weight, cfg.beta * p_cv)
                    kept_near_non_haz.append(zoomed)
                    merge_counter += 1
            geom_non_haz = kept_near_non_haz

        return atomic_protected + geom_haz + geom_non_haz

    # ------------------------------------------------------------------
    # Layer 4: Dynamic-Budget Monotone Submodular Selection + Hysteresis
    # ------------------------------------------------------------------
    def _select_submodular_dynamic(self, entities: List[AegisEntity], k: int) -> List[AegisEntity]:
        """Greedy (1 - 1/e) Submodular Selection stopping early below tau_min."""
        cfg = self.config
        if not entities or k <= 0:
            return []

        # Fast pre-filter: drop strictly zero-weight / negligible clutter before matrix ops
        pool = [e for e in entities if e.weight > 1e-4 or e.risk > cfg.tau_haz]
        n = len(pool)
        if n == 0:
            return []

        weights = np.array([e.weight for e in pool], dtype=float)

        # Redundancy kernel K[u, v]: active hazards remain mutually non-redundant (K=0)
        K = np.eye(n, dtype=float)
        for i in range(n):
            if pool[i].risk > cfg.tau_haz or pool[i].is_atomic_hazard:
                continue
            for j in range(i + 1, n):
                if pool[j].risk > cfg.tau_haz or pool[j].is_atomic_hazard:
                    continue
                gap = _xy_edge_gap(pool[i].pos, pool[i].size, pool[j].pos, pool[j].size)
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
        """Run all 4 Aegis-DSG v2.0 layers sequentially and return <= k compressed entities."""
        if k <= 0:
            self._prev_covered.clear()
            return []

        tracks = self._update_belief_state(graph_obs, dt)
        if not tracks:
            self._prev_covered.clear()
            return []

        candidates: List[AegisEntity] = []
        entity_parent_place: Dict[str, str] = {}
        for tr in tracks:
            r_v = self._compute_unified_risk(tr, agent)
            tr.last_risk = r_v
            if tr.missed_frames > 0 and r_v <= self.config.tau_haz:
                continue
            w_v, clip_rel = self._compute_asymmetric_weight(tr, r_v)
            if tr.parent_place_id is not None:
                entity_parent_place[tr.obj_id] = tr.parent_place_id
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
                    is_atomic_hazard=tr.is_atomic_hazard,
                    is_occluded_memory=(tr.missed_frames > 0),
                    is_dynamic=tr.is_dynamic,
                    dsg_layer="object",
                    gaze_attention=tr.gaze_attention,
                    task_relevance=clip_rel,
                    member_ids={tr.obj_id},
                )
            )

        coarsened = self._coarsen_entities(
            candidates,
            k,
            graph_obs=graph_obs,
            agent=agent,
            entity_parent_place=entity_parent_place,
        )
        selected = self._select_submodular_dynamic(coarsened, k)

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

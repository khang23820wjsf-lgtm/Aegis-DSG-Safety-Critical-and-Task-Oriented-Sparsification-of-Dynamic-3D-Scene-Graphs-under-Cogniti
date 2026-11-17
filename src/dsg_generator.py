"""Synthetic Dynamic 3D Scene Graph (DSG) Sequence Generator for Phase A.

Generates a sequence of hierarchical 3D scene graphs G_t = (V_t, E_t, X_t)
along a 3D indoor corridor with static obstacles, structural anchors,
benign distractors, and moving pedestrians. Produces both ground-truth
(G_t^gt) and noisy egocentric observations (G_t^obs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import math
import networkx as nx
import numpy as np


NoiseLevel = Literal["clean", "medium", "heavy"]

NOISE_CONFIGS: Dict[NoiseLevel, Dict[str, float]] = {
    "clean": {"pos_std": 0.02, "dropout_base": 0.01, "fp_rate": 0.00, "conf_noise": 0.03},
    "medium": {"pos_std": 0.08, "dropout_base": 0.08, "fp_rate": 0.04, "conf_noise": 0.12},
    "heavy": {"pos_std": 0.18, "dropout_base": 0.18, "fp_rate": 0.10, "conf_noise": 0.22},
}

# Object templates: (size_xyz, z_center, structural_importance C_v, hazard_severity, is_dynamic)
OBJECT_SPECS: Dict[str, Tuple[Tuple[float, float, float], float, float, float, bool]] = {
    "chair": ((0.55, 0.55, 0.85), 0.425, 0.15, 0.75, False),
    "table": ((1.10, 0.70, 0.75), 0.375, 0.30, 0.80, False),
    "low_barrier": ((0.90, 0.20, 1.15), 0.575, 0.20, 0.95, False),
    "stair_edge": ((1.80, 0.35, 0.25), 0.125, 0.90, 1.00, False),
    "door": ((0.20, 1.00, 2.10), 1.050, 0.95, 0.30, False),
    "wall_pillar": ((0.45, 0.45, 2.80), 1.400, 0.70, 0.85, False),
    "pedestrian": ((0.50, 0.45, 1.72), 0.860, 0.10, 0.95, True),
    "wall_painting": ((0.06, 0.90, 0.65), 1.950, 0.05, 0.00, False),
    "ceiling_light": ((0.50, 0.50, 0.12), 2.750, 0.05, 0.00, False),
    "shelf": ((0.40, 1.20, 1.80), 0.900, 0.45, 0.50, False),
}


@dataclass
class AgentState:
    """Egocentric user/agent kinematics in 3D world coordinates (meters)."""

    pos: np.ndarray  # shape (3,) -> (x, y, z)
    vel: np.ndarray  # shape (3,) -> (vx, vy, vz)
    radius: float = 0.35
    height: float = 1.70
    fov_half_angle_rad: float = math.radians(70.0)
    max_range: float = 12.0


@dataclass
class SceneObject:
    """Underlying 3D entity state used to build G_t^gt and G_t^obs."""

    obj_id: str
    label: str
    pos: np.ndarray
    size: np.ndarray
    vel: np.ndarray
    structural_importance: float
    hazard_severity: float
    is_dynamic: bool
    place_id: str


@dataclass
class DSGFrame:
    """Single timestep snapshot containing ground-truth and observed 3D scene graphs."""

    t: int
    timestamp: float
    agent: AgentState
    graph_gt: nx.DiGraph
    graph_obs: nx.DiGraph
    metadata: Dict[str, float] = field(default_factory=dict)


def _create_corridor_objects(
    rng: np.random.Generator,
    corridor_length: float,
    corridor_width: float,
    num_objects: int,
    place_ids: List[str],
) -> List[SceneObject]:
    """Populate a 3D corridor with 30-50 diverse static, structural, and dynamic objects."""
    objects: List[SceneObject] = []

    def assign_place(y: float) -> str:
        idx = min(int((y / corridor_length) * len(place_ids)), len(place_ids) - 1)
        return place_ids[max(0, idx)]

    # 1. Always include a few critical structural hazards (stair edge, doors, low barriers)
    critical_anchors = [
        ("stair_edge", np.array([corridor_width * 0.5, corridor_length - 2.5, 0.125])),
        ("door", np.array([0.12, corridor_length * 0.35, 1.05])),
        ("door", np.array([corridor_width - 0.12, corridor_length * 0.70, 1.05])),
    ]
    for i, (label, pos) in enumerate(critical_anchors):
        size, _, c_v, haz, is_dyn = OBJECT_SPECS[label]
        objects.append(
            SceneObject(
                obj_id=f"obj_{label}_{i}",
                label=label,
                pos=pos.astype(float),
                size=np.array(size, dtype=float),
                vel=np.zeros(3, dtype=float),
                structural_importance=c_v,
                hazard_severity=haz,
                is_dynamic=is_dyn,
                place_id=assign_place(pos[1]),
            )
        )

    # 2. Sample remaining objects up to num_objects
    label_pool = [
        "chair",
        "table",
        "low_barrier",
        "wall_pillar",
        "pedestrian",
        "wall_painting",
        "ceiling_light",
        "shelf",
    ]
    label_probs = [0.16, 0.12, 0.10, 0.08, 0.16, 0.16, 0.12, 0.10]

    for idx in range(len(objects), num_objects):
        label = str(rng.choice(label_pool, p=label_probs))
        size, z_center, c_v, haz, is_dyn = OBJECT_SPECS[label]
        y = float(rng.uniform(2.0, corridor_length - 1.5))

        if label in ("wall_painting", "shelf"):
            on_left = bool(rng.integers(0, 2))
            x = (size[0] / 2.0 + 0.05) if on_left else (corridor_width - size[0] / 2.0 - 0.05)
        elif label == "ceiling_light":
            x = float(rng.uniform(0.8, corridor_width - 0.8))
        else:
            x = float(rng.uniform(0.45, corridor_width - 0.45))

        if is_dyn:
            # Pedestrians walk toward the agent or slightly diagonally across corridor
            vy = float(rng.uniform(-1.5, -0.7))
            vx = float(rng.uniform(-0.25, 0.25))
            vel = np.array([vx, vy, 0.0], dtype=float)
        else:
            vel = np.zeros(3, dtype=float)

        objects.append(
            SceneObject(
                obj_id=f"obj_{label}_{idx}",
                label=label,
                pos=np.array([x, y, z_center], dtype=float),
                size=np.array(size, dtype=float),
                vel=vel,
                structural_importance=c_v,
                hazard_severity=haz,
                is_dynamic=is_dyn,
                place_id=assign_place(y),
            )
        )

    return objects


def _is_future_hazard(
    agent: AgentState,
    obj: SceneObject,
    horizon_s: float = 3.0,
    safety_margin: float = 0.25,
) -> bool:
    """Check ground-truth collision risk over [t, t + horizon_s] for Hazard Recall."""
    if obj.hazard_severity <= 0.0:
        return False

    # Vertical overlap check: ignore overhead objects above agent height + margin
    obj_z_min = obj.pos[2] - obj.size[2] / 2.0
    obj_z_max = obj.pos[2] + obj.size[2] / 2.0
    if obj_z_min > agent.height + 0.15 or obj_z_max < 0.0:
        return False

    # Closest approach in XY plane under constant-velocity rollout
    rel_pos = obj.pos[:2] - agent.pos[:2]
    rel_vel = obj.vel[:2] - agent.vel[:2]
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))

    if rel_speed_sq < 1e-6:
        t_star = 0.0
    else:
        t_star = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon_s))

    closest_xy = np.abs(rel_pos + rel_vel * t_star)
    half_extents = obj.size[:2] / 2.0 + agent.radius + safety_margin
    return bool(np.all(closest_xy <= half_extents))


def _build_hierarchy_skeleton(
    corridor_length: float,
    corridor_width: float,
    place_ids: List[str],
) -> nx.DiGraph:
    """Create Room -> Place hierarchical skeleton of the 3D Scene Graph."""
    g = nx.DiGraph()
    g.add_node(
        "room_corridor_0",
        layer="room",
        label="corridor",
        pos=np.array([corridor_width / 2.0, corridor_length / 2.0, 1.5]),
        size=np.array([corridor_width, corridor_length, 3.0]),
        vel=np.zeros(3),
        confidence=1.0,
        structural_importance=1.0,
        is_hazard_gt=False,
    )

    zone_len = corridor_length / len(place_ids)
    for i, pid in enumerate(place_ids):
        g.add_node(
            pid,
            layer="place",
            label="navigable_zone" if i < len(place_ids) - 1 else "stair_zone",
            pos=np.array([corridor_width / 2.0, (i + 0.5) * zone_len, 0.0]),
            size=np.array([corridor_width, zone_len, 0.1]),
            vel=np.zeros(3),
            confidence=1.0,
            structural_importance=0.85,
            is_hazard_gt=False,
        )
        g.add_edge(pid, "room_corridor_0", relation="belongs_to")
        if i > 0:
            g.add_edge(place_ids[i - 1], pid, relation="connects_to")
            g.add_edge(pid, place_ids[i - 1], relation="connects_to")
    return g


def _add_spatial_edges(g: nx.DiGraph, obj_ids: List[str], near_threshold: float = 1.6) -> None:
    """Add proximity ('near') edges between nearby object nodes."""
    for i in range(len(obj_ids)):
        u = obj_ids[i]
        pos_u = g.nodes[u]["pos"]
        for j in range(i + 1, len(obj_ids)):
            v = obj_ids[j]
            if float(np.linalg.norm(pos_u - g.nodes[v]["pos"])) <= near_threshold:
                g.add_edge(u, v, relation="near")
                g.add_edge(v, u, relation="near")


def generate_dsg_sequence(
    num_frames: int = 100,
    dt: float = 0.1,
    num_objects: int = 40,
    corridor_length: float = 24.0,
    corridor_width: float = 4.0,
    noise_level: NoiseLevel = "medium",
    seed: int = 42,
) -> List[DSGFrame]:
    """Generate a sequence of dynamic 3D Scene Graphs (G_t^gt and G_t^obs).

    Args:
        num_frames: Number of discrete timesteps T.
        dt: Time step duration in seconds (default 0.1s = 10 Hz).
        num_objects: Total object count in the 3D scene (recommended 30-50).
        corridor_length: Length of corridor along +Y axis (meters).
        corridor_width: Width of corridor along X axis (meters).
        noise_level: Observation noise severity ('clean', 'medium', 'heavy').
        seed: Random seed for reproducibility.

    Returns:
        List of DSGFrame objects ordered by t = 0 .. num_frames - 1.
    """
    rng = np.random.default_rng(seed)
    noise_cfg = NOISE_CONFIGS[noise_level]
    place_ids = ["place_entry", "place_mid_1", "place_mid_2", "place_stairs"]
    objects = _create_corridor_objects(rng, corridor_length, corridor_width, num_objects, place_ids)

    sequence: List[DSGFrame] = []
    agent_pos = np.array([corridor_width / 2.0, 1.0, 0.85], dtype=float)
    base_speed = 1.35  # m/s walking speed

    for t in range(num_frames):
        # Smooth lateral sway to simulate natural human walking + slight steering
        vx = 0.18 * math.cos(0.25 * t * dt * 2.0 * math.pi)
        vy = base_speed
        agent = AgentState(
            pos=agent_pos.copy(),
            vel=np.array([vx, vy, 0.0], dtype=float),
        )

        g_gt = _build_hierarchy_skeleton(corridor_length, corridor_width, place_ids)
        g_obs = _build_hierarchy_skeleton(corridor_length, corridor_width, place_ids)
        gt_obj_ids: List[str] = []
        obs_obj_ids: List[str] = []

        for obj in objects:
            # Update object's place assignment for dynamic objects
            zone_idx = min(int((obj.pos[1] / corridor_length) * len(place_ids)), len(place_ids) - 1)
            obj.place_id = place_ids[max(0, zone_idx)]
            is_haz = _is_future_hazard(agent, obj)

            g_gt.add_node(
                obj.obj_id,
                layer="object",
                label=obj.label,
                pos=obj.pos.copy(),
                size=obj.size.copy(),
                vel=obj.vel.copy(),
                confidence=1.0,
                structural_importance=obj.structural_importance,
                hazard_severity=obj.hazard_severity,
                is_dynamic=obj.is_dynamic,
                is_hazard_gt=is_haz,
            )
            g_gt.add_edge(obj.obj_id, obj.place_id, relation="belongs_to")
            gt_obj_ids.append(obj.obj_id)

            # Egocentric visibility check (FOV + max_range)
            delta_xy = obj.pos[:2] - agent.pos[:2]
            dist = float(np.linalg.norm(delta_xy))
            if dist > agent.max_range or dist < 1e-4:
                continue

            heading = agent.vel[:2] / (np.linalg.norm(agent.vel[:2]) + 1e-8)
            cos_angle = float(np.clip(np.dot(delta_xy / dist, heading), -1.0, 1.0))
            angle = math.acos(cos_angle)
            if angle > agent.fov_half_angle_rad:
                continue

            # Edge-of-FOV & distance-based dropout and confidence degradation
            fov_ratio = angle / agent.fov_half_angle_rad
            dropout_prob = noise_cfg["dropout_base"] * (0.5 + 1.5 * (fov_ratio ** 2))
            if rng.random() < dropout_prob:
                continue

            noisy_pos = obj.pos + rng.normal(0.0, noise_cfg["pos_std"], size=3)
            noisy_vel = obj.vel + rng.normal(0.0, noise_cfg["pos_std"] * 0.8, size=3) if obj.is_dynamic else np.zeros(3)
            raw_conf = 0.98 - 0.35 * (fov_ratio ** 2) - 0.15 * (dist / agent.max_range)
            conf = float(np.clip(raw_conf + rng.normal(0.0, noise_cfg["conf_noise"]), 0.20, 0.99))

            g_obs.add_node(
                obj.obj_id,
                layer="object",
                label=obj.label,
                pos=noisy_pos,
                size=obj.size.copy(),
                vel=noisy_vel,
                confidence=conf,
                structural_importance=obj.structural_importance,
                hazard_severity=obj.hazard_severity,
                is_dynamic=obj.is_dynamic,
                is_hazard_gt=is_haz,
            )
            g_obs.add_edge(obj.obj_id, obj.place_id, relation="belongs_to")
            obs_obj_ids.append(obj.obj_id)

        # Occasional false positive ghost obstacles in G_t^obs
        if rng.random() < noise_cfg["fp_rate"] * 3.0:
            fp_id = f"fp_ghost_{t}"
            fp_y = agent.pos[1] + float(rng.uniform(2.0, 7.0))
            fp_x = float(rng.uniform(0.5, corridor_width - 0.5))
            g_obs.add_node(
                fp_id,
                layer="object",
                label="unknown_ghost",
                pos=np.array([fp_x, fp_y, 0.5], dtype=float),
                size=np.array([0.4, 0.4, 0.8], dtype=float),
                vel=np.zeros(3),
                confidence=float(rng.uniform(0.25, 0.50)),
                structural_importance=0.05,
                hazard_severity=0.40,
                is_dynamic=False,
                is_hazard_gt=False,
            )
            obs_obj_ids.append(fp_id)

        _add_spatial_edges(g_gt, gt_obj_ids)
        _add_spatial_edges(g_obs, obs_obj_ids)

        sequence.append(
            DSGFrame(
                t=t,
                timestamp=round(t * dt, 3),
                agent=agent,
                graph_gt=g_gt,
                graph_obs=g_obs,
                metadata={
                    "num_gt_objects": float(len(gt_obj_ids)),
                    "num_obs_objects": float(len(obs_obj_ids)),
                    "num_gt_hazards": float(sum(1 for n in gt_obj_ids if g_gt.nodes[n]["is_hazard_gt"])),
                },
            )
        )

        # Step kinematics forward for t + 1
        agent_pos = agent_pos + agent.vel * dt
        agent_pos[0] = float(np.clip(agent_pos[0], 0.5, corridor_width - 0.5))
        if agent_pos[1] >= corridor_length - 1.5:
            break

        for obj in objects:
            if obj.is_dynamic:
                obj.pos = obj.pos + obj.vel * dt
                # Bounce pedestrians softly off corridor walls
                if obj.pos[0] < 0.4 or obj.pos[0] > corridor_width - 0.4:
                    obj.vel[0] *= -1.0
                    obj.pos[0] = float(np.clip(obj.pos[0], 0.4, corridor_width - 0.4))
                if obj.pos[1] < 0.5:
                    obj.pos[1] = corridor_length - 1.5

    return sequence

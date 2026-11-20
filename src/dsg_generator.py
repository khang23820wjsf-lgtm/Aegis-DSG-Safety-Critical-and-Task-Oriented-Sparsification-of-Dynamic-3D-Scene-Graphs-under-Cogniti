"""Synthetic Dynamic 3D Scene Graph (DSG) Sequence Generator for Phase A.

Supports 4 diverse 3D indoor scenarios:
  - 'corridor': Straight 4m x 24m hallway with oncoming pedestrians and static hazards.
  - 'l_turn_corner': L-shaped hallway with a 90-degree blind corner and wall occlusion.
  - 'crowded_lobby': Wide 10m x 18m open hall with dense clusters and cross-traffic pedestrians.
  - 'vertical_hazards': 3D height-stratified hallway (head-height signs, stair drops, ceiling lights, floor mats).

Produces both ground-truth (G_t^gt) and noisy egocentric observations (G_t^obs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import math
import networkx as nx
import numpy as np


NoiseLevel = Literal["clean", "medium", "heavy"]
ScenarioType = Literal["corridor", "l_turn_corner", "crowded_lobby", "vertical_hazards"]

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
    "overhead_sign": ((1.10, 0.14, 0.40), 1.550, 0.35, 0.95, False),  # Head-height hazard (z: 1.35..1.75m)
    "door": ((0.20, 1.00, 2.10), 1.050, 0.95, 0.30, False),
    "wall_pillar": ((0.45, 0.45, 2.80), 1.400, 0.70, 0.85, False),
    "pedestrian": ((0.50, 0.45, 1.72), 0.860, 0.10, 0.95, True),
    "wall_painting": ((0.06, 0.90, 0.65), 2.150, 0.05, 0.00, False),  # Above head height
    "ceiling_light": ((0.50, 0.50, 0.12), 2.750, 0.05, 0.00, False),  # High overhead
    "floor_mat": ((1.20, 0.80, 0.02), 0.010, 0.05, 0.00, False),      # Flat harmless rug
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
    gaze_dir: Optional[np.ndarray] = None  # optional unit 3D world eye-gaze ray (x, y, z)


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


def _make_obj(
    obj_id: str,
    label: str,
    pos: np.ndarray,
    place_id: str,
    vel: np.ndarray | None = None,
) -> SceneObject:
    size, _, c_v, haz, is_dyn = OBJECT_SPECS[label]
    return SceneObject(
        obj_id=obj_id,
        label=label,
        pos=pos.astype(float),
        size=np.array(size, dtype=float),
        vel=np.zeros(3, dtype=float) if vel is None else vel.astype(float),
        structural_importance=c_v,
        hazard_severity=haz,
        is_dynamic=is_dyn,
        place_id=place_id,
    )


def _create_corridor_objects(
    rng: np.random.Generator,
    corridor_length: float,
    corridor_width: float,
    num_objects: int,
    place_ids: List[str],
) -> List[SceneObject]:
    """Populate a straight 3D corridor with static, structural, and oncoming dynamic objects."""
    objects: List[SceneObject] = []

    def assign_place(y: float) -> str:
        idx = min(int((y / corridor_length) * len(place_ids)), len(place_ids) - 1)
        return place_ids[max(0, idx)]

    critical_anchors = [
        ("stair_edge", np.array([corridor_width * 0.5, corridor_length - 2.5, 0.125])),
        ("door", np.array([0.12, corridor_length * 0.35, 1.05])),
        ("door", np.array([corridor_width - 0.12, corridor_length * 0.70, 1.05])),
    ]
    for i, (label, pos) in enumerate(critical_anchors):
        objects.append(_make_obj(f"obj_{label}_{i}", label, pos, assign_place(pos[1])))

    label_pool = [
        "chair", "table", "low_barrier", "wall_pillar",
        "pedestrian", "wall_painting", "ceiling_light", "shelf",
    ]
    label_probs = [0.16, 0.12, 0.10, 0.08, 0.16, 0.16, 0.12, 0.10]

    for idx in range(len(objects), num_objects):
        label = str(rng.choice(label_pool, p=label_probs))
        size, z_center, _, _, is_dyn = OBJECT_SPECS[label]
        y = float(rng.uniform(2.0, corridor_length - 1.5))

        if label in ("wall_painting", "shelf"):
            on_left = bool(rng.integers(0, 2))
            x = (size[0] / 2.0 + 0.05) if on_left else (corridor_width - size[0] / 2.0 - 0.05)
        elif label == "ceiling_light":
            x = float(rng.uniform(0.8, corridor_width - 0.8))
        else:
            x = float(rng.uniform(0.45, corridor_width - 0.45))

        vel = np.array([float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-1.5, -0.7)), 0.0]) if is_dyn else None
        objects.append(_make_obj(f"obj_{label}_{idx}", label, np.array([x, y, z_center]), assign_place(y), vel))

    return objects


def _create_l_turn_objects(
    rng: np.random.Generator,
    num_objects: int,
    place_ids: List[str],
) -> List[SceneObject]:
    """L-shaped hallway: Leg 1 (x in [0,4], y in [0,13]), Leg 2 (x in [4,16], y in [9,13])."""
    objects: List[SceneObject] = []

    # Guaranteed blind-corner surprises right after the 90-degree right turn
    blind_surprises = [
        ("low_barrier", np.array([5.8, 11.1, 0.575]), None),
        ("pedestrian", np.array([9.5, 11.0, 0.860]), np.array([-1.25, 0.0, 0.0])),
        ("chair", np.array([7.2, 10.6, 0.425]), None),
        ("stair_edge", np.array([14.2, 11.0, 0.125]), None),
        ("door", np.array([0.12, 6.0, 1.05]), None),
        ("door", np.array([10.5, 12.88, 1.05]), None),
    ]
    for i, (label, pos, vel) in enumerate(blind_surprises):
        pid = place_ids[2] if pos[0] > 4.0 else place_ids[1]
        objects.append(_make_obj(f"obj_{label}_{i}", label, pos, pid, vel))

    label_pool = ["chair", "table", "low_barrier", "pedestrian", "wall_painting", "ceiling_light", "shelf"]
    label_probs = [0.18, 0.12, 0.10, 0.18, 0.16, 0.14, 0.12]

    for idx in range(len(objects), num_objects):
        label = str(rng.choice(label_pool, p=label_probs))
        _, z_center, _, _, is_dyn = OBJECT_SPECS[label]
        in_leg2 = bool(rng.random() < 0.55)

        if not in_leg2:
            x = float(rng.uniform(0.45, 3.55))
            y = float(rng.uniform(2.5, 12.2))
            pid = place_ids[0] if y < 6.5 else place_ids[1]
            vel = np.array([float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-1.3, -0.7)), 0.0]) if is_dyn else None
        else:
            x = float(rng.uniform(4.5, 15.0))
            y = float(rng.uniform(9.45, 12.55))
            pid = place_ids[2] if x < 10.0 else place_ids[3]
            vel = np.array([float(rng.uniform(-1.4, -0.7)), float(rng.uniform(-0.15, 0.15)), 0.0]) if is_dyn else None

        objects.append(_make_obj(f"obj_{label}_{idx}", label, np.array([x, y, z_center]), pid, vel))

    return objects


def _create_crowded_lobby_objects(
    rng: np.random.Generator,
    num_objects: int,
    place_ids: List[str],
) -> List[SceneObject]:
    """Wide 10m x 18m open lobby with dense furniture clusters and crossing pedestrians."""
    objects: List[SceneObject] = []
    width, length = 10.0, 18.0

    def assign_place(y: float) -> str:
        idx = min(int((y / length) * len(place_ids)), len(place_ids) - 1)
        return place_ids[max(0, idx)]

    # Structural anchors (pillars & exit doors)
    anchors = [
        ("door", np.array([5.0, length - 0.2, 1.05])),
        ("wall_pillar", np.array([3.2, 7.0, 1.40])),
        ("wall_pillar", np.array([6.8, 7.0, 1.40])),
        ("wall_pillar", np.array([3.2, 12.5, 1.40])),
        ("wall_pillar", np.array([6.8, 12.5, 1.40])),
    ]
    for i, (label, pos) in enumerate(anchors):
        objects.append(_make_obj(f"obj_{label}_{i}", label, pos, assign_place(pos[1])))

    # High density of cross-traffic pedestrians (~32%) + clustered chairs/tables
    label_pool = ["pedestrian", "chair", "table", "low_barrier", "ceiling_light", "floor_mat"]
    label_probs = [0.34, 0.24, 0.16, 0.08, 0.10, 0.08]

    for idx in range(len(objects), num_objects):
        label = str(rng.choice(label_pool, p=label_probs))
        _, z_center, _, _, is_dyn = OBJECT_SPECS[label]
        x = float(rng.uniform(0.8, width - 0.8))
        y = float(rng.uniform(2.5, length - 1.5))

        vel = None
        if is_dyn:
            # Mix of cross-traffic (+X/-X) and oncoming (-Y)
            mode = int(rng.integers(0, 3))
            if mode == 0:
                vel = np.array([float(rng.uniform(0.8, 1.5)), float(rng.uniform(-0.4, 0.2)), 0.0])
            elif mode == 1:
                vel = np.array([float(rng.uniform(-1.5, -0.8)), float(rng.uniform(-0.4, 0.2)), 0.0])
            else:
                vel = np.array([float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-1.4, -0.8)), 0.0])

        objects.append(_make_obj(f"obj_{label}_{idx}", label, np.array([x, y, z_center]), assign_place(y), vel))

    return objects


def _create_vertical_hazards_objects(
    rng: np.random.Generator,
    corridor_length: float,
    corridor_width: float,
    num_objects: int,
    place_ids: List[str],
) -> List[SceneObject]:
    """3D height-stratified hallway testing overhead head-height signs vs harmless ceiling/floor distractors."""
    objects: List[SceneObject] = []

    def assign_place(y: float) -> str:
        idx = min(int((y / corridor_length) * len(place_ids)), len(place_ids) - 1)
        return place_ids[max(0, idx)]

    # Guaranteed vertical 3D hazards along the center path
    vertical_anchors = [
        ("overhead_sign", np.array([2.0, 6.5, 1.55])),
        ("stair_edge", np.array([2.0, 11.0, 0.125])),
        ("overhead_sign", np.array([2.0, 15.5, 1.55])),
        ("low_barrier", np.array([2.1, 18.5, 0.575])),
        ("stair_edge", np.array([2.0, 21.5, 0.125])),
    ]
    for i, (label, pos) in enumerate(vertical_anchors):
        objects.append(_make_obj(f"obj_{label}_{i}", label, pos, assign_place(pos[1])))

    # Heavy mix of harmless overhead/floor distractors (2D filters fail here!) + real 3D hazards
    label_pool = [
        "overhead_sign", "low_barrier", "stair_edge", "pedestrian", "chair",
        "ceiling_light", "wall_painting", "floor_mat",
    ]
    label_probs = [0.12, 0.10, 0.06, 0.10, 0.10, 0.20, 0.16, 0.16]

    for idx in range(len(objects), num_objects):
        label = str(rng.choice(label_pool, p=label_probs))
        size, z_center, _, _, is_dyn = OBJECT_SPECS[label]
        y = float(rng.uniform(2.0, corridor_length - 1.5))

        if label == "wall_painting":
            on_left = bool(rng.integers(0, 2))
            x = (size[0] / 2.0 + 0.05) if on_left else (corridor_width - size[0] / 2.0 - 0.05)
        else:
            x = float(rng.uniform(0.6, corridor_width - 0.6))

        vel = np.array([float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-1.3, -0.7)), 0.0]) if is_dyn else None
        objects.append(_make_obj(f"obj_{label}_{idx}", label, np.array([x, y, z_center]), assign_place(y), vel))

    return objects


def _is_future_hazard(
    agent: AgentState,
    obj: SceneObject,
    horizon_s: float = 3.0,
    safety_margin: float = 0.25,
) -> bool:
    """Check ground-truth 3D collision risk over [t, t + horizon_s] for Hazard Recall."""
    if obj.hazard_severity <= 0.0:
        return False

    # Vertical overlap check: ignore overhead objects above agent height + margin
    obj_z_min = obj.pos[2] - obj.size[2] / 2.0
    obj_z_max = obj.pos[2] + obj.size[2] / 2.0
    if obj_z_min > agent.height + 0.15 or obj_z_max < 0.05:
        return False

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


def _is_occluded_by_l_corner(agent_pos: np.ndarray, obj_pos: np.ndarray) -> bool:
    """Simple ray-block check for the inner corner wall (x >= 4.0, y <= 9.0) in l_turn_corner."""
    ax, ay = float(agent_pos[0]), float(agent_pos[1])
    ox, oy = float(obj_pos[0]), float(obj_pos[1])
    # If agent is still in leg 1 (y < 9.0) and object is around the corner in leg 2 (x > 4.0)
    if ay < 9.0 and ox > 4.0:
        # Check where line segment (a -> o) crosses x = 4.0
        if abs(ox - ax) > 1e-6:
            t_cross = (4.0 - ax) / (ox - ax)
            if 0.0 < t_cross < 1.0:
                y_cross = ay + t_cross * (oy - ay)
                if y_cross <= 9.0:
                    return True
    return False


def _build_hierarchy_skeleton(
    bounds_x: float,
    bounds_y: float,
    place_ids: List[str],
    scenario: ScenarioType,
) -> nx.DiGraph:
    """Create Room -> Place hierarchical skeleton of the 3D Scene Graph."""
    g = nx.DiGraph()
    g.add_node(
        "room_0",
        layer="room",
        label=scenario,
        pos=np.array([bounds_x / 2.0, bounds_y / 2.0, 1.5]),
        size=np.array([bounds_x, bounds_y, 3.0]),
        vel=np.zeros(3),
        confidence=1.0,
        structural_importance=1.0,
        is_hazard_gt=False,
    )

    zone_len = bounds_y / len(place_ids)
    for i, pid in enumerate(place_ids):
        if scenario == "l_turn_corner" and i >= 2:
            px, py = 6.5 + (i - 2) * 5.0, 11.0
        else:
            px, py = bounds_x / 2.0, (i + 0.5) * zone_len
        g.add_node(
            pid,
            layer="place",
            label="navigable_zone" if i < len(place_ids) - 1 else "exit_zone",
            pos=np.array([px, py, 0.0]),
            size=np.array([4.0, zone_len, 0.1]),
            vel=np.zeros(3),
            confidence=1.0,
            structural_importance=0.85,
            is_hazard_gt=False,
        )
        g.add_edge(pid, "room_0", relation="belongs_to")
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
    num_frames: int = 120,
    dt: float = 0.1,
    num_objects: int = 42,
    corridor_length: float = 24.0,
    corridor_width: float = 4.0,
    noise_level: NoiseLevel = "medium",
    seed: int = 42,
    scenario: ScenarioType = "corridor",
) -> List[DSGFrame]:
    """Generate a sequence of dynamic 3D Scene Graphs (G_t^gt and G_t^obs).

    Args:
        num_frames: Number of discrete timesteps T.
        dt: Time step duration in seconds (default 0.1s = 10 Hz).
        num_objects: Total object count in the 3D scene.
        corridor_length: Length for corridor-based scenarios (meters).
        corridor_width: Width for corridor-based scenarios (meters).
        noise_level: Observation noise severity ('clean', 'medium', 'heavy').
        seed: Random seed for reproducibility.
        scenario: One of 'corridor', 'l_turn_corner', 'crowded_lobby', 'vertical_hazards'.

    Returns:
        List of DSGFrame objects ordered by t = 0 .. T - 1.
    """
    rng = np.random.default_rng(seed)
    noise_cfg = NOISE_CONFIGS[noise_level]
    place_ids = ["place_entry", "place_mid_1", "place_mid_2", "place_exit"]

    if scenario == "corridor":
        bounds_x, bounds_y = corridor_width, corridor_length
        objects = _create_corridor_objects(rng, bounds_y, bounds_x, num_objects, place_ids)
        agent_pos = np.array([bounds_x / 2.0, 1.0, 0.85], dtype=float)
    elif scenario == "l_turn_corner":
        bounds_x, bounds_y = 16.0, 13.0
        objects = _create_l_turn_objects(rng, num_objects, place_ids)
        agent_pos = np.array([2.0, 1.2, 0.85], dtype=float)
    elif scenario == "crowded_lobby":
        bounds_x, bounds_y = 10.0, 18.0
        objects = _create_crowded_lobby_objects(rng, max(num_objects, 48), place_ids)
        agent_pos = np.array([5.0, 1.0, 0.85], dtype=float)
    elif scenario == "vertical_hazards":
        bounds_x, bounds_y = corridor_width, corridor_length
        objects = _create_vertical_hazards_objects(rng, bounds_y, bounds_x, num_objects, place_ids)
        agent_pos = np.array([bounds_x / 2.0, 1.0, 0.85], dtype=float)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    sequence: List[DSGFrame] = []
    base_speed = 1.35

    for t in range(num_frames):
        # Compute scenario-specific agent velocity
        if scenario == "l_turn_corner":
            if agent_pos[1] < 9.5:
                vel = np.array([0.0, base_speed, 0.0], dtype=float)
            elif agent_pos[0] < 3.5:
                # Smooth 90-degree right turn arc around (2.0, 11.0)
                prog = float(np.clip((agent_pos[1] - 9.5) / 1.5, 0.0, 1.0))
                angle = (math.pi / 2.0) * (1.0 - prog)
                vel = np.array([base_speed * math.cos(angle), base_speed * math.sin(angle), 0.0], dtype=float)
            else:
                vel = np.array([base_speed, 0.0, 0.0], dtype=float)
        else:
            vx = 0.18 * math.cos(0.25 * t * dt * 2.0 * math.pi)
            vel = np.array([vx, base_speed, 0.0], dtype=float)

        agent = AgentState(pos=agent_pos.copy(), vel=vel)
        g_gt = _build_hierarchy_skeleton(bounds_x, bounds_y, place_ids, scenario)
        g_obs = _build_hierarchy_skeleton(bounds_x, bounds_y, place_ids, scenario)
        gt_obj_ids: List[str] = []
        obs_obj_ids: List[str] = []

        for obj in objects:
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

            # Egocentric visibility check (FOV + max_range + corner wall occlusion)
            delta_xy = obj.pos[:2] - agent.pos[:2]
            dist = float(np.linalg.norm(delta_xy))
            if dist > agent.max_range or dist < 1e-4:
                continue
            if scenario == "l_turn_corner" and _is_occluded_by_l_corner(agent.pos, obj.pos):
                continue

            heading = agent.vel[:2] / (np.linalg.norm(agent.vel[:2]) + 1e-8)
            cos_angle = float(np.clip(np.dot(delta_xy / dist, heading), -1.0, 1.0))
            angle = math.acos(cos_angle)
            if angle > agent.fov_half_angle_rad:
                continue

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

        # Occasional false-positive ghost obstacles in G_t^obs
        if rng.random() < noise_cfg["fp_rate"] * 3.0:
            fp_id = f"fp_ghost_{t}"
            heading = agent.vel[:2] / (np.linalg.norm(agent.vel[:2]) + 1e-8)
            ortho = np.array([-heading[1], heading[0]])
            fp_xy = agent.pos[:2] + heading * float(rng.uniform(2.0, 7.0)) + ortho * float(rng.uniform(-1.5, 1.5))
            g_obs.add_node(
                fp_id,
                layer="object",
                label="unknown_ghost",
                pos=np.array([fp_xy[0], fp_xy[1], 0.5], dtype=float),
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
        if scenario == "l_turn_corner":
            if agent_pos[0] >= 15.0:
                break
        else:
            agent_pos[0] = float(np.clip(agent_pos[0], 0.5, bounds_x - 0.5))
            if agent_pos[1] >= bounds_y - 1.5:
                break

        for obj in objects:
            if not obj.is_dynamic:
                continue
            obj.pos = obj.pos + obj.vel * dt
            if scenario == "l_turn_corner":
                if obj.pos[0] > 4.0:
                    obj.pos[1] = float(np.clip(obj.pos[1], 9.4, 12.6))
                    if obj.pos[0] < 4.2:
                        obj.pos[0] = 15.0
                else:
                    obj.pos[0] = float(np.clip(obj.pos[0], 0.4, 3.6))
                    if obj.pos[1] < 0.8:
                        obj.pos[1] = 12.2
            else:
                if obj.pos[0] < 0.4 or obj.pos[0] > bounds_x - 0.4:
                    obj.vel[0] *= -1.0
                    obj.pos[0] = float(np.clip(obj.pos[0], 0.4, bounds_x - 0.4))
                if obj.pos[1] < 0.5:
                    obj.pos[1] = bounds_y - 1.5

    return sequence

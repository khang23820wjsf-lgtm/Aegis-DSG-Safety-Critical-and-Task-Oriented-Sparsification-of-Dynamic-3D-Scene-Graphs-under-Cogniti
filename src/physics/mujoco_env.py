"""MuJoCo 4.0 Real 3D Rigid-Body Physics Environment for 3DSSG Rooms.

Compiles any real 3DSSG room graph into a solid 3D MJCF world where:
- Furniture, walls, and dynamic pedestrians are solid 3D rigid bodies (cannot walk through).
- Collisions are measured directly from MuJoCo's C++ contact solver (data.contact).
- The agent navigates closed-loop using ONLY the <= k compressed entities from a sparsifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Optional, Set, Tuple
import math
import time
import mujoco
import mujoco.viewer
import networkx as nx
import numpy as np

from src.aegis_v3_modules import point_to_obb_clearance_2d
from src.dsg_generator import AgentState, NOISE_CONFIGS, NoiseLevel
from src.phase_b.coarsening import CompressedEntity


@dataclass
class EpisodeMetrics:
    scan_id: str
    method_name: str
    reached_goal: bool
    physical_collisions: int
    path_length_m: float
    spl: float
    hazard_coverage_mean: float
    steps: int


def _is_collidable_entity(pos_z: float, size_z: float, haz: float, agent_h: float = 1.70) -> bool:
    if haz <= 0.0:
        return False
    z_min = pos_z - size_z / 2.0
    z_max = pos_z + size_z / 2.0
    return not (z_min > agent_h + 0.15 or z_max < 0.05)


def build_mjcf_from_3dssg(
    g_room: nx.DiGraph,
    room_size: Tuple[float, float] = (8.0, 10.0),
) -> Tuple[str, Dict[str, dict], List[dict]]:
    """Compile a 3DSSG NetworkX graph into a valid MuJoCo MJCF XML string."""
    w_x, l_y = room_size
    wall_h = 2.8
    obj_geoms_xml: List[str] = []
    obj_meta: Dict[str, dict] = {}

    for nid, d in g_room.nodes(data=True):
        if d.get("layer") != "object":
            continue
        pos = d["pos"]
        size = d["size"]
        lbl = str(d.get("label", "object"))
        haz = float(d.get("hazard_severity", 0.6))

        # Keep start area (y < 1.5) and goal area (y > l_y - 1.2) clear for fair spawning
        if pos[1] < 1.5 or pos[1] > l_y - 1.2:
            continue

        hx, hy, hz = max(0.12, size[0] / 2.0), max(0.12, size[1] / 2.0), max(0.05, size[2] / 2.0)
        z_c = max(hz, float(pos[2]))

        # Color by physical risk / height
        if not _is_collidable_entity(z_c, size[2], haz):
            rgba = "0.25 0.78 0.45 0.55"  # Harmless / overhead (green translucent)
            contype = "0"  # Overhead / floor mats don't block floor agent
            conaffinity = "0"
        elif haz >= 0.8:
            rgba = "0.85 0.32 0.25 0.95"  # High-risk obstacle (red/terracotta)
            contype = "1"
            conaffinity = "1"
        else:
            rgba = "0.72 0.54 0.34 0.95"  # Standard furniture (wood)
            contype = "1"
            conaffinity = "1"

        geom_name = f"geom_{nid}"
        obj_geoms_xml.append(
            f'    <geom name="{geom_name}" type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {z_c:.3f}" '
            f'size="{hx:.3f} {hy:.3f} {hz:.3f}" rgba="{rgba}" contype="{contype}" conaffinity="{conaffinity}"/>'
        )
        obj_meta[nid] = {
            "geom_name": geom_name,
            "label": lbl,
            "pos": np.array([pos[0], pos[1], z_c], dtype=float),
            "size": np.array([2 * hx, 2 * hy, 2 * hz], dtype=float),
            "vel": np.zeros(3, dtype=float),
            "structural_importance": float(d.get("structural_importance", 0.2)),
            "hazard_severity": haz,
            "is_dynamic": False,
        }

    # Add 2 dynamic solid pedestrians crossing the real 3DSSG room
    dyn_peds = [
        {"id": "dyn_ped_0", "x": 1.5, "y": l_y * 0.42, "vx": 0.85, "vy": -0.15},
        {"id": "dyn_ped_1", "x": w_x - 1.5, "y": l_y * 0.68, "vx": -0.80, "vy": -0.20},
    ]
    ped_bodies_xml = []
    for p in dyn_peds:
        ped_bodies_xml.append(f"""
    <body name="{p['id']}" mocap="true" pos="{p['x']:.2f} {p['y']:.2f} 0.85">
      <geom name="geom_{p['id']}" type="capsule" size="0.24 0.60" rgba="0.95 0.18 0.32 1" contype="1" conaffinity="1"/>
    </body>""")

    mjcf = f"""<mujoco model="aegis_3dssg_room">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.14 0.18 0.24" rgb2="0.19 0.24 0.31" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="12 12" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="{w_x/2:.1f} {l_y/2:.1f} 6.0" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="{w_x/2:.1f} {l_y/2:.1f} 0" size="{w_x} {l_y} 0.1" material="floor_mat" contype="1" conaffinity="1"/>

    <!-- Solid Room Walls -->
    <geom name="wall_left"   type="box" pos="0.0 {l_y/2:.2f} {wall_h/2:.2f}" size="0.10 {l_y/2:.2f} {wall_h/2:.2f}" rgba="0.35 0.42 0.52 0.85"/>
    <geom name="wall_right"  type="box" pos="{w_x:.2f} {l_y/2:.2f} {wall_h/2:.2f}" size="0.10 {l_y/2:.2f} {wall_h/2:.2f}" rgba="0.35 0.42 0.52 0.85"/>
    <geom name="wall_back"   type="box" pos="{w_x/2:.2f} 0.0 {wall_h/2:.2f}" size="{w_x/2:.2f} 0.10 {wall_h/2:.2f}" rgba="0.35 0.42 0.52 0.85"/>
    <geom name="wall_front"  type="box" pos="{w_x/2:.2f} {l_y:.2f} {wall_h/2:.2f}" size="{w_x/2:.2f} 0.10 {wall_h/2:.2f}" rgba="0.35 0.42 0.52 0.85"/>

    <!-- Goal Beacon (non-colliding visual marker) -->
    <geom name="goal_marker" type="cylinder" pos="{w_x/2:.2f} {l_y - 0.65:.2f} 0.04" size="0.65 0.04" rgba="0.15 0.90 0.45 0.7" contype="0" conaffinity="0"/>

    <!-- Real 3DSSG Scanned Objects -->
{chr(10).join(obj_geoms_xml)}

    <!-- Dynamic Pedestrians -->
{"".join(ped_bodies_xml)}

    <!-- Solid 3D Agent Capsule (cannot walk through furniture!) -->
    <body name="agent" pos="{w_x/2:.2f} 0.75 0.85">
      <joint name="agent_x" type="slide" axis="1 0 0" damping="2.0"/>
      <joint name="agent_y" type="slide" axis="0 1 0" damping="2.0"/>
      <geom name="agent_geom" type="capsule" size="0.28 0.55" mass="65.0" rgba="0.98 0.65 0.10 1" friction="0.1 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="act_vx" joint="agent_x" kv="180"/>
    <velocity name="act_vy" joint="agent_y" kv="180"/>
  </actuator>
</mujoco>
"""
    return mjcf, obj_meta, dyn_peds


def _update_local_costmap(
    occ: np.ndarray,
    entities: List[CompressedEntity],
    agent: AgentState,
    res: float,
) -> None:
    """Strictly sparsifier-fed recency-decayed local costmap: ONLY <= k static entities are rasterized."""
    w_cells, h_cells = occ.shape
    # Decay transient noisy measurements (standard probabilistic occupancy filtering)
    occ[1:-1, 1:-1] *= 0.88
    inflate = agent.radius + 0.04
    for e in entities:
        # Skip dynamic pedestrians (handled in real-time by DWA velocity cone) and non-collidable overhead/floor items
        if getattr(e, "is_dynamic", False) or float(np.linalg.norm(e.vel[:2])) > 0.15:
            continue
        if e.pos[2] - e.size[2] / 2.0 > agent.height + 0.15 or e.pos[2] + e.size[2] / 2.0 < 0.05:
            continue
        x0 = max(1, int((e.pos[0] - e.size[0] / 2.0 - inflate) / res))
        x1 = min(w_cells - 1, int((e.pos[0] + e.size[0] / 2.0 + inflate) / res) + 1)
        y0 = max(1, int((e.pos[1] - e.size[1] / 2.0 - inflate) / res))
        y1 = min(h_cells - 1, int((e.pos[1] + e.size[1] / 2.0 + inflate) / res) + 1)
        obb_size = getattr(e, "obb_size", None)
        obb_yaw = float(getattr(e, "obb_yaw", 0.0))
        if obb_size is not None and abs(obb_yaw) > 1e-3:
            # Exact rotated OBB rasterization — preserves diagonal corridors!
            for ix in range(x0, x1):
                cx = (ix + 0.5) * res
                for iy in range(y0, y1):
                    cy = (iy + 0.5) * res
                    if point_to_obb_clearance_2d(np.array([cx, cy]), e.pos[:2], obb_size[:2], obb_yaw, margin=inflate) <= 0.0:
                        occ[ix, iy] = min(1.0, occ[ix, iy] + 0.55)
        else:
            occ[x0:x1, y0:y1] = np.minimum(1.0, occ[x0:x1, y0:y1] + 0.55)


def _astar_subgoal(
    occ: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    res: float,
    lookahead_m: float = 1.6,
) -> np.ndarray:
    """Fast 2D A* on the strictly sparsifier-fed local occupancy grid to escape U-shaped traps."""
    w_cells, h_cells = occ.shape
    sx = int(np.clip(start_xy[0] / res, 1, w_cells - 2))
    sy = int(np.clip(start_xy[1] / res, 1, h_cells - 2))
    gx = int(np.clip(goal_xy[0] / res, 1, w_cells - 2))
    gy = int(np.clip(goal_xy[1] / res, 1, h_cells - 2))

    # 8-connected A*
    open_heap: List[Tuple[float, float, int, int]] = [(math.hypot(gx - sx, gy - sy), 0.0, sx, sy)]
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    neighbors = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))

    found: Optional[Tuple[int, int]] = None
    best_node = (sx, sy)
    best_h = math.hypot(gx - sx, gy - sy)

    while open_heap:
        _, g_curr, cx, cy = heapq.heappop(open_heap)
        if g_curr > g_score.get((cx, cy), 1e9) + 1e-6:
            continue
        h_curr = math.hypot(gx - cx, gy - cy)
        if h_curr < best_h:
            best_h = h_curr
            best_node = (cx, cy)
        if h_curr <= 2.0:
            found = (cx, cy)
            break
        for dx, dy, cost in neighbors:
            nx_c, ny_c = cx + dx, cy + dy
            if not (1 <= nx_c < w_cells - 1 and 1 <= ny_c < h_cells - 1):
                continue
            # Allow stepping out if currently inside an inflated margin cell
            if occ[nx_c, ny_c] >= 0.45 and math.hypot(nx_c - sx, ny_c - sy) > 2.2:
                continue
            ng = g_curr + cost
            if ng < g_score.get((nx_c, ny_c), 1e9):
                g_score[(nx_c, ny_c)] = ng
                came_from[(nx_c, ny_c)] = (cx, cy)
                heapq.heappush(open_heap, (ng + math.hypot(gx - nx_c, gy - ny_c), ng, nx_c, ny_c))

    target = found if found is not None else best_node
    path = [target]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()

    # Pick waypoint ~lookahead_m ahead along the A* path
    for cx, cy in path:
        wx, wy = (cx + 0.5) * res, (cy + 0.5) * res
        if math.hypot(wx - start_xy[0], wy - start_xy[1]) >= lookahead_m:
            return np.array([wx, wy], dtype=float)
    return goal_xy.copy()


def _plan_dwa_velocity(
    agent: AgentState,
    entities: List[CompressedEntity],
    subgoal_xy: np.ndarray,
    room_size: Tuple[float, float],
    stall_steps: int,
    bumped_last_step: bool = False,
) -> Tuple[float, float]:
    """Closed-loop Local-A* + DWA planner operating ONLY on the <= k compressed entities."""
    w_x, l_y = room_size
    to_goal = subgoal_xy - agent.pos[:2]
    goal_dir = to_goal / (float(np.linalg.norm(to_goal)) + 1e-8)
    base_ang = math.atan2(goal_dir[1], goal_dir[0])

    # Only dodge collidable entities
    active = [e for e in entities if e.pos[2] - e.size[2] / 2.0 <= agent.height + 0.15 and e.pos[2] + e.size[2] / 2.0 >= 0.05]

    if stall_steps >= 5 or bumped_last_step:
        # Tactile recovery: back up slightly and steer perpendicular toward open room center
        side = 1.0 if agent.pos[0] < w_x / 2.0 else -1.0
        if stall_steps >= 10:
            side = 1.0 if ((stall_steps // 6) % 2 == 0) else -1.0
        return side * 0.95, -0.45

    best_score = -1e9
    best_v = (0.0, 0.0)
    found_safe = False
    margin = agent.radius + 0.08

    # Full 270-degree fan so agent can smoothly follow A* detours around U-traps
    angles_deg = (0, -18, 18, -36, 36, -58, 58, -82, 82, -110, 110, -140, 140)
    for spd in (1.35, 0.90, 0.50):
        for deg in angles_deg:
            ang = base_ang + math.radians(deg)
            vx, vy = spd * math.cos(ang), spd * math.sin(ang)
            min_clr = 9.0
            hit = False
            for s in range(1, 5):
                dt = 0.25 * s
                px, py = agent.pos[0] + vx * dt, agent.pos[1] + vy * dt
                w_clr = min(px - 0.36, (w_x - 0.36) - px, py - 0.32, (l_y - 0.32) - py)
                if w_clr < 0.0:
                    hit = True
                    break
                min_clr = min(min_clr, w_clr)
                for e in active:
                    ox, oy = e.pos[0] + e.vel[0] * dt, e.pos[1] + e.vel[1] * dt
                    obb_s = getattr(e, "obb_size", None)
                    obb_y = float(getattr(e, "obb_yaw", 0.0))
                    if obb_s is not None and abs(obb_y) > 1e-3:
                        clr = point_to_obb_clearance_2d(
                            np.array([px, py]), np.array([ox, oy]), obb_s[:2], obb_y, margin=margin
                        )
                    else:
                        dx = max(abs(px - ox) - (e.size[0] / 2.0 + margin), 0.0)
                        dy = max(abs(py - oy) - (e.size[1] / 2.0 + margin), 0.0)
                        clr = math.hypot(dx, dy)
                    if clr <= 0.02:
                        hit = True
                        break
                    min_clr = min(min_clr, clr)
                if hit:
                    break
            if not hit:
                found_safe = True
                progress = float(vx * goal_dir[0] + vy * goal_dir[1])
                score = 1.8 * progress + 0.9 * min(min_clr, 1.2) - 0.004 * abs(deg)
                if score > best_score:
                    best_score, best_v = score, (vx, vy)

    if found_safe:
        return best_v
    # Soft fallback: step slowly toward subgoal instead of freezing permanently
    return 0.45 * float(goal_dir[0]), 0.45 * float(goal_dir[1])


def run_mujoco_episode(
    scan_id: str,
    g_room: nx.DiGraph,
    sparsifier,
    k: int = 5,
    noise_level: NoiseLevel = "medium",
    max_steps: int = 180,
    seed: int = 42,
    room_size: Tuple[float, float] = (8.0, 10.0),
    launch_viewer: bool = False,
) -> EpisodeMetrics:
    """Run one closed-loop episode inside MuJoCo 3D rigid-body physics."""
    mjcf, obj_meta, dyn_peds = build_mjcf_from_3dssg(g_room, room_size)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(seed)
    noise_cfg = NOISE_CONFIGS[noise_level]
    sparsifier.reset()

    agent_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mocap_ids = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["id"])] for p in dyn_peds]

    w_x, l_y = room_size
    start_xy = np.array([w_x / 2.0, 0.75], dtype=float)
    goal_xy = np.array([w_x / 2.0, l_y - 0.65], dtype=float)
    shortest_dist = float(np.linalg.norm(goal_xy - start_xy))

    # Strictly sparsifier-fed recency-decayed local occupancy costmap (starts completely empty!)
    occ_res = 0.20
    local_occ = np.zeros((int(w_x / occ_res), int(l_y / occ_res)), dtype=float)
    local_occ[0, :] = 1.0
    local_occ[-1, :] = 1.0
    local_occ[:, 0] = 1.0
    local_occ[:, -1] = 1.0

    prev_xy = start_xy.copy()
    path_len = 0.0
    reached = False
    hit_geoms: Set[int] = set()
    recalls: List[float] = []
    stall_steps = 0
    bumped_last_step = False

    viewer = mujoco.viewer.launch_passive(model, data) if launch_viewer else None
    try:
        for step in range(max_steps):
            if viewer is not None and not viewer.is_running():
                break

            # Read true MuJoCo agent kinematics
            ax, ay = float(data.qpos[0] + start_xy[0]), float(data.qpos[1] + start_xy[1])
            avx, avy = float(data.qvel[0]), float(data.qvel[1])
            agent = AgentState(
                pos=np.array([ax, ay, 0.85], dtype=float),
                vel=np.array([avx, avy, 0.0], dtype=float),
                radius=0.30,
            )

            # Build observed DSG G_t^obs from live MuJoCo state
            g_obs = nx.DiGraph()
            heading = agent.vel[:2] / (float(np.linalg.norm(agent.vel[:2])) + 1e-8)
            if float(np.linalg.norm(agent.vel[:2])) < 0.1:
                heading = np.array([0.0, 1.0])

            all_entities = dict(obj_meta)
            for p in dyn_peds:
                all_entities[p["id"]] = {
                    "label": "pedestrian",
                    "pos": np.array([p["x"], p["y"], 0.85], dtype=float),
                    "size": np.array([0.48, 0.48, 1.70], dtype=float),
                    "vel": np.array([p["vx"], p["vy"], 0.0], dtype=float),
                    "structural_importance": 0.10,
                    "hazard_severity": 0.95,
                    "is_dynamic": True,
                }

            obs_hazards: Set[str] = set()
            for nid, m in all_entities.items():
                delta = m["pos"][:2] - agent.pos[:2]
                dist = float(np.linalg.norm(delta))
                if dist > agent.max_range or dist < 0.05:
                    continue
                ang = math.acos(float(np.clip(np.dot(delta / dist, heading), -1.0, 1.0)))
                if ang > agent.fov_half_angle_rad:
                    continue

                fov_ratio = ang / agent.fov_half_angle_rad
                if rng.random() < noise_cfg["dropout_base"] * (0.5 + 1.5 * fov_ratio ** 2):
                    continue

                # GT hazard check over 3.0s
                is_haz = False
                if _is_collidable_entity(m["pos"][2], m["size"][2], m["hazard_severity"]):
                    rv = m["vel"][:2] - agent.vel[:2]
                    v2 = float(np.dot(rv, rv))
                    ts = 0.0 if v2 < 1e-6 else float(np.clip(-np.dot(delta, rv) / v2, 0.0, 3.0))
                    cxy = np.abs(delta + rv * ts)
                    if np.all(cxy <= m["size"][:2] / 2.0 + agent.radius + 0.25):
                        is_haz = True
                        obs_hazards.add(nid)

                conf = float(np.clip(0.98 - 0.35 * fov_ratio ** 2 - 0.12 * (dist / agent.max_range) + rng.normal(0, noise_cfg["conf_noise"]), 0.20, 0.99))
                g_obs.add_node(
                    nid,
                    layer="object",
                    label=m["label"],
                    pos=m["pos"] + rng.normal(0, noise_cfg["pos_std"], size=3),
                    size=m["size"].copy(),
                    vel=m["vel"].copy(),
                    confidence=conf,
                    structural_importance=m["structural_importance"],
                    hazard_severity=m["hazard_severity"],
                    is_dynamic=m["is_dynamic"],
                    is_hazard_gt=is_haz,
                )

            # Query sparsifier (supports both select_entities and select)
            if hasattr(sparsifier, "select_entities"):
                entities = sparsifier.select_entities(g_obs, agent, k)
                covered = set().union(*(e.member_ids for e in entities)) if entities else set()
            else:
                sel = sparsifier.select(g_obs, agent, k)
                covered = sel
                entities = [
                    CompressedEntity(
                        entity_id=nid,
                        label=str(g_obs.nodes[nid]["label"]),
                        pos=g_obs.nodes[nid]["pos"].copy(),
                        size=g_obs.nodes[nid]["size"].copy(),
                        vel=g_obs.nodes[nid]["vel"].copy(),
                        score=1.0,
                        member_ids={nid},
                    )
                    for nid in sel
                    if nid in g_obs
                ]

            recalls.append(1.0 if not obs_hazards else len(covered & obs_hazards) / len(obs_hazards))

            # Update strictly sparsifier-fed local occupancy memory and plan local A* subgoal
            _update_local_costmap(local_occ, entities, agent, occ_res)
            subgoal_xy = _astar_subgoal(local_occ, agent.pos[:2], goal_xy, occ_res, lookahead_m=1.6)

            # Closed-loop Local-A* + DWA command using ONLY <= k compressed entities
            vx_cmd, vy_cmd = _plan_dwa_velocity(
                agent, entities, subgoal_xy, room_size, stall_steps, bumped_last_step=bumped_last_step
            )
            data.ctrl[0] = vx_cmd
            data.ctrl[1] = vy_cmd

            # Step MuJoCo 3D rigid-body physics (20 substeps * 0.005s = 0.1s)
            step_bumped = False
            for _ in range(20):
                for p, mid in zip(dyn_peds, mocap_ids):
                    p["x"] += p["vx"] * 0.005
                    if p["x"] < 0.7 or p["x"] > w_x - 0.7:
                        p["vx"] *= -1.0
                    data.mocap_pos[mid] = [p["x"], p["y"], 0.85]

                mujoco.mj_step(model, data)

                # Inspect real MuJoCo C++ contact solver for physical collisions!
                for ci in range(data.ncon):
                    con = data.contact[ci]
                    g1, g2 = int(con.geom1), int(con.geom2)
                    if agent_geom_id in (g1, g2):
                        other = g2 if g1 == agent_geom_id else g1
                        if other != floor_geom_id:
                            hit_geoms.add(other)
                            step_bumped = True
            bumped_last_step = step_bumped

            if viewer is not None:
                viewer.sync()
                time.sleep(0.06)

            curr_xy = np.array([data.qpos[0] + start_xy[0], data.qpos[1] + start_xy[1]])
            step_dist = float(np.linalg.norm(curr_xy - prev_xy))
            path_len += step_dist
            stall_steps = (stall_steps + 1) if step_dist < 0.03 else 0
            prev_xy = curr_xy

            if float(np.linalg.norm(curr_xy - goal_xy)) <= 0.85:
                reached = True
                break
    finally:
        if viewer is not None:
            viewer.close()

    spl = (shortest_dist / max(path_len, shortest_dist)) if reached else 0.0
    return EpisodeMetrics(
        scan_id=scan_id,
        method_name=sparsifier.name,
        reached_goal=reached,
        physical_collisions=len(hit_geoms),
        path_length_m=round(path_len, 2),
        spl=round(spl, 3),
        hazard_coverage_mean=float(np.mean(recalls)) if recalls else 1.0,
        steps=step + 1,
    )

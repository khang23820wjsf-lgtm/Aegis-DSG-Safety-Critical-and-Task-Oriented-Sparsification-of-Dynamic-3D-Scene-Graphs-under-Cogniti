"""MuJoCo 4.0 Real 3D Rigid-Body Physics Environment for 3DSSG Rooms.

Compiles any real 3DSSG room graph into a solid 3D MJCF world where:
- Furniture, walls, and dynamic pedestrians are solid 3D rigid bodies (cannot walk through).
- Collisions are measured directly from MuJoCo's C++ contact solver (data.contact).
- The agent navigates closed-loop using ONLY the <= k compressed entities from a sparsifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import math
import time
import mujoco
import mujoco.viewer
import networkx as nx
import numpy as np

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


def _plan_dwa_velocity(
    agent: AgentState,
    entities: List[CompressedEntity],
    goal_xy: np.ndarray,
    room_size: Tuple[float, float],
    stall_steps: int,
) -> Tuple[float, float]:
    """Closed-loop Mini-DWA planner operating ONLY on the <= k compressed entities."""
    w_x, l_y = room_size
    to_goal = goal_xy - agent.pos[:2]
    goal_dir = to_goal / (float(np.linalg.norm(to_goal)) + 1e-8)
    base_ang = math.atan2(goal_dir[1], goal_dir[0])

    # Only dodge collidable entities
    active = [e for e in entities if e.pos[2] - e.size[2] / 2.0 <= agent.height + 0.15 and e.pos[2] + e.size[2] / 2.0 >= 0.05]

    if stall_steps >= 6:
        # Gap-seeking sidestep around rigid obstacle
        best_x, best_clr = w_x / 2.0, -1.0
        probe_y = min(l_y - 0.5, agent.pos[1] + 1.0)
        for px in np.linspace(0.6, w_x - 0.6, 15):
            clr = min(px - 0.4, (w_x - 0.4) - px)
            for e in active:
                dx = max(abs(px - e.pos[0]) - (e.size[0] / 2.0 + agent.radius), 0.0)
                dy = max(abs(probe_y - e.pos[1]) - (e.size[1] / 2.0 + agent.radius), 0.0)
                clr = min(clr, math.hypot(dx, dy))
            if clr > best_clr:
                best_clr, best_x = clr, float(px)
        dx = best_x - agent.pos[0]
        return (math.copysign(1.1, dx) if abs(dx) > 0.1 else 0.8), -0.20

    best_score = -1e9
    best_v = (0.0, 0.0)
    found_safe = False
    margin = agent.radius + 0.12

    for spd in (1.35, 0.85, 0.45):
        for deg in (0, -18, 18, -36, 36, -58, 58, -82, 82):
            ang = base_ang + math.radians(deg)
            vx, vy = spd * math.cos(ang), spd * math.sin(ang)
            min_clr = 9.0
            hit = False
            for s in range(1, 6):
                dt = 0.28 * s
                px, py = agent.pos[0] + vx * dt, agent.pos[1] + vy * dt
                w_clr = min(px - 0.42, (w_x - 0.42) - px, py - 0.35, (l_y - 0.35) - py)
                if w_clr < 0.0:
                    hit = True
                    break
                min_clr = min(min_clr, w_clr)
                for e in active:
                    ox, oy = e.pos[0] + e.vel[0] * dt, e.pos[1] + e.vel[1] * dt
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
                score = 1.6 * progress + 1.1 * min(min_clr, 1.4) - 0.006 * abs(deg)
                if score > best_score:
                    best_score, best_v = score, (vx, vy)

    return best_v if found_safe else (0.0, 0.0)


def run_mujoco_episode(
    scan_id: str,
    g_room: nx.DiGraph,
    sparsifier,
    k: int = 5,
    noise_level: NoiseLevel = "medium",
    max_steps: int = 130,
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

    prev_xy = start_xy.copy()
    path_len = 0.0
    reached = False
    hit_geoms: Set[int] = set()
    recalls: List[float] = []
    stall_steps = 0

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

            # Closed-loop DWA command using ONLY <= k compressed entities
            vx_cmd, vy_cmd = _plan_dwa_velocity(agent, entities, goal_xy, room_size, stall_steps)
            data.ctrl[0] = vx_cmd
            data.ctrl[1] = vy_cmd

            # Step MuJoCo 3D rigid-body physics (20 substeps * 0.005s = 0.1s)
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

"""Loader for Meta Project Aria Digital Twin (ADT, ICCV 2023) Real Egocentric Sequences.

Parses official ADT sequence files:
  - instances.json (object categories, static/dynamic motion types)
  - 3d_bounding_box.csv (exact metric 3D AABB extents)
  - scene_objects.csv (exact 3D world poses for static and dynamic objects)
  - 2d_bounding_box.csv (exact 2D RGB projections & visibility_ratio occlusion)
  - aria_trajectory.csv (exact 6DoF wearable glasses pose & world velocity)

Converts Meta ADT's Y-up coordinate frame (x, y_up, z) into Aegis-DSG's
Z-up world frame (x_horiz, y_horiz, z_up) and builds DSGFrame sequences directly
compatible with src/aegis_core.py and all baselines.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState, DSGFrame


DEFAULT_ADT_SEQ_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "aria_adt"
    / "Apartment_release_golden_skeleton_seq100_10s_sample_M1292"
)

# Categories that are tiny tabletop decor or architectural surfaces, not walking obstacles
BENIGN_SMALL_CATEGORIES = {
    "book", "hook", "vase", "cup", "mug", "bowl", "plate", "bottle", "pen", "pencil",
    "remote", "phone", "cellphone", "clock", "picture", "frame", "painting", "poster",
    "pillow", "cushion", "towel", "cloth", "paper", "notebook", "candle", "decoration",
    "ornament", "spoon", "fork", "coasters", "coaster", "napkin",
    "wall", "floor", "ceiling", "window", "curtain", "blinds", "rug", "mat", "carpet",
    "switch", "outlet", "light", "lamp", "light_fixture", "chandelier", "speaker",
}

ATOMIC_INTRINSIC_KEYWORDS = ("knife", "blade", "scissors", "stove", "fire", "kettle", "stair", "person", "human")
LANDMARK_CATEGORIES = {"door", "stair", "stairs", "cabinet", "wardrobe", "counter", "refrigerator", "shelf", "bookshelf"}


def _infer_adt_priors(
    category: str,
    prototype_name: str,
    size_xyz: np.ndarray,
    z_center: float,
    is_dynamic: bool,
) -> Tuple[str, float, float, bool]:
    """Derive (clean_label, C_v, hazard_severity, is_atomic_hazard) from ADT category + prototype."""
    cat = category.lower().strip()
    proto = prototype_name.lower().strip()
    combined = f"{cat} {proto}"
    vol = float(size_xyz[0] * size_xyz[1] * size_xyz[2])
    z_min = z_center - float(size_xyz[2]) / 2.0
    z_max = z_center + float(size_xyz[2]) / 2.0

    # Refine generic labels using exact prototype_name (e.g. 'kitchen utensil' + 'FakeButcherKnife' -> 'knife')
    clean_label = cat
    if "knife" in combined or "blade" in combined:
        clean_label = "knife"
    elif "scissors" in combined:
        clean_label = "scissors"

    c_v = 0.85 if any(k in combined for k in LANDMARK_CATEGORIES) else 0.15

    # Intrinsic / Atomic Safety-Critical Hazards (NEVER suppressed by tiny volume!)
    # Note: Do NOT flag all ADT 'dynamic' (movable) tableware like trays/containers as atomic!
    if any(k in combined for k in ATOMIC_INTRINSIC_KEYWORDS):
        return clean_label, c_v, 1.0, True

    if z_min > 1.80 or z_max < 0.08:
        return clean_label, c_v, 0.0, False
    if any(k in cat for k in BENIGN_SMALL_CATEGORIES) and vol < 0.08:
        return clean_label, c_v, 0.0, False
    if any(k in cat for k in ("chair", "table", "desk", "sofa", "couch", "bed", "stool", "bench", "ottoman")):
        return clean_label, c_v, 0.85, False
    if vol >= 0.06:
        return clean_label, c_v, 0.72, False
    return clean_label, c_v, 0.0, False


def load_aria_adt_sequence(
    seq_dir: Path = DEFAULT_ADT_SEQ_DIR,
    max_frames: int = 150,
    stride: int = 2,
) -> Tuple[List[DSGFrame], List[Dict[str, Tuple[int, int, int, int]]], List[int]]:
    """Load an official Meta Aria Digital Twin sequence into DSGFrames + 2D RGB boxes.

    Args:
        seq_dir: Directory containing extracted ADT main_groundtruth & trajectory files.
        max_frames: Maximum number of sampled frames to load.
        stride: Frame stride on the 30 Hz RGB stream (stride=2 -> 15 Hz, stride=3 -> 10 Hz).

    Returns:
        (frames, per_frame_bboxes_2d, video_frame_indices)
    """
    instances: Dict[str, dict] = json.loads((seq_dir / "instances.json").read_text(encoding="utf-8"))

    # 1. Load exact 3D bounding box extents (ADT Y-up -> Aegis Z-up: [dx, dz, dy])
    sizes_3d: Dict[str, np.ndarray] = {}
    with open(seq_dir / "3d_bounding_box.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["object_uid"]
            dx = float(row["p_local_obj_xmax[m]"]) - float(row["p_local_obj_xmin[m]"])
            dy_up = float(row["p_local_obj_ymax[m]"]) - float(row["p_local_obj_ymin[m]"])
            dz = float(row["p_local_obj_zmax[m]"]) - float(row["p_local_obj_zmin[m]"])
            sizes_3d[uid] = np.array([max(0.05, dx), max(0.05, dz), max(0.05, dy_up)], dtype=float)

    # 2. Load 3D object world positions (static: timestamp=-1; dynamic: per timestamp)
    static_pos: Dict[str, np.ndarray] = {}
    dynamic_pos: Dict[int, Dict[str, np.ndarray]] = defaultdict(dict)
    with open(seq_dir / "scene_objects.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["object_uid"]
            ts = int(row["timestamp[ns]"])
            # Map ADT (x, y_up, z) -> Aegis (x, z, y_up)
            pos = np.array([float(row["t_wo_x[m]"]), float(row["t_wo_z[m]"]), float(row["t_wo_y[m]"])], dtype=float)
            if ts == -1:
                static_pos[uid] = pos
            else:
                dynamic_pos[ts][uid] = pos

    # 3. Load 2D RGB bounding boxes & exact occlusion visibility_ratio (stream 214-1)
    boxes_by_ts: Dict[int, Dict[str, Tuple[int, int, int, int, float]]] = defaultdict(dict)
    with open(seq_dir / "2d_bounding_box.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["stream_id"] != "214-1":
                continue
            ts = int(row["timestamp[ns]"])
            uid = row["object_uid"]
            vis = float(row["visibility_ratio[%]"])
            if vis < 0.08:
                continue
            boxes_by_ts[ts][uid] = (
                int(float(row["x_min[pixel]"])),
                int(float(row["y_min[pixel]"])),
                int(float(row["x_max[pixel]"])),
                int(float(row["y_max[pixel]"])),
                vis,
            )

    # 4. Load 6DoF wearable Aria glasses trajectory (convert us -> ns)
    traj_ts: List[int] = []
    traj_pos: List[np.ndarray] = []
    with open(seq_dir / "aria_trajectory.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts_ns = int(row["tracking_timestamp_us"]) * 1000
            pos = np.array(
                [float(row["tx_world_device"]), float(row["tz_world_device"]), float(row["ty_world_device"])],
                dtype=float,
            )
            traj_ts.append(ts_ns)
            traj_pos.append(pos)

    traj_ts_arr = np.array(traj_ts, dtype=np.int64)
    rgb_timestamps = sorted(boxes_by_ts.keys())
    dyn_ts_arr = np.array(sorted(dynamic_pos.keys()), dtype=np.int64) if dynamic_pos else np.array([], dtype=np.int64)

    frames: List[DSGFrame] = []
    bboxes_per_frame: List[Dict[str, Tuple[int, int, int, int]]] = []
    video_indices: List[int] = []

    prev_agent_pos: np.ndarray | None = None
    prev_ts_s: float | None = None
    prev_obj_pos: Dict[str, np.ndarray] = {}

    for vid_idx in range(0, len(rgb_timestamps), stride):
        if len(frames) >= max_frames:
            break
        ts = rgb_timestamps[vid_idx]
        ts_s = ts * 1e-9

        # Nearest 6DoF head pose
        ti = int(np.argmin(np.abs(traj_ts_arr - ts)))
        apos = traj_pos[ti].copy()
        if prev_agent_pos is not None and prev_ts_s is not None and (ts_s - prev_ts_s) > 1e-4:
            avel = (apos - prev_agent_pos) / (ts_s - prev_ts_s)
        else:
            avel = np.array([0.0, 0.5, 0.0], dtype=float)
        dt = (ts_s - prev_ts_s) if (prev_ts_s is not None and (ts_s - prev_ts_s) > 1e-4) else (stride / 30.0)
        prev_agent_pos = apos.copy()
        prev_ts_s = ts_s

        agent = AgentState(pos=apos, vel=avel, radius=0.32, height=max(1.5, float(apos[2]) + 0.15), max_range=8.0)

        # Nearest dynamic object snapshot
        dyn_map: Dict[str, np.ndarray] = {}
        if len(dyn_ts_arr) > 0:
            di = int(np.argmin(np.abs(dyn_ts_arr - ts)))
            if abs(int(dyn_ts_arr[di]) - ts) < 100_000_000:
                dyn_map = dynamic_pos[int(dyn_ts_arr[di])]

        g_gt = nx.DiGraph()
        g_obs = nx.DiGraph()
        g_obs.add_node("room_aria_apt", layer="room", label="aria_apartment", pos=apos.copy(), size=np.array([10.0, 10.0, 3.0]), vel=np.zeros(3), confidence=1.0, structural_importance=1.0, hazard_severity=0.0)

        frame_boxes: Dict[str, Tuple[int, int, int, int]] = {}
        vis_boxes = boxes_by_ts[ts]
        curr_obj_pos: Dict[str, np.ndarray] = {}

        for uid, (x1, y1, x2, y2, vis_ratio) in vis_boxes.items():
            meta = instances.get(uid)
            if meta is None:
                continue
            pos = dyn_map.get(uid, static_pos.get(uid))
            if pos is None:
                continue
            size = sizes_3d.get(uid, np.array([0.35, 0.35, 0.45]))
            cat = str(meta.get("category", "object"))
            proto = str(meta.get("prototype_name", ""))
            is_dyn = (meta.get("motion_type", "static") == "dynamic")

            vel = np.zeros(3, dtype=float)
            if is_dyn and uid in prev_obj_pos:
                vel = (pos - prev_obj_pos[uid]) / max(dt, 0.03)
            curr_obj_pos[uid] = pos.copy()

            clean_label, c_v, haz, is_atomic = _infer_adt_priors(cat, proto, size, float(pos[2]), is_dyn)
            # Ground-truth safety hazard check: geometric collision OR close-range atomic hazard (<= 2.0m)
            is_haz = False
            if haz > 0.0:
                rel = pos[:2] - agent.pos[:2]
                dist_xy = float(np.linalg.norm(rel))
                if is_atomic and dist_xy <= 2.0:
                    is_haz = True
                else:
                    rv = vel[:2] - agent.vel[:2]
                    v2 = float(np.dot(rv, rv))
                    t_star = 0.0 if v2 < 1e-6 else float(np.clip(-np.dot(rel, rv) / v2, 0.0, 3.0))
                    cxy = np.abs(rel + rv * t_star)
                    if dist_xy <= 3.5 and np.all(cxy <= size[:2] / 2.0 + agent.radius + 0.30):
                        is_haz = True

            nid = f"adt_{uid}"
            # Map real ADT visibility_ratio [0.08..1.0] to observation confidence Q_v
            conf = float(np.clip(0.35 + 0.64 * vis_ratio, 0.35, 0.99))
            node_attrs = dict(
                layer="object",
                label=clean_label,
                pos=pos.copy(),
                size=size.copy(),
                vel=vel.copy(),
                confidence=conf,
                structural_importance=c_v,
                hazard_severity=haz,
                is_dynamic=is_dyn,
                is_atomic_hazard=is_atomic,
                is_hazard_gt=is_haz,
            )
            g_gt.add_node(nid, **node_attrs)
            g_obs.add_node(nid, **node_attrs)
            g_obs.add_edge(nid, "room_aria_apt", relation="belongs_to")
            frame_boxes[nid] = (x1, y1, x2, y2)

        prev_obj_pos = curr_obj_pos
        frames.append(
            DSGFrame(
                t=len(frames),
                timestamp=round((ts - rgb_timestamps[0]) * 1e-9, 3),
                agent=agent,
                graph_gt=g_gt,
                graph_obs=g_obs,
                metadata={
                    "num_obs_objects": float(len(frame_boxes)),
                    "num_gt_hazards": float(sum(1 for _, d in g_obs.nodes(data=True) if d.get("is_hazard_gt", False))),
                },
            )
        )
        bboxes_per_frame.append(frame_boxes)
        video_indices.append(vid_idx)

    return frames, bboxes_per_frame, video_indices

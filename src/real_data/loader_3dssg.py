"""Loader for the official 3DSSG (3RScan, Wald et al., CVPR 2020) Real 3D Scene Graph Dataset.

Loads real scanned indoor rooms (1,482 scans) from data/3dssg/3DSSG/{objects,relationships}.json,
preserves real semantic labels, affordances, and spatial relationships, and embeds each scan
into a physically consistent 3D metric scene graph via constraint-aware layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import networkx as nx
import numpy as np


DEFAULT_3DSSG_DIR = Path(__file__).resolve().parents[2] / "data" / "3dssg" / "3DSSG"

# Approximate real-world metric sizes (width, depth, height in meters) for common NYU40/3DSSG labels
APPROX_SIZES: Dict[str, Tuple[float, float, float]] = {
    "chair": (0.55, 0.55, 0.85),
    "armchair": (0.80, 0.80, 0.90),
    "table": (1.20, 0.75, 0.75),
    "desk": (1.30, 0.70, 0.75),
    "coffee table": (1.00, 0.60, 0.45),
    "sofa": (1.90, 0.85, 0.85),
    "couch": (1.90, 0.85, 0.85),
    "bed": (1.80, 2.00, 0.60),
    "cabinet": (0.90, 0.50, 1.30),
    "shelf": (0.90, 0.35, 1.60),
    "bookshelf": (0.90, 0.35, 1.70),
    "wardrobe": (1.10, 0.60, 2.00),
    "door": (0.15, 0.90, 2.10),
    "lamp": (0.35, 0.35, 0.55),
    "ceiling light": (0.45, 0.45, 0.12),
    "picture": (0.06, 0.70, 0.55),
    "tv": (0.12, 1.00, 0.65),
    "box": (0.45, 0.45, 0.40),
    "trash can": (0.35, 0.35, 0.50),
    "plant": (0.40, 0.40, 0.80),
    "stool": (0.40, 0.40, 0.50),
    "nightstand": (0.50, 0.45, 0.55),
    "counter": (1.40, 0.65, 0.90),
    "refrigerator": (0.75, 0.75, 1.80),
    "toilet": (0.45, 0.65, 0.75),
    "sink": (0.60, 0.50, 0.85),
}

IGNORE_STRUCTURAL_SURFACES = {"floor", "wall", "ceiling", "window", "curtain", "blinds"}


def _infer_affordance_scores(label: str, affordances: List[str], z_center: float, height: float) -> Tuple[float, float]:
    """Derive structural_importance (C_v) and hazard_severity from real 3DSSG affordances + 3D height."""
    lbl = label.lower()
    aff_set = {a.lower() for a in affordances}

    # Structural anchors (doors, stairs, cabinets/shelves near walls)
    if any(k in lbl for k in ("door", "stair", "entrance")):
        c_v = 0.92
    elif "opening" in aff_set or any(k in lbl for k in ("cabinet", "wardrobe", "shelf", "counter")):
        c_v = 0.55
    else:
        c_v = 0.18

    # Physical collision severity grounded in 3D height + affordances
    z_min = z_center - height / 2.0
    z_max = z_center + height / 2.0
    if z_min > 1.85 or z_max < 0.05 or "walking on" in aff_set:
        haz = 0.0
    elif any(k in lbl for k in ("stair", "barrier")):
        haz = 1.0
    elif any(k in lbl for k in ("chair", "table", "desk", "sofa", "couch", "bed", "stool", "box", "trash")):
        haz = 0.82
    elif any(k in lbl for k in ("door", "wall")):
        haz = 0.35
    else:
        haz = 0.65

    return c_v, haz


def load_3dssg_scan(
    scan_index: int = 0,
    data_dir: Path = DEFAULT_3DSSG_DIR,
    room_size: Tuple[float, float] = (8.0, 10.0),
    seed: int = 42,
) -> Tuple[str, nx.DiGraph]:
    """Load a real 3DSSG scan by index and embed it into a 3D metric NetworkX Scene Graph.

    Args:
        scan_index: Index into the 1,482 real scanned rooms in 3DSSG.
        data_dir: Path containing objects.json and relationships.json.
        room_size: (width_x, length_y) in meters for metric layout normalization.
        seed: Random seed for reproducible force-directed 2D embedding of 3DSSG relations.

    Returns:
        (scan_id, nx.DiGraph) with Room -> Object nodes, real affordances, and real 3DSSG edges.
    """
    objs_all = json.loads((data_dir / "objects.json").read_text(encoding="utf-8"))["scans"]
    rels_all = json.loads((data_dir / "relationships.json").read_text(encoding="utf-8"))["scans"]

    scan_obj = objs_all[scan_index]
    scan_id = str(scan_obj["scan"])
    rel_map = {str(r["scan"]): r["relationships"] for r in rels_all}
    raw_rels = rel_map.get(scan_id, [])

    # Build relational graph over non-surface foreground objects
    obj_by_id: Dict[str, dict] = {}
    for o in scan_obj["objects"]:
        lbl = str(o.get("label", "object")).lower()
        if lbl in IGNORE_STRUCTURAL_SURFACES:
            continue
        obj_by_id[str(o["id"])] = o

    rel_graph = nx.Graph()
    for oid in obj_by_id:
        rel_graph.add_node(oid)

    elevated_ids = set()
    support_parent: Dict[str, str] = {}
    valid_edges: List[Tuple[str, str, str]] = []

    for r in raw_rels:
        u, v, _, pred = str(r[0]), str(r[1]), r[2], str(r[3]).lower()
        if u in obj_by_id and v in obj_by_id and u != v:
            valid_edges.append((u, v, pred))
            if pred in ("close by", "standing on", "supported by", "attached to", "next to"):
                rel_graph.add_edge(u, v, weight=2.0)
            elif pred in ("left", "right", "front", "behind"):
                rel_graph.add_edge(u, v, weight=0.6)
            if pred in ("standing on", "supported by", "lying on"):
                support_parent[u] = v
        elif u in obj_by_id and pred in ("hanging on", "attached to"):
            elevated_ids.add(u)

    # Embed real 3DSSG proximity topology into 2D room space using spring layout
    w_x, l_y = room_size
    if len(rel_graph) > 0:
        pos_2d = nx.spring_layout(rel_graph, seed=seed, k=1.4 / max(1.0, np.sqrt(len(rel_graph))), iterations=60)
        xs = np.array([p[0] for p in pos_2d.values()])
        ys = np.array([p[1] for p in pos_2d.values()])
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
    else:
        pos_2d, x_min, x_max, y_min, y_max = {}, 0.0, 1.0, 0.0, 1.0

    # Assemble hierarchical 3D Scene Graph
    g = nx.DiGraph()
    g.add_node(
        f"room_{scan_id[:8]}",
        layer="room",
        label="real_3dssg_room",
        pos=np.array([w_x / 2.0, l_y / 2.0, 1.5]),
        size=np.array([w_x, l_y, 3.0]),
        vel=np.zeros(3),
        confidence=1.0,
        structural_importance=1.0,
        hazard_severity=0.0,
        is_dynamic=False,
        is_hazard_gt=False,
    )

    for oid, o in obj_by_id.items():
        lbl = str(o.get("label", "object")).lower()
        affs = list(o.get("affordances", []))
        size = np.array(APPROX_SIZES.get(lbl, (0.50, 0.50, 0.65)), dtype=float)

        px_raw, py_raw = pos_2d[oid]
        # Map into interior [0.8 .. w_x - 0.8] x [1.8 .. l_y - 1.2] leaving start/goal clearance
        nx_norm = (px_raw - x_min) / max(x_max - x_min, 1e-4)
        ny_norm = (py_raw - y_min) / max(y_max - y_min, 1e-4)
        x = 0.8 + nx_norm * (w_x - 1.6)
        y = 1.8 + ny_norm * (l_y - 3.0)

        # Height assignment respecting real 3DSSG support / hanging relations
        if "ceiling" in lbl or "lamp" in lbl and oid in elevated_ids:
            z = 2.65
        elif oid in elevated_ids or "picture" in lbl or "tv" in lbl:
            z = 1.95
        elif oid in support_parent:
            parent_lbl = str(obj_by_id[support_parent[oid]].get("label", "")).lower()
            parent_h = APPROX_SIZES.get(parent_lbl, (0.5, 0.5, 0.75))[2]
            z = parent_h + size[2] / 2.0
        else:
            z = size[2] / 2.0

        c_v, haz = _infer_affordance_scores(lbl, affs, z, size[2])
        nid = f"3dssg_{oid}"
        g.add_node(
            nid,
            layer="object",
            label=lbl,
            affordances=affs,
            pos=np.array([x, y, z], dtype=float),
            size=size,
            vel=np.zeros(3, dtype=float),
            confidence=1.0,
            structural_importance=c_v,
            hazard_severity=haz,
            is_dynamic=False,
            is_hazard_gt=False,
        )
        g.add_edge(nid, f"room_{scan_id[:8]}", relation="belongs_to")

    for u, v, pred in valid_edges:
        g.add_edge(f"3dssg_{u}", f"3dssg_{v}", relation=pred)

    return scan_id, g

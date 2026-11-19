"""Real-World Photo -> 3D Scene Graph -> Aegis-DSG v2 Visualization Pipeline.

Takes ANY real RGB indoor room photo, runs:
  1. YOLOv8 (ultralytics) for real 2D object detection & confidence Q_v
  2. MiDaS_small (PyTorch Hub + timm) for real monocular 3D depth estimation
  3. 3D Pinhole Back-Projection -> Real NetworkX 3D Scene Graph G^obs
  4. AegisV2Sparsifier (k=5) -> Compressed k-entity safety subgraph G'
  5. 3-Panel Publication Visualization: Raw Perception | Aegis-v2 View | 3D Scene Graph
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import networkx as nx
import numpy as np
import torch
from ultralytics import YOLO

from src.dsg_generator import AgentState
from src.phase_b.aegis_v2 import AegisV2Sparsifier


# Real-world COCO class priors for structural importance C_v & physical hazard severity
COCO_PRIORS: Dict[str, Tuple[float, float]] = {
    "person": (0.15, 0.95),
    "chair": (0.18, 0.85),
    "couch": (0.25, 0.85),
    "dining table": (0.30, 0.88),
    "bed": (0.30, 0.82),
    "bench": (0.20, 0.85),
    "potted plant": (0.12, 0.72),
    "suitcase": (0.10, 0.85),
    "backpack": (0.10, 0.70),
    "bicycle": (0.12, 0.92),
    "dog": (0.10, 0.85),
    "cat": (0.08, 0.65),
    "refrigerator": (0.65, 0.80),
    "oven": (0.55, 0.82),
    "sink": (0.55, 0.60),
    "tv": (0.30, 0.35),
    "laptop": (0.10, 0.15),
    "vase": (0.05, 0.20),
    "bottle": (0.05, 0.10),
    "cup": (0.05, 0.10),
    "bowl": (0.05, 0.10),
    "book": (0.05, 0.08),
    "clock": (0.15, 0.00),
}


def estimate_depth_map(img_rgb: np.ndarray) -> np.ndarray:
    """Estimate indoor metric depth (meters, 1.2m..7.5m) using MiDaS_small."""
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        inp = transforms.small_transform(img_rgb)
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        # MiDaS outputs inverse relative depth (higher = closer) -> map to realistic indoor room scale [1.0m .. 4.6m]
        p_min, p_max = float(np.percentile(pred, 2)), float(np.percentile(pred, 98))
        inv_norm = np.clip((pred - p_min) / max(p_max - p_min, 1e-6), 0.0, 1.0)
        depth_m = 4.6 - 3.6 * inv_norm
        return depth_m.astype(np.float32)
    except Exception as exc:
        print(f"[Warn] MiDaS unavailable ({exc}), using geometric perspective depth fallback.")
        h, w = img_rgb.shape[:2]
        ys = np.linspace(4.6, 1.0, h, dtype=np.float32)[:, None]
        return np.repeat(ys, w, axis=1)


def build_3d_scene_graph_from_photo(
    img_rgb: np.ndarray,
    yolo_model: YOLO,
    depth_map: np.ndarray,
    conf_thresh: float = 0.22,
) -> Tuple[nx.DiGraph, Dict[str, Tuple[int, int, int, int]]]:
    """Run YOLOv8 + Depth lifting to build a real NetworkX 3D Scene Graph G^obs."""
    h, w = img_rgb.shape[:2]
    fx = fy = 0.85 * w  # Approximate standard indoor camera focal length
    cx_img, cy_img = w / 2.0, h / 2.0

    results = yolo_model.predict(img_rgb, conf=conf_thresh, verbose=False)[0]
    g_obs = nx.DiGraph()
    g_obs.add_node(
        "room_real_photo",
        layer="room",
        label="real_indoor_room",
        pos=np.array([0.0, 4.0, 1.5]),
        size=np.array([8.0, 8.0, 3.0]),
        vel=np.zeros(3),
        confidence=1.0,
        structural_importance=1.0,
        hazard_severity=0.0,
    )

    bboxes_2d: Dict[str, Tuple[int, int, int, int]] = {}
    obj_ids: List[str] = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = str(yolo_model.names[cls_id])

        # Robust median depth inside center 50% of bounding box
        bx1, bx2 = max(0, x1 + (x2 - x1) // 4), min(w, x2 - (x2 - x1) // 4)
        by1, by2 = max(0, y1 + (y2 - y1) // 4), min(h, y2 - (y2 - y1) // 4)
        patch = depth_map[by1:max(by1 + 1, by2), bx1:max(bx1 + 1, bx2)]
        y_fwd = float(np.median(patch))

        # 3D pinhole back-projection (camera frame: +X right, +Y forward, +Z up from floor)
        u_c, v_c = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        x_right = (u_c - cx_img) * y_fwd / fx
        z_up = np.clip(1.35 + (cy_img - v_c) * y_fwd / fy, 0.20, 2.75)

        w_3d = max(0.25, ((x2 - x1) * y_fwd) / fx)
        h_3d = max(0.25, ((y2 - y1) * y_fwd) / fy)
        d_3d = min(w_3d, 0.90)

        c_v, haz = COCO_PRIORS.get(label, (0.15, 0.50))
        # Overhead wall/ceiling items (e.g. clock/tv high up) get low collision severity
        if z_up - h_3d / 2.0 > 1.80:
            haz = 0.0

        nid = f"obj_{label.replace(' ', '_')}_{i}"
        g_obs.add_node(
            nid,
            layer="object",
            label=label,
            pos=np.array([x_right, y_fwd, z_up], dtype=float),
            size=np.array([w_3d, d_3d, h_3d], dtype=float),
            vel=np.array([0.0, -0.8, 0.0]) if label == "person" else np.zeros(3),
            confidence=conf,
            structural_importance=c_v,
            hazard_severity=haz,
            is_dynamic=(label == "person"),
        )
        g_obs.add_edge(nid, "room_real_photo", relation="belongs_to")
        bboxes_2d[nid] = (x1, y1, x2, y2)
        obj_ids.append(nid)

    # Add 3D spatial proximity ('near') edges
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            u, v = obj_ids[i], obj_ids[j]
            if float(np.linalg.norm(g_obs.nodes[u]["pos"] - g_obs.nodes[v]["pos"])) <= 1.8:
                g_obs.add_edge(u, v, relation="near")
                g_obs.add_edge(v, u, relation="near")

    return g_obs, bboxes_2d


def _cluster_adjacent_same_class(
    entities,
    g_obs: nx.DiGraph,
    gap_thresh_m: float = 0.85,
):
    """Proactively merge adjacent same-class static objects (e.g. dining chairs) into 1 Super-Node."""
    from src.phase_b.coarsening import CompressedEntity, _merge_pair, _xy_edge_gap

    out = list(entities)
    merge_idx = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                base_a = a.label.split("_group")[0]
                base_b = b.label.split("_group")[0]
                is_dyn_a = any(g_obs.nodes[m].get("is_dynamic", False) for m in a.member_ids if m in g_obs)
                is_dyn_b = any(g_obs.nodes[m].get("is_dynamic", False) for m in b.member_ids if m in g_obs)
                if base_a == base_b and not is_dyn_a and not is_dyn_b:
                    if _xy_edge_gap(a, b) <= gap_thresh_m:
                        merged = _merge_pair(a, b, merge_idx)
                        merged.label = f"{base_a}_group(x{len(merged.member_ids)})"
                        merge_idx += 1
                        out = [e for idx, e in enumerate(out) if idx not in (i, j)]
                        out.append(merged)
                        changed = True
                        break
            if changed:
                break
    out.sort(key=lambda e: e.score, reverse=True)
    return out


def visualize_real_photo_aegis(
    img_rgb: np.ndarray,
    g_obs: nx.DiGraph,
    bboxes_2d: Dict[str, Tuple[int, int, int, int]],
    k: int,
    out_path: Path,
    tau_min: float = 0.20,
) -> None:
    """Run AegisV2Sparsifier with dynamic floor threshold (S >= tau_min) + spatial clustering."""
    agent = AgentState(pos=np.array([0.0, 0.0, 1.35]), vel=np.array([0.0, 1.35, 0.0]))
    aegis = AegisV2Sparsifier()
    aegis.reset()
    # Run 2 ticks so persistent real detections pass the 1-frame transient FP filter
    _ = aegis.select_entities(g_obs, agent, max(k, 8))
    raw_entities = aegis.select_entities(g_obs, agent, max(k, 8))

    # 1. Dynamic thresholding: drop low-importance clutter (S < tau_min) unless inside 2.2m Safety Bubble
    filtered = [
        e for e in raw_entities
        if e.score >= tau_min or float(np.linalg.norm(e.pos[:2])) <= 2.2
    ]
    # 2. Proactively group adjacent same-class static objects (e.g., 3 dining chairs -> chair_group)
    entities = _cluster_adjacent_same_class(filtered, g_obs)[:k]

    covered_ids = set().union(*(e.member_ids for e in entities)) if entities else set()
    obj_ids = [n for n, d in g_obs.nodes(data=True) if d.get("layer") == "object"]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=140)

    # --- Panel 1: Raw Perception (Cluttered Information Overload) ---
    ax1 = axes[0]
    ax1.imshow(img_rgb)
    for nid in obj_ids:
        x1, y1, x2, y2 = bboxes_2d[nid]
        d = g_obs.nodes[nid]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#f1c40f", lw=1.4)
        ax1.add_patch(rect)
        ax1.text(
            x1, max(10, y1 - 3), f"{d['label']} ({d['confidence']:.2f})",
            color="black", fontsize=7.2,
            bbox=dict(facecolor="#f1c40f", alpha=0.8, pad=1, edgecolor="none"),
        )
    ax1.set_title(f"1. Raw Perception G_obs ({len(obj_ids)} objects)\nInformation Overload!", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # --- Panel 2: Aegis-v2 Dynamic-Budget Compressed View (|V'| <= k, S >= tau_min) ---
    ax2 = axes[1]
    dimmed = (img_rgb.astype(np.float32) * 0.50).astype(np.uint8)
    for nid in covered_ids:
        if nid in bboxes_2d:
            x1, y1, x2, y2 = bboxes_2d[nid]
            dimmed[y1:y2, x1:x2] = img_rgb[y1:y2, x1:x2]
    ax2.imshow(dimmed)

    used_label_y: List[Tuple[int, int]] = []

    def place_label(x: int, y: int) -> Tuple[int, int]:
        ly = max(14, y - 5)
        for ux, uy in used_label_y:
            if abs(x - ux) < 130 and abs(ly - uy) < 22:
                ly = uy + 20
        used_label_y.append((x, ly))
        return x, ly

    for ent in entities:
        if ent.is_super_node:
            xs1 = [bboxes_2d[m][0] for m in ent.member_ids if m in bboxes_2d]
            ys1 = [bboxes_2d[m][1] for m in ent.member_ids if m in bboxes_2d]
            xs2 = [bboxes_2d[m][2] for m in ent.member_ids if m in bboxes_2d]
            ys2 = [bboxes_2d[m][3] for m in ent.member_ids if m in bboxes_2d]
            if xs1:
                sx1, sy1, sx2, sy2 = min(xs1), min(ys1), max(xs2), max(ys2)
                ax2.add_patch(Rectangle((sx1, sy1), sx2 - sx1, sy2 - sy1, fill=False, edgecolor="#f97316", lw=2.8, ls="--"))
                lx, ly = place_label(sx1, sy1)
                ax2.text(
                    lx, ly, f"SUPER-NODE: {ent.label} | ~{ent.pos[1]:.1f}m | S={ent.score:.2f}",
                    color="white", fontsize=8.2, fontweight="bold",
                    bbox=dict(facecolor="#ea580c", alpha=0.92, pad=1.8, edgecolor="none"),
                )
        else:
            nid = ent.entity_id
            if nid not in bboxes_2d:
                continue
            x1, y1, x2, y2 = bboxes_2d[nid]
            d = g_obs.nodes[nid]
            col = "#ef4444" if d["hazard_severity"] >= 0.7 else "#38bdf8"
            ax2.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=2.5))
            lx, ly = place_label(x1, y1)
            ax2.text(
                lx, ly, f"{d['label']} | ~{d['pos'][1]:.1f}m | S={ent.score:.2f}",
                color="white", fontsize=8.2, fontweight="bold",
                bbox=dict(facecolor=col, alpha=0.92, pad=1.8, edgecolor="none"),
            )

    ax2.set_title(
        f"2. Aegis-v2 Dynamic Budget (|V'|={len(entities)} <= k={k}, S >= {tau_min})\nFiltered {len(obj_ids) - len(covered_ids)} clutter nodes",
        fontsize=11, fontweight="bold",
    )
    ax2.axis("off")

    # --- Panel 3: Top-Down 3D Scene Graph Extracted from Photo ---
    ax3 = axes[2]
    ax3.set_facecolor("#0f172a")
    ax3.add_patch(Wedge((0, 0), 4.6, 90 - 38, 90 + 38, facecolor="#38bdf8", alpha=0.14, edgecolor="#38bdf8", ls="--", lw=1.2, label="Collision Cone (±38°)"))
    ax3.add_patch(Circle((0, 0), 2.2, fill=False, edgecolor="#4ade80", ls="--", lw=1.4, label="Safety Bubble (2.2m)"))

    # Draw dropped clutter nodes faintly
    for nid in obj_ids:
        if nid in covered_ids:
            continue
        p = g_obs.nodes[nid]["pos"]
        ax3.scatter(p[0], p[1], c="#64748b", s=34, alpha=0.45, zorder=2)

    # Draw retained Aegis entities (individual or Super-Node!) with clean non-overlapping offsets
    offsets = [(0.15, 0.14), (-1.15, 0.22), (0.18, -0.35), (-1.10, -0.30), (0.15, 0.35)]
    for idx, ent in enumerate(entities):
        p = ent.pos
        ox, oy = offsets[idx % len(offsets)]
        if ent.is_super_node:
            ax3.add_patch(
                Rectangle(
                    (p[0] - ent.size[0] / 2, p[1] - ent.size[1] / 2),
                    ent.size[0], ent.size[1],
                    fill=True, facecolor="#f97316", alpha=0.22, edgecolor="#f97316", lw=2.2, ls="--", zorder=4,
                )
            )
            ax3.scatter(p[0], p[1], c="#f97316", s=130, marker="s", edgecolors="white", linewidths=1.2, zorder=5)
            ax3.text(p[0] + ox, p[1] + oy, f"{ent.label}\n(~{p[1]:.1f}m)", color="#fdba74", fontsize=7.8, fontweight="bold", zorder=6)
        else:
            d = g_obs.nodes[ent.entity_id]
            col = "#ef4444" if d["hazard_severity"] >= 0.7 else "#38bdf8"
            ax3.scatter(p[0], p[1], c=col, s=115, edgecolors="white", linewidths=1.2, zorder=5)
            ax3.text(p[0] + ox, p[1] + oy, f"{d['label']}\n(~{p[1]:.1f}m)", color="#f8fafc", fontsize=7.8, fontweight="bold", zorder=6)

    ax3.scatter(0, 0, c="#f59e0b", s=180, marker="^", edgecolors="white", zorder=7, label="Camera / User")
    ax3.set_title("3. Extracted 3D Scene Graph (Top-Down)\nCalibrated Est. Depth | Clean Non-Overlapping Layout", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X Lateral (m)")
    ax3.set_ylabel("Y Est. Depth Forward (m)")
    ax3.set_xlim(-3.2, 3.2)
    ax3.set_ylim(-0.4, 5.4)
    ax3.grid(True, color="#1e293b", alpha=0.8)
    ax3.legend(loc="lower left", fontsize=8, facecolor="#0f172a", edgecolor="#475569", framealpha=0.92, labelcolor="white")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved real-photo Aegis-DSG visualization to: {out_path.resolve()} ({len(obj_ids)} raw -> {len(entities)} kept entities)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/real_photos/cluttered_office.jpg", help="Path to any real room photo")
    parser.add_argument("--k", type=int, default=5, help="Cognitive budget k")
    parser.add_argument("--out", type=str, default="results/real_photo_aegis.png", help="Output visualization path")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"1) Loading YOLOv8n detector on {args.image}...")
    yolo = YOLO("yolov8n.pt")
    print("2) Estimating 3D monocular depth map (MiDaS_small)...")
    depth_map = estimate_depth_map(img_rgb)
    print("3) Lifting detections to 3D Scene Graph & running Aegis-v2...")
    g_obs, bboxes_2d = build_3d_scene_graph_from_photo(img_rgb, yolo, depth_map)
    visualize_real_photo_aegis(img_rgb, g_obs, bboxes_2d, args.k, Path(args.out))


if __name__ == "__main__":
    main()

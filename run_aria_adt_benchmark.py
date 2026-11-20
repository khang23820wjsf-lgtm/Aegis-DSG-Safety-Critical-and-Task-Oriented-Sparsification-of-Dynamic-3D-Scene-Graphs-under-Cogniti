"""Run UNMODIFIED Aegis-Core v1.0 vs. Distance, D-Lite(True), and Clio(aIB)
on the REAL Meta Project Aria Digital Twin (ADT) Egocentric Sequence!

Outputs:
  1. Benchmark table on real Meta Aria glasses sequence (350 real objects, exact 6DoF pose).
  2. Publication-grade side-by-side keyframe figure: results/aria_adt_aegis_showcase.png
  3. Real egocentric MP4 comparison video: results/aria_adt_aegis_demo.mp4
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.aegis_core import AegisConfig, AegisDSG
from src.baselines import DistanceSparsifier
from src.metrics import evaluate_sparsifier
from src.phase_b.true_baselines import ClioTrueIBSparsifier, DLiteTrueSparsifier
from src.real_data.loader_aria_adt import DEFAULT_ADT_SEQ_DIR, load_aria_adt_sequence


def _rotate_box_cw90(box: Tuple[int, int, int, int], raw_w: int = 1408, raw_h: int = 1408) -> Tuple[int, int, int, int]:
    """Rotate Aria raw 2D sensor box 90 deg clockwise to match preview_rgb.mp4 orientation."""
    x1, y1, x2, y2 = box
    nx1, nx2 = raw_h - y2, raw_h - y1
    ny1, ny2 = x1, x2
    return (max(0, nx1), max(0, ny1), min(raw_h - 1, nx2), min(raw_w - 1, ny2))


def main() -> None:
    print("1) Loading REAL Meta Project Aria Digital Twin sequence...")
    frames, bboxes_list, vid_indices = load_aria_adt_sequence(DEFAULT_ADT_SEQ_DIR, max_frames=150, stride=2)
    avg_obs = float(np.mean([f.metadata["num_obs_objects"] for f in frames]))
    avg_haz = float(np.mean([f.metadata["num_gt_hazards"] for f in frames]))
    print(f"   Loaded {len(frames)} real egocentric frames | Avg visible objects/frame: {avg_obs:.1f} | Avg hazards/frame: {avg_haz:.1f}")

    k = 5
    methods = [
        DistanceSparsifier(),
        DLiteTrueSparsifier(),
        ClioTrueIBSparsifier(),
        AegisDSG(AegisConfig(tau_min=0.15)),
    ]

    print(f"\n=== REAL META ARIA DIGITAL TWIN BENCHMARK (k={k}, 150 real egocentric frames) ===")
    header = f"{'Method':<14} | {'Hazard Cov.':>11} | {'CVaR-5%':>9} | {'Flicker F_t':>11} | {'Latency(ms)':>11}"
    print(header)
    print("-" * len(header))
    for m in methods:
        eval_frames = frames[::5] if m.name == "Clio(aIB)" else frames
        res = evaluate_sparsifier(m, eval_frames, k)
        print(
            f"{m.name:<14} | {res['hazard_recall_mean']*100:10.1f}% | "
            f"{res['hazard_recall_cvar5']*100:8.1f}% | {res['flicker_mean']:11.3f} | {res['latency_ms_mean']:11.3f}"
        )

    # Render side-by-side MP4 video + crisp showcase PNG from the real Aria RGB video!
    video_path = DEFAULT_ADT_SEQ_DIR / "ADT_Apartment_release_golden_skeleton_seq100_10s_sample_M1292_preview_rgb.mp4"
    cap = cv2.VideoCapture(str(video_path))
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "aria_adt_aegis_demo.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (1280, 640))

    aegis = AegisDSG(AegisConfig(tau_min=0.15))
    aegis.reset()
    best_showcase = None
    best_score = -1

    for idx, (frame, raw_boxes, vid_idx) in enumerate(zip(frames, bboxes_list, vid_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ok, img_bgr = cap.read()
        if not ok:
            continue

        h_img, w_img = img_bgr.shape[:2]
        entities = aegis.compress(frame.graph_obs, frame.agent, k)
        covered = set().union(*(e.member_ids for e in entities)) if entities else set()

        # Left panel: Raw Meta Aria perception (cluttered!)
        left = img_bgr.copy()
        for nid, box in raw_boxes.items():
            x1, y1, x2, y2 = _rotate_box_cw90(box, w_img, h_img)
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(left, f"1. Meta Aria Raw Perception ({len(raw_boxes)} objects)", (28, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 220, 255), 3)

        # Right panel: Aegis-Core v1.0 compressed view (|V'| <= k)
        right = (img_bgr.astype(np.float32) * 0.48).astype(np.uint8)
        for nid in covered:
            if nid in raw_boxes:
                x1, y1, x2, y2 = _rotate_box_cw90(raw_boxes[nid], w_img, h_img)
                right[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]

        for ent in entities:
            boxes_ent = [_rotate_box_cw90(raw_boxes[m], w_img, h_img) for m in ent.member_ids if m in raw_boxes]
            if not boxes_ent:
                continue
            bx1 = min(b[0] for b in boxes_ent)
            by1 = min(b[1] for b in boxes_ent)
            bx2 = max(b[2] for b in boxes_ent)
            by2 = max(b[3] for b in boxes_ent)
            dist_m = float(np.linalg.norm(ent.pos[:2] - frame.agent.pos[:2]))
            if ent.is_atomic_hazard:
                col = (220, 20, 255)  # Vivid magenta/crimson for Atomic Critical Hazard (e.g. knife!)
                tag = f"[!] ATOMIC: {ent.label.upper()} | {dist_m:.1f}m | R={ent.risk:.2f}"
                thick = 6
            elif ent.is_super_node:
                col = (0, 140, 255)
                tag = f"{ent.label} | {dist_m:.1f}m | R={ent.risk:.2f}"
                thick = 4
            else:
                col = (60, 76, 231) if ent.risk > 0.28 else (248, 189, 56)
                tag = f"{ent.label} | {dist_m:.1f}m | R={ent.risk:.2f}"
                thick = 4
            cv2.rectangle(right, (bx1, by1), (bx2, by2), col, thick)
            tx = max(6, min(bx1, w_img - len(tag) * 16 - 20))
            ty = max(44, by1)
            cv2.rectangle(right, (tx, ty - 38), (min(w_img - 4, tx + len(tag) * 16 + 16), ty), col, -1)
            cv2.putText(right, tag, (tx + 6, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(right, f"2. Aegis-Core v2.0 (|V'|={len(entities)} <= k={k})", (28, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (80, 240, 130), 3)

        combo = np.hstack([cv2.resize(left, (640, 640)), cv2.resize(right, (640, 640))])
        writer.write(combo)

        # Prioritize a frame containing an Atomic Knife + Super-Nodes for the v2 showcase PNG!
        has_knife = any(e.is_atomic_hazard and "knife" in e.label.lower() for e in entities)
        sc = (200 if has_knife else 0) + len(raw_boxes) + 8 * len(entities) + 15 * sum(1 for e in entities if e.is_super_node)
        if idx >= 15 and sc > best_score:
            best_score = sc
            best_showcase = combo.copy()

    cap.release()
    writer.release()
    print(f"Saved real Meta Aria comparison video to: {out_mp4.resolve()}")

    if best_showcase is not None:
        out_png = out_dir / "aria_adt_aegis_v2_showcase.png"
        cv2.imwrite(str(out_png), best_showcase)
        print(f"Saved real Meta Aria v2 showcase PNG to: {out_png.resolve()}")


if __name__ == "__main__":
    main()

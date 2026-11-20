"""Preview and visualize the generated Dynamic 3D Scene Graph sequence."""

from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

import sys as _sys; from pathlib import Path as _Path; _ROOT = str(_Path(__file__).resolve().parents[2]); (_sys.path.insert(0, _ROOT) if _ROOT not in _sys.path else None)
from src.dsg_generator import DSGFrame, generate_dsg_sequence


def plot_dsg_snapshot(frame: DSGFrame, out_path: Path) -> None:
    """Render 3D hierarchical view and 2D top-down view of a single DSGFrame."""
    fig = plt.figure(figsize=(14, 6), dpi=130)

    # --- Panel 1: 3D Hierarchical Scene Graph ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    g = frame.graph_obs

    for u, v, edata in g.edges(data=True):
        pu = g.nodes[u]["pos"]
        pv = g.nodes[v]["pos"]
        rel = edata.get("relation", "")
        if rel == "belongs_to":
            ax3d.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], color="#95a5a6", lw=0.7, alpha=0.55)
        elif rel == "near":
            ax3d.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], color="#3498db", ls="--", lw=0.6, alpha=0.45)

    for nid, data in g.nodes(data=True):
        pos = data["pos"]
        layer = data["layer"]
        if layer == "room":
            ax3d.scatter(*pos, c="#8e44ad", s=160, marker="^", label="Room" if "Room" not in ax3d.get_legend_handles_labels()[1] else "")
        elif layer == "place":
            ax3d.scatter(*pos, c="#2980b9", s=90, marker="s", label="Place" if "Place" not in ax3d.get_legend_handles_labels()[1] else "")
        else:
            is_haz = data.get("is_hazard_gt", False)
            is_dyn = data.get("is_dynamic", False)
            color = "#e74c3c" if is_haz else "#2ecc71"
            marker = "D" if is_dyn else "o"
            ax3d.scatter(*pos, c=color, s=45, marker=marker, edgecolors="k", linewidths=0.4)

    ap = frame.agent.pos
    av = frame.agent.vel
    ax3d.scatter(*ap, c="#f39c12", s=140, marker="*", edgecolors="k", label="Agent")
    ax3d.quiver(ap[0], ap[1], ap[2], av[0], av[1], av[2], length=1.2, color="#d35400", lw=2)

    ax3d.set_title(f"3D Observed DSG (t={frame.t}, {frame.timestamp:.1f}s)\nNodes: {g.number_of_nodes()} | Edges: {g.number_of_edges()}")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_xlim(0, 4.2)
    ax3d.set_zlim(0, 3.0)

    # --- Panel 2: Top-Down (BEV) View with FOV & Hazards ---
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_facecolor("#f8f9fa")
    ax2d.axvline(0.0, color="#2c3e50", lw=2)
    ax2d.axvline(4.0, color="#2c3e50", lw=2)

    heading_deg = math.degrees(math.atan2(av[1], av[0]))
    fov_deg = math.degrees(frame.agent.fov_half_angle_rad)
    wedge = Wedge(
        (ap[0], ap[1]),
        frame.agent.max_range,
        heading_deg - fov_deg,
        heading_deg + fov_deg,
        facecolor="#f1c40f",
        alpha=0.14,
        edgecolor="#f39c12",
        ls="--",
        label="Agent FOV",
    )
    ax2d.add_patch(wedge)

    # Plot ground truth faintly and observed brightly
    for nid, data in frame.graph_gt.nodes(data=True):
        if data["layer"] != "object":
            continue
        p = data["pos"]
        ax2d.scatter(p[0], p[1], c="#bdc3c7", s=30, alpha=0.6, zorder=2)

    for nid, data in g.nodes(data=True):
        if data["layer"] != "object":
            continue
        p = data["pos"]
        v = data["vel"]
        is_haz = data.get("is_hazard_gt", False)
        color = "#e74c3c" if is_haz else "#27ae60"
        ax2d.scatter(p[0], p[1], c=color, s=60, edgecolors="k", linewidths=0.5, zorder=4)
        ax2d.text(p[0] + 0.08, p[1] + 0.1, f"{data['label']} ({data['confidence']:.2f})", fontsize=7, alpha=0.85)
        if data.get("is_dynamic", False):
            ax2d.arrow(p[0], p[1], v[0], v[1], head_width=0.12, color="#c0392b", lw=1.2, zorder=5)

    ax2d.scatter(ap[0], ap[1], c="#f39c12", s=160, marker="*", edgecolors="k", zorder=6, label="Agent")
    ax2d.arrow(ap[0], ap[1], av[0], av[1], head_width=0.15, color="#d35400", lw=2, zorder=6)

    ax2d.set_title("Top-Down View: Red=Hazard, Green=Benign, Gray=Out-of-FOV/Dropped")
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_xlim(-0.5, 4.5)
    ax2d.set_ylim(max(0, ap[1] - 2.0), ap[1] + 13.0)
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    seq = generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=42)
    print(f"Generated {len(seq)} DSG frames (dt=0.1s).")
    avg_gt = np.mean([f.metadata["num_gt_objects"] for f in seq])
    avg_obs = np.mean([f.metadata["num_obs_objects"] for f in seq])
    avg_haz = np.mean([f.metadata["num_gt_hazards"] for f in seq])
    print(f"Avg GT objects/frame: {avg_gt:.1f} | Avg Observed objects/frame: {avg_obs:.1f} | Avg GT hazards/frame: {avg_haz:.1f}")

    sample_idx = min(25, len(seq) - 1)
    out_file = Path("results/dsg_preview.png")
    plot_dsg_snapshot(seq[sample_idx], out_file)
    print(f"Saved preview image to {out_file.resolve()}")


if __name__ == "__main__":
    main()

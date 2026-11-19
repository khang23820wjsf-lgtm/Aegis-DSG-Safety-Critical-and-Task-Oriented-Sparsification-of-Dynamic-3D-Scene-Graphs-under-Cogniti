"""Visual Demo: How a computer transforms raw 3D/2D Space into a Hierarchical Scene Graph.

Demonstrates the exact 4-step Hydra pipeline using only NumPy, NetworkX, and Matplotlib:
  1. Occupancy Grid (Walls + Furniture + Doorway)
  2. ESDF (Euclidean Signed Distance Field: distance from every free cell to nearest obstacle)
  3. GVD Sparsification -> Places Topological Graph (obstacle-free navigable nodes & edges)
  4. Doorway Dilation -> Automatic Room Segmentation -> Hierarchical Scene Graph (Room -> Place -> Object)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def build_floorplan(h: int = 80, w: int = 120, res: float = 0.1) -> Tuple[np.ndarray, List[Dict]]:
    """Create a 2-room floorplan (12m x 8m at 0.1m/cell) connected by a narrow 0.9m doorway."""
    grid = np.ones((h, w), dtype=bool)  # True = free space, False = obstacle/wall

    # Outer walls (thickness = 2 cells)
    grid[:2, :] = False
    grid[-2:, :] = False
    grid[:, :2] = False
    grid[:, -2:] = False

    # Dividing wall at x = 60 (6.0m) with a narrow doorway at y in [36..44] (0.9m wide)
    grid[:, 59:62] = False
    grid[36:45, 59:62] = True

    # Add furniture obstacles (tables/chairs)
    objects = [
        {"id": "obj_table_1", "label": "Table", "y0": 15, "y1": 25, "x0": 20, "x1": 34},
        {"id": "obj_chair_1", "label": "Chair", "y0": 55, "y1": 62, "x0": 25, "x1": 32},
        {"id": "obj_table_2", "label": "Table", "y0": 22, "y1": 32, "x0": 82, "x1": 96},
        {"id": "obj_chair_2", "label": "Chair", "y0": 54, "y1": 61, "x0": 88, "x1": 95},
    ]
    for obj in objects:
        grid[obj["y0"]:obj["y1"], obj["x0"]:obj["x1"]] = False

    return grid, objects


def compute_esdf(grid: np.ndarray, res: float = 0.1) -> np.ndarray:
    """Compute exact Euclidean distance (meters) from each free cell to the nearest obstacle cell."""
    obs_y, obs_x = np.where(~grid)
    obs_pts = np.stack([obs_y, obs_x], axis=1).astype(np.float32)

    h, w = grid.shape
    esdf = np.zeros((h, w), dtype=np.float32)
    free_y, free_x = np.where(grid)
    free_pts = np.stack([free_y, free_x], axis=1).astype(np.float32)

    # Chunked vectorized Euclidean distance (fast, zero scipy dependency)
    chunk = 512
    min_dists = np.empty(len(free_pts), dtype=np.float32)
    for start in range(0, len(free_pts), chunk):
        pts = free_pts[start:start + chunk]
        diff = pts[:, None, :] - obs_pts[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_dists[start:start + chunk] = np.sqrt(np.min(d2, axis=1)) * res

    esdf[free_y, free_x] = min_dists
    return esdf


def has_line_of_sight(grid: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    """Check if straight segment between two grid cells stays completely in free space."""
    y1, x1 = p1
    y2, x2 = p2
    steps = max(abs(y2 - y1), abs(x2 - x1), 1) * 2
    ys = np.linspace(y1, y2, steps).round().astype(int)
    xs = np.linspace(x1, x2, steps).round().astype(int)
    return bool(np.all(grid[ys, xs]))


def extract_places_graph(grid: np.ndarray, esdf: np.ndarray, res: float = 0.1) -> nx.Graph:
    """Extract GVD-like free-space skeleton and sparsify into a Places topological graph."""
    h, w = grid.shape
    candidates: List[Tuple[float, int, int]] = []

    # 1. Find local ridge points (higher ESDF than most neighbors) + doorway center points
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            val = float(esdf[y, x])
            if val < 0.35:
                continue
            patch = esdf[y - 2:y + 3, x - 2:x + 3]
            if val >= float(np.percentile(patch, 82)):
                candidates.append((val, y, x))

    # Sort descending by clearance so wide open room centers are picked first
    candidates.sort(reverse=True)

    # 2. Sparsify with minimum spacing (~1.1m = 11 cells)
    min_spacing_cells = 11.0
    places: List[Tuple[int, int, float]] = []
    for val, y, x in candidates:
        if all((y - py) ** 2 + (x - px) ** 2 >= min_spacing_cells ** 2 for py, px, _ in places):
            places.append((y, x, val))

    # 3. Build Places Graph: connect nearby places with unobstructed line of sight
    g_places = nx.Graph()
    for i, (y, x, val) in enumerate(places):
        g_places.add_node(f"place_{i}", pos=(x * res, y * res), cell=(y, x), esdf=val, layer="place")

    max_edge_dist = 2.6  # meters
    node_list = list(g_places.nodes(data=True))
    for i in range(len(node_list)):
        u, du = node_list[i]
        for j in range(i + 1, len(node_list)):
            v, dv = node_list[j]
            dist = float(np.hypot(du["pos"][0] - dv["pos"][0], du["pos"][1] - dv["pos"][1]))
            if dist <= max_edge_dist and has_line_of_sight(grid, du["cell"], dv["cell"]):
                g_places.add_edge(u, v, relation="traversable", weight=dist)

    return g_places


def segment_rooms_and_build_dsg(
    g_places: nx.Graph,
    objects: List[Dict],
    doorway_dilation_m: float = 0.65,
    res: float = 0.1,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """Hydra-style Room Detection via ESDF Dilation + Hierarchical DSG assembly."""
    # Step A: Temporarily remove narrow-clearance Place nodes (blocks doorways!)
    wide_nodes = [n for n, d in g_places.nodes(data=True) if d["esdf"] >= doorway_dilation_m]
    sub_wide = g_places.subgraph(wide_nodes)

    room_assignment: Dict[str, int] = {}
    room_centroids: Dict[int, np.ndarray] = {}

    # Each connected component of wide places is automatically discovered as a Room!
    comps = [c for c in nx.connected_components(sub_wide) if len(c) >= 2]
    for r_idx, comp in enumerate(comps):
        pts = np.array([g_places.nodes[n]["pos"] for n in comp])
        room_centroids[r_idx] = np.mean(pts, axis=0)
        for n in comp:
            room_assignment[n] = r_idx

    # Assign remaining doorway/narrow Place nodes to nearest discovered Room
    for n, d in g_places.nodes(data=True):
        if n not in room_assignment and room_centroids:
            p = np.array(d["pos"])
            nearest_r = min(room_centroids, key=lambda r: float(np.linalg.norm(p - room_centroids[r])))
            room_assignment[n] = nearest_r

    # Step B: Assemble Hierarchical Scene Graph (Room -> Place -> Object)
    dsg = g_places.copy()
    for r_idx, center in room_centroids.items():
        r_id = f"room_{r_idx}"
        dsg.add_node(r_id, pos=(float(center[0]), float(center[1])), layer="room")

    for n in g_places.nodes():
        r_idx = room_assignment.get(n, 0)
        dsg.add_edge(n, f"room_{r_idx}", relation="belongs_to")

    # Add Room <-> Room doorway edge if any Place edge crosses between two rooms
    for u, v in g_places.edges():
        ru, rv = room_assignment.get(u, 0), room_assignment.get(v, 0)
        if ru != rv:
            dsg.add_edge(f"room_{ru}", f"room_{rv}", relation="connected_via_door")

    # Attach Object nodes to their nearest Place node
    for obj in objects:
        ox = 0.5 * (obj["x0"] + obj["x1"]) * res
        oy = 0.5 * (obj["y0"] + obj["y1"]) * res
        dsg.add_node(obj["id"], pos=(ox, oy), label=obj["label"], layer="object")
        nearest_place = min(
            g_places.nodes(),
            key=lambda p: float(np.hypot(ox - g_places.nodes[p]["pos"][0], oy - g_places.nodes[p]["pos"][1])),
        )
        dsg.add_edge(obj["id"], nearest_place, relation="near_place")

    return dsg, room_assignment


def render_4_steps(
    grid: np.ndarray,
    esdf: np.ndarray,
    g_places: nx.Graph,
    dsg: nx.Graph,
    room_assign: Dict[str, int],
    out_path: Path,
    res: float = 0.1,
) -> None:
    """Plot the 4-step Space -> Scene Graph transformation."""
    h, w = grid.shape
    extent = [0, w * res, 0, h * res]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=135)

    # Panel 1: Raw Occupancy Grid
    ax = axes[0, 0]
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Bước 1: Bản đồ ô vuông (Đen=Tường/Đồ, Trắng=Trống)", fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Panel 2: ESDF Distance Field
    ax = axes[0, 1]
    im = ax.imshow(esdf, origin="lower", extent=extent, cmap="inferno")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Khoảng cách tới vật cản (m)")
    ax.set_title("Bước 2: Bản đồ ESDF (Sáng=Rộng, Tối=Chật/Cửa hẹp)", fontweight="bold")
    ax.set_xlabel("X (m)")

    # Panel 3: Extracted Places Topological Graph
    ax = axes[1, 0]
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray", alpha=0.35)
    for u, v in g_places.edges():
        pu, pv = g_places.nodes[u]["pos"], g_places.nodes[v]["pos"]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#3498db", lw=1.4, alpha=0.75)
    for n, d in g_places.nodes(data=True):
        color = "#e74c3c" if d["esdf"] < 0.65 else "#2980b9"
        ax.scatter(d["pos"][0], d["pos"][1], c=color, s=45, edgecolors="k", linewidths=0.5, zorder=4)
    ax.set_title("Bước 3: Rút xương GVD -> Đồ thị Places (Đỏ=Nút cửa hẹp)", fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Panel 4: Final Hierarchical 3D Scene Graph (Rooms -> Places -> Objects)
    ax = axes[1, 1]
    ax.imshow(grid, origin="lower", extent=extent, cmap="gray", alpha=0.25)
    room_colors = ["#1abc9c", "#9b59b6", "#f39c12"]

    # Draw edges by hierarchy relation
    for u, v, ed in dsg.edges(data=True):
        pu, pv = dsg.nodes[u]["pos"], dsg.nodes[v]["pos"]
        rel = ed.get("relation", "")
        if rel == "belongs_to":
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#7f8c8d", ls=":", lw=0.9, alpha=0.6)
        elif rel == "near_place":
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#e67e22", ls="--", lw=1.2)
        elif rel == "connected_via_door":
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color="#e74c3c", lw=3.0, zorder=5)

    # Draw nodes by layer
    for n, d in dsg.nodes(data=True):
        layer = d.get("layer")
        p = d["pos"]
        if layer == "place":
            c = room_colors[room_assign.get(n, 0) % len(room_colors)]
            ax.scatter(p[0], p[1], c=c, s=38, edgecolors="k", linewidths=0.4, zorder=4)
        elif layer == "room":
            r_idx = int(n.split("_")[1])
            c = room_colors[r_idx % len(room_colors)]
            ax.scatter(p[0], p[1], c=c, s=260, marker="^", edgecolors="k", linewidths=1.2, zorder=6)
            ax.text(p[0], p[1] + 0.45, f"ROOM {r_idx}", ha="center", fontweight="bold", fontsize=9)
        elif layer == "object":
            ax.scatter(p[0], p[1], c="#e74c3c", s=85, marker="s", edgecolors="k", zorder=5)
            ax.text(p[0], p[1] - 0.55, d["label"], ha="center", fontsize=8, color="#c0392b", fontweight="bold")

    ax.set_title("Bước 4: Cắt cửa -> Tự gom thành 2 ROOM + Nối phân tầng DSG", fontweight="bold")
    ax.set_xlabel("X (m)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    grid, objects = build_floorplan()
    esdf = compute_esdf(grid)
    g_places = extract_places_graph(grid, esdf)
    dsg, room_assign = segment_rooms_and_build_dsg(g_places, objects)

    rooms = [n for n, d in dsg.nodes(data=True) if d.get("layer") == "room"]
    places = [n for n, d in dsg.nodes(data=True) if d.get("layer") == "place"]
    objs = [n for n, d in dsg.nodes(data=True) if d.get("layer") == "object"]
    print(f"Space -> Scene Graph complete!")
    print(f"Discovered: {len(rooms)} Rooms | {len(places)} Places | {len(objs)} Objects | {dsg.number_of_edges()} Edges")

    out_file = Path("results/space_to_graph.png")
    render_4_steps(grid, esdf, g_places, dsg, room_assign, out_file)
    print(f"Saved 4-step visualization to: {out_file.resolve()}")


if __name__ == "__main__":
    main()

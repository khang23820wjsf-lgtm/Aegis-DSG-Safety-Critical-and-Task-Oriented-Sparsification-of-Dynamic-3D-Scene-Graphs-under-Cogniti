"""Aegis-V4 (CCBS + Online ACI) Master Breakthrough Benchmark Suite.

Compares 5 DSG sparsifiers:
  1. Distance
  2. D-Lite(True)
  3. Clio(aIB)
  4. Aegis-Core-v3 (v3.0-frozen)
  5. Aegis-V4 (CCBS+ACI)

Across two real-world evaluation suites:
  (A) 6 Unseen Real Meta Project Aria Digital Twin sequences (eyegaze + 3D poses + Place hierarchy):
      - Hazard Coverage (%), CVaR-5% (%), Unmerged Atomic Recall (%)
      - Counterfactual Control Regret ||u*(G) - u*(G')||_2 (overall & CBF-active frames)
      - Online ACI Non-Exchangeable Coverage Convergence across streaming activity shifts
      - Temporal Flicker (Jaccard) & Latency (ms/frame)
  (B) 10 Real 3DSSG Scanned Rooms in MuJoCo 4.0 Rigid-Body Physics:
      - Local-A* + DWA across all baselines vs Aegis-V4 + CBF-QP Safety Filter on <= k compressed graph
      - Success Rate (%), Real Physical Collisions, SPL, and Hazard Coverage (%)

Outputs:
  - results/aegis_v4_results.json
  - results/aegis_v4_breakthrough.png
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parents[2])
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import networkx as nx
import numpy as np

from src.aegis_core import AegisConfig, AegisDSG, AegisEntity
from src.aegis_v4.adaptive_conformal import OnlineACISafetyCalibrator
from src.aegis_v4.aegis_v4_core import AegisV4CCBS, AegisV4Config
from src.aegis_v4.cbf_sparsifier import (
    build_cbf_constraints,
    compute_graph_control_regret,
    solve_cbf_qp,
)
from src.baselines import DistanceSparsifier
from src.dsg_generator import AgentState, NOISE_CONFIGS, NoiseLevel
from src.metrics import compute_hazard_recall, compute_jaccard_flicker
from src.phase_b.true_baselines import ClioTrueIBSparsifier, DLiteTrueSparsifier
from src.physics.mujoco_env import (
    EpisodeMetrics,
    _astar_subgoal,
    _is_collidable_entity,
    _plan_dwa_velocity,
    _update_local_costmap,
    build_mjcf_from_3dssg,
    run_mujoco_episode,
)
from src.real_data.loader_3dssg import load_3dssg_scan
from src.real_data.loader_aria_adt import load_aria_adt_sequence


ADT_ROOT = Path("data/aria_adt")
TEST_SEQS = [
    "Apartment_release_work_seq139_M1292",
    "Apartment_release_work_seq106_M1292",
    "Apartment_release_decoration_seq135_M1292",
    "Apartment_release_decoration_seq140_M1292",
    "Apartment_release_multiskeleton_party_seq127_M1292",
    "Apartment_release_multiuser_clean_seq120_M1292",
]


def _extract_selected_entities_and_ids(
    sparsifier: Any,
    graph_obs: nx.DiGraph,
    agent: AgentState,
    k: int,
) -> Tuple[List[Any], Set[str], Set[str]]:
    """Run sparsifier and return (sparse_entities, covered_ids, unmerged_ids)."""
    if hasattr(sparsifier, "compress"):
        entities = sparsifier.compress(graph_obs, agent, k)
        covered = set().union(*(e.member_ids for e in entities)) if entities else set()
        unmerged = set()
        for e in entities:
            if not getattr(e, "is_super_node", False) and len(e.member_ids) == 1:
                unmerged |= e.member_ids
        return entities, covered, unmerged

    sel = sparsifier.select(graph_obs, agent, k)
    entities = [(nid, graph_obs.nodes[nid]) for nid in sel if nid in graph_obs]
    return entities, sel, sel


def evaluate_aria_method(
    sparsifier: Any,
    sequences: List[list],
    k: int = 5,
    dt: float = 0.1,
) -> Dict[str, Any]:
    """Evaluate a sparsifier across the 6 unseen Meta Project Aria sequences."""
    all_recalls: List[float] = []
    cvar5_per_seq: List[float] = []
    flickers: List[float] = []
    latencies_ms: List[float] = []
    regrets_all: List[float] = []
    regrets_active: List[float] = []
    atomic_unmerged_hits = 0
    atomic_total = 0

    for seq in sequences:
        sparsifier.reset()
        seq_recalls: List[float] = []
        prev_covered: Set[str] = set()

        for idx, frame in enumerate(seq):
            g_obs = frame.graph_obs
            agent = frame.agent

            full_objs = [
                (nid, d) for nid, d in g_obs.nodes(data=True) if d.get("layer") == "object"
            ]
            obs_hazards = {nid for nid, d in full_objs if d.get("is_hazard_gt", False)}
            obs_atomics = {
                nid
                for nid, d in full_objs
                if d.get("is_hazard_gt", False) and d.get("is_atomic_hazard", False)
            }

            t0 = time.perf_counter()
            sparse_entities, covered, unmerged = _extract_selected_entities_and_ids(
                sparsifier, g_obs, agent, k
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            rec = compute_hazard_recall(covered, obs_hazards)
            seq_recalls.append(rec)
            all_recalls.append(rec)

            if obs_atomics:
                atomic_total += len(obs_atomics)
                atomic_unmerged_hits += len(unmerged & obs_atomics)

            if idx > 0:
                flickers.append(compute_jaccard_flicker(prev_covered, covered))
            prev_covered = covered

            # Counterfactual Control Regret ||u*(G_full) - u*(G_sparse)||_2
            u_nom = agent.vel[:2]
            c_full = build_cbf_constraints(full_objs, agent, dt=dt, gamma_cbf=0.10, safety_margin=0.15)
            u_full = solve_cbf_qp(u_nom, c_full, u_max=2.0)
            c_sparse = build_cbf_constraints(sparse_entities, agent, dt=dt, gamma_cbf=0.10, safety_margin=0.15)
            u_sparse = solve_cbf_qp(u_nom, c_sparse, u_max=2.0)
            regret = float(np.linalg.norm(u_full - u_sparse))
            regrets_all.append(regret)

            if float(np.linalg.norm(u_full - u_nom)) > 1e-4:
                regrets_active.append(regret)

        arr = np.sort(np.asarray(seq_recalls, dtype=float))
        w_n = max(1, int(math.ceil(0.05 * len(arr))))
        cvar5_per_seq.append(float(np.mean(arr[:w_n])))

    reg_arr = np.asarray(regrets_all, dtype=float)
    return {
        "name": sparsifier.name,
        "hazard_cov_pct": round(float(np.mean(all_recalls)) * 100.0, 2),
        "cvar5_pct": round(float(np.mean(cvar5_per_seq)) * 100.0, 2),
        "unmerged_atomic_recall_pct": round(
            (atomic_unmerged_hits / atomic_total * 100.0) if atomic_total > 0 else 100.0, 2
        ),
        "atomic_count": atomic_total,
        "control_regret_mean": round(float(np.mean(reg_arr)), 5),
        "control_regret_p95": round(float(np.percentile(reg_arr, 95)), 5),
        "control_regret_active_mean": round(
            float(np.mean(regrets_active)) if regrets_active else 0.0, 5
        ),
        "active_cbf_frames": len(regrets_active),
        "flicker_mean": round(float(np.mean(flickers)) if flickers else 0.0, 4),
        "latency_ms_mean": round(float(np.mean(latencies_ms)), 3),
    }


def evaluate_online_aci_stream(sequences: List[list], k: int = 5) -> Dict[str, Any]:
    """Stream all 6 unseen Meta Aria sequences continuously to verify Online ACI convergence."""
    v4 = AegisV4CCBS(AegisV4Config(enable_online_aci=True, aci_alpha_target=0.05, aci_eta=0.02))
    # Track both Aegis-V4's deployment proxy ACI and an explicit causal ACI monitor on hazard coverage
    eval_aci = OnlineACISafetyCalibrator(alpha_target=0.05, eta=0.02, window_size=120, default_tau=0.10)

    cum_cov_trace: List[float] = []
    alpha_trace: List[float] = []
    tau_trace: List[float] = []
    seq_boundaries: List[int] = []
    step_idx = 0

    for s_idx, seq in enumerate(sequences):
        if s_idx > 0:
            seq_boundaries.append(step_idx)
        # Keep ACI state across sequence transitions (non-exchangeable shift!)
        # Only clear per-room track IDs since each Aria sequence has its own object IDs
        v4._tracks.clear()
        v4._prev_covered.clear()
        v4._persistence.clear()
        v4.gaze_modulator.reset()

        for frame in seq:
            g_obs = frame.graph_obs
            agent = frame.agent
            obs_hazards = {
                nid for nid, d in g_obs.nodes(data=True) if d.get("layer") == "object" and d.get("is_hazard_gt", False)
            }
            ents = v4.compress(g_obs, agent, k)
            covered = set().union(*(e.member_ids for e in ents)) if ents else set()

            if obs_hazards:
                retained_scores = [
                    float(np.clip(e.risk + e.delta_cf, 0.0, 1.0))
                    for e in ents
                    if (e.member_ids & obs_hazards)
                ]
                missed = len(obs_hazards - covered) > 0
                min_score = min(retained_scores) if (retained_scores and not missed) else 0.0
                # Causal online ACI update (Gibbs & Candes): err_t = 1 iff e_t > q_t (with guard)
                res = eval_aci.update(min_retained_hazard_score=min_score)
            else:
                res = eval_aci.update(min_retained_hazard_score=1.0)

            cum_cov_trace.append(1.0 - res.empirical_miscoverage)
            alpha_trace.append(res.alpha_next)
            tau_trace.append(res.tau_t)
            step_idx += 1

    return {
        "total_frames": step_idx,
        "seq_boundaries": seq_boundaries,
        "target_coverage": 0.95,
        "final_empirical_coverage": round(eval_aci.empirical_coverage, 4),
        "final_empirical_miscoverage": round(eval_aci.empirical_miscoverage, 4),
        "final_alpha_t": round(eval_aci.alpha_t, 4),
        "proxy_aci_summary": v4.aci.summary(),
        "cum_cov_trace": [round(x, 4) for x in cum_cov_trace],
        "alpha_trace": [round(x, 4) for x in alpha_trace],
    }


def _wall_boundary_entities(room_size: Tuple[float, float]) -> List[AegisEntity]:
    """Represent the 4 static room walls as thin entities for the CBF-QP safety filter."""
    w_x, l_y = room_size
    return [
        AegisEntity("wall_L", "wall", np.array([0.0, l_y / 2.0, 1.0]), np.array([0.20, l_y, 2.0]), np.zeros(3), 1.0, 1.0, 1.0),
        AegisEntity("wall_R", "wall", np.array([w_x, l_y / 2.0, 1.0]), np.array([0.20, l_y, 2.0]), np.zeros(3), 1.0, 1.0, 1.0),
        AegisEntity("wall_B", "wall", np.array([w_x / 2.0, 0.0, 1.0]), np.array([w_x, 0.20, 2.0]), np.zeros(3), 1.0, 1.0, 1.0),
        AegisEntity("wall_F", "wall", np.array([w_x / 2.0, l_y, 1.0]), np.array([w_x, 0.20, 2.0]), np.zeros(3), 1.0, 1.0, 1.0),
    ]


def run_mujoco_cbf_episode(
    scan_id: str,
    g_room: nx.DiGraph,
    sparsifier: AegisV4CCBS,
    k: int = 5,
    noise_level: NoiseLevel = "medium",
    max_steps: int = 180,
    seed: int = 42,
    room_size: Tuple[float, float] = (8.0, 10.0),
) -> EpisodeMetrics:
    """Closed-loop MuJoCo 4.0 episode using Aegis-V4's CBF-QP safety filter on <= k compressed entities."""
    mjcf, obj_meta, dyn_peds = build_mjcf_from_3dssg(g_room, room_size)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(seed)
    noise_cfg = NOISE_CONFIGS[noise_level]
    sparsifier.reset()

    agent_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mocap_ids = [
        model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["id"])]
        for p in dyn_peds
    ]

    w_x, l_y = room_size
    start_xy = np.array([w_x / 2.0, 0.75], dtype=float)
    goal_xy = np.array([w_x / 2.0, l_y - 0.65], dtype=float)
    shortest_dist = float(np.linalg.norm(goal_xy - start_xy))
    wall_ents = _wall_boundary_entities(room_size)

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
    last_subgoal = goal_xy.copy()

    for step in range(max_steps):
        ax, ay = float(data.qpos[0] + start_xy[0]), float(data.qpos[1] + start_xy[1])
        avx, avy = float(data.qvel[0]), float(data.qvel[1])
        agent = AgentState(
            pos=np.array([ax, ay, 0.85], dtype=float),
            vel=np.array([avx, avy, 0.0], dtype=float),
            radius=0.30,
        )

        # Build observed DSG G_t^obs (identical sensor model to run_mujoco_episode)
        g_obs = nx.DiGraph()
        spd_xy = float(np.linalg.norm(agent.vel[:2]))
        heading = agent.vel[:2] / spd_xy if spd_xy >= 0.1 else np.array([0.0, 1.0])

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

            is_haz = False
            if _is_collidable_entity(m["pos"][2], m["size"][2], m["hazard_severity"]):
                rv = m["vel"][:2] - agent.vel[:2]
                v2 = float(np.dot(rv, rv))
                ts = 0.0 if v2 < 1e-6 else float(np.clip(-np.dot(delta, rv) / v2, 0.0, 3.0))
                cxy = np.abs(delta + rv * ts)
                if np.all(cxy <= m["size"][:2] / 2.0 + agent.radius + 0.25):
                    is_haz = True
                    obs_hazards.add(nid)

            conf = float(
                np.clip(
                    0.98 - 0.35 * fov_ratio ** 2 - 0.12 * (dist / agent.max_range)
                    + rng.normal(0, noise_cfg["conf_noise"]),
                    0.20,
                    0.99,
                )
            )
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

        # Query Aegis-V4 CCBS using true live agent velocity -> <= k compressed entities
        entities = sparsifier.compress(g_obs, agent, k, dt=0.1)
        covered = set().union(*(e.member_ids for e in entities)) if entities else set()
        recalls.append(1.0 if not obs_hazards else len(covered & obs_hazards) / len(obs_hazards))

        # Strictly sparsifier-fed local occupancy grid -> A* subgoal -> Nominal DWA velocity
        _update_local_costmap(local_occ, entities, agent, occ_res)
        subgoal_xy = _astar_subgoal(local_occ, agent.pos[:2], goal_xy, occ_res, lookahead_m=1.6)
        last_subgoal = subgoal_xy

        vx_nom, vy_nom = _plan_dwa_velocity(
            agent, entities, subgoal_xy, room_size, stall_steps, bumped_last_step=bumped_last_step
        )
        u_nom = np.array([vx_nom, vy_nom], dtype=float)

        # Deadlock-Free Tangential CBF-QP safety filter on <= k compressed entities (+ room walls)
        # Uses safety_margin=0.08 m aligned with DWA's clearance margin (agent.radius + 0.08).
        u_safe = sparsifier.compute_safe_velocity(
            entities=list(entities) + wall_ents,
            agent=agent,
            u_nom=u_nom,
            dt=0.1,
            u_max=1.40,
            gamma_cbf=0.35,
            safety_margin=0.08,
            enable_deadlock_escape=True,
        )

        data.ctrl[0] = float(u_safe[0])
        data.ctrl[1] = float(u_safe[1])

        # Step MuJoCo 3D rigid-body physics (20 substeps * 0.005s = 0.1s)
        step_bumped = False
        for _ in range(20):
            for p, mid in zip(dyn_peds, mocap_ids):
                p["x"] += p["vx"] * 0.005
                if p["x"] < 0.7 or p["x"] > w_x - 0.7:
                    p["vx"] *= -1.0
                data.mocap_pos[mid] = [p["x"], p["y"], 0.85]

            mujoco.mj_step(model, data)

            for ci in range(data.ncon):
                con = data.contact[ci]
                g1, g2 = int(con.geom1), int(con.geom2)
                if agent_geom_id in (g1, g2):
                    other = g2 if g1 == agent_geom_id else g1
                    if other != floor_geom_id:
                        hit_geoms.add(other)
                        step_bumped = True
        bumped_last_step = step_bumped

        curr_xy = np.array([data.qpos[0] + start_xy[0], data.qpos[1] + start_xy[1]])
        step_dist = float(np.linalg.norm(curr_xy - prev_xy))
        path_len += step_dist
        stall_steps = (stall_steps + 1) if step_dist < 0.03 else 0
        prev_xy = curr_xy

        if float(np.linalg.norm(curr_xy - goal_xy)) <= 0.85:
            reached = True
            break

    spl = (shortest_dist / max(path_len, shortest_dist)) if reached else 0.0
    return EpisodeMetrics(
        scan_id=scan_id,
        method_name=f"{sparsifier.name} + CBF-QP",
        reached_goal=reached,
        physical_collisions=len(hit_geoms),
        path_length_m=round(path_len, 2),
        spl=round(spl, 3),
        hazard_coverage_mean=float(np.mean(recalls)) if recalls else 1.0,
        steps=step + 1,
    )


def _summarize_mujoco(label: str, episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
    return {
        "name": label,
        "success_rate_pct": round(float(np.mean([e.reached_goal for e in episodes])) * 100.0, 1),
        "collisions_mean": round(float(np.mean([e.physical_collisions for e in episodes])), 2),
        "spl_mean": round(float(np.mean([e.spl for e in episodes])), 3),
        "hazard_cov_pct": round(float(np.mean([e.hazard_coverage_mean for e in episodes])) * 100.0, 1),
    }


def plot_v4_breakthrough(
    aria_results: List[Dict[str, Any]],
    aci_stream: Dict[str, Any],
    mujoco_results: List[Dict[str, Any]],
    out_png: Path,
) -> None:
    """Generate clean, non-overlapping 4-panel publication-grade breakthrough figure."""
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#333333", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.8), dpi=160)

    short_names = ["Distance", "D-Lite\n(True)", "Clio\n(aIB)", "Aegis-v3\n(Frozen)", "Aegis-V4\n(CCBS+ACI)"]
    palette = ["#5dade2", "#9b59b6", "#7f8c8d", "#f39c12", "#c0392b"]

    # ------------------------------------------------------------------
    # Panel (a): Counterfactual Control Regret ||u*(G) - u*(G')||_2 on Meta Aria
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    reg_active = [r["control_regret_active_mean"] for r in aria_results]
    reg_all = [r["control_regret_mean"] for r in aria_results]
    x = np.arange(len(short_names))
    w = 0.36
    b1 = ax.bar(x - w / 2, reg_active, w, color=palette, alpha=0.95, edgecolor="#222", linewidth=0.6, label="Active-CBF Frames")
    b2 = ax.bar(x + w / 2, reg_all, w, color=palette, alpha=0.45, edgecolor="#222", linewidth=0.6, hatch="//", label="All Frames")
    y_max_a = max(max(reg_active), max(reg_all), 0.05) * 1.28
    for bar, val in zip(b1, reg_active):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + y_max_a * 0.025,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )
    ax.set_title("(a) Counterfactual Control Regret ||u*(G) - u*(G')||₂ ↓ (6 Unseen Meta Aria)", fontweight="bold", fontsize=10.5, pad=10)
    ax.set_ylabel("Control Distortion (m/s) ↓")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8.8, fontweight="bold")
    ax.set_ylim(0, y_max_a)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.3)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel (b): Meta Aria Safety Metrics (Hazard Cov, CVaR-5%, Unmerged Atomic Recall)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    cov_vals = [r["hazard_cov_pct"] for r in aria_results]
    cvar_vals = [r["cvar5_pct"] for r in aria_results]
    atom_vals = [r["unmerged_atomic_recall_pct"] for r in aria_results]
    w3 = 0.25
    bb1 = ax.bar(x - w3, cov_vals, w3, label="Hazard Coverage (%)", color="#2980b9", alpha=0.92)
    bb2 = ax.bar(x, cvar_vals, w3, label="CVaR-5% Worst Frames (%)", color="#34495e", alpha=0.90)
    bb3 = ax.bar(x + w3, atom_vals, w3, label="Unmerged Atomic Recall (%)", color="#e74c3c", alpha=0.92)
    for bars in (bb1, bb2, bb3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.4,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
            )
    ax.set_title("(b) Safety Coverage & Atomic Preservation on 6 Unseen Meta Aria (k=5)", fontweight="bold", fontsize=10.5, pad=10)
    ax.set_ylabel("Recall / Coverage (%) ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8.8, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel (c): Online ACI Non-Exchangeable Coverage Convergence
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    t_steps = np.arange(1, len(aci_stream["cum_cov_trace"]) + 1)
    cov_pct = np.asarray(aci_stream["cum_cov_trace"]) * 100.0
    ax.plot(t_steps, cov_pct, color="#c0392b", linewidth=2.1, label="Online ACI Empirical Coverage (%)")
    ax.axhline(95.0, color="#16a085", linestyle="--", linewidth=1.6, label="Target Coverage 1 - α = 95.0%")
    for b_idx, b_step in enumerate(aci_stream["seq_boundaries"]):
        ax.axvline(b_step, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.75)
        if b_idx == 0:
            ax.text(b_step + 3, 100.3, "Activity Shifts →", fontsize=8.0, color="#555555", va="bottom")
    final_cov = aci_stream["final_empirical_coverage"] * 100.0
    ax.annotate(
        f"Final: {final_cov:.1f}%\n(α_T={aci_stream['final_alpha_t']:.3f})",
        xy=(t_steps[-1], cov_pct[-1]),
        xytext=(max(10, t_steps[-1] * 0.56), 86.5),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.1),
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8f6", edgecolor="#c0392b", alpha=0.95),
    )
    ax.set_title("(c) Non-Exchangeable Online ACI Convergence (6 Streaming Aria Shifts)", fontweight="bold", fontsize=10.5, pad=10)
    ax.set_xlabel("Streaming Egocentric Frame (t)")
    ax.set_ylabel("Cumulative Hazard Coverage (%)")
    ax.set_ylim(82, 103)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.3)
    ax.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel (d): 10 Real 3DSSG Rooms in MuJoCo 4.0 (Success Rate & Physical Collisions)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    mj_labels = [
        "Distance\n(DWA)",
        "D-Lite\n(DWA)",
        "Clio\n(DWA)",
        "Aegis-v3\n(DWA)",
        "Aegis-V4\n(DWA)",
        "Aegis-V4\n(CBF-QP)",
    ]
    mj_colors = ["#5dade2", "#9b59b6", "#7f8c8d", "#f39c12", "#e67e22", "#c0392b"]
    xm = np.arange(len(mujoco_results))
    sr_vals = [m["success_rate_pct"] for m in mujoco_results]
    col_vals = [m["collisions_mean"] for m in mujoco_results]

    bars_sr = ax.bar(xm, sr_vals, 0.56, color=mj_colors[: len(mujoco_results)], alpha=0.90, edgecolor="#222", linewidth=0.6, label="Success Rate (%) ↑")
    ax.set_ylabel("Success Rate (%) ↑", color="#222222")
    ax.set_ylim(0, 122)
    ax.set_xticks(xm)
    ax.set_xticklabels(mj_labels[: len(mujoco_results)], fontsize=8.3, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    ax2 = ax.twinx()
    line_col = ax2.plot(xm, col_vals, color="#1a252f", marker="o", linewidth=2.0, markersize=6.5, label="Physical Collisions ↓")
    ax2.set_ylabel("Mean Physical Collisions ↓", color="#1a252f")
    col_top = max(max(col_vals), 1.0) * 1.42
    ax2.set_ylim(0, col_top)

    for idx_m, (bar, sr, col) in enumerate(zip(bars_sr, sr_vals, col_vals)):
        # Place SR% cleanly inside the lower-middle of each bar so it NEVER collides with collision line
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(12.0, sr * 0.22),
            f"{sr:.0f}%",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
        )
        # Place collision value cleanly above each marker
        ax2.text(
            xm[idx_m],
            col + col_top * 0.045,
            f"{col:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#1a252f",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.92, fontsize=8.0)
    ax.set_title("(d) 10 Real 3DSSG Rooms × MuJoCo 4.0 Rigid-Body Physics (k=5)", fontweight="bold", fontsize=10.5, pad=10)

    fig.suptitle(
        "Aegis-V4 (CCBS + Online ACI): Counterfactual Control-Barrier Sparsification on Real Meta Aria & 3DSSG × MuJoCo 4.0",
        fontsize=12.2,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955], h_pad=2.4, w_pad=2.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("AEGIS-V4 (CCBS + Online ACI) BREAKTHROUGH BENCHMARK SUITE")
    print("=" * 78)

    # ------------------------------------------------------------------
    # (A) 6 Unseen Real Meta Project Aria Digital Twin Sequences
    # ------------------------------------------------------------------
    print("\n[A] Loading 6 Unseen Real Meta Project Aria sequences...", flush=True)
    test_data = [load_aria_adt_sequence(ADT_ROOT / s, max_frames=35, stride=8)[0] for s in TEST_SEQS]

    aria_methods = [
        DistanceSparsifier(),
        DLiteTrueSparsifier(),
        ClioTrueIBSparsifier(),
        AegisDSG(AegisConfig()),
        AegisV4CCBS(AegisV4Config()),
    ]

    aria_results: List[Dict[str, Any]] = []
    print("\n--- (A) 6 Unseen Meta Project Aria Results (k=5) ---")
    header = f"{'Method':22s} | {'Cov%':>6s} | {'CVaR5%':>6s} | {'Atom%':>6s} | {'Regret(All)':>11s} | {'Regret(Act)':>11s} | {'Flicker':>7s} | {'ms/fr':>6s}"
    print(header)
    print("-" * len(header))
    for m in aria_methods:
        res = evaluate_aria_method(m, test_data, k=5)
        aria_results.append(res)
        print(
            f"{res['name']:22s} | {res['hazard_cov_pct']:6.2f} | {res['cvar5_pct']:6.2f} | "
            f"{res['unmerged_atomic_recall_pct']:6.2f} | {res['control_regret_mean']:11.5f} | "
            f"{res['control_regret_active_mean']:11.5f} | {res['flicker_mean']:7.4f} | {res['latency_ms_mean']:6.2f}"
        )

    print("\n--- (A2) Non-Exchangeable Online ACI Convergence (6 Streaming Aria Shifts) ---")
    aci_stream = evaluate_online_aci_stream(test_data, k=5)
    print(
        f"  Frames={aci_stream['total_frames']} | Target Cov=95.0% | "
        f"Final Empirical Cov={aci_stream['final_empirical_coverage']*100:.2f}% | "
        f"alpha_T={aci_stream['final_alpha_t']:.4f} | Proxy ACI={aci_stream['proxy_aci_summary']}"
    )

    # ------------------------------------------------------------------
    # (B) 10 Real 3DSSG Scanned Rooms in MuJoCo 4.0 Rigid-Body Physics
    # ------------------------------------------------------------------
    print("\n[B] Running 10 Real 3DSSG Scanned Rooms in MuJoCo 4.0 Rigid-Body Physics (k=5, medium noise)...")
    scans = [load_3dssg_scan(i, seed=42 + i) for i in range(10)]

    mj_dwa_factories = [
        ("Distance (DWA)", lambda: DistanceSparsifier()),
        ("D-Lite(True) (DWA)", lambda: DLiteTrueSparsifier()),
        ("Clio(aIB) (DWA)", lambda: ClioTrueIBSparsifier()),
        ("Aegis-Core-v3 (DWA)", lambda: AegisDSG(AegisConfig())),
        ("Aegis-V4 (DWA)", lambda: AegisV4CCBS(AegisV4Config())),
    ]

    mujoco_results: List[Dict[str, Any]] = []
    mj_header = f"{'Method':28s} | {'SR%':>6s} | {'Collisions':>10s} | {'SPL':>6s} | {'Cov%':>6s}"
    print("\n--- (B) 10 Real 3DSSG × MuJoCo 4.0 Results (k=5) ---")
    print(mj_header)
    print("-" * len(mj_header))

    for label, factory in mj_dwa_factories:
        eps = [
            run_mujoco_episode(sid, g_room, factory(), k=5, noise_level="medium", seed=42 + idx)
            for idx, (sid, g_room) in enumerate(scans)
        ]
        summ = _summarize_mujoco(label, eps)
        mujoco_results.append(summ)
        print(
            f"{summ['name']:28s} | {summ['success_rate_pct']:6.1f} | {summ['collisions_mean']:10.2f} | "
            f"{summ['spl_mean']:6.3f} | {summ['hazard_cov_pct']:6.1f}"
        )

    # Aegis-V4 + CBF-QP Safety Filter directly on <= k compressed graph
    eps_v4_cbf = [
        run_mujoco_cbf_episode(
            sid, g_room, AegisV4CCBS(AegisV4Config()), k=5, noise_level="medium", seed=42 + idx
        )
        for idx, (sid, g_room) in enumerate(scans)
    ]
    summ_v4_cbf = _summarize_mujoco("Aegis-V4 (CCBS+CBF-QP)", eps_v4_cbf)
    mujoco_results.append(summ_v4_cbf)
    print(
        f"{summ_v4_cbf['name']:28s} | {summ_v4_cbf['success_rate_pct']:6.1f} | "
        f"{summ_v4_cbf['collisions_mean']:10.2f} | {summ_v4_cbf['spl_mean']:6.3f} | "
        f"{summ_v4_cbf['hazard_cov_pct']:6.1f}"
    )

    # Save JSON and 4-panel breakthrough figure
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "aria_unseen_6seq_k5": aria_results,
        "online_aci_stream": aci_stream,
        "mujoco_3dssg_10rooms_k5": mujoco_results,
    }
    out_json = out_dir / "aegis_v4_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_png = out_dir / "aegis_v4_breakthrough.png"
    plot_v4_breakthrough(aria_results, aci_stream, mujoco_results, out_png)

    print(f"\nSaved JSON results -> {out_json.resolve()}")
    print(f"Saved Breakthrough Figure -> {out_png.resolve()}")


if __name__ == "__main__":
    main()

# Phase B Architecture Design — Aegis-DSG v2

Building directly on Phase A findings (EXP 1–4) and MIT SPARK's **Hydra / D-Lite / Clio** ecosystem, Phase B introduces 4 core modules under `src/phase_b/`:

## 1. Free-Space-Preserving Graph Coarsening (`src/phase_b/coarsening.py`)
- **Problem (Phase A EXP 3):** $37.4\%$ of missed hazards at $k=5$ occur when $>k$ hazards appear simultaneously.
- **Solution:** Inspired by Hydra's Layer-3 `Places` (navigable free space), when high-risk hazards exceed budget $k$, iteratively merge pairs of adjacent hazards into a **Super-Node** (enclosing 3D bounding box) **ONLY IF** the physical clearance gap between them is $< 0.70\text{m}$ (impassable by human shoulders).
- **Guarantee:** Preserves all navigable narrow gaps (avoids Over-Coarsening / Freezing Robot) while covering $>k$ hazards within budget $k$.

## 2. FP-Aware Asymmetric Safety Score (`src/phase_b/aegis_v2.py`)
- **Problem (Phase A EXP 2):** Single-frame low-confidence false-positive ghosts (`fp_ghost`) abused $\gamma = 1.5$ to steal top-$k$ slots under heavy noise.
- **Solution:** Track temporal observation persistence (`seen_frames`). Transient 1-frame ghosts with $Q_v < 0.45$ receive damped uncertainty weight ($0.2\gamma$), while persistent occluded real hazards (`seen_frames >= 2`) receive full asymmetric priority boost.

## 3. Adapted Robotics Baselines (`src/phase_b/robotics_baselines.py`)
- **D-Lite (Adapted):** Topological shortest-path & structural anchor preservation toward the global goal, capped at $k$ nodes.
- **Clio (Adapted):** Task-driven semantic relevance & semantic clustering for `"avoid obstacles and navigate safely forward"`, capped at $k$ nodes without geometric TTC/hysteresis.

## 4. Closed-Loop Python DWA Evaluation (`src/phase_b/closed_loop_dwa.py` + `run_phase_b.py`)
- Evaluates `Distance`, `D-Lite (Adapted)`, `Clio (Adapted)`, `Phase A Proposed`, and `Phase B Aegis-v2` in closed loop using a standardized DWA + Gap-Seeking proxy planner.
- Metrics: **Collisions / 10m**, **Effective Hazard Coverage** (counting super-node coverage), **SPL**, and **Flicker $F_t$**.

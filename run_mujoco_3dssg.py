"""Mũi 1A + 1B Benchmark: Real 3DSSG Scanned Rooms inside MuJoCo 4.0 3D Rigid-Body Physics.

Usage:
  python run_mujoco_3dssg.py          # Headless benchmark across 10 real 3DSSG rooms
  python run_mujoco_3dssg.py --view   # Launch interactive 3D MuJoCo OpenGL GUI window!
"""

from __future__ import annotations

import argparse
from typing import List
import numpy as np

from src.aegis_core import AegisConfig, AegisDSG
from src.baselines import DistanceSparsifier
from src.phase_b.true_baselines import ClioTrueIBSparsifier, DLiteTrueSparsifier
from src.physics.mujoco_env import run_mujoco_episode
from src.real_data.loader_3dssg import load_3dssg_scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="Open interactive 3D MuJoCo GUI window")
    parser.add_argument("--scan", type=int, default=0, help="3DSSG scan index for --view mode")
    parser.add_argument("--k", type=int, default=5, help="Cognitive budget k")
    args = parser.parse_args()

    if args.view:
        scan_id, g_room = load_3dssg_scan(args.scan)
        print(f"Launching 3D MuJoCo Viewer on REAL 3DSSG scan {scan_id} (Aegis-Core v2.0, k={args.k})...")
        m = run_mujoco_episode(scan_id, g_room, AegisDSG(AegisConfig()), k=args.k, launch_viewer=True)
        print(f"Finished: reached={m.reached_goal} | real_collisions={m.physical_collisions} | SPL={m.spl} | coverage={m.hazard_coverage_mean*100:.1f}%")
        return

    scan_indices = list(range(10))
    methods = [
        DistanceSparsifier(),
        DLiteTrueSparsifier(),
        ClioTrueIBSparsifier(),
        AegisDSG(AegisConfig()),
    ]

    print(f"\n=== REAL 3DSSG (10 Scanned Rooms) x MuJoCo 4.0 Rigid-Body Physics (Local-A*+DWA, k={args.k}, medium noise) ===")
    header = f"{'Method':<14} | {'Success Rate':>12} | {'Real Collisions':>15} | {'SPL (↑)':>8} | {'Hazard Coverage':>15}"
    print(header)
    print("-" * len(header))

    scans = [load_3dssg_scan(i, seed=42 + i) for i in scan_indices]
    for method in methods:
        ep_results = [
            run_mujoco_episode(sid, g_room, method, k=args.k, noise_level="medium", seed=42 + idx)
            for idx, (sid, g_room) in enumerate(scans)
        ]
        sr = float(np.mean([r.reached_goal for r in ep_results])) * 100
        cols = float(np.mean([r.physical_collisions for r in ep_results]))
        spl = float(np.mean([r.spl for r in ep_results]))
        cov = float(np.mean([r.hazard_coverage_mean for r in ep_results])) * 100
        print(f"{method.name:<14} | {sr:11.1f}% | {cols:15.2f} | {spl:8.3f} | {cov:14.1f}%")


if __name__ == "__main__":
    main()

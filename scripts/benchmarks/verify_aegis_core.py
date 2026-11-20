"""End-to-end verification of the unified AegisDSG core (src/aegis_core.py)
across 4 unseen environments (medium & heavy noise) and 10 real 3DSSG rooms in MuJoCo 4.0.
"""

from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path; _ROOT = str(_Path(__file__).resolve().parents[2]); (_sys.path.insert(0, _ROOT) if _ROOT not in _sys.path else None)

import numpy as np

from src.aegis_core import AegisDSG
from src.dsg_generator import generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.physics.mujoco_env import run_mujoco_episode
from src.real_data.loader_3dssg import load_3dssg_scan


def main() -> None:
    aegis = AegisDSG()
    print("=== 1) Unified AegisDSG Core on Unseen Test Scenes (k=5, seeds 100..104) ===")
    print(f"{'Noise':<6} | {'Scenario':<16} | {'Hazard Cov':>10} | {'CVaR-5%':>8} | {'Flicker':>7} | {'Avg |V|':>8} | {'Latency':>8}")
    print("-" * 79)

    for noise in ("medium", "heavy"):
        for sc in ("corridor", "l_turn_corner", "crowded_lobby", "vertical_hazards"):
            seqs = [
                generate_dsg_sequence(120, 0.1, 42, noise_level=noise, seed=s, scenario=sc)
                for s in range(100, 105)
            ]
            runs = [evaluate_sparsifier(aegis, seq, 5) for seq in seqs]
            counts = []
            for seq in seqs:
                aegis.reset()
                for f in seq:
                    counts.append(len(aegis.compress(f.graph_obs, f.agent, 5)))
            cov = float(np.mean([r["hazard_recall_mean"] for r in runs])) * 100
            cvar = float(np.mean([r["hazard_recall_cvar5"] for r in runs])) * 100
            fl = float(np.mean([r["flicker_mean"] for r in runs]))
            lat = float(np.mean([r["latency_ms_mean"] for r in runs]))
            print(f"{noise:<6} | {sc:<16} | {cov:9.1f}% | {cvar:7.1f}% | {fl:7.3f} | {np.mean(counts):5.2f}/5  | {lat:6.2f}ms")
        print("-" * 79)

    print("\n=== 2) Unified AegisDSG Core on 10 Real 3DSSG Scanned Rooms x MuJoCo 4.0 (k=5) ===")
    eps = [
        run_mujoco_episode(*load_3dssg_scan(i, seed=42 + i), aegis, k=5, noise_level="medium", seed=42 + i)
        for i in range(10)
    ]
    sr = float(np.mean([e.reached_goal for e in eps])) * 100
    cols = float(np.mean([e.physical_collisions for e in eps]))
    spl = float(np.mean([e.spl for e in eps]))
    cov = float(np.mean([e.hazard_coverage_mean for e in eps])) * 100
    print(f"Success Rate={sr:.1f}% | Real MuJoCo Collisions={cols:.2f} | SPL={spl:.3f} | Hazard Coverage={cov:.1f}%")


if __name__ == "__main__":
    main()

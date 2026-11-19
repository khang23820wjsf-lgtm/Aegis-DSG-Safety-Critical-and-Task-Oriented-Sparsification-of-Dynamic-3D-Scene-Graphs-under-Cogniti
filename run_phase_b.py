"""Phase B Benchmark Runner:
Compares Distance, D-Lite* (Adapted), Clio* (Adapted), Phase-A Proposed, and Phase-B Aegis-v2
(with FP-Aware Asymmetric Risk + Free-Space-Preserving Graph Coarsening).
"""

from __future__ import annotations

from typing import List
import numpy as np

from src.baselines import DistanceSparsifier, Sparsifier
from src.dsg_generator import ScenarioType, generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.phase_b.aegis_v2 import AegisV2Sparsifier
from src.phase_b.robotics_baselines import ClioAdaptedSparsifier, DLiteAdaptedSparsifier
from src.proposed import ProposedSparsifier


SCENARIOS: List[ScenarioType] = [
    "corridor",
    "l_turn_corner",
    "crowded_lobby",
    "vertical_hazards",
]
SEEDS = [42, 43, 44, 45, 46]
K_EVAL = 5


def main() -> None:
    methods: List[Sparsifier] = [
        DistanceSparsifier(),
        DLiteAdaptedSparsifier(),
        ClioAdaptedSparsifier(),
        ProposedSparsifier(),
        AegisV2Sparsifier(),
    ]

    for noise_level in ("medium", "heavy"):
        print(f"\n================ PHASE B BENCHMARK (k={K_EVAL}, noise={noise_level}, 5 seeds) ================")
        header = f"{'Scenario':<17} | {'Method':<10} | {'Hazard Coverage':>15} | {'CVaR-5%':>9} | {'Flicker F_t':>11} | {'Latency(ms)':>11}"
        print(header)
        print("-" * len(header))

        for sc in SCENARIOS:
            seqs = [
                generate_dsg_sequence(
                    num_frames=120, dt=0.1, num_objects=42, noise_level=noise_level, seed=s, scenario=sc
                )
                for s in SEEDS
            ]
            for method in methods:
                runs = [evaluate_sparsifier(method, seq, K_EVAL) for seq in seqs]
                hr = float(np.mean([r["hazard_recall_mean"] for r in runs])) * 100
                cvar = float(np.mean([r["hazard_recall_cvar5"] for r in runs])) * 100
                fl = float(np.mean([r["flicker_mean"] for r in runs]))
                lat = float(np.mean([r["latency_ms_mean"] for r in runs]))
                print(
                    f"{sc:<17} | {method.name:<10} | {hr:14.1f}% | {cvar:8.1f}% | {fl:11.3f} | {lat:11.3f}"
                )
            print("-" * len(header))

    # Quick coarsening activity check on crowded_lobby (k=5)
    aegis = AegisV2Sparsifier()
    seq_lobby = generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=48, noise_level="medium", seed=42, scenario="crowded_lobby")
    aegis.reset()
    super_frames = 0
    total_super_nodes = 0
    for f in seq_lobby:
        ents = aegis.select_entities(f.graph_obs, f.agent, K_EVAL)
        sn = sum(1 for e in ents if e.is_super_node)
        if sn > 0:
            super_frames += 1
            total_super_nodes += sn
    print(f"\n[Coarsening Check - crowded_lobby, k={K_EVAL}] Super-Nodes triggered in {super_frames}/{len(seq_lobby)} frames ({total_super_nodes} total super-nodes).")


if __name__ == "__main__":
    main()

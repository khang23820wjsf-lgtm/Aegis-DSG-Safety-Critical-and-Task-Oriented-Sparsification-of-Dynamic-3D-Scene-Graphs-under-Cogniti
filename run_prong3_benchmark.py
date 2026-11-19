"""Mũi 3 Grand Benchmark: True D-Lite (Dijkstra) & True Clio (Agglomerative IB)
vs. Distance, GNN-Aegis (PyTorch GAT), Submod+Coars (1-1/e), and Aegis-v2!

Evaluated on UNSEEN test scenes (seeds 100..104) across all 4 environments at k=5.
"""

from __future__ import annotations

from typing import List
import numpy as np

from src.baselines import DistanceSparsifier, Sparsifier
from src.dsg_generator import ScenarioType, generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.phase_b.aegis_v2 import AegisV2Sparsifier
from src.phase_b.gnn_aegis import GNNAegisSparsifier
from src.phase_b.submodular_aegis import SubmodularAegisSparsifier
from src.phase_b.true_baselines import ClioTrueIBSparsifier, DLiteTrueSparsifier


SCENARIOS: List[ScenarioType] = ["corridor", "l_turn_corner", "crowded_lobby", "vertical_hazards"]
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5]
TEST_SEEDS = [100, 101, 102, 103, 104]
K_EVAL = 5


def main() -> None:
    print("Training GNN-Aegis (2-layer PyTorch GAT) on training split (seeds 0..5)...")
    train_seqs = [
        generate_dsg_sequence(num_frames=80, dt=0.1, num_objects=42, noise_level="medium", seed=s, scenario=sc)
        for sc in SCENARIOS
        for s in TRAIN_SEEDS
    ]
    gnn = GNNAegisSparsifier(fn_weight=5.0, hysteresis_bonus=0.8)
    gnn.train_on_sequences(train_seqs, epochs=10, lr=5e-3)

    methods: List[Sparsifier] = [
        DistanceSparsifier(),
        DLiteTrueSparsifier(),
        ClioTrueIBSparsifier(),
        gnn,
        SubmodularAegisSparsifier(enable_coarsening=True),
        AegisV2Sparsifier(),
    ]

    print(f"\n=== GRAND BENCHMARK: TRUE BASELINES vs. AEGIS FAMILY (k={K_EVAL}, unseen seeds 100..104) ===")
    header = f"{'Scenario':<17} | {'Method':<13} | {'Hazard Cov.':>11} | {'CVaR-5%':>9} | {'Flicker F_t':>11} | {'Latency(ms)':>11}"
    print(header)
    print("-" * len(header))

    for sc in SCENARIOS:
        test_seqs = [
            generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=s, scenario=sc)
            for s in TEST_SEEDS
        ]
        for m in methods:
            runs = [evaluate_sparsifier(m, seq, K_EVAL) for seq in test_seqs]
            hr = float(np.mean([r["hazard_recall_mean"] for r in runs])) * 100
            cvar = float(np.mean([r["hazard_recall_cvar5"] for r in runs])) * 100
            fl = float(np.mean([r["flicker_mean"] for r in runs]))
            lat = float(np.mean([r["latency_ms_mean"] for r in runs]))
            print(f"{sc:<17} | {m.name:<13} | {hr:10.1f}% | {cvar:8.1f}% | {fl:11.3f} | {lat:11.3f}")
        print("-" * len(header))


if __name__ == "__main__":
    main()

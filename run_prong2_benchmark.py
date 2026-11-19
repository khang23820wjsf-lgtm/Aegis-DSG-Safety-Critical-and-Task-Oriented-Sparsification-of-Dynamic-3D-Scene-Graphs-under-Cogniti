"""Mũi 2 Benchmark: Principled Optimization (Submodular 1 - 1/e) & Learned GNN (PyTorch GAT).

Strictly follows scene-level train/test split:
- Train seeds: [0..7] across all 4 environments (for GNN-Aegis).
- Unseen Test seeds: [100..104] across all 4 environments (zero data leakage!).
"""

from __future__ import annotations

from typing import List
import time
import numpy as np

from src.baselines import DistanceSparsifier, Sparsifier
from src.dsg_generator import ScenarioType, generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.phase_b.gnn_aegis import GNNAegisSparsifier
from src.phase_b.submodular_aegis import SubmodularAegisSparsifier
from src.proposed import ProposedSparsifier


SCENARIOS: List[ScenarioType] = ["corridor", "l_turn_corner", "crowded_lobby", "vertical_hazards"]
TRAIN_SEEDS = [0, 1, 2, 3, 4, 5]
TEST_SEEDS = [100, 101, 102, 103, 104]
K_EVAL = 5


def main() -> None:
    print("1) Generating scene-level training split for GNN-Aegis (6 seeds x 4 scenarios)...")
    train_seqs = [
        generate_dsg_sequence(num_frames=80, dt=0.1, num_objects=42, noise_level="medium", seed=s, scenario=sc)
        for sc in SCENARIOS
        for s in TRAIN_SEEDS
    ]

    gnn = GNNAegisSparsifier(fn_weight=5.0, hysteresis_bonus=0.8)
    t0 = time.perf_counter()
    final_loss = gnn.train_on_sequences(train_seqs, epochs=10, lr=5e-3)
    print(f"   Trained 2-layer PyTorch GAT in {time.perf_counter() - t0:.2f}s (final Asymmetric BCE loss={final_loss:.4f}).")

    methods: List[Sparsifier] = [
        DistanceSparsifier(),
        ProposedSparsifier(),
        GNNAegisSparsifier.__new__(GNNAegisSparsifier) if False else gnn,
        SubmodularAegisSparsifier(enable_coarsening=False),  # Pure Submodular (1-1/e) without coarsening
        SubmodularAegisSparsifier(enable_coarsening=True),   # Full Submodular + Free-Space Coarsening
    ]
    methods[3].name = "Submod-Pure"
    methods[4].name = "Submod+Coars"

    print(f"\n=== MŨI 2 BENCHMARK ON UNSEEN TEST SCENES (k={K_EVAL}, medium noise, seeds 100..104) ===")
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

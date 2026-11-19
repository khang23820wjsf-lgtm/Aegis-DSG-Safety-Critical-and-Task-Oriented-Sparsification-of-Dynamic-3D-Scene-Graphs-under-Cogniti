"""Multi-Environment Benchmark for Phase A (4 Diverse 3D Scenarios)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from src.baselines import DistanceSparsifier, RandomSparsifier, SemanticSparsifier, Sparsifier
from src.dsg_generator import ScenarioType, generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.proposed import ProposedSparsifier


SCENARIOS: List[ScenarioType] = [
    "corridor",
    "l_turn_corner",
    "crowded_lobby",
    "vertical_hazards",
]
SCENARIO_TITLES: Dict[ScenarioType, str] = {
    "corridor": "1. Straight Corridor",
    "l_turn_corner": "2. 90° Blind Corner",
    "crowded_lobby": "3. Crowded Lobby",
    "vertical_hazards": "4. 3D Vertical Hazards",
}
SEEDS = [42, 43, 44, 45, 46]
K_EVAL = 5


def run_multi_scenario() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate all 4 methods across all 4 environments (5 seeds, k=5, medium noise)."""
    methods: List[Sparsifier] = [
        RandomSparsifier(seed=0),
        DistanceSparsifier(),
        SemanticSparsifier(),
        ProposedSparsifier(),
    ]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sc in SCENARIOS:
        seqs = [
            generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=s, scenario=sc)
            for s in SEEDS
        ]
        results[sc] = {}
        for method in methods:
            runs = [evaluate_sparsifier(method, seq, K_EVAL) for seq in seqs]
            agg: Dict[str, float] = {}
            for key in runs[0]:
                vals = np.array([r[key] for r in runs], dtype=float)
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_ci95"] = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
            results[sc][method.name] = agg

    return results


def print_summary(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print(f"\n=== Multi-Scenario Benchmark (k={K_EVAL}, medium noise, 5 seeds) ===")
    header = f"{'Scenario':<18} | {'Method':<10} | {'Hazard Recall':>16} | {'CVaR-5%':>10} | {'Flicker F_t':>11}"
    print(header)
    print("-" * len(header))
    for sc in SCENARIOS:
        for mname in ("Random", "Distance", "Semantic", "Proposed"):
            d = results[sc][mname]
            hr = f"{d['hazard_recall_mean_mean']*100:5.1f}% ± {d['hazard_recall_mean_ci95']*100:4.1f}%"
            cvar = f"{d['hazard_recall_cvar5_mean']*100:6.1f}%"
            fl = f"{d['flicker_mean_mean']:.3f}"
            print(f"{sc:<18} | {mname:<10} | {hr:>16} | {cvar:>10} | {fl:>11}")
        print("-" * len(header))


def plot_scenarios(results: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    """Plot grouped bar chart of Hazard Recall and Flicker across all 4 environments."""
    method_names = ["Random", "Semantic", "Distance", "Proposed"]
    colors = ["#95a5a6", "#27ae60", "#2980b9", "#e74c3c"]
    x = np.arange(len(SCENARIOS))
    width = 0.19

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=140)

    # Panel 1: Hazard Recall
    ax1 = axes[0]
    for i, (mname, col) in enumerate(zip(method_names, colors)):
        means = [results[sc][mname]["hazard_recall_mean_mean"] * 100 for sc in SCENARIOS]
        cis = [results[sc][mname]["hazard_recall_mean_ci95"] * 100 for sc in SCENARIOS]
        ax1.bar(x + (i - 1.5) * width, means, width, yerr=cis, capsize=3, label=mname, color=col, alpha=0.92)

    ax1.set_title(f"Hazard Recall Across 4 Environments (k={K_EVAL})", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Hazard Recall (%) ↑")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCENARIO_TITLES[sc] for sc in SCENARIOS], fontsize=9)
    ax1.set_ylim(0, 108)
    ax1.grid(axis="y", alpha=0.3, ls="--")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Flicker Rate
    ax2 = axes[1]
    for i, (mname, col) in enumerate(zip(method_names, colors)):
        means = [results[sc][mname]["flicker_mean_mean"] for sc in SCENARIOS]
        cis = [results[sc][mname]["flicker_mean_ci95"] for sc in SCENARIOS]
        ax2.bar(x + (i - 1.5) * width, means, width, yerr=cis, capsize=3, label=mname, color=col, alpha=0.92)

    ax2.set_title(f"Temporal Flicker F_t Across 4 Environments (k={K_EVAL})", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Jaccard Flicker F_t ↓ (lower is better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIO_TITLES[sc] for sc in SCENARIOS], fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3, ls="--")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved multi-scenario chart to: {out_path.resolve()}")


def main() -> None:
    results = run_multi_scenario()
    print_summary(results)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenarios_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_scenarios(results, out_dir / "scenarios_benchmark.png")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from src.baselines import DistanceSparsifier, RandomSparsifier, SemanticSparsifier, Sparsifier
from src.dsg_generator import generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.proposed import ProposedSparsifier


K_BUDGETS = [3, 5, 7, 10, 15]
SEEDS = [42, 43, 44, 45, 46]


def run_all() -> Dict[str, Dict[int, Dict[str, float]]]:
    """Run multi-seed benchmark and aggregate mean & 95% CI for each method and k."""
    methods: List[Sparsifier] = [
        RandomSparsifier(seed=0),
        DistanceSparsifier(),
        SemanticSparsifier(),
        ProposedSparsifier(),
    ]

    # Pre-generate sequences per seed so every method sees the exact same data
    sequences = {
        seed: generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=seed)
        for seed in SEEDS
    }

    summary: Dict[str, Dict[int, Dict[str, float]]] = {m.name: {} for m in methods}

    for method in methods:
        for k in K_BUDGETS:
            seed_metrics = [evaluate_sparsifier(method, sequences[s], k) for s in SEEDS]
            agg: Dict[str, float] = {}
            for key in seed_metrics[0]:
                vals = np.array([m[key] for m in seed_metrics], dtype=float)
                mean = float(np.mean(vals))
                ci95 = float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
                agg[f"{key}_mean"] = mean
                agg[f"{key}_ci95"] = ci95
            summary[method.name][k] = agg

    return summary


def print_table(summary: Dict[str, Dict[int, Dict[str, float]]]) -> None:
    """Print a clean ASCII summary table to stdout."""
    print("\n=== Phase A Benchmark Summary (5 seeds, medium noise, 42 objects) ===")
    header = f"{'Method':<10} | {'k':>2} | {'Hazard Recall':>17} | {'CVaR-5% Recall':>16} | {'Flicker F_t':>15} | {'Latency (ms)':>12}"
    print(header)
    print("-" * len(header))
    for method_name, by_k in summary.items():
        for k in K_BUDGETS:
            d = by_k[k]
            hr = f"{d['hazard_recall_mean_mean']*100:5.1f}% ± {d['hazard_recall_mean_ci95']*100:4.1f}%"
            cvar = f"{d['hazard_recall_cvar5_mean']*100:5.1f}% ± {d['hazard_recall_cvar5_ci95']*100:4.1f}%"
            fl = f"{d['flicker_mean_mean']:.3f} ± {d['flicker_mean_ci95']:.3f}"
            lat = f"{d['latency_ms_mean_mean']:.3f}"
            print(f"{method_name:<10} | {k:>2} | {hr:>17} | {cvar:>16} | {fl:>15} | {lat:>12}")
        print("-" * len(header))


def plot_pareto(summary: Dict[str, Dict[int, Dict[str, float]]], out_path: Path) -> None:
    """Plot Hazard Recall and Flicker Rate across cognitive budgets k."""
    styles = {
        "Proposed": {"color": "#e74c3c", "marker": "o", "lw": 2.6},
        "Distance": {"color": "#2980b9", "marker": "s", "lw": 2.0},
        "Semantic": {"color": "#27ae60", "marker": "^", "lw": 2.0},
        "Random": {"color": "#7f8c8d", "marker": "x", "lw": 1.6},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    # Panel 1: Hazard Recall vs k
    ax1 = axes[0]
    for name, st in styles.items():
        means = [summary[name][k]["hazard_recall_mean_mean"] * 100 for k in K_BUDGETS]
        cis = [summary[name][k]["hazard_recall_mean_ci95"] * 100 for k in K_BUDGETS]
        ax1.errorbar(K_BUDGETS, means, yerr=cis, label=name, capsize=4, **st)

    ax1.set_title("Safety Preservation: Budget k vs. Hazard Recall", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cognitive Budget k (max objects per frame)")
    ax1.set_ylabel("Hazard Recall (%) ↑")
    ax1.set_xticks(K_BUDGETS)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="lower right")

    # Panel 2: Flicker Rate vs k
    ax2 = axes[1]
    for name, st in styles.items():
        means = [summary[name][k]["flicker_mean_mean"] for k in K_BUDGETS]
        cis = [summary[name][k]["flicker_mean_ci95"] for k in K_BUDGETS]
        ax2.errorbar(K_BUDGETS, means, yerr=cis, label=name, capsize=4, **st)

    ax2.set_title("Temporal Stability: Budget k vs. Flicker Rate", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cognitive Budget k (max objects per frame)")
    ax2.set_ylabel("Jaccard Flicker F_t ↓ (lower is more stable)")
    ax2.set_xticks(K_BUDGETS)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved Pareto plot to: {out_path.resolve()}")


def main() -> None:
    summary = run_all()
    print_table(summary)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "benchmark_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_pareto(summary, results_dir / "benchmark_pareto.png")


if __name__ == "__main__":
    main()

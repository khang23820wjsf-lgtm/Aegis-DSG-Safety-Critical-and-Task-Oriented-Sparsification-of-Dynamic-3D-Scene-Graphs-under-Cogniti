"""Run Phase A Verification Suite:
- EXP 1: Noise Stress Test (clean / medium / heavy)
- EXP 2: Ablation Study (Full vs. No Asymmetric vs. No Hysteresis vs. No Semantic)
- EXP 3: Error Analysis at k=5 (Overflow vs. Scoring Misses)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List
import numpy as np

from src.baselines import DistanceSparsifier
from src.dsg_generator import NoiseLevel, generate_dsg_sequence
from src.metrics import evaluate_sparsifier
from src.proposed import ProposedSparsifier


SEEDS = [42, 43, 44, 45, 46]
K_EVAL = 5


def _mean_metrics(method, sequences, k: int) -> Dict[str, float]:
    runs = [evaluate_sparsifier(method, seq, k) for seq in sequences]
    return {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}


def run_exp1_noise() -> None:
    print("\n================ EXP 1: NOISE STRESS TEST (k=5, 5 seeds) ================")
    print(f"{'Noise':<8} | {'Method':<10} | {'Hazard Recall':>13} | {'CVaR-5%':>9} | {'Flicker F_t':>11}")
    print("-" * 64)
    for level in ("clean", "medium", "heavy"):
        seqs = [
            generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level=level, seed=s)
            for s in SEEDS
        ]
        for method in (DistanceSparsifier(), ProposedSparsifier()):
            m = _mean_metrics(method, seqs, K_EVAL)
            print(
                f"{level:<8} | {method.name:<10} | {m['hazard_recall_mean']*100:12.1f}% | "
                f"{m['hazard_recall_cvar5']*100:8.1f}% | {m['flicker_mean']:11.3f}"
            )
        print("-" * 64)


def run_exp2_ablation() -> None:
    print("\n================ EXP 2: ABLATION STUDY (k=5, 5 seeds) ================")
    variants = [
        ("Full Proposed", ProposedSparsifier()),
        ("No Asymmetric (g=0)", ProposedSparsifier(gamma=0.0)),
        ("No Hysteresis (h=0)", ProposedSparsifier(hysteresis_bonus=0.0)),
        ("No Semantic (b=0)", ProposedSparsifier(beta=0.0)),
    ]
    for level in ("medium", "heavy"):
        seqs = [
            generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level=level, seed=s)
            for s in SEEDS
        ]
        print(f"\n-- Noise: {level} --")
        print(f"{'Variant':<22} | {'Hazard Recall':>13} | {'CVaR-5%':>9} | {'Flicker F_t':>11}")
        print("-" * 66)
        for label, method in variants:
            m = _mean_metrics(method, seqs, K_EVAL)
            print(
                f"{label:<22} | {m['hazard_recall_mean']*100:12.1f}% | "
                f"{m['hazard_recall_cvar5']*100:8.1f}% | {m['flicker_mean']:11.3f}"
            )


def run_exp3_error_analysis() -> None:
    print("\n================ EXP 3: ERROR ANALYSIS AT k=5 (medium noise, 5 seeds) ================")
    seqs = [
        generate_dsg_sequence(num_frames=120, dt=0.1, num_objects=42, noise_level="medium", seed=s)
        for s in SEEDS
    ]
    method = ProposedSparsifier()

    total_frames = 0
    total_obs_hazards = 0
    missed_due_to_overflow = 0
    missed_due_to_scoring = 0
    overflow_frames = 0
    missed_labels: Counter[str] = Counter()

    for seq in seqs:
        method.reset()
        for frame in seq:
            total_frames += 1
            obs_hazards = {
                nid
                for nid, d in frame.graph_obs.nodes(data=True)
                if d.get("layer") == "object" and d.get("is_hazard_gt", False)
            }
            total_obs_hazards += len(obs_hazards)
            selected = method.select(frame.graph_obs, frame.agent, K_EVAL)
            missed = obs_hazards - selected

            if len(obs_hazards) > K_EVAL:
                overflow_frames += 1
                # At least (len(obs_hazards) - K_EVAL) misses are unavoidable without coarsening
                unavoidable = len(obs_hazards) - K_EVAL
                missed_due_to_overflow += min(len(missed), unavoidable)
                missed_due_to_scoring += max(0, len(missed) - unavoidable)
            else:
                missed_due_to_scoring += len(missed)

            for nid in missed:
                missed_labels[frame.graph_obs.nodes[nid]["label"]] += 1

    total_missed = missed_due_to_overflow + missed_due_to_scoring
    print(f"Total frames analyzed        : {total_frames}")
    print(f"Frames with >{K_EVAL} hazards (overflow): {overflow_frames} ({overflow_frames/total_frames*100:.1f}%)")
    print(f"Total observed hazards       : {total_obs_hazards}")
    print(f"Total missed hazards         : {total_missed} ({total_missed/max(1,total_obs_hazards)*100:.1f}%)")
    if total_missed > 0:
        print(
            f"  -> Unavoidable (Budget Overflow > k): {missed_due_to_overflow} "
            f"({missed_due_to_overflow/total_missed*100:.1f}% of all misses)"
        )
        print(
            f"  -> Scoring/Ranking Errors (<= k)    : {missed_due_to_scoring} "
            f"({missed_due_to_scoring/total_missed*100:.1f}% of all misses)"
        )
        print("Top missed object classes    :", dict(missed_labels.most_common(5)))


if __name__ == "__main__":
    run_exp1_noise()
    run_exp2_ablation()
    run_exp3_error_analysis()

"""Evaluation metrics for DSG sparsification (Hazard Recall, Flicker, Latency)."""

from __future__ import annotations

from typing import Dict, List, Set
import time
import numpy as np

from src.baselines import Sparsifier
from src.dsg_generator import DSGFrame


def compute_hazard_recall(selected_ids: Set[str], obs_hazard_ids: Set[str]) -> float:
    """Fraction of observed ground-truth hazards retained in the compressed graph."""
    if not obs_hazard_ids:
        return 1.0
    return len(selected_ids & obs_hazard_ids) / len(obs_hazard_ids)


def compute_jaccard_flicker(prev_ids: Set[str], curr_ids: Set[str]) -> float:
    """Symmetric Jaccard distance F_t = |V'_t Δ V'_{t-1}| / |V'_t ∪ V'_{t-1}|."""
    union = prev_ids | curr_ids
    if not union:
        return 0.0
    sym_diff = prev_ids ^ curr_ids
    return len(sym_diff) / len(union)


def evaluate_sparsifier(
    sparsifier: Sparsifier,
    sequence: List[DSGFrame],
    k: int,
) -> Dict[str, float]:
    """Run a sparsifier across a full DSG sequence and return summary metrics."""
    sparsifier.reset()
    recalls: List[float] = []
    flickers: List[float] = []
    latencies_ms: List[float] = []
    prev_selected: Set[str] = set()

    for idx, frame in enumerate(sequence):
        obs_hazards = {
            nid
            for nid, d in frame.graph_obs.nodes(data=True)
            if d.get("layer") == "object" and d.get("is_hazard_gt", False)
        }

        t0 = time.perf_counter()
        selected = sparsifier.select(frame.graph_obs, frame.agent, k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        recalls.append(compute_hazard_recall(selected, obs_hazards))
        if idx > 0:
            flickers.append(compute_jaccard_flicker(prev_selected, selected))
        prev_selected = selected

    recalls_arr = np.array(recalls, dtype=float)
    # CVaR 5% (average of worst 5% frames)
    worst_count = max(1, int(np.ceil(0.05 * len(recalls_arr))))
    worst_5_recall = float(np.mean(np.sort(recalls_arr)[:worst_count]))

    return {
        "hazard_recall_mean": float(np.mean(recalls_arr)),
        "hazard_recall_cvar5": worst_5_recall,
        "flicker_mean": float(np.mean(flickers)) if flickers else 0.0,
        "latency_ms_mean": float(np.mean(latencies_ms)),
    }

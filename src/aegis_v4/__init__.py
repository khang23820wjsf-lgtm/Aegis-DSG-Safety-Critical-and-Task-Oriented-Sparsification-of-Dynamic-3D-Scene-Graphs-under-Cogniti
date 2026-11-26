"""Aegis-V4: Counterfactual Control-Barrier Sparsification (CCBS) + Online ACI."""

from src.aegis_v4.cbf_sparsifier import (
    CBFConstraint,
    CBFSparsificationResult,
    build_cbf_constraints,
    compute_counterfactual_regrets,
    compute_graph_control_regret,
    solve_cbf_qp,
)

__all__ = [
    "CBFConstraint",
    "CBFSparsificationResult",
    "build_cbf_constraints",
    "solve_cbf_qp",
    "compute_counterfactual_regrets",
    "compute_graph_control_regret",
]

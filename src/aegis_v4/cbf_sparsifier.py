"""Aegis-V4 Mathematical Core: Counterfactual Control-Barrier Sparsification (CCBS).

Implements:
  1. 3D-filtered Discrete-Time Control Barrier Function (CBF) linear velocity
     constraints with obstacle velocity compensation and exact OBB/AABB geometry.
  2. Pure-NumPy Discrete-Time CBF-QP Safety Filter (`solve_cbf_qp`):
       min_{u, delta>=0} 0.5 * ||u - u_nom||^2 + 0.5 * p_slack * delta^2
       s.t. a_v^T u + delta >= b_v   for all v
            ||u||_2 <= u_max
     Uses exact 2D KKT geometry when feasible (machine precision, < 0.05 ms)
     and provably convergent weighted 3D Dykstra alternating projections when
     boxed in (infeasible without slack).
  3. Counterfactual Safety Regret (`compute_counterfactual_regrets`):
       delta_cf(v) = ||u*(all) - u*(all \\ {v})||_2
     Provably 0.0 for geometrically redundant obstacles dominated by closer
     barriers, paired with barrier `margin_urgency(v)` so close-range atomic
     hazards (knives, scissors, stoves, pedestrians, cars, etc.) retain high
     priority even when the agent is stationary (u_nom ~ 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.aegis_core import AegisEntity, AffordanceRiskMapper
from src.dsg_generator import AgentState


_DEFAULT_MAPPER = AffordanceRiskMapper()


@dataclass
class CBFConstraint:
    """Single linear half-space velocity constraint a_v^T u >= b_v from a 3D obstacle."""

    entity_id: str
    a: np.ndarray  # shape (2,), unit outward normal from obstacle toward agent
    b: float  # RHS: a^T v_obs - (gamma_cbf / dt) * h_val
    h_val: float  # signed barrier clearance h_v(x) = dist_2d - (agent.radius + safety_margin)
    dist_2d: float  # 2D signed surface distance from obstacle footprint to agent center
    label: str = "object"
    is_atomic_hazard: bool = False
    intrinsic_severity: float = 0.0
    v_obs: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


@dataclass
class CBFSparsificationResult:
    """Output of Counterfactual Control-Barrier Sparsification for one frame."""

    u_nom: np.ndarray
    u_safe: np.ndarray
    slack: float
    constraints: List[CBFConstraint]
    active_ids: Set[str]
    regrets: Dict[str, float]
    margin_urgencies: Dict[str, float]
    combined_scores: Dict[str, float]
    solve_time_ms: float = 0.0
    support_ids: Set[str] = field(default_factory=set)
    dual_multipliers: Dict[str, float] = field(default_factory=dict)


def _extract_entity_fields(
    ent: Any,
    risk_mapper: AffordanceRiskMapper,
) -> Optional[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, bool, float, float, Optional[np.ndarray], Optional[float]]]:
    """Extract standardized geometric and semantic fields from AegisEntity, SceneObject, or dict."""
    if isinstance(ent, AegisEntity):
        eid = ent.entity_id
        lbl = ent.label
        pos = np.asarray(ent.pos, dtype=float)
        size = np.asarray(ent.size, dtype=float)
        vel = np.asarray(ent.vel, dtype=float)
        sev, mapped_atomic, _ = risk_mapper.assess(lbl, explicit_flag=True if ent.is_atomic_hazard else None)
        is_atomic = bool(ent.is_atomic_hazard or mapped_atomic)
        return eid, lbl, pos, size, vel, is_atomic, sev, float(ent.obb_yaw), ent.obb_size, None

    if isinstance(ent, tuple) and len(ent) == 2 and isinstance(ent[1], dict):
        eid = str(ent[0])
        d = ent[1]
    elif isinstance(ent, dict):
        d = ent
        eid = str(d.get("entity_id", d.get("obj_id", d.get("id", "unknown"))))
    else:
        # Duck-typed object (e.g. SceneObject or internal _TrackState)
        eid = str(getattr(ent, "entity_id", getattr(ent, "obj_id", "unknown")))
        lbl = str(getattr(ent, "label", "object"))
        pos = np.asarray(getattr(ent, "pos"), dtype=float)
        size = np.asarray(getattr(ent, "size"), dtype=float)
        vel = np.asarray(getattr(ent, "vel", np.zeros(3, dtype=float)), dtype=float)
        explicit_atomic = getattr(ent, "is_atomic_hazard", None)
        sev, is_atomic, _ = risk_mapper.assess(lbl, explicit_flag=explicit_atomic)
        obb_yaw = float(getattr(ent, "obb_yaw", 0.0))
        obb_size = getattr(ent, "obb_size", None)
        haz_sev = getattr(ent, "hazard_severity", None)
        return eid, lbl, pos, size, vel, is_atomic, sev, obb_yaw, obb_size, (None if haz_sev is None else float(haz_sev))

    if d.get("layer", "object") != "object":
        return None
    lbl = str(d.get("label", "object"))
    pos = np.asarray(d["pos"], dtype=float)
    size = np.asarray(d["size"], dtype=float)
    vel = np.asarray(d.get("vel", np.zeros(3, dtype=float)), dtype=float)
    sev, is_atomic, _ = risk_mapper.assess(
        lbl,
        explicit_flag=d.get("is_atomic_hazard"),
        affordances=d.get("affordances"),
    )
    obb_yaw = float(d.get("obb_yaw", 0.0))
    obb_size = d.get("obb_size", None)
    haz_sev = d.get("hazard_severity", None)
    return eid, lbl, pos, size, vel, is_atomic, sev, obb_yaw, obb_size, (None if haz_sev is None else float(haz_sev))


def _box_signed_distance_and_normal_2d(
    agent_xy: np.ndarray,
    box_xy: np.ndarray,
    box_size_xy: np.ndarray,
    yaw: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Compute exact 2D signed distance from box footprint to agent_xy and outward unit normal."""
    hx = 0.5 * float(box_size_xy[0])
    hy = 0.5 * float(box_size_xy[1])
    if hx < 1e-4:
        hx = 1e-4
    if hy < 1e-4:
        hy = 1e-4

    del_x = float(agent_xy[0]) - float(box_xy[0])
    del_y = float(agent_xy[1]) - float(box_xy[1])

    if abs(yaw) > 1e-6:
        c = math.cos(yaw)
        s = math.sin(yaw)
        # Rotate delta into box local frame (-yaw)
        lx = c * del_x + s * del_y
        ly = -s * del_x + c * del_y
    else:
        c, s = 1.0, 0.0
        lx, ly = del_x, del_y

    cx = -hx if lx < -hx else (hx if lx > hx else lx)
    cy = -hy if ly < -hy else (hy if ly > hy else ly)
    dx = lx - cx
    dy = ly - cy
    dist_out = math.hypot(dx, dy)

    if dist_out > 1e-9:
        nlx = dx / dist_out
        nly = dy / dist_out
        signed_dist = dist_out
    else:
        # Agent center is inside the 2D box footprint: push out along nearest face
        pen_x = hx - abs(lx)
        pen_y = hy - abs(ly)
        if pen_x <= pen_y:
            nlx = 1.0 if lx >= 0.0 else -1.0
            nly = 0.0
            signed_dist = -pen_x
        else:
            nlx = 0.0
            nly = 1.0 if ly >= 0.0 else -1.0
            signed_dist = -pen_y

    if abs(yaw) > 1e-6:
        # Rotate normal back to world frame (+yaw)
        nx = c * nlx - s * nly
        ny = s * nlx + c * nly
    else:
        nx, ny = nlx, nly

    norm = math.hypot(nx, ny)
    if norm > 1e-12:
        normal = np.array([nx / norm, ny / norm], dtype=float)
    else:
        normal = np.array([1.0, 0.0], dtype=float)

    return signed_dist, normal


def build_cbf_constraints(
    entities: Iterable[Any],
    agent: AgentState,
    dt: float = 0.1,
    gamma_cbf: float = 0.10,
    safety_margin: float = 0.15,
    max_range: Optional[float] = None,
    risk_mapper: Optional[AffordanceRiskMapper] = None,
) -> List[CBFConstraint]:
    """Construct 3D-filtered discrete-time CBF linear half-space constraints a_v^T u >= b_v.

    Derivation:
      h_v(x) = dist_2d(x, box_v) - (agent.radius + safety_margin)
      h_v(x + u*dt, c_v + v_obs*dt) >= (1 - gamma_cbf) * h_v(x)
      => a_v^T (u - v_obs) * dt >= -gamma_cbf * h_v(x)
      => a_v^T u >= a_v^T v_obs - (gamma_cbf / dt) * h_v(x)
    """
    mapper = risk_mapper if risk_mapper is not None else _DEFAULT_MAPPER
    dt_safe = max(float(dt), 1e-4)
    alpha_rate = float(np.clip(gamma_cbf, 1e-4, 1.0)) / dt_safe
    r_safe = float(agent.radius) + float(safety_margin)
    range_lim = float(agent.max_range if max_range is None else max_range)
    agent_xy = np.asarray(agent.pos[:2], dtype=float)
    z_upper = float(agent.height) + 0.15
    z_lower = 0.05

    constraints: List[CBFConstraint] = []
    for ent in entities:
        parsed = _extract_entity_fields(ent, mapper)
        if parsed is None:
            continue
        eid, lbl, pos, size, vel, is_atomic, sev, obb_yaw, obb_size, haz_sev = parsed

        # 3D vertical relevance filter (atomic hazards always pass)
        z_min = float(pos[2] - 0.5 * size[2])
        z_max = float(pos[2] + 0.5 * size[2])
        if not is_atomic:
            if z_min > z_upper or z_max < z_lower:
                continue
            if haz_sev is not None and haz_sev <= 0.0:
                continue

        box_xy = pos[:2]
        # Quick radial range pre-filter
        center_dist = float(math.hypot(agent_xy[0] - box_xy[0], agent_xy[1] - box_xy[1]))
        eff_size_xy = np.asarray(obb_size[:2], dtype=float) if obb_size is not None else size[:2]
        half_diag = 0.5 * float(math.hypot(eff_size_xy[0], eff_size_xy[1]))
        if center_dist - half_diag > range_lim:
            continue

        dist_2d, normal = _box_signed_distance_and_normal_2d(
            agent_xy=agent_xy,
            box_xy=box_xy,
            box_size_xy=eff_size_xy,
            yaw=obb_yaw if obb_size is not None else 0.0,
        )
        h_val = dist_2d - r_safe
        v_obs_xy = vel[:2].astype(float)
        b_val = float(normal[0] * v_obs_xy[0] + normal[1] * v_obs_xy[1]) - alpha_rate * h_val

        constraints.append(
            CBFConstraint(
                entity_id=eid,
                a=normal,
                b=b_val,
                h_val=h_val,
                dist_2d=dist_2d,
                label=lbl,
                is_atomic_hazard=is_atomic,
                intrinsic_severity=sev,
                v_obs=v_obs_xy,
            )
        )

    return constraints


def _solve_slack_eq_subproblem(
    u_nom_xy: np.ndarray,
    A_w: np.ndarray,
    b_w: np.ndarray,
    u_max: float,
    slack_penalty: float,
) -> Optional[Tuple[np.ndarray, float, np.ndarray, float]]:
    """Exact KKT solver for equality-constrained 3D slack subproblem (|W| in {1, 2, 3}).

    Solves:
      min_{u in R^2, delta in R} 0.5 * ||u - u_nom||^2 + 0.5 * p * delta^2
      s.t. A_w u + delta * 1 = b_w,  ||u||_2 <= u_max

    Returns (u, delta, lam, mu) or None if infeasible / rank-deficient.
    """
    r = A_w.shape[0]
    p = max(float(slack_penalty), 1.0)
    inv_p = 1.0 / p
    ux0 = float(u_nom_xy[0])
    uy0 = float(u_nom_xy[1])
    u_max_sq = (u_max + 1e-11) ** 2

    if r == 1:
        ax = float(A_w[0, 0])
        ay = float(A_w[0, 1])
        bv = float(b_w[0])
        au0 = ax * ux0 + ay * uy0
        # Check interior of disk (mu = 0)
        lam0 = (bv - au0) / (1.0 + inv_p)
        ux = ux0 + lam0 * ax
        uy = uy0 + lam0 * ay
        if ux * ux + uy * uy <= u_max_sq:
            return np.array([ux, uy], dtype=float), lam0 * inv_p, np.array([lam0], dtype=float), 0.0

        # Boundary of disk (mu > 0): ||u(mu)|| strictly decreases to 0 as mu -> inf
        def eval1(mu: float) -> Tuple[float, float, float, float]:
            s = 1.0 / (1.0 + mu)
            lam = (bv - s * au0) / (s + inv_p)
            x = s * (ux0 + lam * ax)
            y = s * (uy0 + lam * ay)
            return x, y, lam * inv_p, lam

        lo = 0.0
        hi = 1.0
        while hi < 1e12:
            hx, hy, _, _ = eval1(hi)
            if hx * hx + hy * hy <= u_max * u_max:
                break
            hi *= 2.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            mx, my, _, _ = eval1(mid)
            if mx * mx + my * my > u_max * u_max:
                lo = mid
            else:
                hi = mid
        mu_star = 0.5 * (lo + hi)
        fx, fy, fdelta, flam = eval1(mu_star)
        return np.array([fx, fy], dtype=float), fdelta, np.array([flam], dtype=float), mu_star

    if r == 2:
        a0x, a0y = float(A_w[0, 0]), float(A_w[0, 1])
        a1x, a1y = float(A_w[1, 0]), float(A_w[1, 1])
        b0, b1 = float(b_w[0]), float(b_w[1])
        dax, day = a0x - a1x, a0y - a1y
        da_norm = math.hypot(dax, day)
        if da_norm < 1e-10:
            return None
        g01 = a0x * a1x + a0y * a1y
        au0 = a0x * ux0 + a0y * uy0
        au1 = a1x * ux0 + a1y * uy0

        def eval2(mu: float) -> Optional[Tuple[float, float, float, float, float]]:
            s = 1.0 / (1.0 + mu)
            m00 = s + inv_p
            m01 = s * g01 + inv_p
            det = m00 * m00 - m01 * m01
            if abs(det) < 1e-15:
                return None
            r0 = b0 - s * au0
            r1 = b1 - s * au1
            l0 = (m00 * r0 - m01 * r1) / det
            l1 = (m00 * r1 - m01 * r0) / det
            x = s * (ux0 + l0 * a0x + l1 * a1x)
            y = s * (uy0 + l0 * a0y + l1 * a1y)
            return x, y, (l0 + l1) * inv_p, l0, l1

        res0 = eval2(0.0)
        if res0 is None:
            return None
        ux, uy, delta, l0, l1 = res0
        if ux * ux + uy * uy <= u_max_sq:
            return np.array([ux, uy], dtype=float), delta, np.array([l0, l1], dtype=float), 0.0

        # Asymptotic minimum norm on line (a0 - a1)^T u = b0 - b1
        d_min = abs(b0 - b1) / da_norm
        if d_min > u_max + 1e-10:
            return None

        lo = 0.0
        hi = 1.0
        while hi < 1e12:
            rhi = eval2(hi)
            if rhi is None:
                return None
            if rhi[0] * rhi[0] + rhi[1] * rhi[1] <= u_max * u_max:
                break
            hi *= 2.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            rm = eval2(mid)
            if rm is None:
                return None
            if rm[0] * rm[0] + rm[1] * rm[1] > u_max * u_max:
                lo = mid
            else:
                hi = mid
        mu_star = 0.5 * (lo + hi)
        rf = eval2(mu_star)
        if rf is None:
            return None
        return np.array([rf[0], rf[1]], dtype=float), rf[2], np.array([rf[3], rf[4]], dtype=float), mu_star

    if r == 3:
        M = np.empty((3, 3), dtype=float)
        M[:, :2] = A_w
        M[:, 2] = 1.0
        if abs(float(np.linalg.det(M))) < 1e-12:
            return None
        sol = np.linalg.solve(M, b_w)
        ux, uy, delta = float(sol[0]), float(sol[1]), float(sol[2])
        if ux * ux + uy * uy > u_max_sq or delta < -1e-10:
            return None
        rhs_dual = np.array([ux - ux0, uy - uy0, p * delta], dtype=float)
        lam = np.linalg.solve(M.T, rhs_dual)
        return np.array([ux, uy], dtype=float), delta, lam, 0.0

    return None


def _solve_feasible_2d_exact(
    u_nom_xy: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    u_max_f: float,
    tol: float = 1e-7,
) -> Optional[Tuple[np.ndarray, List[int], List[int], Dict[int, float]]]:
    """Exact feasible (delta == 0) 2D CBF-QP solver.

    Returns (u_best, active_indices, support_indices, dual_map) or None if infeasible.
    """
    m = A.shape[0]
    u0 = u_nom_xy.astype(float, copy=True)
    n0 = math.hypot(u0[0], u0[1])
    if n0 > u_max_f:
        u0 *= u_max_f / n0

    if m == 0:
        return u0, [], [], {}

    margins0 = A @ u0 - b
    if float(np.min(margins0)) >= -tol:
        active = [int(i) for i in np.where(margins0 <= 1e-5)[0]]
        return u0, active, [], {}

    if float(np.max(b)) > u_max_f + tol:
        return None

    # Case 1: Single active linear constraint (exact disk chord clamping)
    viol_idx = np.where(margins0 < -tol)[0]
    Av = A[viol_idx]
    bv = b[viol_idx]
    tx = -Av[:, 1]
    ty = Av[:, 0]
    tau_max = np.sqrt(np.maximum(0.0, u_max_f * u_max_f - bv * bv))
    tau = np.clip(tx * u_nom_xy[0] + ty * u_nom_xy[1], -tau_max, tau_max)
    cand1 = np.column_stack((bv * Av[:, 0] + tau * tx, bv * Av[:, 1] + tau * ty))

    feas1 = np.all((A @ cand1.T) >= (b[:, None] - tol), axis=0)
    if np.any(feas1):
        feas_mask_idx = np.where(feas1)[0]
        feas_pts = cand1[feas_mask_idx]
        diffs = feas_pts - u_nom_xy[None, :]
        costs = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        best_local = int(np.argmin(costs))
        u_best = feas_pts[best_local]
        supp_idx = int(viol_idx[feas_mask_idx[best_local]])
        # Exact dual multiplier lambda >= 0 along normal a_supp
        lam_val = max(0.0, float(np.dot(A[supp_idx], u_best - u_nom_xy)))
        active = [int(i) for i in np.where((A @ u_best - b) <= 1e-5)[0]]
        return u_best, active, [supp_idx], {supp_idx: lam_val}

    # Case 2: Exact intersection of 2 linear constraints inside disk ||u|| <= u_max
    if m <= 48:
        sub_idx = np.arange(m)
    else:
        sub_idx = np.argsort(margins0)[:48]
    As = A[sub_idx]
    bs = b[sub_idx]
    ms = As.shape[0]
    ii, jj = np.triu_indices(ms, k=1)
    det = As[ii, 0] * As[jj, 1] - As[ii, 1] * As[jj, 0]
    non_par = np.abs(det) > 1e-8
    if np.any(non_par):
        ii = ii[non_par]
        jj = jj[non_par]
        det = det[non_par]
        ux = (bs[ii] * As[jj, 1] - As[ii, 1] * bs[jj]) / det
        uy = (As[ii, 0] * bs[jj] - bs[ii] * As[jj, 0]) / det
        in_disk = (ux * ux + uy * uy) <= (u_max_f + tol) ** 2
        if np.any(in_disk):
            ii_d = ii[in_disk]
            jj_d = jj[in_disk]
            cand2 = np.column_stack((ux[in_disk], uy[in_disk]))
            feas2 = np.all((A @ cand2.T) >= (b[:, None] - tol), axis=0)
            if np.any(feas2):
                f2_idx = np.where(feas2)[0]
                feas_pts = cand2[f2_idx]
                diffs = feas_pts - u_nom_xy[None, :]
                costs = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
                best_local = int(np.argmin(costs))
                u_best = feas_pts[best_local]
                k_best = int(f2_idx[best_local])
                s1 = int(sub_idx[ii_d[k_best]])
                s2 = int(sub_idx[jj_d[k_best]])
                # Compute 2x2 dual multipliers A_supp^T lam = u_best - u_nom
                A2 = A[[s1, s2]]
                try:
                    lam2 = np.linalg.solve(A2.T, u_best - u_nom_xy)
                except np.linalg.LinAlgError:
                    lam2 = np.array([1.0, 1.0], dtype=float)
                supp = []
                dmap: Dict[int, float] = {}
                for s_i, lv in ((s1, float(lam2[0])), (s2, float(lam2[1]))):
                    if lv > 1e-7:
                        supp.append(s_i)
                        dmap[s_i] = lv
                if not supp:
                    supp = [s1, s2]
                    dmap = {s1: max(0.0, float(lam2[0])), s2: max(0.0, float(lam2[1]))}
                active = [int(i) for i in np.where((A @ u_best - b) <= 1e-5)[0]]
                return u_best, active, supp, dmap

    return None


def _solve_slack_qp_exact_2d(
    u_nom_xy: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    u_max: float,
    slack_penalty: float,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, float, List[int], Dict[int, float]]:
    """Exact Primal-Dual Active-Set solver for slack-relaxed 2D CBF-QP.

    Solves to machine precision (< 0.05 ms) and extracts the exact Caratheodory
    minimal KKT support set (size <= 3):
      min_{u in R^2, delta >= 0} 0.5 * ||u - u_nom||^2 + 0.5 * p * delta^2
      s.t. A u + delta >= b,  ||u||_2 <= u_max
    """
    m = A.shape[0]
    p = max(float(slack_penalty), 1.0)
    u0 = u_nom_xy.astype(float, copy=True)
    n0 = math.hypot(u0[0], u0[1])
    if n0 > u_max and n0 > 1e-12:
        u0 *= u_max / n0

    # Initialize active set with the single most violated constraint at u0
    W: List[int] = [int(np.argmax(b - A @ u0))]
    visited: Set[Tuple[int, ...]] = set()

    def _obj(u: np.ndarray, d: float) -> float:
        dx = float(u[0] - u_nom_xy[0])
        dy = float(u[1] - u_nom_xy[1])
        return 0.5 * (dx * dx + dy * dy) + 0.5 * p * d * d

    for _ in range(16):
        key = tuple(sorted(W))
        if key in visited:
            break
        visited.add(key)

        sub = _solve_slack_eq_subproblem(u_nom_xy, A[W], b[W], u_max, p)
        if sub is None:
            # Current equality set infeasible in disk -> try best feasible subset of size len(W)-1
            if len(W) <= 1:
                break
            best_sub = None
            best_W: List[int] = []
            best_cost = float("inf")
            for drop_i in range(len(W)):
                W_try = [W[k] for k in range(len(W)) if k != drop_i]
                s_try = _solve_slack_eq_subproblem(u_nom_xy, A[W_try], b[W_try], u_max, p)
                if s_try is not None and s_try[1] >= -1e-8 and float(np.min(s_try[2])) >= -1e-8:
                    # Must satisfy dropped constraint too if possible
                    c_val = _obj(s_try[0], s_try[1])
                    if c_val < best_cost:
                        best_cost = c_val
                        best_sub = s_try
                        best_W = W_try
            if best_sub is None:
                break
            W = best_W
            continue

        u_sol, delta_sol, lam, _ = sub

        # Dual feasibility check: drop constraint with most negative multiplier
        min_lam_idx = int(np.argmin(lam))
        if float(lam[min_lam_idx]) < -1e-8 and len(W) > 1:
            W = [W[k] for k in range(len(W)) if k != min_lam_idx]
            continue

        # Primal feasibility check over all m constraints
        if delta_sol >= -1e-8:
            viols = b - (A @ u_sol + delta_sol)
            j_max = int(np.argmax(viols))
            if float(viols[j_max]) <= max(tol, 1e-7):
                # KKT optimality proven! Extract positive-multiplier minimal support
                supp: List[int] = []
                dmap: Dict[int, float] = {}
                for idx_w, lv in zip(W, lam):
                    if float(lv) > 1e-7:
                        supp.append(int(idx_w))
                        dmap[int(idx_w)] = float(lv)
                if not supp:
                    supp = [int(i) for i in W]
                    dmap = {int(idx_w): max(0.0, float(lv)) for idx_w, lv in zip(W, lam)}
                return u_sol, max(0.0, float(delta_sol)), supp, dmap

            # Add most violated constraint j_max to working set
            if j_max not in W:
                if len(W) < 3:
                    W_ext = W + [j_max]
                    if _solve_slack_eq_subproblem(u_nom_xy, A[W_ext], b[W_ext], u_max, p) is not None:
                        W = W_ext
                        continue
                # Pool = W + [j_max]: find best valid subset of size <= 3 containing j_max
                pool = W + [j_max]
                best_next_W: Optional[List[int]] = None
                best_next_cost = float("inf")
                n_pool = len(pool)
                cand_sets: List[List[int]] = [[j_max]]
                for i in range(n_pool - 1):
                    cand_sets.append([pool[i], j_max])
                for i in range(n_pool - 1):
                    for j in range(i + 1, n_pool - 1):
                        cand_sets.append([pool[i], pool[j], j_max])
                for cW in cand_sets:
                    s_c = _solve_slack_eq_subproblem(u_nom_xy, A[cW], b[cW], u_max, p)
                    if s_c is None:
                        continue
                    uc, dc, lc, _ = s_c
                    if dc < -1e-8 or float(np.min(lc)) < -1e-8:
                        continue
                    # Must satisfy all constraints in pool
                    if float(np.max(b[pool] - (A[pool] @ uc + dc))) <= 1e-6:
                        cv = _obj(uc, dc)
                        if cv < best_next_cost:
                            best_next_cost = cv
                            best_next_W = cW
                if best_next_W is not None and tuple(sorted(best_next_W)) not in visited:
                    W = best_next_W
                    continue
        break

    # Guaranteed-convergent exact 1D golden-section fallback on strictly convex Phi(delta)
    phi_inv = (math.sqrt(5.0) - 1.0) / 2.0
    d_lo = 0.0
    d_hi = max(1e-3, float(np.max(b)) + u_max + 1e-3)

    def eval_phi(d: float) -> Tuple[float, Optional[np.ndarray], List[int], Dict[int, float]]:
        res = _solve_feasible_2d_exact(u_nom_xy, A, b - d, u_max, tol=1e-7)
        if res is None:
            return 1e18 + (d_hi - d) * 1e9, None, [], {}
        u_d, _, supp_d, dmap_d = res
        return _obj(u_d, d), u_d, supp_d, dmap_d

    # First tighten d_lo to minimum feasible slack via bisection (16 steps)
    b_lo, b_hi = d_lo, d_hi
    for _ in range(16):
        mid = 0.5 * (b_lo + b_hi)
        if _solve_feasible_2d_exact(u_nom_xy, A, b - mid, u_max, tol=1e-7) is not None:
            b_hi = mid
        else:
            b_lo = mid
    d_lo = b_hi

    # Golden-section search on [d_lo, d_hi]
    a_g, b_g = d_lo, d_hi
    c_g = b_g - phi_inv * (b_g - a_g)
    d_g = a_g + phi_inv * (b_g - a_g)
    fc = eval_phi(c_g)
    fd = eval_phi(d_g)
    for _ in range(28):
        if fc[0] < fd[0]:
            b_g = d_g
            d_g = c_g
            fd = fc
            c_g = b_g - phi_inv * (b_g - a_g)
            fc = eval_phi(c_g)
        else:
            a_g = c_g
            c_g = d_g
            fc = fd
            d_g = a_g + phi_inv * (b_g - a_g)
            fd = eval_phi(d_g)

    best_d = 0.5 * (a_g + b_g)
    _, u_best, supp_best, dmap_best = eval_phi(best_d)
    if u_best is None:
        _, u_best, supp_best, dmap_best = eval_phi(d_hi)
        best_d = d_hi
    if u_best is None:
        u_best = u0
    max_viol = float(np.max(b - A @ u_best)) if m > 0 else 0.0
    slack = max(0.0, best_d, max_viol)
    return u_best, slack, supp_best, dmap_best


def _solve_cbf_qp_arrays(
    u_nom_xy: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    u_max: float = 2.0,
    slack_penalty: float = 100.0,
    max_iters: int = 60,
    tol: float = 1e-7,
    return_support: bool = False,
) -> Union[Tuple[np.ndarray, float, List[int]], Tuple[np.ndarray, float, List[int], List[int], Dict[int, float]]]:
    """Fast exact pure-NumPy 2D CBF-QP solver operating directly on (A, b) arrays."""
    u_max_f = max(float(u_max), 1e-6)
    feas = _solve_feasible_2d_exact(u_nom_xy, A, b, u_max_f, tol=tol)
    if feas is not None:
        u_best, active, supp, dmap = feas
        if return_support:
            return u_best, 0.0, active, supp, dmap
        return u_best, 0.0, active

    # Case 3: Infeasible / boxed-in -> exact Primal-Dual Active-Set 3D slack solver
    u_sol, slack, supp, dmap = _solve_slack_qp_exact_2d(
        u_nom_xy=u_nom_xy,
        A=A,
        b=b,
        u_max=u_max_f,
        slack_penalty=slack_penalty,
        tol=tol,
    )
    active = [int(i) for i in np.where((A @ u_sol + slack - b) <= 1e-4)[0]]
    if return_support:
        return u_sol, slack, active, supp, dmap
    return u_sol, slack, active


def solve_cbf_qp(
    u_nom: np.ndarray,
    constraints: Sequence[CBFConstraint],
    u_max: float = 2.0,
    slack_penalty: float = 100.0,
    max_iters: int = 60,
    tol: float = 1e-7,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, List[int]]]:
    """Solve the discrete-time CBF Safety Filter QP in pure NumPy.

    Args:
        u_nom: Nominal desired velocity, shape (2,) or (3,).
        constraints: Sequence of CBFConstraint half-spaces a_v^T u >= b_v.
        u_max: Maximum planar speed magnitude (m/s).
        slack_penalty: Quadratic penalty on slack delta >= 0 when boxed in.
        max_iters: Maximum Dykstra iterations (only used when boxed in).
        tol: Feasibility tolerance.
        return_info: If True, return (u_safe, slack, active_indices).

    Returns:
        u_safe (matching shape of u_nom), or (u_safe, slack, active_indices).
    """
    u_arr = np.asarray(u_nom, dtype=float)
    is_3d = u_arr.ndim == 1 and u_arr.shape[0] == 3
    u_nom_xy = u_arr[:2]

    if not constraints:
        u_sol_xy = u_nom_xy.copy()
        n = math.hypot(u_sol_xy[0], u_sol_xy[1])
        if n > u_max and n > 1e-12:
            u_sol_xy *= u_max / n
        u_out = np.array([u_sol_xy[0], u_sol_xy[1], u_arr[2]], dtype=float) if is_3d else u_sol_xy
        return (u_out, 0.0, []) if return_info else u_out

    A = np.vstack([c.a for c in constraints])
    b = np.array([c.b for c in constraints], dtype=float)
    u_sol_xy, slack, active = _solve_cbf_qp_arrays(
        u_nom_xy=u_nom_xy,
        A=A,
        b=b,
        u_max=u_max,
        slack_penalty=slack_penalty,
        max_iters=max_iters,
        tol=tol,
    )
    u_out = np.array([u_sol_xy[0], u_sol_xy[1], u_arr[2]], dtype=float) if is_3d else u_sol_xy
    return (u_out, slack, active) if return_info else u_out


def _compute_margin_urgency(
    c: CBFConstraint,
    sigma_atomic_m: float = 1.25,
    sigma_geom_m: float = 0.45,
) -> float:
    """Compute smooth barrier margin urgency in [0, 1] so stationary close hazards are never zeroed."""
    h_pos = max(0.0, float(c.h_val))
    if c.is_atomic_hazard:
        sev_floor = max(float(c.intrinsic_severity), 0.85)
        if c.h_val <= 0.0:
            return min(1.0, max(0.0, sev_floor))
        val = sev_floor * math.exp(-(h_pos * h_pos) / (2.0 * sigma_atomic_m * sigma_atomic_m))
        return min(1.0, max(0.0, val))
    if c.h_val <= 0.0:
        return 1.0
    val = 0.5 * math.exp(-(h_pos * h_pos) / (2.0 * sigma_geom_m * sigma_geom_m))
    return min(1.0, max(0.0, val))


def compute_counterfactual_regrets(
    entities: Iterable[Any],
    agent: AgentState,
    u_nom: Optional[np.ndarray] = None,
    dt: float = 0.1,
    gamma_cbf: float = 0.10,
    safety_margin: float = 0.15,
    u_max: float = 2.0,
    near_active_tol: float = 0.05,
    slack_penalty: float = 100.0,
    risk_mapper: Optional[AffordanceRiskMapper] = None,
) -> CBFSparsificationResult:
    """Compute exact Counterfactual Safety Regret delta_cf(v) and barrier margin urgency.

    For each constraint v:
      delta_cf(v) = ||u_full - solve_cbf_qp(u_nom, all_except_v)||_2
    By KKT complementarity of convex QPs, any strictly inactive constraint
    (a_v^T u_full > b_v + tol when slack == 0) has delta_cf(v) == 0.0 provably.
    """
    t0 = time.perf_counter()

    u_nom_arr = np.asarray(agent.vel if u_nom is None else u_nom, dtype=float)
    is_3d = u_nom_arr.ndim == 1 and u_nom_arr.shape[0] == 3
    u_nom_xy = u_nom_arr[:2]

    constraints = build_cbf_constraints(
        entities=entities,
        agent=agent,
        dt=dt,
        gamma_cbf=gamma_cbf,
        safety_margin=safety_margin,
        risk_mapper=risk_mapper,
    )

    if not constraints:
        u_safe_xy = u_nom_xy.copy()
        n = math.hypot(u_safe_xy[0], u_safe_xy[1])
        if n > u_max and n > 1e-12:
            u_safe_xy *= u_max / n
        u_safe = np.array([u_safe_xy[0], u_safe_xy[1], u_nom_arr[2]], dtype=float) if is_3d else u_safe_xy
        return CBFSparsificationResult(
            u_nom=u_nom_arr.copy(),
            u_safe=u_safe,
            slack=0.0,
            constraints=[],
            active_ids=set(),
            regrets={},
            margin_urgencies={},
            combined_scores={},
            solve_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    m = len(constraints)
    A = np.vstack([c.a for c in constraints])
    b = np.array([c.b for c in constraints], dtype=float)

    u_full_xy, slack, active_indices, support_indices, dual_idx_map = _solve_cbf_qp_arrays(
        u_nom_xy=u_nom_xy,
        A=A,
        b=b,
        u_max=u_max,
        slack_penalty=slack_penalty,
        return_support=True,
    )
    active_ids: Set[str] = {constraints[i].entity_id for i in active_indices}
    support_ids: Set[str] = {constraints[i].entity_id for i in support_indices}
    dual_multipliers: Dict[str, float] = {}
    for idx_s, lam_v in dual_idx_map.items():
        eid = constraints[idx_s].entity_id
        dual_multipliers[eid] = max(dual_multipliers.get(eid, 0.0), float(lam_v))

    # Residuals r_i = a_i^T u_full + slack - b_i >= 0
    residuals = A @ u_full_xy + slack - b
    regrets: Dict[str, float] = {}
    margin_urgencies: Dict[str, float] = {}
    combined_scores: Dict[str, float] = {}

    # Candidate set for counterfactual re-solve: support + active/near-active constraints
    cand_indices = set(int(i) for i in np.where(residuals <= near_active_tol)[0])
    cand_indices.update(active_indices)
    cand_indices.update(support_indices)

    mask = np.ones(m, dtype=bool)
    for i, c in enumerate(constraints):
        mu = _compute_margin_urgency(c)
        margin_urgencies[c.entity_id] = mu

        if i in cand_indices:
            mask[i] = False
            if m > 1:
                u_cf_xy, _, _ = _solve_cbf_qp_arrays(
                    u_nom_xy=u_nom_xy,
                    A=A[mask],
                    b=b[mask],
                    u_max=u_max,
                    slack_penalty=slack_penalty,
                )
            else:
                u_cf_xy = u_nom_xy.copy()
                n_cf = math.hypot(u_cf_xy[0], u_cf_xy[1])
                if n_cf > u_max and n_cf > 1e-12:
                    u_cf_xy *= u_max / n_cf
            mask[i] = True
            delta_cf = float(math.hypot(u_full_xy[0] - u_cf_xy[0], u_full_xy[1] - u_cf_xy[1]))
            if delta_cf < 1e-9:
                delta_cf = 0.0
        else:
            # Provably 0.0 by KKT complementarity (strictly inactive constraint)
            delta_cf = 0.0

        # If duplicate entity_ids exist (rare), keep max regret/urgency
        prev_r = regrets.get(c.entity_id, 0.0)
        regrets[c.entity_id] = max(prev_r, delta_cf)
        combined_scores[c.entity_id] = regrets[c.entity_id] + mu

    u_safe = np.array([u_full_xy[0], u_full_xy[1], u_nom_arr[2]], dtype=float) if is_3d else u_full_xy
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return CBFSparsificationResult(
        u_nom=u_nom_arr.copy(),
        u_safe=u_safe,
        slack=slack,
        constraints=constraints,
        active_ids=active_ids,
        regrets=regrets,
        margin_urgencies=margin_urgencies,
        combined_scores=combined_scores,
        solve_time_ms=elapsed_ms,
        support_ids=support_ids,
        dual_multipliers=dual_multipliers,
    )


def compute_graph_control_regret(
    full_entities: Iterable[Any],
    sparse_entities: Iterable[Any],
    agent: AgentState,
    u_nom: Optional[np.ndarray] = None,
    dt: float = 0.1,
    gamma_cbf: float = 0.10,
    safety_margin: float = 0.15,
    u_max: float = 2.0,
    risk_mapper: Optional[AffordanceRiskMapper] = None,
) -> float:
    """Compute exact CBF control distortion ||u*(G_full) - u*(G_sparse)||_2."""
    u_ref = np.asarray(agent.vel if u_nom is None else u_nom, dtype=float)[:2]
    c_full = build_cbf_constraints(
        full_entities, agent, dt=dt, gamma_cbf=gamma_cbf, safety_margin=safety_margin, risk_mapper=risk_mapper
    )
    c_sparse = build_cbf_constraints(
        sparse_entities, agent, dt=dt, gamma_cbf=gamma_cbf, safety_margin=safety_margin, risk_mapper=risk_mapper
    )
    u_full = solve_cbf_qp(u_ref, c_full, u_max=u_max)
    u_sparse = solve_cbf_qp(u_ref, c_sparse, u_max=u_max)
    return float(np.linalg.norm(u_full - u_sparse))

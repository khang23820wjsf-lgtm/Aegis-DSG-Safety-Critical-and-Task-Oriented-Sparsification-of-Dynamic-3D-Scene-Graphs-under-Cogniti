"""Aegis-V4 Non-Exchangeable Online Adaptive Conformal Inference (ACI) & Epistemic Gaze-Regret Modulator.

This module provides two core theoretical components for Aegis-V4:

1. `OnlineACISafetyCalibrator` (Gibbs & Candès, NeurIPS 2021):
   Distribution-free, **non-exchangeable** online conformal safety calibration for
   temporally correlated egocentric video streams under arbitrary distribution shift.
   Maintains a rolling window of recent non-conformity scores
       e_t = 1 - min_retained_hazard_score
   and an adaptive miscoverage level `alpha_t` updated online via:
       alpha_{t+1} = clip(alpha_t + eta * (alpha_target - err_t), alpha_min, alpha_max)
   with a finite-window conservative guard so that long-run empirical miscoverage
   converges to `alpha_target` with zero exchangeability or i.i.d. assumptions.
   Also provides a self-supervised proxy feedback mode (`update_proxy`) using
   CBF-QP slack violations and sudden TTC surprises when ground-truth labels are
   unavailable at deployment time.

2. `EpistemicGazeModulator`:
   Tracks per-entity 3D foveal fixation EMA `A_gaze(v, t) in [0, 1]` from `agent.gaze_dir`
   and computes Epistemic Counterfactual Priority:
       priority(v) = (delta_cf(v) + r_v) * (1 + lambda_epistemic * (1 - A_gaze(v, t)))
   guaranteeing that unattended peripheral obstacles that would alter the safe CBF
   control action receive maximum epistemic priority, while already-fixated low-regret
   clutter yields cognitive bandwidth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from src.dsg_generator import AgentState


# ==============================================================================
# 1. Non-Exchangeable Online Adaptive Conformal Inference (ACI)
# ==============================================================================
@dataclass(frozen=True)
class ACIStepResult:
    """Per-step diagnostic record produced by `OnlineACISafetyCalibrator`."""

    step: int
    alpha_t: float
    alpha_next: float
    q_t: float
    tau_t: float
    nonconformity_score: Optional[float]
    err_t: int
    conservative_guard_active: bool
    empirical_miscoverage: float


class OnlineACISafetyCalibrator:
    """Non-Exchangeable Online Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021).

    Guarantees long-run empirical miscoverage convergence to `alpha_target` under
    arbitrary distribution shift and temporal autocorrelation (zero exchangeability
    or i.i.d. assumptions):
        |(1 / T) * sum_{t=1}^T err_t - alpha_target| <= (alpha_max - alpha_min) / (T * eta)

    At each timestep t:
      - Non-conformity score: e_t = 1 - min_retained_hazard_score (higher => harder to cover).
      - Dynamic non-conformity quantile: q_t = quantile(window, 1 - alpha_t) with
        finite-sample conservative guard when ceil((n + 1)(1 - alpha_t)) > n or
        alpha_t reaches the floor `alpha_min`.
      - Dynamic safety retention threshold: tau_t = max(0.0, 1.0 - q_t).
        A hazard with score s_t is covered iff s_t >= tau_t (equivalently e_t <= q_t).
      - Online adaptive level update:
            alpha_{t+1} = clip(alpha_t + eta * (alpha_target - err_t), alpha_min, alpha_max).
    """

    def __init__(
        self,
        alpha_target: float = 0.05,
        eta: float = 0.02,
        window_size: int = 200,
        alpha_min: float = 0.001,
        alpha_max: float = 0.50,
        default_tau: float = 0.12,
    ) -> None:
        if not (0.0 < alpha_target < 1.0):
            raise ValueError("alpha_target must lie in (0, 1)")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not (0.0 < alpha_min < alpha_max < 1.0):
            raise ValueError("Must satisfy 0 < alpha_min < alpha_max < 1")

        self.alpha_target = float(alpha_target)
        self.eta = float(eta)
        self.window_size = int(window_size)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.default_tau = float(np.clip(default_tau, 0.0, 1.0))

        self.alpha_t: float = float(np.clip(alpha_target, alpha_min, alpha_max))
        self._window: Deque[float] = deque(maxlen=self.window_size)
        self._err_history: List[int] = []
        self._alpha_history: List[float] = []
        self._tau_history: List[float] = []
        self._q_history: List[float] = []
        self._step: int = 0
        self._last_proxy_ttc: Optional[float] = None

    def reset(self) -> None:
        """Reset online state, rolling window, and history back to initial conditions."""
        self.alpha_t = float(np.clip(self.alpha_target, self.alpha_min, self.alpha_max))
        self._window.clear()
        self._err_history.clear()
        self._alpha_history.clear()
        self._tau_history.clear()
        self._q_history.clear()
        self._step = 0
        self._last_proxy_ttc = None

    def _compute_quantile_and_guard(self) -> Tuple[float, bool]:
        """Compute finite-window conformal quantile q_t = Q(window, 1 - alpha_t) and guard flag."""
        n = len(self._window)
        if n == 0:
            return 1.0 - self.default_tau, False

        # Finite-sample conservative guard:
        # Triggers when ceil((n + 1)(1 - alpha_t)) > n OR alpha_t hits the floor alpha_min.
        # In Gibbs & Candès (2021), Q(1 - alpha_t) = +inf in this regime so err_t = 0 is
        # guaranteed even under adversarial distribution drift, preventing alpha_t pinning.
        level_idx = int(math.ceil((n + 1) * (1.0 - self.alpha_t)))
        if level_idx > n or self.alpha_t <= self.alpha_min + 1e-12:
            # For bounded [0, 1] scores, q_t = 1.0 gives tau_t = 0.0; for unbounded
            # scores, conservative_guard_active=True enforces err_t = 0.
            max_w = float(max(self._window))
            return max(1.0, max_w), True

        sorted_w = np.sort(np.asarray(self._window, dtype=float))
        idx = max(0, min(n - 1, level_idx - 1))
        return float(sorted_w[idx]), False

    @property
    def q_t(self) -> float:
        """Current non-conformity quantile threshold q_t = quantile(window, 1 - alpha_t)."""
        q, _ = self._compute_quantile_and_guard()
        return q

    @property
    def tau_t(self) -> float:
        """Current dynamic safety retention threshold tau_t = max(0.0, 1.0 - q_t)."""
        q, guard = self._compute_quantile_and_guard()
        if guard:
            return 0.0
        return float(max(0.0, 1.0 - q))

    @property
    def conservative_guard_active(self) -> bool:
        """Whether the finite-window conservative guard is currently active."""
        _, guard = self._compute_quantile_and_guard()
        return guard

    @property
    def step_count(self) -> int:
        """Total number of online update steps processed."""
        return self._step

    @property
    def empirical_miscoverage(self) -> float:
        """Long-run cumulative empirical miscoverage rate (1/T) * sum(err_t)."""
        if not self._err_history:
            return 0.0
        return float(np.mean(self._err_history))

    @property
    def empirical_coverage(self) -> float:
        """Long-run cumulative empirical coverage rate 1 - empirical_miscoverage."""
        return 1.0 - self.empirical_miscoverage

    @property
    def window_empirical_miscoverage(self) -> float:
        """Recent empirical miscoverage rate over the last `window_size` steps."""
        if not self._err_history:
            return 0.0
        recent = self._err_history[-self.window_size :]
        return float(np.mean(recent))

    @property
    def alpha_history(self) -> List[float]:
        """Copy of the adaptive alpha_t trajectory."""
        return list(self._alpha_history)

    @property
    def tau_history(self) -> List[float]:
        """Copy of the dynamic safety threshold tau_t trajectory."""
        return list(self._tau_history)

    @property
    def err_history(self) -> List[int]:
        """Copy of the binary miscoverage indicator err_t trajectory."""
        return list(self._err_history)

    def get_calibrated_tau_min(
        self,
        default_tau_min: float = 0.12,
        floor: float = 0.005,
    ) -> float:
        """Return a bounded stopping threshold suitable for submodular selection layers.

        During cold-start warmup (fewer than ceil(1 / alpha_target) samples and no
        observed miscoverage errors yet), preserves `default_tau_min` to avoid a
        spurious finite-window guard drop on clean scenes.
        """
        n = len(self._window)
        if n == 0:
            return float(default_tau_min)
        min_warmup = int(math.ceil(1.0 / max(self.alpha_target, 1e-3)))
        if n < min_warmup and not any(self._err_history):
            return float(default_tau_min)
        return float(np.clip(self.tau_t, floor, default_tau_min))

    def predict_threshold(self) -> Tuple[float, float, bool]:
        """Return (tau_t, q_t, conservative_guard_active) prior to observing step t."""
        q, guard = self._compute_quantile_and_guard()
        tau = 0.0 if guard else float(max(0.0, 1.0 - q))
        return tau, q, guard

    def update(
        self,
        min_retained_hazard_score: Optional[float] = None,
        nonconformity_score: Optional[float] = None,
        err_t: Optional[int] = None,
    ) -> ACIStepResult:
        """Perform one causal online ACI update step.

        Args:
            min_retained_hazard_score: Minimum score among ground-truth/active hazards at step t.
                Converted to non-conformity score `e_t = 1 - clip(score, 0, 1)`.
            nonconformity_score: Directly supplied non-conformity score `e_t`. Takes precedence
                over `min_retained_hazard_score` if both are provided.
            err_t: Optional explicit binary miscoverage / safety violation indicator (0 or 1).
                If omitted, computed causally as `0` when the finite-window conservative guard
                is active, else `1` iff `e_t > q_t`.

        Returns:
            ACIStepResult capturing thresholds before update and the new `alpha_{t+1}`.
        """
        q_curr, guard = self._compute_quantile_and_guard()
        tau_curr = 0.0 if guard else float(max(0.0, 1.0 - q_curr))

        e_t: Optional[float] = None
        if nonconformity_score is not None:
            e_t = float(nonconformity_score)
        elif min_retained_hazard_score is not None:
            score_clipped = float(np.clip(min_retained_hazard_score, 0.0, 1.0))
            e_t = 1.0 - score_clipped

        if err_t is not None:
            err = 1 if bool(err_t) else 0
        elif e_t is not None:
            if guard:
                err = 0
            else:
                err = 1 if (e_t > q_curr + 1e-12) else 0
        else:
            err = 0

        alpha_curr = self.alpha_t
        alpha_next = float(
            np.clip(
                alpha_curr + self.eta * (self.alpha_target - float(err)),
                self.alpha_min,
                self.alpha_max,
            )
        )
        self.alpha_t = alpha_next

        if e_t is not None:
            self._window.append(e_t)

        self._step += 1
        self._err_history.append(err)
        self._alpha_history.append(alpha_curr)
        self._tau_history.append(tau_curr)
        self._q_history.append(q_curr)

        return ACIStepResult(
            step=self._step,
            alpha_t=alpha_curr,
            alpha_next=alpha_next,
            q_t=q_curr,
            tau_t=tau_curr,
            nonconformity_score=e_t,
            err_t=err,
            conservative_guard_active=guard,
            empirical_miscoverage=self.empirical_miscoverage,
        )

    def update_proxy(
        self,
        cbf_slack: float = 0.0,
        min_ttc: float = float("inf"),
        min_retained_score: Optional[float] = None,
        slack_threshold: float = 1e-3,
        ttc_surprise_threshold: float = 1.2,
        prev_min_ttc: Optional[float] = None,
        ttc_drop_threshold: float = 1.0,
    ) -> ACIStepResult:
        """Self-supervised deployment feedback mode without ground-truth hazard labels.

        Uses online control-barrier slack violations (`cbf_slack > slack_threshold`)
        and sudden Time-To-Collision surprises (`min_ttc < ttc_surprise_threshold` or
        abrupt TTC drop) as a self-supervised miscoverage error signal `err_t`.
        """
        slack_val = max(0.0, float(cbf_slack))
        ttc_val = float(min_ttc)
        ref_prev_ttc = self._last_proxy_ttc if prev_min_ttc is None else float(prev_min_ttc)

        slack_violation = slack_val > float(slack_threshold)
        imminent_ttc = math.isfinite(ttc_val) and ttc_val < float(ttc_surprise_threshold)
        sudden_ttc_drop = (
            ref_prev_ttc is not None
            and math.isfinite(ref_prev_ttc)
            and math.isfinite(ttc_val)
            and (ref_prev_ttc - ttc_val) > float(ttc_drop_threshold)
            and ttc_val < 2.5
        )
        proxy_err = 1 if (slack_violation or imminent_ttc or sudden_ttc_drop) else 0

        if min_retained_score is not None:
            e_t = 1.0 - float(np.clip(min_retained_score, 0.0, 1.0))
        else:
            slack_scale = max(float(slack_threshold) * 5.0, 1e-4)
            slack_nc = 1.0 - math.exp(-slack_val / slack_scale)
            ttc_scale = max(float(ttc_surprise_threshold), 0.2)
            ttc_nc = math.exp(-max(0.0, ttc_val) / ttc_scale) if math.isfinite(ttc_val) else 0.0
            e_t = float(np.clip(max(slack_nc, ttc_nc), 0.0, 1.0))

        self._last_proxy_ttc = ttc_val if math.isfinite(ttc_val) else None
        return self.update(nonconformity_score=e_t, err_t=proxy_err)

    def summary(self) -> Dict[str, float]:
        """Return compact JSON-serializable summary metrics of ACI convergence."""
        return {
            "steps": float(self._step),
            "alpha_target": self.alpha_target,
            "alpha_current": round(self.alpha_t, 5),
            "tau_current": round(self.tau_t, 5),
            "q_current": round(self.q_t, 5),
            "empirical_miscoverage": round(self.empirical_miscoverage, 5),
            "empirical_coverage": round(self.empirical_coverage, 5),
            "window_empirical_miscoverage": round(self.window_empirical_miscoverage, 5),
            "abs_miscoverage_error": round(abs(self.empirical_miscoverage - self.alpha_target), 5),
        }


# ==============================================================================
# 2. Active Epistemic Gaze-Regret Modulator
# ==============================================================================
@dataclass(frozen=True)
class EpistemicPriorityResult:
    """Per-entity output of `EpistemicGazeModulator`."""

    entity_id: str
    priority: float
    gaze_attention: float
    epistemic_multiplier: float
    delta_cf: float
    kinematic_risk: float


class EpistemicGazeModulator:
    """Active Epistemic Gaze-Regret Modulator for Aegis-V4.

    Tracks per-track 3D foveal fixation EMA `A_gaze(v, t) in [0, 1]` from `agent.gaze_dir`
    and computes Epistemic Counterfactual Priority:

        priority(v) = (delta_cf(v) + r_v) * (1 + lambda_epistemic * (1 - A_gaze(v, t)))

    Properties:
      - An unattended peripheral obstacle (A_gaze ~ 0) that alters the safe CBF control
        action (delta_cf > 0) receives maximum epistemic boost `(1 + lambda_epistemic)`.
      - An already-fixated (A_gaze -> 1) low-regret (delta_cf == 0) entity receives no
        epistemic boost (`multiplier -> 1.0`), yielding bandwidth to unseen threats.
      - Gracefully degrades when `agent.gaze_dir is None` (treats `A_gaze = 0.0` so
        conservative safety priority is preserved without crashing).
    """

    def __init__(
        self,
        lambda_epistemic: float = 0.60,
        fovea_half_angle_deg: float = 15.0,
        ema_decay: float = 0.65,
    ) -> None:
        if lambda_epistemic < 0.0:
            raise ValueError("lambda_epistemic must be >= 0")
        if fovea_half_angle_deg <= 0.0:
            raise ValueError("fovea_half_angle_deg must be > 0")
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError("ema_decay must lie in [0, 1)")

        self.lambda_epistemic = float(lambda_epistemic)
        self.fovea_half_angle_deg = float(fovea_half_angle_deg)
        self.ema_decay = float(ema_decay)
        self._fovea_rad = math.radians(self.fovea_half_angle_deg)
        self._gaze_ema: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear all per-track visual fixation EMA states."""
        self._gaze_ema.clear()

    def get_fixation(self, entity_id: str) -> float:
        """Return current visual fixation EMA A_gaze(v, t) in [0, 1]."""
        return float(self._gaze_ema.get(entity_id, 0.0))

    def prune_tracks(self, active_ids: Iterable[str]) -> None:
        """Drop EMA state for tracks no longer present in the active scene graph."""
        active_set = set(active_ids)
        stale = [eid for eid in self._gaze_ema if eid not in active_set]
        for eid in stale:
            del self._gaze_ema[eid]

    def instantaneous_attention(self, entity_pos: np.ndarray, agent: AgentState) -> float:
        """Compute instantaneous 3D foveal visual alignment in [0, 1] for a single entity."""
        gaze = agent.gaze_dir
        if gaze is None:
            return 0.0
        gx, gy, gz = float(gaze[0]), float(gaze[1]), float(gaze[2])
        g_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
        if g_norm < 1e-6:
            return 0.0

        ap = agent.pos
        rx = float(entity_pos[0]) - float(ap[0])
        ry = float(entity_pos[1]) - float(ap[1])
        rz = float(entity_pos[2]) - float(ap[2])
        dist_3d = math.sqrt(rx * rx + ry * ry + rz * rz)
        if dist_3d < 1e-4:
            return 1.0

        cos_g = (rx * gx + ry * gy + rz * gz) / (dist_3d * g_norm)
        if cos_g > 1.0:
            cos_g = 1.0
        elif cos_g < -1.0:
            cos_g = -1.0
        ang_g = math.acos(cos_g)
        f = self._fovea_rad
        return math.exp(-(ang_g * ang_g) / (2.0 * f * f))

    def update_fixation(
        self,
        entity_id: str,
        entity_pos: np.ndarray,
        agent: AgentState,
    ) -> float:
        """Update and return per-track visual fixation EMA A_gaze(v, t) in [0, 1]."""
        if agent.gaze_dir is None:
            self._gaze_ema[entity_id] = 0.0
            return 0.0

        inst_att = self.instantaneous_attention(entity_pos, agent)
        prev = self._gaze_ema.get(entity_id, 0.0)
        att = self.ema_decay * prev + (1.0 - self.ema_decay) * inst_att
        att_clipped = 0.0 if att < 0.0 else (1.0 if att > 1.0 else att)
        self._gaze_ema[entity_id] = att_clipped
        return att_clipped

    def compute_priority(
        self,
        entity_id: str,
        entity_pos: np.ndarray,
        agent: AgentState,
        delta_cf: float,
        r_v: float,
        update_ema: bool = True,
    ) -> EpistemicPriorityResult:
        """Compute Epistemic Counterfactual Priority for a single entity.

        Formula:
            priority(v) = (delta_cf(v) + r_v) * (1 + lambda_epistemic * (1 - A_gaze(v, t)))
        """
        a_gaze = (
            self.update_fixation(entity_id, entity_pos, agent)
            if update_ema
            else self.get_fixation(entity_id)
        )
        d_cf = max(0.0, float(delta_cf))
        risk = max(0.0, float(r_v))
        mult = 1.0 + self.lambda_epistemic * (1.0 - a_gaze)
        priority = (d_cf + risk) * mult
        return EpistemicPriorityResult(
            entity_id=entity_id,
            priority=float(priority),
            gaze_attention=a_gaze,
            epistemic_multiplier=float(mult),
            delta_cf=d_cf,
            kinematic_risk=risk,
        )

    def modulate_batch(
        self,
        entities: Sequence[Any],
        agent: AgentState,
        delta_cf_map: Mapping[str, float],
        risk_map: Mapping[str, float],
        update_ema: bool = True,
        prune_stale: bool = False,
    ) -> Dict[str, EpistemicPriorityResult]:
        """Compute Epistemic Counterfactual Priority for a collection of entities.

        Accepts entity objects (having `.entity_id` or `.obj_id`, and `.pos`) or dicts
        (having `"entity_id"` or `"obj_id"` or `"id"`, and `"pos"`).
        """
        results: Dict[str, EpistemicPriorityResult] = {}
        active_ids: List[str] = []
        for ent in entities:
            if isinstance(ent, Mapping):
                eid = str(ent.get("entity_id", ent.get("obj_id", ent.get("id", ""))))
                pos = np.asarray(ent["pos"], dtype=float)
            else:
                eid = str(getattr(ent, "entity_id", getattr(ent, "obj_id", "")))
                pos = np.asarray(getattr(ent, "pos"), dtype=float)
            active_ids.append(eid)
            results[eid] = self.compute_priority(
                entity_id=eid,
                entity_pos=pos,
                agent=agent,
                delta_cf=float(delta_cf_map.get(eid, 0.0)),
                r_v=float(risk_map.get(eid, 0.0)),
                update_ema=update_ema,
            )

        if prune_stale:
            self.prune_tracks(active_ids)
        return results


__all__ = [
    "ACIStepResult",
    "OnlineACISafetyCalibrator",
    "EpistemicPriorityResult",
    "EpistemicGazeModulator",
]

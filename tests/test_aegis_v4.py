"""Comprehensive verification tests for Aegis-V4 (CCBS + Online ACI).

Verifies:
  (a) Discrete-Time CBF-QP safety filter forward invariance, velocity compensation,
      3D vertical filtering, and boxed-in slack feasibility.
  (b) Counterfactual Safety Regret delta_cf(v) == 0 for geometrically redundant
      occluded obstacles, delta_cf(v) > 0 for active blocking hazards, and
      non-zero margin urgency for stationary close-range atomic hazards.
  (c) Non-Exchangeable Online ACI (Gibbs & Candes, NeurIPS 2021) miscoverage
      tracking under autocorrelated drifting/regime-shifting sequences,
      finite-window conservative guard, and self-supervised proxy feedback.
  (d) Epistemic Gaze-Regret prioritization of unattended peripheral CBF-active
      hazards over fixated hazards (unit & end-to-end AegisV4CCBS).
  (e) End-to-end AegisV4CCBS preservation of non-coarsenable Atomic Hazards
      (knife, acid_beaker), strictly lower Counterfactual Control Regret than
      Distance, closed-loop compute_safe_velocity, and edge cases.
"""

from __future__ import annotations

import math
import time
import unittest

import networkx as nx
import numpy as np

from src.aegis_v4 import (
    CBFConstraint,
    build_cbf_constraints,
    compute_counterfactual_regrets,
    compute_graph_control_regret,
    solve_cbf_qp,
)
from src.aegis_v4.adaptive_conformal import (
    EpistemicGazeModulator,
    OnlineACISafetyCalibrator,
)
from src.aegis_v4.aegis_v4_core import AegisV4CCBS, AegisV4Config
from src.dsg_generator import AgentState


def _make_graph(objs: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for o in objs:
        g.add_node(
            o["id"],
            layer=o.get("layer", "object"),
            label=o.get("label", "chair"),
            pos=np.array(o["pos"], dtype=float),
            size=np.array(o.get("size", [0.5, 0.5, 0.8]), dtype=float),
            vel=np.array(o.get("vel", [0.0, 0.0, 0.0]), dtype=float),
            confidence=o.get("conf", 0.95),
            structural_importance=o.get("cv", 0.15),
            hazard_severity=o.get("haz", 0.85),
            is_dynamic=o.get("dyn", False),
        )
    return g


class TestAegisV4CBF(unittest.TestCase):
    """(a) & (b): CBF-QP Forward Invariance & Counterfactual Safety Regret."""

    def setUp(self) -> None:
        self.agent = AgentState(
            pos=np.array([0.0, 0.0, 0.85], dtype=float),
            vel=np.array([0.0, 1.4, 0.0], dtype=float),
            radius=0.30,
            height=1.70,
        )

    def test_a1_cbf_qp_forward_invariance_single_obstacle(self) -> None:
        """CBF-QP must never output a velocity violating active obstacle barrier constraints."""
        dt = 0.1
        gamma_cbf = 0.10
        safety_margin = 0.15
        u_nom = np.array([0.0, 1.5, 0.0], dtype=float)

        # Wall centered at y=1.0, size y=0.4 => front surface at y=0.80.
        # dist_2d = 0.80, r_safe = 0.30 + 0.15 = 0.45 => h = 0.35 > 0.
        # Nominal velocity 1.5 m/s would violate h(x + u*dt) >= (1 - gamma) * h(x).
        obs = [
            {
                "id": "wall_front",
                "label": "wall",
                "pos": [0.0, 1.0, 1.0],
                "size": [2.0, 0.4, 2.0],
                "vel": [0.0, 0.0, 0.0],
            }
        ]
        constraints = build_cbf_constraints(
            obs, self.agent, dt=dt, gamma_cbf=gamma_cbf, safety_margin=safety_margin
        )
        self.assertEqual(len(constraints), 1)
        c = constraints[0]
        self.assertGreater(c.h_val, 0.0)

        u_safe, slack, active = solve_cbf_qp(u_nom, constraints, u_max=2.0, return_info=True)
        self.assertEqual(u_safe.shape, (3,))
        self.assertAlmostEqual(slack, 0.0, places=7)
        self.assertEqual(active, [0])

        # Verify linear barrier constraint a^T u_safe >= b
        self.assertGreaterEqual(float(np.dot(c.a, u_safe[:2])), c.b - 1e-6)
        self.assertLessEqual(float(np.linalg.norm(u_safe[:2])), 2.0 + 1e-6)

        # Verify actual discrete-time geometric forward invariance: h(x_next) >= (1 - gamma) * h(x)
        next_agent = AgentState(
            pos=self.agent.pos + u_safe * dt,
            vel=u_safe,
            radius=self.agent.radius,
            height=self.agent.height,
        )
        c_next = build_cbf_constraints(
            obs, next_agent, dt=dt, gamma_cbf=gamma_cbf, safety_margin=safety_margin
        )[0]
        self.assertGreaterEqual(c_next.h_val, (1.0 - gamma_cbf) * c.h_val - 1e-6)
        self.assertLess(u_safe[1], u_nom[1])  # velocity was actively braked

    def test_a2_cbf_qp_multi_obstacle_corner_and_optimality(self) -> None:
        """Feasible multi-obstacle corner satisfies ALL barriers and minimizes ||u - u_nom||."""
        u_nom = np.array([1.2, 1.2], dtype=float)
        # Two walls forming an L-corner ahead-right; feasible region is u_x <= 0.3, u_y <= 0.4
        obs = [
            {"id": "wall_n", "label": "wall", "pos": [0.0, 0.90, 1.0], "size": [3.0, 0.4, 2.0]},
            {"id": "wall_e", "label": "wall", "pos": [0.85, 0.0, 1.0], "size": [0.4, 3.0, 2.0]},
        ]
        constraints = build_cbf_constraints(obs, self.agent, dt=0.1, gamma_cbf=0.10, safety_margin=0.15)
        self.assertEqual(len(constraints), 2)

        u_safe, slack, active = solve_cbf_qp(u_nom, constraints, u_max=2.0, return_info=True)
        self.assertAlmostEqual(slack, 0.0, places=7)
        self.assertEqual(len(active), 2)
        for c in constraints:
            self.assertGreaterEqual(float(np.dot(c.a, u_safe)), c.b - 1e-6)
        self.assertLessEqual(float(np.linalg.norm(u_safe)), 2.0 + 1e-6)

        # Verify KKT optimality: no random feasible point in disk has strictly lower cost
        rng = np.random.default_rng(42)
        opt_cost = float(np.sum((u_safe - u_nom) ** 2))
        for _ in range(500):
            cand = rng.uniform(-2.0, 2.0, size=2)
            if np.linalg.norm(cand) > 2.0:
                continue
            if all(float(np.dot(c.a, cand)) >= c.b - 1e-9 for c in constraints):
                self.assertGreaterEqual(float(np.sum((cand - u_nom) ** 2)), opt_cost - 1e-6)

    def test_a3_cbf_qp_dynamic_obstacle_velocity_compensation(self) -> None:
        """Oncoming dynamic pedestrian induces stronger braking via obstacle velocity compensation."""
        u_nom = np.array([0.0, 1.2, 0.0], dtype=float)
        static_obs = [{"id": "p_static", "label": "pedestrian", "pos": [0.0, 1.2, 0.9], "size": [0.5, 0.5, 1.7], "vel": [0.0, 0.0, 0.0]}]
        oncoming_obs = [{"id": "p_moving", "label": "pedestrian", "pos": [0.0, 1.2, 0.9], "size": [0.5, 0.5, 1.7], "vel": [0.0, -1.3, 0.0]}]

        c_stat = build_cbf_constraints(static_obs, self.agent, dt=0.1, gamma_cbf=0.10, safety_margin=0.15)
        c_mov = build_cbf_constraints(oncoming_obs, self.agent, dt=0.1, gamma_cbf=0.10, safety_margin=0.15)

        u_stat = solve_cbf_qp(u_nom, c_stat, u_max=2.0)
        u_mov, slack, _ = solve_cbf_qp(u_nom, c_mov, u_max=2.0, return_info=True)
        self.assertAlmostEqual(slack, 0.0, places=6)
        # Moving oncoming pedestrian must force strictly lower forward velocity than static one
        self.assertLess(u_mov[1], u_stat[1] - 0.5)
        self.assertGreaterEqual(float(np.dot(c_mov[0].a, u_mov[:2])), c_mov[0].b - 1e-6)

    def test_a4_cbf_3d_vertical_filter_and_boxed_in_slack(self) -> None:
        """3D filter drops overhead/floor clutter; boxed-in infeasible QP uses slack cleanly."""
        obs_3d = [
            {"id": "ceiling_lamp", "label": "lamp", "pos": [0.0, 0.8, 2.5], "size": [0.4, 0.4, 0.2]},
            {"id": "flat_rug", "label": "rug", "pos": [0.0, 0.8, 0.01], "size": [1.0, 1.0, 0.02]},
            {"id": "blocking_chair", "label": "chair", "pos": [0.0, 0.8, 0.45], "size": [0.5, 0.5, 0.9]},
        ]
        cs = build_cbf_constraints(obs_3d, self.agent)
        c_ids = {c.entity_id for c in cs}
        self.assertEqual(c_ids, {"blocking_chair"})

        # Boxed-in infeasible constraints: opposite half-spaces requiring u_y >= 1.0 and -u_y >= 1.0
        infeasible_cs = [
            CBFConstraint(entity_id="c1", a=np.array([0.0, 1.0]), b=1.0, h_val=-0.1, dist_2d=0.35),
            CBFConstraint(entity_id="c2", a=np.array([0.0, -1.0]), b=1.0, h_val=-0.1, dist_2d=0.35),
        ]
        u_box, slack, _ = solve_cbf_qp(np.array([0.0, 0.5]), infeasible_cs, u_max=1.5, return_info=True)
        self.assertGreater(slack, 0.5)
        self.assertLessEqual(float(np.linalg.norm(u_box)), 1.5 + 1e-6)
        for c in infeasible_cs:
            self.assertGreaterEqual(float(np.dot(c.a, u_box)) + slack, c.b - 1e-4)

    def test_b1_counterfactual_regret_zero_for_occluded_redundant_obstacle(self) -> None:
        """delta_cf(v) == 0 for a redundant obstacle behind a closer barrier; > 0 for active blocker."""
        u_nom = np.array([0.0, 1.5, 0.0], dtype=float)
        entities = [
            {
                "id": "front_wall",
                "label": "wall",
                "pos": [0.0, 0.90, 1.0],
                "size": [2.0, 0.3, 2.0],
            },
            {
                "id": "behind_chair",
                "label": "chair",
                "pos": [0.0, 2.20, 0.45],
                "size": [0.5, 0.5, 0.9],
            },
            {
                "id": "far_side_table",
                "label": "table",
                "pos": [3.5, 0.0, 0.45],
                "size": [0.6, 0.6, 0.8],
            },
        ]
        res = compute_counterfactual_regrets(entities, self.agent, u_nom=u_nom, dt=0.1, gamma_cbf=0.10)

        self.assertGreater(res.regrets["front_wall"], 0.3)
        self.assertEqual(res.regrets["behind_chair"], 0.0)
        self.assertEqual(res.regrets["far_side_table"], 0.0)
        self.assertIn("front_wall", res.active_ids)
        self.assertNotIn("behind_chair", res.active_ids)

        # Cross-check exact definition: delta_cf(front_wall) == ||u_full - u_without_front||_2
        cs_no_front = build_cbf_constraints(entities[1:], self.agent, dt=0.1, gamma_cbf=0.10)
        u_no_front = solve_cbf_qp(u_nom, cs_no_front, u_max=2.0)
        expected_delta = float(np.linalg.norm(res.u_safe[:2] - u_no_front[:2]))
        self.assertAlmostEqual(res.regrets["front_wall"], expected_delta, places=6)

    def test_b2_stationary_atomic_hazard_margin_urgency(self) -> None:
        """Stationary agent (u_nom=0) near an atomic hazard still yields high margin urgency."""
        stationary_agent = AgentState(pos=np.array([0.0, 0.0, 0.85]), vel=np.array([0.0, 0.0, 0.0]))
        entities = [
            {"id": "knife_1", "label": "knife", "pos": [0.0, 0.55, 0.90], "size": [0.05, 0.25, 0.03]},
            {"id": "far_chair", "label": "chair", "pos": [3.0, 3.0, 0.45], "size": [0.5, 0.5, 0.8]},
        ]
        res = compute_counterfactual_regrets(entities, stationary_agent, u_nom=np.zeros(3))
        self.assertGreater(res.margin_urgencies["knife_1"], 0.80)
        self.assertLess(res.margin_urgencies["far_chair"], 0.05)

    def test_a5_cbf_qp_realtime_latency(self) -> None:
        """Pure-NumPy solve_cbf_qp must run well under 1.0 ms/frame on 30 obstacles."""
        rng = np.random.default_rng(123)
        obs = [
            {
                "id": f"o_{i}",
                "label": "chair",
                "pos": [float(rng.uniform(-4.0, 4.0)), float(rng.uniform(0.8, 5.0)), 0.45],
                "size": [0.5, 0.5, 0.8],
            }
            for i in range(30)
        ]
        cs = build_cbf_constraints(obs, self.agent)
        u_nom = np.array([0.0, 1.4], dtype=float)
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            _ = solve_cbf_qp(u_nom, cs, u_max=2.0)
        avg_ms = (time.perf_counter() - t0) * 1000.0 / iters
        self.assertLess(avg_ms, 1.0)


class TestAegisV4OnlineACIAndGaze(unittest.TestCase):
    """(c) & (d): Non-Exchangeable Online ACI & Epistemic Gaze-Regret Modulator."""

    def test_c1_online_aci_tracks_alpha_under_non_exchangeable_drift(self) -> None:
        """Online ACI converges to target miscoverage alpha under AR(1) + drift + regime shocks."""
        rng = np.random.default_rng(2026)
        T = 1500
        alpha_target = 0.05
        eta = 0.02
        alpha_min = 0.001
        alpha_max = 0.50

        # Non-exchangeable, strongly autocorrelated, drifting + regime-shifting scores
        e_seq = np.zeros(T, dtype=float)
        ar = 0.0
        for t in range(T):
            ar = 0.90 * ar + rng.normal(0.0, 0.06)
            drift = 0.30 * (t / T)
            shock = 0.25 if (500 <= t < 900) else 0.0
            sinusoid = 0.08 * math.sin(2.0 * math.pi * t / 180.0)
            e_seq[t] = float(np.clip(0.20 + ar + drift + shock + sinusoid, 0.0, 1.0))

        cal = OnlineACISafetyCalibrator(
            alpha_target=alpha_target,
            eta=eta,
            window_size=150,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        for t in range(T):
            cal.update(nonconformity_score=e_seq[t])

        # Exact Gibbs & Candes (2021) theoretical finite-T miscoverage bound:
        # |(1/T) sum err_t - alpha_target| <= (alpha_max - alpha_min) / (T * eta) + 1/T
        theo_bound = (alpha_max - alpha_min) / (T * eta) + (1.0 / T)
        abs_err = abs(cal.empirical_miscoverage - alpha_target)
        self.assertLessEqual(abs_err, theo_bound + 1e-6)

        # Static split conformal calibrated on first 200 frames fails badly under shift
        static_q = float(np.quantile(e_seq[:200], 1.0 - alpha_target))
        static_miscov = float(np.mean(e_seq[200:] > static_q))
        self.assertGreater(abs(static_miscov - alpha_target), 3.0 * abs_err)

    def test_c2_online_aci_conservative_guard_and_proxy_mode(self) -> None:
        """Finite-window guard enforces err_t=0 at floor; proxy mode tightens alpha_t on slack/TTC."""
        cal = OnlineACISafetyCalibrator(alpha_target=0.05, eta=0.05, window_size=10, alpha_min=0.001)
        # With n=5 samples and alpha_t=0.05: ceil(6 * 0.95) = 6 > 5 -> conservative guard MUST activate
        for s in [0.9, 0.85, 0.8, 0.75, 0.7]:
            cal.update(min_retained_hazard_score=s)
        self.assertTrue(cal.conservative_guard_active)
        self.assertEqual(cal.tau_t, 0.0)

        # Proxy mode: slack violation + imminent TTC triggers err_t=1 and tightens alpha_t
        cal_proxy = OnlineACISafetyCalibrator(alpha_target=0.05, eta=0.04, window_size=50)
        for _ in range(10):
            cal_proxy.update_proxy(cbf_slack=0.0, min_ttc=float("inf"), min_retained_score=0.9)
        alpha_before = cal_proxy.alpha_t
        step_viol = cal_proxy.update_proxy(cbf_slack=0.12, min_ttc=0.5, min_retained_score=0.4)
        self.assertEqual(step_viol.err_t, 1)
        self.assertLess(step_viol.alpha_next, alpha_before)

    def test_d1_epistemic_gaze_prioritizes_unattended_cbf_hazard(self) -> None:
        """Unattended peripheral CBF-active hazard outranks fixated hazard and fixated clutter."""
        gaze_left = np.array([-0.8, 1.8, -0.35], dtype=float)
        gaze_left /= np.linalg.norm(gaze_left)
        agent = AgentState(
            pos=np.array([2.0, 0.0, 0.85], dtype=float),
            vel=np.array([0.0, 1.0, 0.0], dtype=float),
            gaze_dir=gaze_left,
        )
        mod = EpistemicGazeModulator(lambda_epistemic=0.60, fovea_half_angle_deg=15.0, ema_decay=0.50)

        pos_left = np.array([1.2, 1.8, 0.5], dtype=float)   # directly fixated
        pos_right = np.array([2.8, 1.8, 0.5], dtype=float)  # unattended peripheral

        # Sustain fixation for 5 frames
        for _ in range(5):
            r_left = mod.compute_priority("haz_left", pos_left, agent, delta_cf=0.40, r_v=0.50)
            r_right = mod.compute_priority("haz_right", pos_right, agent, delta_cf=0.40, r_v=0.50)
            r_clutter = mod.compute_priority("clutter_left", pos_left, agent, delta_cf=0.0, r_v=0.10)

        self.assertGreater(r_left.gaze_attention, 0.90)
        self.assertLess(r_right.gaze_attention, 0.05)
        self.assertGreater(r_right.priority, r_left.priority * 1.4)
        self.assertGreater(r_left.priority, r_clutter.priority * 5.0)

        # Graceful fallback when gaze_dir is None
        agent_no_gaze = AgentState(pos=agent.pos, vel=agent.vel, gaze_dir=None)
        r_fallback = mod.compute_priority("haz_right", pos_right, agent_no_gaze, delta_cf=0.40, r_v=0.50)
        self.assertEqual(r_fallback.gaze_attention, 0.0)
        self.assertAlmostEqual(r_fallback.epistemic_multiplier, 1.60, places=6)

    def test_d2_aegis_v4_end_to_end_epistemic_gaze_selection(self) -> None:
        """Inside AegisV4CCBS(k=1), unattended peripheral hazard beats fixated symmetric hazard."""
        gaze_left = np.array([-0.8, 1.8, -0.35], dtype=float)
        gaze_left /= np.linalg.norm(gaze_left)
        agent = AgentState(
            pos=np.array([2.0, 0.0, 0.85], dtype=float),
            vel=np.array([0.0, 1.0, 0.0], dtype=float),
            gaze_dir=gaze_left,
        )
        g = _make_graph([
            {"id": "obs_left_fixated", "label": "chair", "pos": [1.2, 1.8, 0.5], "haz": 0.75},
            {"id": "obs_right_unattended", "label": "table", "pos": [2.8, 1.8, 0.5], "haz": 0.75},
        ])
        v4 = AegisV4CCBS(AegisV4Config(enable_epistemic_gaze=True, lambda_epistemic=0.60))
        out = v4.compress(g, agent, k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_id, "obs_right_unattended")
        self.assertLess(out[0].gaze_attention, 0.1)


class TestAegisV4CCBSIntegration(unittest.TestCase):
    """(e): AegisV4CCBS Atomic Hazard Preservation, Lower Control Regret vs Distance, & Edge Cases."""

    def setUp(self) -> None:
        self.agent = AgentState(
            pos=np.array([2.0, 0.0, 0.85], dtype=float),
            vel=np.array([0.0, 1.35, 0.0], dtype=float),
            radius=0.30,
            height=1.70,
        )

    def test_e1_preserves_non_coarsenable_atomic_hazards(self) -> None:
        """AegisV4CCBS never merges or drops Atomic Hazards (knife, acid_beaker) under tight k."""
        v4 = AegisV4CCBS(AegisV4Config(min_navigable_gap_m=0.70))
        g = _make_graph([
            {"id": "knife_1", "label": "knife", "pos": [2.0, 0.8, 0.9], "size": [0.05, 0.25, 0.03], "haz": 1.0},
            {"id": "acid_1", "label": "hydrochloric_acid_beaker", "pos": [2.15, 0.8, 0.9], "size": [0.15, 0.15, 0.2], "haz": 0.95},
            {"id": "box_1", "label": "cardboard_box", "pos": [1.85, 0.8, 0.5], "size": [0.4, 0.4, 0.4], "haz": 0.70},
            {"id": "box_2", "label": "cardboard_box", "pos": [2.35, 0.8, 0.5], "size": [0.4, 0.4, 0.4], "haz": 0.70},
            {"id": "tray_1", "label": "tray", "pos": [2.0, 1.1, 0.9], "size": [0.4, 0.3, 0.05], "haz": 0.65},
        ])
        out = v4.compress(g, self.agent, k=3)
        self.assertLessEqual(len(out), 3)
        by_id = {e.entity_id: e for e in out}
        for atom_id in ("knife_1", "acid_1"):
            self.assertIn(atom_id, by_id)
            self.assertTrue(by_id[atom_id].is_atomic_hazard)
            self.assertFalse(by_id[atom_id].is_super_node)

    def test_e2_lower_counterfactual_control_regret_than_distance(self) -> None:
        """AegisV4CCBS achieves strictly lower Counterfactual Control Regret than Distance."""
        # Agent moves +Y at 1.4 m/s.
        # 3 closer non-blocking objects sit BEHIND or overhead/side, while 1 CBF-active
        # oncoming blocking obstacle sits directly ahead at y=1.35m (slightly farther than
        # the 3 closer clutter items at dist ~0.8-1.0m).
        agent = AgentState(
            pos=np.array([0.0, 0.0, 0.85], dtype=float),
            vel=np.array([0.0, 1.4, 0.0], dtype=float),
            radius=0.30,
            height=1.70,
        )
        objs = [
            {"id": "behind_1", "label": "chair", "pos": [-0.4, -0.95, 0.45], "size": [0.4, 0.4, 0.8], "haz": 0.2},
            {"id": "behind_2", "label": "chair", "pos": [0.4, -0.95, 0.45], "size": [0.4, 0.4, 0.8], "haz": 0.2},
            {"id": "overhead_1", "label": "lamp", "pos": [0.0, 0.5, 2.55], "size": [0.4, 0.4, 0.15], "haz": 0.0},
            {"id": "blocker_ahead", "label": "pedestrian", "pos": [0.0, 1.35, 0.90], "size": [0.5, 0.5, 1.7], "vel": [0.0, -0.8, 0.0], "haz": 0.95, "dyn": True},
        ]
        g = _make_graph(objs)
        k = 2

        # Distance baseline picks the k=2 closest by Euclidean 3D distance (misses blocker_ahead)
        dists = sorted(objs, key=lambda o: float(np.linalg.norm(np.array(o["pos"]) - agent.pos)))
        dist_selected = dists[:k]
        regret_distance = compute_graph_control_regret(objs, dist_selected, agent, u_nom=agent.vel)

        # Aegis-V4 retains the CBF-active blocker_ahead -> near-zero control regret!
        v4 = AegisV4CCBS()
        v4_selected = v4.compress(g, agent, k=k, dt=0.1, u_nom=agent.vel)
        regret_v4 = compute_graph_control_regret(objs, v4_selected, agent, u_nom=agent.vel)

        self.assertGreater(regret_distance, 0.5)
        self.assertLess(regret_v4, 1e-5)
        self.assertLess(regret_v4, regret_distance)

        # Verify compute_safe_velocity on compressed graph matches full-graph safety
        u_safe_v4, slack_v4, _ = v4.compute_safe_velocity(v4_selected, agent, agent.vel, return_info=True)
        self.assertAlmostEqual(slack_v4, 0.0, places=6)
        c_full = build_cbf_constraints(objs, agent, dt=0.1, gamma_cbf=v4.config.gamma_cbf, safety_margin=v4.config.cbf_safety_margin)
        for c in c_full:
            self.assertGreaterEqual(float(np.dot(c.a, u_safe_v4[:2])), c.b - 1e-5)

    def test_e3_edge_cases_and_api_methods(self) -> None:
        """Verify empty graph, k=0, select(), select_entities(), and reset()."""
        v4 = AegisV4CCBS()
        self.assertEqual(v4.compress(nx.DiGraph(), self.agent, k=5), [])
        g = _make_graph([{"id": "c1", "pos": [2.0, 1.5, 0.45]}])
        self.assertEqual(v4.compress(g, self.agent, k=0), [])

        ents = v4.select_entities(g, self.agent, k=3)
        self.assertEqual(len(ents), 1)
        ids = v4.select(g, self.agent, k=3)
        self.assertEqual(ids, {"c1"})

        v4.reset()
        self.assertEqual(v4.aci.step_count, 0)
        self.assertIsNone(v4.last_cbf_result)

    def test_f1_exact_active_set_slack_qp_and_caratheodory_support(self) -> None:
        """Exact Active-Set 3D slack QP matches analytical KKT optimum and extracts |S_KKT| <= 3."""
        # Single deeply penetrating constraint: a=(0,1), b=4.50, u_nom=(0,0), u_max=2.0, p=100
        # Analytical KKT optimum: u* = (0.0, 2.0), slack* = 2.50
        cs = [
            CBFConstraint(entity_id="enclosure", a=np.array([0.0, 1.0]), b=4.50, h_val=-4.50, dist_2d=-4.05),
            CBFConstraint(entity_id="inactive_1", a=np.array([1.0, 0.0]), b=-1.50, h_val=1.50, dist_2d=1.95),
            CBFConstraint(entity_id="inactive_2", a=np.array([0.0, -1.0]), b=-3.00, h_val=3.00, dist_2d=3.45),
        ]
        u_sol, slack, active = solve_cbf_qp(np.array([0.0, 0.0]), cs, u_max=2.0, slack_penalty=100.0, return_info=True)
        self.assertAlmostEqual(float(u_sol[0]), 0.0, places=6)
        self.assertAlmostEqual(float(u_sol[1]), 2.0, places=6)
        self.assertAlmostEqual(slack, 2.50, places=6)
        self.assertEqual(active, [0])

        # Verify Caratheodory support size <= 3 on multi-obstacle scene
        agent = AgentState(pos=np.array([0.0, 0.0, 0.85]), vel=np.array([0.0, 1.5, 0.0]))
        objs = [
            {"id": f"w_{i}", "label": "chair", "pos": [0.4 * (i - 2), 0.7, 0.5], "size": [0.5, 0.5, 0.8]}
            for i in range(6)
        ]
        res = compute_counterfactual_regrets(objs, agent, u_nom=agent.vel)
        self.assertLessEqual(len(res.support_ids), 3)
        self.assertGreaterEqual(len(res.support_ids), 1)

    def test_f2_cobinding_constraints_kkt_support_zero_regret(self) -> None:
        """Two identical co-binding blockers both have LOO delta_cf == 0, yet KKT support yields 0 regret."""
        agent = AgentState(pos=np.array([0.0, 0.0, 0.85]), vel=np.array([0.0, 1.4, 0.0]))
        objs = [
            {"id": "twin_a", "label": "table", "pos": [0.0, 0.85, 0.5], "size": [1.2, 0.4, 0.8], "haz": 0.4},
            {"id": "twin_b", "label": "desk", "pos": [0.0, 0.85, 0.5], "size": [1.2, 0.4, 0.8], "haz": 0.4},
            {"id": "clutter_1", "label": "chair", "pos": [-1.0, -0.8, 0.5], "size": [0.4, 0.4, 0.8], "haz": 0.2},
            {"id": "clutter_2", "label": "chair", "pos": [1.0, -0.8, 0.5], "size": [0.4, 0.4, 0.8], "haz": 0.2},
        ]
        res = compute_counterfactual_regrets(objs, agent, u_nom=agent.vel)
        # Both twins mask each other under Leave-One-Out
        self.assertAlmostEqual(res.regrets["twin_a"], 0.0, places=6)
        self.assertAlmostEqual(res.regrets["twin_b"], 0.0, places=6)
        # Yet Caratheodory KKT support identifies one of them
        self.assertTrue(bool(res.support_ids & {"twin_a", "twin_b"}))

        v4 = AegisV4CCBS()
        sel = v4.compress(_make_graph(objs), agent, k=2, dt=0.1, u_nom=agent.vel)
        reg = compute_graph_control_regret(objs, sel, agent, u_nom=agent.vel)
        self.assertLess(reg, 1e-5)

    def test_f3_self_supervised_proxy_aci_no_pinning_inside_enclosure(self) -> None:
        """Inside a large enclosing box (slack > 2.0), self-supervised ACI does not pin at alpha_min."""
        agent = AgentState(pos=np.array([0.0, 0.0, 0.85]), vel=np.array([0.0, 0.4, 0.0]))
        g = _make_graph([
            {"id": "room_shell", "label": "shelter", "pos": [0.0, 0.0, 1.5], "size": [10.0, 10.0, 3.0], "haz": 0.7},
            {"id": "chair_1", "label": "chair", "pos": [0.0, 1.8, 0.45], "size": [0.5, 0.5, 0.8], "haz": 0.7},
        ])
        v4 = AegisV4CCBS(AegisV4Config(enable_online_aci=True, aci_alpha_target=0.05))
        for _ in range(50):
            v4.compress(g, agent, k=5)
        self.assertGreater(v4.aci.empirical_coverage, 0.85)
        self.assertGreater(v4.aci.alpha_t, v4.aci.alpha_min + 1e-3)

    def test_f4_deadlock_free_tangential_cbf_qp(self) -> None:
        """Deadlock escape breaks collinear wall stall while strictly satisfying all CBF barriers."""
        agent = AgentState(pos=np.array([0.0, 0.0, 0.85]), vel=np.array([0.0, 1.2, 0.0]))
        wall = _make_graph([
            {"id": "flat_wall", "label": "wall", "pos": [0.0, 0.70, 1.0], "size": [4.0, 0.4, 2.0], "haz": 0.9},
        ])
        v4 = AegisV4CCBS()
        ents = v4.compress(wall, agent, k=3)
        u_nom = np.array([0.0, 1.2], dtype=float)
        u_plain = v4.compute_safe_velocity(ents, agent, u_nom, dt=0.1, u_max=1.4, gamma_cbf=0.2, safety_margin=0.05, enable_deadlock_escape=False)
        u_esc, slack_esc, _ = v4.compute_safe_velocity(ents, agent, u_nom, dt=0.1, u_max=1.4, gamma_cbf=0.2, safety_margin=0.05, enable_deadlock_escape=True, return_info=True)
        # Plain CBF-QP stalls near 0 speed; deadlock escape slides tangentially
        self.assertLess(float(np.linalg.norm(u_plain)), 0.25)
        self.assertGreater(float(np.linalg.norm(u_esc)), 0.50)
        # Every CBF barrier constraint must remain strictly satisfied
        cs = build_cbf_constraints(ents, agent, dt=0.1, gamma_cbf=0.2, safety_margin=0.05)
        for c in cs:
            self.assertGreaterEqual(float(np.dot(c.a, u_esc)) + slack_esc, c.b - 1e-5)

    def test_f5_no_gaze_sensor_no_weight_inflation(self) -> None:
        """When agent.gaze_dir is None (no eye-tracker), AegisV4CCBS must not inflate weights by 1+lambda."""
        agent_no_gaze = AgentState(
            pos=np.array([0.0, 0.0, 0.85]),
            vel=np.array([0.0, 1.0, 0.0]),
            gaze_dir=None,
        )
        g = _make_graph([
            {"id": "chair_far", "label": "chair", "pos": [1.5, 2.2, 0.45], "size": [0.5, 0.5, 0.8], "haz": 0.6},
        ])
        v4_on = AegisV4CCBS(AegisV4Config(enable_epistemic_gaze=True, lambda_epistemic=0.60, tau_min=0.0))
        v4_off = AegisV4CCBS(AegisV4Config(enable_epistemic_gaze=False, lambda_epistemic=0.60, tau_min=0.0))
        out_on = v4_on.compress(g, agent_no_gaze, k=3)
        out_off = v4_off.compress(g, agent_no_gaze, k=3)
        self.assertEqual(len(out_on), 1)
        self.assertEqual(len(out_off), 1)
        self.assertAlmostEqual(out_on[0].weight, out_off[0].weight, places=6)
        self.assertAlmostEqual(out_on[0].epistemic_priority, out_on[0].delta_cf + out_on[0].risk, places=6)

    def test_f6_degenerate_collinear_and_tiny_umax_qp(self) -> None:
        """Exact CBF-QP handles nearly parallel/duplicate constraints and tiny u_max without drift."""
        from src.aegis_v4.cbf_sparsifier import _solve_cbf_qp_arrays

        cases = [
            (
                np.array([0.3, 0.1]),
                np.array([[1.0, 0.0], [1.0, 1e-6], [-1.0, 1e-6]]),
                np.array([2.0, 2.5, 1.5]),
                0.5,
            ),
            (
                np.array([0.8, 0.2]),
                np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]),
                np.array([1.5, 1.5, 1.5, 1.5]),
                1.0,
            ),
            (
                np.array([2.0, -1.5]),
                np.array([[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]]),
                np.array([1.2, 1.4, 1.1]),
                0.05,
            ),
        ]
        for u_nom, A, b, u_max in cases:
            u_sol, slack, _, supp, _ = _solve_cbf_qp_arrays(
                u_nom, A, b, u_max=u_max, slack_penalty=100.0, return_support=True
            )
            self.assertLessEqual(float(np.linalg.norm(u_sol)), u_max + 1e-5)
            self.assertGreaterEqual(float(np.min(A @ u_sol + slack - b)), -1e-5)
            self.assertGreaterEqual(slack, -1e-7)
            self.assertLessEqual(len(supp), 3)
            if supp:
                u_s, _, _ = _solve_cbf_qp_arrays(u_nom, A[supp], b[supp], u_max=u_max, slack_penalty=100.0)
                self.assertLess(float(np.linalg.norm(u_sol - u_s)), 1e-4)


if __name__ == "__main__":
    unittest.main()

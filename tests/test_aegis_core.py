"""Unit tests verifying every mathematical and safety guarantee of AegisDSG (src/aegis_core.py)."""

from __future__ import annotations

import unittest
import networkx as nx
import numpy as np

from src.aegis_core import AegisConfig, AegisDSG, AffordanceRiskMapper
from src.aegis_v3_modules import (
    ConformalSafetyCalibrator,
    ZeroShotCLIPCache,
    compute_min_area_obb_2d,
    entity_corners_2d,
)
from src.dsg_generator import AgentState


def _make_graph(objs: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for o in objs:
        g.add_node(
            o["id"],
            layer="object",
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


class TestAegisCore(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AgentState(pos=np.array([2.0, 0.0, 0.85]), vel=np.array([0.0, 1.35, 0.0]))

    def test_edge_cases_empty_and_zero_k(self) -> None:
        aegis = AegisDSG()
        self.assertEqual(aegis.compress(nx.DiGraph(), self.agent, k=5), [])
        g = _make_graph([{"id": "c1", "pos": [2.0, 2.0, 0.4]}])
        self.assertEqual(aegis.compress(g, self.agent, k=0), [])

    def test_layer1_fp_ghost_damping_and_object_permanence(self) -> None:
        aegis = AegisDSG()
        # Frame 1: real high-conf obstacle + 1-frame low-conf FP ghost at same risk distance
        g1 = _make_graph([
            {"id": "real_chair", "pos": [2.0, 2.5, 0.4], "conf": 0.92},
            {"id": "fp_ghost", "pos": [2.1, 2.5, 0.4], "conf": 0.30, "label": "unknown_ghost"},
        ])
        out1 = aegis.compress(g1, self.agent, k=1)
        self.assertEqual(len(out1), 1)
        self.assertEqual(out1[0].entity_id, "real_chair")

        # Frame 2: real_chair seen again -> becomes persistent
        g2 = _make_graph([{"id": "real_chair", "pos": [2.0, 2.5, 0.4], "conf": 0.90}])
        _ = aegis.compress(g2, self.agent, k=3)

        # Frame 3: real_chair temporarily occluded (missing from g3) -> Object Permanence retains it!
        out3 = aegis.compress(nx.DiGraph(), self.agent, k=3)
        self.assertEqual(len(out3), 1)
        self.assertEqual(out3[0].entity_id, "real_chair")
        self.assertTrue(out3[0].is_occluded_memory)

    def test_layer2_no_side_swipe_blind_spot_and_height_filter(self) -> None:
        aegis = AegisDSG()
        # Side chair 0.9m to agent's right (outside ±38° cone!) vs harmless overhead light & far door
        g = _make_graph([
            {"id": "side_chair", "pos": [2.9, 0.4, 0.4], "conf": 0.90, "haz": 0.85},
            {"id": "ceiling_light", "pos": [2.0, 1.5, 2.75], "size": [0.5, 0.5, 0.12], "haz": 0.0, "cv": 0.05},
            {"id": "far_door", "pos": [2.0, 9.0, 1.05], "size": [0.2, 1.0, 2.1], "haz": 0.3, "cv": 0.95},
        ])
        out = aegis.compress(g, self.agent, k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_id, "side_chair")

    def test_layer3_free_space_preserving_coarsening(self) -> None:
        aegis = AegisDSG(AegisConfig(min_navigable_gap_m=0.70, proactive_same_class_merge=True))
        # c1 & c2 touch (gap = 0.10m < 0.70m -> MUST merge); c3 is 1.2m away (gap >= 0.70m -> MUST NOT merge!)
        g = _make_graph([
            {"id": "c1", "label": "chair", "pos": [1.0, 2.0, 0.4], "size": [0.5, 0.5, 0.8]},
            {"id": "c2", "label": "chair", "pos": [1.6, 2.0, 0.4], "size": [0.5, 0.5, 0.8]},
            {"id": "c3", "label": "chair", "pos": [3.4, 2.0, 0.4], "size": [0.5, 0.5, 0.8]},
        ])
        out = aegis.compress(g, self.agent, k=5)
        self.assertEqual(len(out), 2)
        super_nodes = [e for e in out if e.is_super_node]
        self.assertEqual(len(super_nodes), 1)
        self.assertEqual(super_nodes[0].member_ids, {"c1", "c2"})

    def test_layer4_dynamic_budget_drops_low_score_clutter(self) -> None:
        aegis = AegisDSG(AegisConfig(tau_min=0.15))
        # Only 1 real hazard + 4 distant harmless clutter items (S << 0.15) with k=5
        g = _make_graph([
            {"id": "real_haz", "pos": [2.0, 2.0, 0.5], "haz": 0.95},
            {"id": "junk_1", "pos": [0.2, 10.0, 2.2], "haz": 0.0, "cv": 0.05},
            {"id": "junk_2", "pos": [3.8, 10.5, 2.2], "haz": 0.0, "cv": 0.05},
            {"id": "junk_3", "pos": [0.2, 11.0, 2.7], "haz": 0.0, "cv": 0.05},
            {"id": "junk_4", "pos": [3.8, 11.5, 2.7], "haz": 0.0, "cv": 0.05},
        ])
        out = aegis.compress(g, self.agent, k=5)
        # Dynamic budget stops at 1 entity instead of padding 4 junk items to fill k=5!
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_id, "real_haz")

    def test_v2_atomic_hazard_never_erased_by_coarsening(self) -> None:
        aegis = AegisDSG(AegisConfig(min_navigable_gap_m=0.70))
        # A sharp knife sits 0.05m from a cutting_board and tray (overflow at k=2):
        # The knife MUST NOT be swallowed into a generic no_go_cluster!
        g = _make_graph([
            {"id": "k1", "label": "knife", "pos": [2.0, 0.7, 0.9], "size": [0.05, 0.25, 0.03], "haz": 1.0},
            {"id": "b1", "label": "cutting_board", "pos": [2.1, 0.7, 0.9], "size": [0.4, 0.3, 0.04], "haz": 0.7},
            {"id": "t1", "label": "tray", "pos": [1.9, 0.7, 0.9], "size": [0.4, 0.3, 0.05], "haz": 0.7},
            {"id": "c1", "label": "chair", "pos": [2.0, 1.4, 0.4], "size": [0.5, 0.5, 0.8], "haz": 0.85},
        ])
        out = aegis.compress(g, self.agent, k=2)
        self.assertEqual(len(out), 2)
        atomic_out = [e for e in out if e.entity_id == "k1"]
        self.assertEqual(len(atomic_out), 1)
        self.assertFalse(atomic_out[0].is_super_node)
        self.assertTrue(atomic_out[0].is_atomic_hazard)
        self.assertEqual(atomic_out[0].label, "knife")

    def test_v2_multi_domain_affordance_mapper(self) -> None:
        """Verify domain-agnostic protection on construction/lab scenes (not just indoor homes)."""
        aegis = AegisDSG(AegisConfig(min_navigable_gap_m=0.70))
        g = _make_graph([
            {"id": "lab_acid", "label": "hydrochloric_acid_beaker", "pos": [2.0, 0.8, 0.9], "size": [0.15, 0.15, 0.2], "haz": 0.9},
            {"id": "wire1", "label": "exposed_wire", "pos": [2.1, 0.8, 0.9], "size": [0.1, 0.3, 0.05], "haz": 0.9},
            {"id": "box1", "label": "cardboard_box", "pos": [1.85, 0.8, 0.5], "size": [0.4, 0.4, 0.4], "haz": 0.7},
            {"id": "box2", "label": "cardboard_box", "pos": [2.30, 0.8, 0.5], "size": [0.4, 0.4, 0.4], "haz": 0.7},
        ])
        out = aegis.compress(g, self.agent, k=3)
        ids = {e.entity_id for e in out}
        self.assertIn("lab_acid", ids)
        self.assertIn("wire1", ids)
        for e in out:
            if e.entity_id in ("lab_acid", "wire1"):
                self.assertTrue(e.is_atomic_hazard)
                self.assertFalse(e.is_super_node)

    def test_v2_zero_shot_embedding_offline_cache(self) -> None:
        """Verify zero-shot semantic embedding hook caches unseen classes in O(1) at runtime."""
        calls = {"n": 0}

        def mock_clip_encoder(text: str) -> np.ndarray:
            calls["n"] += 1
            t = text.lower()
            # Map unseen exotic label 'machete_tool' close to sharp_puncture prototype
            if "sharp" in t or "machete" in t:
                return np.array([1.0, 0.0, 0.0])
            if "heat" in t or "flame" in t:
                return np.array([0.0, 1.0, 0.0])
            return np.array([0.0, 0.0, 1.0])

        mapper = AffordanceRiskMapper(embedding_fn=mock_clip_encoder, embedding_threshold=0.75)
        init_calls = calls["n"]  # 6 prototype axes + 1 benign contrastive baseline encoded once at startup
        self.assertEqual(init_calls, 7)

        sev1, atom1, axis1 = mapper.assess("exotic_machete_tool")
        self.assertTrue(atom1)
        self.assertGreaterEqual(sev1, 0.90)
        self.assertEqual(axis1, "emb:sharp_puncture")
        after_first = calls["n"]

        # Repeated runtime calls hit O(1) cache -> ZERO encoder calls!
        for _ in range(100):
            _, atom_rep, _ = mapper.assess("exotic_machete_tool")
            self.assertTrue(atom_rep)
        self.assertEqual(calls["n"], after_first)

    def test_v3_eye_gaze_unattended_hazard_priority(self) -> None:
        """Breakthrough 1: Unattended peripheral hazards outrank fixated hazards at identical kinematics."""
        aegis = AegisDSG(AegisConfig(enable_gaze_attention=True, gaze_unattended_boost=0.30))
        # Agent walks straight +Y, but eyes are fixated directly on the LEFT obstacle (-0.8, 1.8)
        gaze_left = np.array([-0.8, 1.8, -0.35], dtype=float)
        gaze_left /= np.linalg.norm(gaze_left)
        agent_gaze = AgentState(pos=np.array([2.0, 0.0, 0.85]), vel=np.array([0.0, 1.0, 0.0]), gaze_dir=gaze_left)
        g = _make_graph([
            {"id": "obs_left_seen", "label": "chair", "pos": [1.2, 1.8, 0.5], "haz": 0.75},
            {"id": "obs_right_unseen", "label": "table", "pos": [2.8, 1.8, 0.5], "haz": 0.75},
        ])
        out = aegis.compress(g, agent_gaze, k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_id, "obs_right_unseen")
        self.assertLess(out[0].gaze_attention, 0.1)

    def test_v3_split_conformal_safety_calibration(self) -> None:
        """Breakthrough 2: Finite-sample Split Conformal quantile satisfies P(score >= tau) >= 1 - alpha."""
        rng = np.random.default_rng(7)
        calib_scores = rng.uniform(0.05, 1.0, size=200)
        calib_risks = rng.uniform(0.10, 1.0, size=200)
        cal = ConformalSafetyCalibrator(alpha=0.05)
        rep = cal.calibrate_from_scores(calib_scores, calib_risks, default_tau_min=0.25, default_tau_haz=0.35)
        self.assertGreaterEqual(rep.calib_empirical_coverage, 0.95)
        self.assertLessEqual(rep.calibrated_tau_min, 0.25)

    def test_v3_obb_rotating_calipers_diagonal_row(self) -> None:
        """Breakthrough 3: Rotating-Calipers OBB merges a 45-deg diagonal furniture row without AABB bloat."""
        pts = np.vstack([
            entity_corners_2d(np.array([1.0, 1.0]), np.array([0.4, 0.4])),
            entity_corners_2d(np.array([1.5, 1.5]), np.array([0.4, 0.4])),
            entity_corners_2d(np.array([2.0, 2.0]), np.array([0.4, 0.4])),
        ])
        aabb_area = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        _, obb_wh, obb_yaw, obb_area = compute_min_area_obb_2d(pts)
        # OBB area on a 45-degree diagonal strip must be strictly much tighter than AABB!
        self.assertLess(obb_area, 0.65 * aabb_area)
        self.assertGreater(abs(obb_yaw), 0.3)

        # Verify Layer 3 merges the 3 diagonal chairs into ONE rotated OBB Super-Node
        aegis = AegisDSG(AegisConfig(enable_obb_supernodes=True, min_navigable_gap_m=0.70, max_aabb_bloat_ratio=2.2))
        g = _make_graph([
            {"id": "d1", "label": "chair", "pos": [1.0, 1.5, 0.4], "size": [0.45, 0.45, 0.8]},
            {"id": "d2", "label": "chair", "pos": [1.5, 2.0, 0.4], "size": [0.45, 0.45, 0.8]},
            {"id": "d3", "label": "chair", "pos": [2.0, 2.5, 0.4], "size": [0.45, 0.45, 0.8]},
        ])
        out = aegis.compress(g, self.agent, k=5)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].is_super_node)
        self.assertIsNotNone(out[0].obb_size)
        self.assertEqual(out[0].member_ids, {"d1", "d2", "d3"})

    def test_v3_cross_layer_hierarchical_semantic_zoom(self) -> None:
        """Breakthrough 4: Keeps near atomic knife at Object layer while zooming distant clutter to Place layer."""
        aegis = AegisDSG(AegisConfig(enable_hierarchical_zoom=True, hierarchical_zoom_dist_m=3.0, tau_min=0.05))
        g = _make_graph([
            {"id": "k1", "label": "knife", "pos": [2.0, 0.7, 0.9], "size": [0.05, 0.25, 0.03], "haz": 1.0},
            {"id": "far1", "label": "vase", "pos": [2.0, 5.0, 0.8], "size": [0.2, 0.2, 0.3], "haz": 0.0, "cv": 0.4},
            {"id": "far2", "label": "lamp", "pos": [2.6, 5.4, 0.8], "size": [0.3, 0.3, 0.5], "haz": 0.0, "cv": 0.4},
            {"id": "far3", "label": "book", "pos": [1.4, 5.2, 0.8], "size": [0.2, 0.2, 0.2], "haz": 0.0, "cv": 0.4},
        ])
        g.add_node("place_far", layer="place", label="dining_corner", pos=np.array([2.0, 5.2, 1.0]), size=np.array([2.5, 2.5, 2.4]), structural_importance=0.85)
        for nid in ("far1", "far2", "far3"):
            g.add_edge(nid, "place_far", relation="in_place")

        out = aegis.compress(g, self.agent, k=3)
        by_id = {e.entity_id: e for e in out}
        self.assertIn("k1", by_id)
        self.assertEqual(by_id["k1"].dsg_layer, "object")
        place_nodes = [e for e in out if e.dsg_layer == "place"]
        self.assertEqual(len(place_nodes), 1)
        self.assertEqual(place_nodes[0].member_ids, {"far1", "far2", "far3"})

    def test_v3_zero_shot_clip_cache_task_routing(self) -> None:
        """Breakthrough 5: Offline-cached open-vocab task engine boosts task-relevant items at O(1) lookup."""
        clip_cache = ZeroShotCLIPCache()  # uses lightweight fallback without downloading weights during unit test
        clip_cache.set_task_prompt("brew hot coffee mug", vocabulary=["coffee_mug", "wall_painting"])
        aegis = AegisDSG(AegisConfig(tau_min=0.05, clip_task_weight=0.50), clip_cache=clip_cache)
        g = _make_graph([
            {"id": "mug1", "label": "coffee_mug", "pos": [2.0, 4.0, 0.9], "haz": 0.0, "cv": 0.1},
            {"id": "art1", "label": "wall_painting", "pos": [2.0, 4.0, 0.9], "haz": 0.0, "cv": 0.1},
        ])
        out = aegis.compress(g, self.agent, k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].entity_id, "mug1")
        self.assertGreater(out[0].task_relevance, 0.0)


if __name__ == "__main__":
    unittest.main()

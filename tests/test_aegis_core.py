"""Unit tests verifying every mathematical and safety guarantee of AegisDSG (src/aegis_core.py)."""

from __future__ import annotations

import unittest
import networkx as nx
import numpy as np

from src.aegis_core import AegisConfig, AegisDSG, AffordanceRiskMapper
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
        init_calls = calls["n"]  # 6 prototype axes encoded once at startup
        self.assertEqual(init_calls, 6)

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


if __name__ == "__main__":
    unittest.main()

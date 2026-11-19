"""Proposed Task-Oriented DSG Sparsifier (Dynamic Collision Cone + Asymmetric Risk + Hysteresis)."""

from __future__ import annotations

from typing import Dict, Set
import math
import networkx as nx
import numpy as np

from src.dsg_generator import AgentState


class ProposedSparsifier:
    """Task-oriented sparsifier combining Dynamic Collision Cone, Asymmetric Risk, and Hysteresis."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.20,
        gamma: float = 1.50,
        tau_haz: float = 0.30,
        horizon_s: float = 3.2,
        cone_half_angle_deg: float = 35.0,
        hysteresis_bonus: float = 0.18,
        theta_out: float = 0.12,
        min_persistence_frames: int = 2,
    ) -> None:
        self.name = "Proposed"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau_haz = tau_haz
        self.horizon_s = horizon_s
        self.cone_half_angle_rad = math.radians(cone_half_angle_deg)
        self.hysteresis_bonus = hysteresis_bonus
        self.theta_out = theta_out
        self.min_persistence_frames = min_persistence_frames

        self._prev_selected: Set[str] = set()
        self._persistence: Dict[str, int] = {}

    def reset(self) -> None:
        """Clear temporal hysteresis memory at the start of a new sequence."""
        self._prev_selected.clear()
        self._persistence.clear()

    def _compute_collision_risk(self, data: dict, agent: AgentState) -> float:
        """Compute dynamic collision risk R_v in [0, 1] using reachable forward cone & TTC."""
        pos = data["pos"]
        size = data["size"]
        vel = data["vel"]

        # 1. Height relevance filter (ignore overhead objects above head & flat floor mats)
        z_min = pos[2] - size[2] / 2.0
        z_max = pos[2] + size[2] / 2.0
        if z_min > agent.height + 0.15 or z_max < 0.05:
            return 0.0

        # 2. Relative kinematics in XY plane
        rel_pos = pos[:2] - agent.pos[:2]
        dist = float(np.linalg.norm(rel_pos))
        rel_vel = vel[:2] - agent.vel[:2]
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))

        if rel_speed_sq < 1e-6:
            ttc = 0.0
        else:
            ttc = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, self.horizon_s))

        closest_vec = rel_pos + rel_vel * ttc
        # Account for object half-extents + agent safety radius
        half_ext = size[:2] / 2.0 + agent.radius + 0.20
        excess = np.maximum(np.abs(closest_vec) - half_ext, 0.0)
        d_min = float(np.linalg.norm(excess))

        # 3. Reachable forward cone alignment
        agent_speed = float(np.linalg.norm(agent.vel[:2]))
        if agent_speed > 1e-4 and dist > 1e-4:
            heading = agent.vel[:2] / agent_speed
            cos_ang = float(np.clip(np.dot(rel_pos / dist, heading), -1.0, 1.0))
            angle = math.acos(cos_ang)
        else:
            angle = 0.0

        # Smooth cone factor: 1.0 inside reachable cone OR inside personal safety bubble (< 2.2m)
        # Prevents side-swipe blind spots when steering sharply around front obstacles.
        in_safety_bubble = dist <= 2.2
        if in_safety_bubble or angle <= self.cone_half_angle_rad:
            cone_factor = 1.0
        else:
            cone_factor = math.exp(-((angle - self.cone_half_angle_rad) / 0.35) ** 2)

        # Objects moving toward agent can still enter the cone even from wider angles
        closing_speed = -float(np.dot(rel_pos / (dist + 1e-6), rel_vel))
        if closing_speed > 0.3:
            cone_factor = max(cone_factor, 0.75)

        # Combine spatial proximity (min of TTC approach & immediate distance inside bubble)
        d_now = float(np.linalg.norm(np.maximum(np.abs(rel_pos) - half_ext, 0.0)))
        eff_dist = min(d_min, d_now) if in_safety_bubble else d_min
        spatial_risk = math.exp(-(eff_dist ** 2) / (2.0 * 0.65 ** 2))
        time_urgency = 1.0 if in_safety_bubble else (1.0 - 0.5 * (ttc / self.horizon_s))
        sev = float(data.get("hazard_severity", 0.8))

        r_v = spatial_risk * time_urgency * cone_factor * max(sev, 0.2)
        return float(np.clip(r_v, 0.0, 1.0))

    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        """Select up to k objects maximizing asymmetric safety score with temporal hysteresis."""
        obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
        if not obj_ids:
            self._prev_selected.clear()
            return set()

        scores: Dict[str, float] = {}
        for nid in obj_ids:
            data = graph_obs.nodes[nid]
            q_v = float(data.get("confidence", 1.0))
            u_v = 1.0 - q_v
            c_v = float(data.get("structural_importance", 0.0))
            r_v = self._compute_collision_risk(data, agent)

            # Asymmetric Safety Cost: uncertainty boosts priority inside hazard cone
            haz_indicator = 1.0 if r_v > self.tau_haz else 0.0
            s_v = self.alpha * r_v * (1.0 + self.gamma * u_v * haz_indicator) + self.beta * q_v * c_v

            # Temporal Hysteresis & Persistence bonus for previously retained nodes
            if nid in self._prev_selected and s_v >= self.theta_out:
                persisted = self._persistence.get(nid, 1)
                if persisted < self.min_persistence_frames:
                    s_v += self.hysteresis_bonus * 1.5
                else:
                    s_v += self.hysteresis_bonus

            scores[nid] = s_v

        ranked = sorted(obj_ids, key=lambda nid: scores[nid], reverse=True)
        selected = set(ranked[:k])

        # Update temporal state
        new_persistence: Dict[str, int] = {}
        for nid in selected:
            new_persistence[nid] = self._persistence.get(nid, 0) + 1
        self._persistence = new_persistence
        self._prev_selected = selected

        return selected

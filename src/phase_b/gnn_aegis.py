"""Mũi 2B: Pure-PyTorch Graph Attention Network (GAT) with Asymmetric Safety Loss.

Learns node retention scores directly over the 3D Scene Graph topology (edges: 'near',
spatial relations) instead of relying solely on hand-crafted alpha/beta/gamma weights.
Trained with an Asymmetric Weighted BCE Loss that penalizes missed hazards (FN)
5x more heavily than false positives (FP).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dsg_generator import AgentState, DSGFrame


class SimpleGATLayer(nn.Module):
    """Single-head Graph Attention layer operating on dense local Active-Window graphs (N <= 50)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim), adj: (N, N) binary adjacency with self-loops
        h = self.lin(x)  # (N, out_dim)
        e_src = (h * self.att_src).sum(dim=-1)  # (N,)
        e_dst = (h * self.att_dst).sum(dim=-1)  # (N,)
        e = F.leaky_relu(e_src.unsqueeze(1) + e_dst.unsqueeze(0), negative_slope=0.2)
        e = e.masked_fill(adj <= 0, -1e9)
        attn = F.softmax(e, dim=-1)
        return attn @ h


class AegisGATNet(nn.Module):
    """2-layer Graph Attention Network predicting safety-critical retention logits per node."""

    def __init__(self, in_dim: int = 11, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gat1 = SimpleGATLayer(in_dim, hidden_dim)
        self.gat2 = SimpleGATLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim + in_dim, 1)  # Residual skip connection

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.gat1(x, adj))
        h = F.elu(self.gat2(h, adj))
        logits = self.head(torch.cat([h, x], dim=-1)).squeeze(-1)
        return logits


def extract_graph_tensors(
    graph_obs: nx.DiGraph,
    agent: AgentState,
    prev_selected: Set[str],
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an observed 3D Scene Graph G_t^obs into (obj_ids, X_feat, Adj, y_gt)."""
    obj_ids = [nid for nid, d in graph_obs.nodes(data=True) if d.get("layer") == "object"]
    n = len(obj_ids)
    if n == 0:
        return [], torch.zeros((0, 11)), torch.zeros((0, 0)), torch.zeros((0,))

    spd = float(np.linalg.norm(agent.vel[:2]))
    heading = agent.vel[:2] / spd if spd > 0.1 else np.array([0.0, 1.0])
    ortho = np.array([-heading[1], heading[0]])

    feats = np.zeros((n, 11), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    id_to_idx = {nid: i for i, nid in enumerate(obj_ids)}

    for i, nid in enumerate(obj_ids):
        d = graph_obs.nodes[nid]
        pos = d["pos"]
        size = d["size"]
        vel = d["vel"]

        rel_xy = pos[:2] - agent.pos[:2]
        fwd = float(np.dot(rel_xy, heading)) / 12.0
        lat = float(np.dot(rel_xy, ortho)) / 6.0
        dist = float(np.linalg.norm(rel_xy)) / 12.0

        rel_v = vel[:2] - agent.vel[:2]
        closing = -float(np.dot(rel_xy / (float(np.linalg.norm(rel_xy)) + 1e-6), rel_v)) / 3.0

        z_min = (pos[2] - size[2] / 2.0) / 2.5
        z_max = (pos[2] + size[2] / 2.0) / 2.5
        q_v = float(d.get("confidence", 1.0))
        u_v = 1.0 - q_v
        c_v = float(d.get("structural_importance", 0.0))
        haz_sev = float(d.get("hazard_severity", 0.5))
        was_sel = 1.0 if nid in prev_selected else 0.0

        feats[i] = [fwd, lat, dist, closing, z_min, z_max, q_v, u_v, c_v, haz_sev, was_sel]
        labels[i] = 1.0 if d.get("is_hazard_gt", False) else 0.0

    # Build adjacency matrix from actual DSG edges ('near' or shared 'place') + self-loops
    adj = np.eye(n, dtype=np.float32)
    for u, v in graph_obs.edges():
        if u in id_to_idx and v in id_to_idx:
            iu, iv = id_to_idx[u], id_to_idx[v]
            adj[iu, iv] = 1.0
            adj[iv, iu] = 1.0

    return obj_ids, torch.from_numpy(feats), torch.from_numpy(adj), torch.from_numpy(labels)


class GNNAegisSparsifier:
    """Learned Graph Attention Network Sparsifier trained with Asymmetric Safety Loss."""

    def __init__(self, fn_weight: float = 5.0, hysteresis_bonus: float = 0.8) -> None:
        self.name = "GNN-Aegis"
        self.fn_weight = fn_weight
        self.hysteresis_bonus = hysteresis_bonus
        self.model = AegisGATNet(in_dim=11, hidden_dim=32)
        self._prev_selected: Set[str] = set()

    def reset(self) -> None:
        self._prev_selected.clear()

    def train_on_sequences(self, train_sequences: List[List[DSGFrame]], epochs: int = 12, lr: float = 5e-3) -> float:
        """Train GAT with Asymmetric Weighted BCE Loss (FN penalized fn_weight x harder than FP)."""
        torch.manual_seed(42)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        pos_weight = torch.tensor([self.fn_weight], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Pre-extract tensors for fast CPU training
        samples = []
        for seq in train_sequences:
            for frame in seq[::2]:  # Subsample every 2nd frame for diversity & speed
                ids, x, adj, y = extract_graph_tensors(frame.graph_obs, frame.agent, set())
                if len(ids) > 0:
                    samples.append((x, adj, y))

        final_loss = 0.0
        for _ in range(epochs):
            total_loss = 0.0
            for x, adj, y in samples:
                opt.zero_grad()
                logits = self.model(x, adj)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            final_loss = total_loss / max(1, len(samples))

        self.model.eval()
        return final_loss

    @torch.no_grad()
    def select(self, graph_obs: nx.DiGraph, agent: AgentState, k: int) -> Set[str]:
        obj_ids, x, adj, _ = extract_graph_tensors(graph_obs, agent, self._prev_selected)
        if not obj_ids:
            self._prev_selected.clear()
            return set()
        if len(obj_ids) <= k:
            self._prev_selected = set(obj_ids)
            return set(obj_ids)

        logits = self.model(x, adj).numpy()
        # Mild temporal hysteresis on logits to reduce flicker
        for i, nid in enumerate(obj_ids):
            if nid in self._prev_selected and logits[i] > -1.0:
                logits[i] += self.hysteresis_bonus

        top_idx = np.argsort(-logits)[:k]
        selected = {obj_ids[i] for i in top_idx}
        self._prev_selected = selected
        return selected

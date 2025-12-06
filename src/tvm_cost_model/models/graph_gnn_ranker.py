"""Graph-based ranker with relational GraphSAGE and attention pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tvm_cost_model.features.graph_encoder import GraphEncoding, TensorGraphEncoding


def _to_tensor(values: Any, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.tensor(values, dtype=dtype, device=device)


@dataclass
class RankerOutput:
    score: torch.Tensor
    attribution: torch.Tensor  # shape: [num_nodes]


class RelationalGraphSAGELayer(nn.Module):
    """GraphSAGE layer with per-edge-type aggregation."""

    def __init__(self, hidden_dim: int, num_edge_types: int, dropout: float = 0.1) -> None:
        super().__init__()  # type: ignore
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.rel_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        num_nodes, hidden_dim = h.shape
        device = h.device
        relation_messages = torch.zeros_like(h)

        if edge_index.numel() > 0 and len(self.rel_linears) > 0:
            # Aggregate neighbor messages per edge type.
            for rel_id, rel_linear in enumerate(self.rel_linears):
                mask = edge_types == rel_id
                if not torch.any(mask):
                    continue
                type_edges = edge_index[mask]
                src = type_edges[:, 0]
                dst = type_edges[:, 1]
                messages = h[src]

                agg = torch.zeros((num_nodes, hidden_dim), device=device, dtype=h.dtype)
                agg.index_add_(0, dst, messages)

                deg = torch.zeros((num_nodes,), device=device, dtype=h.dtype)
                deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
                mean_messages = agg / (deg.clamp(min=1).unsqueeze(-1))

                relation_messages = relation_messages + rel_linear(mean_messages)

        out = self.self_linear(h) + relation_messages
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class GraphGNNRanker(nn.Module):
    """Relational GraphSAGE ranker with attention pooling."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_node_types: int = 16,
        num_edge_types: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()  # type: ignore
        self.hidden_dim = hidden_dim

        # Node encoder.
        self.node_proj = nn.Linear(feature_dim, hidden_dim)
        self.node_type_emb = nn.Embedding(num_node_types, hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder_dropout = nn.Dropout(dropout)

        # Two-layer relational GraphSAGE stack.
        self.gnn_layers = nn.ModuleList(
            [RelationalGraphSAGELayer(hidden_dim, num_edge_types, dropout=dropout) for _ in range(2)]
        )

        # Attention pooling.
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Prediction head.
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoding: GraphEncoding | TensorGraphEncoding) -> RankerOutput:
        device = next(self.parameters()).device
        if isinstance(encoding, TensorGraphEncoding):
            if encoding.node_features.numel() == 0:
                raise ValueError("Encoding has no node features.")
            node_feats = encoding.node_features.to(device)
            type_ids = encoding.node_types.to(device)
            edge_index = encoding.edge_index.to(device)
            edge_types = encoding.edge_types.to(device)
        else:
            if not encoding.node_features:
                raise ValueError("Encoding has no node features.")
            node_feats = _to_tensor(encoding.node_features, dtype=torch.float32, device=device)
            type_ids = _to_tensor(encoding.node_types, dtype=torch.long, device=device)
            edge_index = (
                _to_tensor(encoding.edge_index, dtype=torch.long, device=device)
                if encoding.edge_index
                else torch.empty((0, 2), dtype=torch.long, device=device)
            )
            edge_types = (
                _to_tensor(encoding.edge_types, dtype=torch.long, device=device)
                if encoding.edge_types
                else torch.empty((0,), dtype=torch.long, device=device)
            )
        if edge_types.numel() > 0 and int(torch.max(edge_types)) >= len(self.gnn_layers[0].rel_linears):  # type: ignore
            raise ValueError("Edge type ID exceeds configured capacity for the GNN.")

        # Node encoder: feature projection + type embedding.
        h = self.node_proj(node_feats) + self.node_type_emb(type_ids)
        h = self.encoder_norm(h)
        h = F.relu(h)
        h = self.encoder_dropout(h)

        # Relational GraphSAGE layers with LayerNorm inside each layer.
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_types)

        # Attention pooling.
        attn_logits = self.attn_mlp(h).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=0)
        graph_repr = (attn_weights.unsqueeze(-1) * h).sum(dim=0)

        # Final scalar score.
        score = self.graph_mlp(graph_repr).squeeze(-1)
        return RankerOutput(score=score, attribution=attn_weights)

    def score_pair(self, better: GraphEncoding | TensorGraphEncoding, worse: GraphEncoding | TensorGraphEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        out_better = self.forward(better)
        out_worse = self.forward(worse)
        return out_better.score, out_worse.score

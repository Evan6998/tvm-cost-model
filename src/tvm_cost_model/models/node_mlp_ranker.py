"""Lightweight ranking model over encoded graphs using PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tvm_cost_model.features.graph_encoder import GraphEncoding


def _to_tensor(values: Any, dtype: torch.dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


@dataclass
class RankerOutput:
    score: torch.Tensor
    attribution: torch.Tensor  # shape: [num_nodes]


class NodeMLPRanker(nn.Module):
    """Scores graphs by aggregating node features; offers per-node attribution.

    This version uses a small per-node MLP, layer normalization, and an
    attention-style pooling mechanism to obtain a graph-level representation.
    The per-node attribution is the attention weight assigned to each node.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_node_types: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()  # type: ignore

        # Project raw node features to hidden_dim; node-type embedding is added.
        self.node_feat_proj = nn.Linear(feature_dim, hidden_dim)
        self.node_type_emb = nn.Embedding(num_node_types, hidden_dim)

        # Normalization + nonlinearity on node representations.
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Attention-style pooling to get a single graph embedding.
        self.attn_vector = nn.Linear(hidden_dim, 1)

        # Final MLP that maps graph embedding to a scalar score.
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoding: GraphEncoding) -> RankerOutput:
        if not encoding.node_features:
            raise ValueError("Encoding has no node features.")

        # Shape: [num_nodes, feature_dim]
        node_feats = _to_tensor(encoding.node_features)
        # Shape: [num_nodes]
        type_ids = _to_tensor(encoding.node_types, dtype=torch.long)

        # Initial node representation: projected features + type embedding.
        h_feat = self.node_feat_proj(node_feats)
        h_type = self.node_type_emb(type_ids)
        h = h_feat + h_type

        # Normalize & refine node representation.
        h = self.node_norm(h)
        h = self.node_mlp(h)

        # Attention pooling over nodes.
        # attn_logits: [num_nodes]
        attn_logits = self.attn_vector(h).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=0)

        # Graph representation is the attention-weighted sum of node embeddings.
        graph_repr = (attn_weights.unsqueeze(-1) * h).sum(dim=0)

        # Final scalar score.
        score = self.graph_mlp(graph_repr).squeeze(-1)

        # Use the attention weights as per-node attribution.
        attribution = attn_weights
        return RankerOutput(score=score, attribution=attribution)

    def score_pair(self, better: GraphEncoding, worse: GraphEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return scores for a pair (better, worse)."""

        out_better = self.forward(better)
        out_worse = self.forward(worse)
        return out_better.score, out_worse.score

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

    This is a stepping stone toward a full R-GAT. It embeds node types, applies a
    small MLP to node features, aggregates via mean, and produces a scalar score.
    Node attributions are derived from a softmax over node-level logits.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, num_node_types: int = 16) -> None:
        super().__init__() # type: ignore
        self.node_type_emb = nn.Embedding(num_node_types, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoding: GraphEncoding) -> RankerOutput:
        if not encoding.node_features:
            zero = torch.zeros(1)
            return RankerOutput(score=zero, attribution=zero)

        node_feats = _to_tensor(encoding.node_features)
        type_ids = _to_tensor(encoding.node_types, dtype=torch.long)
        type_embs = self.node_type_emb(type_ids)
        mlp_input = torch.cat([node_feats, type_embs], dim=-1)

        node_logits = self.mlp(mlp_input).squeeze(-1)
        score = node_logits.mean()
        attribution = F.softmax(node_logits, dim=0)
        return RankerOutput(score=score, attribution=attribution)

    def score_pair(self, better: GraphEncoding, worse: GraphEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return scores for a pair (better, worse)."""

        out_better = self.forward(better)
        out_worse = self.forward(worse)
        return out_better.score, out_worse.score

"""Minimal toy training loop to sanity check GraphCostModel on trivial data."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tvm_cost_model.features.graph_encoder import GraphEncoding
from tvm_cost_model.models.graph_cost_model import GraphCostModel
from tvm_cost_model.training.ranking_dataset import EncodedPair


def _make_graph(value: float) -> GraphEncoding:
    return GraphEncoding(
        node_features=[[float(value)]],
        node_types=[0],
        edge_index=[],
        edge_types=[],
        feature_names=["x"],
    )


def build_pairs(repeats: int = 64) -> list[EncodedPair]:
    g1 = _make_graph(0.0)
    g2 = _make_graph(1.0)
    g3 = _make_graph(2.0)

    base_pairs = [
        EncodedPair(better=g3, worse=g1, difficulty="easy"),
        EncodedPair(better=g2, worse=g1, difficulty="easy"),
        EncodedPair(better=g3, worse=g2, difficulty="easy"),
    ]
    return base_pairs * repeats


def main() -> None:
    torch.manual_seed(0)

    pairs = build_pairs()
    model = GraphCostModel(learning_rate=1e-2, margin=0.1, hidden_dim=16)
    model._ensure_model(feature_dim=1)

    # Quick gradient sanity check on the first pair.
    assert model._model is not None
    assert model._optimizer is not None
    first_pair = pairs[0]
    better_score, worse_score = model._model.score_pair(first_pair.better, first_pair.worse)
    target = torch.ones_like(better_score)
    loss = model._margin_loss(
        better_score.unsqueeze(0),
        worse_score.unsqueeze(0),
        target.unsqueeze(0),
    )
    model._optimizer.zero_grad()
    loss.backward()
    grad_norm = next(model._model.parameters()).grad.norm().item()
    print(f"[sanity] initial loss={loss.item():.6f} grad_norm={grad_norm:.6f}")
    model._optimizer.zero_grad()

    avg_loss = model.train_on_pairs(pairs, epochs=50, batch_size=8, val_split=0.0, show_progress=True)
    print(f"Finished toy training. avg_loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()

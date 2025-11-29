"""Graph-based cost model with a lightweight training loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from tvm_cost_model.features.graph_builder import ProgramGraph
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.models.node_mlp_ranker import NodeMLPRanker, RankerOutput
from tvm_cost_model.training.ranking_dataset import EncodedPair


@dataclass
class Prediction:
    score: float
    attribution: dict[str, float]


class GraphCostModel:
    """Learned cost model that supports pointwise and pairwise training."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        margin: float = 0.1,
        weight_decay: float = 1e-4,
        hidden_dim: int = 64,
    ) -> None:
        self._is_trained = False
        self.encoder = GraphEncoder()
        self.learning_rate = learning_rate
        self.margin = margin
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self._model: NodeMLPRanker | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._feature_dim: int | None = None
        self._node_type_capacity: int = 0
        self._margin_loss = nn.MarginRankingLoss(margin=margin)

    def predict(self, graph: ProgramGraph) -> Prediction:
        """Encode a graph and run it through the ranker."""

        encoding = self.encoder.encode(graph)
        self._ensure_model(len(encoding.feature_names))
        assert self._model is not None
        output: RankerOutput = self._model(encoding)
        attribution = {node.name: float(weight.detach().item()) for node, weight in zip(graph.nodes, output.attribution)}
        return Prediction(score=float(output.score.detach().item()), attribution=attribution)

    def update(self, graphs: Sequence[ProgramGraph], scores: Sequence[float]) -> None:
        """Pointwise regression update to keep compatibility with simple callers."""

        if not graphs or not scores:
            return
        if len(graphs) != len(scores):
            raise ValueError("graphs and scores must have the same length")
        self.encoder.prime_feature_names(graphs)
        encodings = [self.encoder.encode(graph) for graph in graphs]
        feature_dim = len(encodings[0].feature_names)
        self._ensure_model(feature_dim)
        assert self._model is not None
        assert self._optimizer is not None

        preds = torch.stack([self._model(enc).score for enc in encodings])
        target = torch.tensor(list(scores), dtype=torch.float32)
        loss = torch.mean((preds - target) ** 2)
        self._optimizer.zero_grad()
        loss.backward()  # type: ignore
        self._optimizer.step()
        self._is_trained = True

    def train_on_pairs(
        self,
        pairs: Sequence[EncodedPair],
        epochs: int,
        batch_size: int = 32,
        val_split: float = 0.2,
        show_progress: bool = True,
    ) -> float:
        """Train on encoded ranking pairs using a margin ranking loss.

        Prints per-epoch train/validation loss and accuracy (better score > worse score).
        """

        if not pairs:
            return 0.0
        feature_dim = len(pairs[0].better.feature_names)
        self._ensure_model(feature_dim)
        assert self._model is not None
        assert self._optimizer is not None

        pairs_list = list(pairs)
        random.shuffle(pairs_list)
        split_idx = max(1, int(len(pairs_list) * (1 - val_split)))
        train_pairs = pairs_list[:split_idx]
        val_pairs = pairs_list[split_idx:] if split_idx < len(pairs_list) else []

        avg_loss = 0.0
        steps = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            train_loss = 0.0
            train_correct = 0
            train_count = 0
            for start in range(0, len(train_pairs), batch_size):
                batch = train_pairs[start : start + batch_size]
                batch_losses: list[torch.Tensor] = []
                for pair in batch:
                    better_score, worse_score = self._model.score_pair(pair.better, pair.worse)
                    target = torch.ones_like(better_score)
                    batch_losses.append(
                        self._margin_loss(
                            better_score.unsqueeze(0),
                            worse_score.unsqueeze(0),
                            target.unsqueeze(0),
                        )
                    )
                    train_correct += int((better_score > worse_score).item())
                    train_count += 1
                loss = torch.stack(batch_losses).mean()
                self._optimizer.zero_grad()
                loss.backward()  # type: ignore
                self._optimizer.step()
                avg_loss += float(loss.detach())
                steps += 1
                train_loss += float(loss.detach())

            val_loss, val_correct, val_count = self._evaluate_pairs(val_pairs)
            train_acc = (train_correct / train_count) if train_count else 0.0
            val_acc = (val_correct / val_count) if val_count else 0.0
            if show_progress:
                print(
                    f"Epoch {epoch + 1}/{epochs} | train_count={train_count} | "
                    f"train_loss={train_loss / max(train_count, 1):.4f} train_acc={train_acc:.3f} | "
                    f"val_loss={(val_loss / max(val_count, 1)):.4f} val_acc={val_acc:.3f}"
                )

        self._is_trained = True
        return avg_loss / max(steps, 1)

    def _ensure_model(self, feature_dim: int) -> None:
        """(Re)initialize the underlying ranker + optimizer if needed."""

        num_node_types = max(self.encoder.node_type_to_id.values(), default=-1) + 1
        capacity = max(num_node_types, 8)
        if (
            self._model is None
            or self._feature_dim != feature_dim
            or capacity > self._node_type_capacity
        ):
            self._model = NodeMLPRanker(
                feature_dim=feature_dim,
                hidden_dim=self.hidden_dim,
                num_node_types=capacity,
            )
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            self._feature_dim = feature_dim
            self._node_type_capacity = capacity

    def _evaluate_pairs(self, pairs: Sequence[EncodedPair]) -> tuple[float, int, int]:
        if not pairs or self._model is None:
            return 0.0, 0, 0
        total_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for pair in pairs:
                better_score, worse_score = self._model.score_pair(pair.better, pair.worse)
                target = torch.ones_like(better_score)
                loss = self._margin_loss(
                    better_score.unsqueeze(0),
                    worse_score.unsqueeze(0),
                    target.unsqueeze(0),
                )
                total_loss += float(loss.detach())
                correct += int((better_score > worse_score).item())
                count += 1
        return total_loss, correct, count

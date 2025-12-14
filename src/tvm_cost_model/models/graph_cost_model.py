"""Graph-based cost model with a lightweight training loop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

import torch
import torch.nn as nn

from tvm_cost_model.features.graph_builder import ProgramGraph
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.models.graph_gnn_ranker import GraphGNNRanker, RankerOutput
from tvm_cost_model.training.ranking_dataset import EncodedPair
from tvm_cost_model.training.losses import (
    ListNetPairwiseLoss,
    AdaptiveMarginRankingLoss,
    compute_hard_pair_weight,
)


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
        device: torch.device | str | None = None,
        use_listnet: bool = True,  # NEW: Use ListNet loss
        use_adaptive_margin: bool = True,  # NEW: Use adaptive margins
        hard_pair_reweight: bool = True,  # NEW: Reweight hard pairs
    ) -> None:
        self._is_trained = False
        self.encoder = GraphEncoder()
        self.learning_rate = learning_rate
        self.margin = margin
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.use_listnet = use_listnet
        self.use_adaptive_margin = use_adaptive_margin
        self.hard_pair_reweight = hard_pair_reweight
        self.device = torch.device(device) if device is not None else _select_device()
        self._model: GraphGNNRanker | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._feature_dim: int | None = None
        self._node_type_capacity: int = 0
        self._edge_type_capacity: int = 0
        
        # Loss functions
        if use_listnet:
            self._listnet_loss = ListNetPairwiseLoss(temperature=0.5)
        if use_adaptive_margin:
            self._adaptive_loss = AdaptiveMarginRankingLoss(base_margin=0.05, margin_scale=0.5)
        
        # Always keep a basic margin loss for evaluation
        self._margin_loss = nn.MarginRankingLoss(margin=margin)

    def predict(self, graph: ProgramGraph) -> Prediction:
        """Encode a graph and run it through the ranker."""

        encoding = self.encoder.encode(graph)
        self._ensure_model(len(encoding.feature_names))
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
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
        tensor_encodings = [self.encoder.to_tensor_encoding(enc, device=self.device) for enc in encodings]
        feature_dim = len(encodings[0].feature_names)
        self._ensure_model(feature_dim)
        assert self._model is not None
        assert self._optimizer is not None

        preds = torch.stack([self._model(enc).score for enc in tensor_encodings])
        target = torch.tensor(list(scores), dtype=torch.float32, device=preds.device)
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
        stage_name: str | None = None,
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
        self._model.train()

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
                
                # Use ListNet loss over batches
                if self.use_listnet and len(batch) > 1:
                    better_scores = []
                    worse_scores = []
                    better_runtimes = []
                    worse_runtimes = []
                    
                    for pair in batch:
                        better_score, worse_score = self._model.score_pair(pair.better, pair.worse)
                        better_scores.append(better_score)
                        worse_scores.append(worse_score)
                        better_runtimes.append(pair.better_runtime)
                        worse_runtimes.append(pair.worse_runtime)
                        train_correct += int((better_score > worse_score).item())
                        train_count += 1
                    
                    loss = self._listnet_loss(
                        torch.stack(better_scores),
                        torch.stack(worse_scores),
                        torch.tensor(better_runtimes, device=self.device),
                        torch.tensor(worse_runtimes, device=self.device),
                    )
                else:
                    # Fallback to pairwise loss for small batches
                    batch_losses: list[torch.Tensor] = []
                    for pair in batch:
                        better_score, worse_score = self._model.score_pair(pair.better, pair.worse)
                        
                        # Apply hard pair reweighting
                        weight = 1.0
                        if self.hard_pair_reweight:
                            weight = compute_hard_pair_weight(
                                pair.better_runtime,
                                pair.worse_runtime,
                                hard_threshold=0.15
                            )
                        
                        # Use adaptive margin or fixed margin
                        if self.use_adaptive_margin:
                            pair_loss = self._adaptive_loss(
                                better_score, worse_score,
                                pair.better_runtime, pair.worse_runtime
                            )
                        else:
                            target = torch.ones_like(better_score)
                            pair_loss = self._margin_loss(
                                better_score.unsqueeze(0),
                                worse_score.unsqueeze(0),
                                target.unsqueeze(0),
                            )
                        
                        batch_losses.append(pair_loss * weight)
                        train_correct += int((better_score > worse_score).item())
                        train_count += 1
                    
                    loss = torch.stack(batch_losses).mean()
                
                # Optimizer step (outside if/else)
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
                prefix = f"[{stage_name}] " if stage_name else ""
                print(
                    f"{prefix}Epoch {epoch + 1}/{epochs} | train_count={train_count} | "
                    f"train_loss={train_loss / max(train_count, 1):.4f} train_acc={train_acc:.3f} | "
                    f"val_loss={(val_loss / max(val_count, 1)):.4f} val_acc={val_acc:.3f}"
                )

        self._is_trained = True
        return avg_loss / max(steps, 1)

    def save(self, path: str | Path) -> None:
        """Persist the ranker weights and encoder state to ``path``.

        Saves model/optimizer state along with encoder vocabularies and
        feature dimensions so the model can be restored for inference or
        continued training.
        """

        if self._model is None or self._feature_dim is None:
            raise RuntimeError("Cannot save an uninitialized model. Train or run predict first.")

        payload: dict[str, Any] = {
            "state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "encoder": {
                "node_type_to_id": self.encoder.node_type_to_id,
                "edge_type_to_id": self.encoder.edge_type_to_id,
                "feature_names": self.encoder.feature_names,
            },
            "config": {
                "learning_rate": self.learning_rate,
                "margin": self.margin,
                "weight_decay": self.weight_decay,
                "hidden_dim": self.hidden_dim,
                "feature_dim": self._feature_dim,
                "node_type_capacity": self._node_type_capacity,
                "edge_type_capacity": self._edge_type_capacity,
                "is_trained": self._is_trained,
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model/optimizer/encoder payload."""

        payload = torch.load(path, map_location=self.device)
        encoder_state = payload.get("encoder", {})
        self.encoder.node_type_to_id = dict(encoder_state.get("node_type_to_id", {}))
        self.encoder.edge_type_to_id = dict(encoder_state.get("edge_type_to_id", {}))
        self.encoder.feature_names = list(encoder_state.get("feature_names", []))

        config = payload.get("config", {})
        self.learning_rate = float(config.get("learning_rate", self.learning_rate))
        self.margin = float(config.get("margin", self.margin))
        self.weight_decay = float(config.get("weight_decay", self.weight_decay))
        self.hidden_dim = int(config.get("hidden_dim", self.hidden_dim))
        feature_dim = int(config.get("feature_dim", len(self.encoder.feature_names)))
        self._margin_loss = nn.MarginRankingLoss(margin=self.margin)

        self._ensure_model(feature_dim)
        if self._model is not None and "state_dict" in payload:
            self._model.load_state_dict(payload["state_dict"])
            self._model.to(self.device)
        if self._optimizer is not None and payload.get("optimizer_state_dict"):
            self._optimizer.load_state_dict(payload["optimizer_state_dict"])
            _move_optimizer_state(self._optimizer, self.device)

        self._is_trained = bool(config.get("is_trained", False))

    def _ensure_model(self, feature_dim: int) -> None:
        """(Re)initialize the underlying ranker + optimizer if needed."""

        num_node_types = max(self.encoder.node_type_to_id.values(), default=-1) + 1
        num_edge_types = max(self.encoder.edge_type_to_id.values(), default=-1) + 1
        capacity = max(num_node_types, 8)
        edge_capacity = max(num_edge_types, 4)
        if (
            self._model is None
            or self._feature_dim != feature_dim
            or capacity > self._node_type_capacity
            or edge_capacity > self._edge_type_capacity
        ):
            self._model = GraphGNNRanker(
                feature_dim=feature_dim,
                hidden_dim=self.hidden_dim,
                num_node_types=capacity,
                num_edge_types=edge_capacity,
            )
            self._model.to(self.device)
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            self._feature_dim = feature_dim
            self._node_type_capacity = capacity
            self._edge_type_capacity = edge_capacity
        elif next(self._model.parameters()).device != self.device:
            self._model.to(self.device)
            if self._optimizer is not None:
                _move_optimizer_state(self._optimizer, self.device)

    def _evaluate_pairs(self, pairs: Sequence[EncodedPair]) -> tuple[float, int, int]:
        if not pairs or self._model is None:
            return 0.0, 0, 0
        total_loss = 0.0
        correct = 0
        count = 0
        prev_training = self._model.training
        self._model.eval()
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
        if prev_training:
            self._model.train()
        return total_loss, correct, count


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

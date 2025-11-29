"""Helpers to build encoded ranking pairs for training."""

from __future__ import annotations

from dataclasses import dataclass
# from typing import List, Sequence

# from tvm_cost_model.data.dataset_builder import MeasurementRecord
# from tvm_cost_model.features.graph_builder import GraphBuilder
# from tvm_cost_model.features.graph_encoder import GraphEncoder, GraphEncoding
# from tvm_cost_model.training.pair_sampling import make_ranking_pairs

from tvm_cost_model.features.graph_encoder import GraphEncoding

@dataclass
class EncodedPair:
    """Pair of encoded graphs with metadata for ranking losses."""

    better: GraphEncoding
    worse: GraphEncoding
    difficulty: str


# def build_encoded_pairs(
#     measurements: Sequence[MeasurementRecord],
#     builder: GraphBuilder,
#     encoder: GraphEncoder,
#     easy_gap: float = 10.0,
#     hard_gap: float = 2.0,
# ) -> List[EncodedPair]:
#     """Generate encoded pairs ready for model consumption."""

#     encoded_pairs: List[EncodedPair] = []
#     for pair in make_ranking_pairs(measurements, easy_gap=easy_gap, hard_gap=hard_gap):
#         better_graph = builder.build(pair.better.scheduled_tir)
#         worse_graph = builder.build(pair.worse.scheduled_tir)
#         encoded_pairs.append(
#             EncodedPair(
#                 better=encoder.encode(better_graph),
#                 worse=encoder.encode(worse_graph),
#                 difficulty=pair.difficulty,
#             )
#         )
#     return encoded_pairs

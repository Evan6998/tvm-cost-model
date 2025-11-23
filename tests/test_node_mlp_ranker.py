import torch

from tvm_cost_model.features.graph_builder import GraphNode, ProgramGraph
from tvm_cost_model.features.graph_encoder import GraphEncoder
from tvm_cost_model.models.node_mlp_ranker import NodeMLPRanker


def _encode_two_nodes():
    graph = ProgramGraph(
        nodes=[GraphNode(name="loop:i", attrs={"extent": 8}), GraphNode(name="compute", attrs={"loop_depth": 1})],
        edges=[],
    )
    encoding = GraphEncoder().encode(graph)
    return encoding


def test_node_mlp_ranker_produces_score_and_attribution():
    encoding = _encode_two_nodes()
    model = NodeMLPRanker(feature_dim=len(encoding.feature_names), hidden_dim=8, num_node_types=8)
    output = model(encoding)

    assert isinstance(output.score, torch.Tensor)
    assert output.score.shape == ()
    assert output.attribution.shape[0] == len(encoding.node_features)
    # Attribution should sum to 1
    assert torch.isclose(output.attribution.sum(), torch.tensor(1.0), atol=1e-5)


def test_node_mlp_ranker_scores_pairs():
    encoding = _encode_two_nodes()
    model = NodeMLPRanker(feature_dim=len(encoding.feature_names), hidden_dim=8, num_node_types=8)
    score_better, score_worse = model.score_pair(encoding, encoding)
    assert score_better.shape == ()
    assert score_worse.shape == ()

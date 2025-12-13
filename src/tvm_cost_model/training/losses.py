"""Advanced loss functions for ranking schedule performance."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetLoss(nn.Module):
    """
    ListNet loss for listwise ranking.
    
    Learns probability distributions over rankings rather than pairwise comparisons.
    More effective for schedules with similar performance (hard pairs).
    
    Reference: "Learning to Rank: From Pairwise Approach to Listwise Approach"
               (Cao et al., ICML 2007)
    
    Args:
        temperature: Controls softness of probability distribution (lower = sharper)
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        predictions: torch.Tensor,  # [batch_size] predicted scores
        target_runtimes: torch.Tensor,  # [batch_size] ground truth runtimes
    ) -> torch.Tensor:
        """
        Compute cross-entropy between predicted ranking distribution and ground truth.
        
        Lower runtime = better schedule, so we negate runtimes when computing probabilities.
        """
        # Convert predictions and runtimes to probability distributions
        # Negative sign: lower runtime = higher probability
        pred_probs = F.softmax(predictions / self.temperature, dim=0)
        true_probs = F.softmax(-target_runtimes / self.temperature, dim=0)
        
        # Cross-entropy between distributions
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))
        
        return loss


class AdaptiveMarginRankingLoss(nn.Module):
    """
    Margin ranking loss with adaptive margins based on runtime differences.
    
    Schedules with 2x runtime difference need larger margins than schedules
    with 1.01x difference. Adaptive margins improve learning on hard pairs.
    
    Reference: "Learning to Rank using Adaptive Margins" (SIGIR 2019)
    
    Args:
        base_margin: Minimum margin for very similar schedules
        margin_scale: How much to scale margin based on performance gap
    """
    
    def __init__(self, base_margin: float = 0.05, margin_scale: float = 0.5):
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
    
    def forward(
        self,
        better_score: torch.Tensor,  # Predicted score for better (faster) schedule
        worse_score: torch.Tensor,   # Predicted score for worse (slower) schedule
        better_runtime: float,       # Ground truth runtime for better schedule
        worse_runtime: float,        # Ground truth runtime for worse schedule
    ) -> torch.Tensor:
        """
        Compute margin ranking loss with margin adapted to runtime gap.
        
        Larger runtime gaps → larger margins → model must separate them more.
        """
        # Compute relative performance gap
        # Example: 10ms vs 20ms → relative_gap = 1.0 (100% slower)
        #          10ms vs 11ms → relative_gap = 0.1 (10% slower)
        relative_gap = abs(worse_runtime - better_runtime) / min(better_runtime, worse_runtime)
        
        # Adaptive margin: grows logarithmically with performance gap
        # log1p(x) = log(1 + x) ensures margin > 0 even for tiny gaps
        adaptive_margin = self.base_margin + self.margin_scale * torch.log1p(
            torch.tensor(relative_gap, device=better_score.device)
        )
        
        # Standard margin ranking loss: better_score should be > worse_score + margin
        # Loss is 0 if better_score - worse_score >= margin
        # Loss increases linearly otherwise
        target = torch.ones_like(better_score)
        loss = F.margin_ranking_loss(
            better_score.unsqueeze(0),
            worse_score.unsqueeze(0),
            target.unsqueeze(0),
            margin=float(adaptive_margin),
        )
        
        return loss


class ListNetPairwiseLoss(nn.Module):
    """
    Hybrid loss: ListNet over batches of pairs.
    
    Combines benefits of listwise ranking with pairwise training.
    For each batch, treat all "better" schedules as one ranked list.
    
    Args:
        temperature: Softmax temperature
        batch_aggregation: How to aggregate over batches ('mean' or 'sum')
    """
    
    def __init__(self, temperature: float = 0.5, batch_aggregation: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.batch_aggregation = batch_aggregation
    
    def forward(
        self,
        better_scores: torch.Tensor,     # [batch_size] scores for better schedules
        worse_scores: torch.Tensor,      # [batch_size] scores for worse schedules  
        better_runtimes: torch.Tensor,   # [batch_size] ground truth runtimes (better)
        worse_runtimes: torch.Tensor,    # [batch_size] ground truth runtimes (worse)
    ) -> torch.Tensor:
        """
        Compute listwise ranking loss over a batch of pairs.
        """
        # Stack all scores and runtimes
        all_scores = torch.cat([better_scores, worse_scores])
        all_runtimes = torch.cat([better_runtimes, worse_runtimes])
        
        # ListNet loss over the combined batch
        pred_probs = F.softmax(all_scores / self.temperature, dim=0)
        true_probs = F.softmax(-all_runtimes / self.temperature, dim=0)
        
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10))
        
        if self.batch_aggregation == 'mean':
            loss = loss / len(all_scores)
        
        return loss


def compute_hard_pair_weight(
    better_runtime: float,
    worse_runtime: float,
    hard_threshold: float = 0.15,
) -> float:
    """
    Compute sample weight for hard pairs (similar runtimes).
    
    Gives higher weight to pairs where schedules are close in performance,
    forcing the model to learn fine-grained distinctions.
    
    Args:
        better_runtime: Runtime of better schedule (ms)
        worse_runtime: Runtime of worse schedule (ms)
        hard_threshold: Relative gap threshold for "hard" pairs (default 15%)
    
    Returns:
        Weight multiplier (1.0 for easy pairs, up to 3.0 for hard pairs)
    """
    relative_gap = abs(worse_runtime - better_runtime) / min(better_runtime, worse_runtime)
    
    if relative_gap < hard_threshold:
        # Hard pair: weight inversely proportional to gap
        # Smaller gap → higher weight
        # Gap of 0.01 (1%) → weight ~ 3.0
        # Gap of 0.15 (15%) → weight ~ 1.0
        weight = 1.0 + (hard_threshold - relative_gap) / hard_threshold * 2.0
    else:
        # Easy pair: standard weight
        weight = 1.0
    
    return weight




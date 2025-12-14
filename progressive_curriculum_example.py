"""Example: Progressive curriculum mixing instead of strict stages."""

def sample_progressive_curriculum_pairs(
    measurements,
    num_pairs: int,
    epoch: int,
    total_epochs: int,
    easy_frac: float = 0.3,
    hard_frac: float = 0.1,
    seed: int = 0,
):
    """
    Progressive curriculum: gradually shift from easy to hard pairs.
    
    Instead of discrete stages, smoothly transition difficulty distribution:
    - Early epochs: 70% easy, 20% medium, 10% hard
    - Mid epochs:   40% easy, 40% medium, 20% hard  
    - Late epochs:  20% easy, 30% medium, 50% hard
    """
    import random
    from tvm_cost_model.training.pair_sampling import sample_ranking_pairs, Difficulty
    
    # Compute progression factor (0 → 1)
    alpha = min(1.0, epoch / total_epochs)
    
    # Compute target proportions (smooth transition)
    easy_proportion = 0.7 - 0.5 * alpha    # 0.7 → 0.2
    medium_proportion = 0.2 + 0.1 * alpha  # 0.2 → 0.3
    hard_proportion = 0.1 + 0.4 * alpha    # 0.1 → 0.5
    
    # Sample pairs for each difficulty
    all_pairs = []
    for difficulty, proportion in [
        (Difficulty.EASY, easy_proportion),
        (Difficulty.MEDIUM, medium_proportion),
        (Difficulty.HARD, hard_proportion),
    ]:
        n_pairs = int(num_pairs * proportion)
        if n_pairs > 0:
            pairs = sample_ranking_pairs(
                measurements,
                num_pairs=n_pairs,
                easy_frac=easy_frac,
                hard_frac=hard_frac,
                seed=seed + epoch,
                allowed_difficulties={difficulty},
            )
            all_pairs.extend(pairs)
    
    # Shuffle mixed pairs
    random.Random(seed + epoch).shuffle(all_pairs)
    return all_pairs


# Usage in training loop:
# for epoch in range(total_epochs):
#     pairs = sample_progressive_curriculum_pairs(
#         measurements, num_pairs=10000, epoch=epoch, total_epochs=200
#     )
#     # Train on pairs...


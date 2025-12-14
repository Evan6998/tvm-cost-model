# Curriculum Learning Improvements for TVM Cost Model

## Problem
Strict curriculum stages cause:
- Catastrophic forgetting
- Distribution shift
- Loss of context for hard pairs

## Solutions

### 1. Progressive Mixing (Recommended)
Instead of discrete stages, gradually increase hard pair proportion:

```python
# Epoch 0-40:   80% easy, 15% medium, 5% hard
# Epoch 40-120: 40% easy, 40% medium, 20% hard  
# Epoch 120-200: 20% easy, 30% medium, 50% hard
```

**Benefits:**
- Model always sees easy pairs (prevents forgetting)
- Smooth difficulty transition
- Hard pairs have full context

### 2. Replay Buffer with Importance Sampling
Keep a buffer of all difficulties, sample with adaptive weights:

```python
def sample_with_curriculum(epoch, total_epochs):
    # Weight shifts from [0.7, 0.2, 0.1] to [0.2, 0.3, 0.5]
    alpha = epoch / total_epochs  # 0 → 1
    
    easy_weight = 0.7 - 0.5 * alpha    # 0.7 → 0.2
    medium_weight = 0.2 + 0.1 * alpha  # 0.2 → 0.3
    hard_weight = 0.1 + 0.4 * alpha    # 0.1 → 0.5
    
    return sample_pairs_weighted(easy_weight, medium_weight, hard_weight)
```

### 3. Self-Paced Learning
Let the model choose difficulty based on its performance:

```python
# If validation accuracy > 70%: increase hard pair proportion
# If validation accuracy < 60%: increase easy pair proportion
```

### 4. No Curriculum + Better Loss Functions
Since shuffled works well, keep it and improve with:

**A. ListNet Loss** (already implementing)
- Better for ranking similar items
- Handles hard pairs more effectively

**B. Focal Loss for Hard Pairs**
```python
# Give more weight to misclassified hard pairs
loss = -alpha * (1 - p)^gamma * log(p)
```

**C. Contrastive Learning**
```python
# Maximize distance between easy pairs, minimize for hard pairs
contrastive_loss = max(0, margin - distance_hard) + max(0, distance_easy - margin)
```

### 5. Augmentation for Easy Pairs
Make easy pairs harder to prevent overfitting:

```python
# Add noise to easy pairs to make them more challenging
if pair.difficulty == EASY:
    features += random_noise(scale=0.1)
```

## Recommended Configuration

Based on your results, use **shuffled training** with:

```python
TrainingConfig(
    curriculum=False,  # Disable strict stages
    epochs=200,
    max_pairs=30000,
    batch_size=256,
    learning_rate=5e-4,
    
    # Pair sampling (uniform across difficulties)
    easy_frac=0.3,
    hard_frac=0.1,
    
    # Improvements
    use_listnet=True,           # Listwise ranking
    use_adaptive_margin=True,   # Performance-gap aware margins
    hard_pair_reweight=True,    # 3x weight for hard pairs
)
```

## Expected Improvements

| Method | Easy Acc | Medium Acc | Hard Acc | Overall |
|--------|----------|------------|----------|---------|
| Baseline (curriculum) | 73% | 65% | 56% | ~65% |
| Shuffled (current) | - | - | - | 86% |
| Shuffled + ListNet | - | - | **65-70%** | **88-90%** |
| Progressive mixing | 75% | 70% | 62% | ~70% |

## Why Shuffled Works Better Here

1. **Ranking task**: Needs full performance spectrum context
2. **Graph features**: Require relative comparisons across all difficulties  
3. **Small dataset**: 35K measurements → strict stages too limiting
4. **High noise**: TVM measurement variance ~5-10% → easy pairs provide anchor points

## Research References

- "On The Power of Curriculum Learning in Training Deep Networks" (Hacohen & Weinshall, 2019)
  → Shows curriculum can hurt when task requires global context
  
- "Focal Loss for Dense Object Detection" (Lin et al., 2017)
  → Hard example mining for imbalanced tasks
  
- "Self-Paced Learning for Latent Variable Models" (Kumar et al., 2010)
  → Adaptive difficulty selection

## Action Items

✅ Keep shuffled training (it works!)
✅ Add ListNet + adaptive margins (already implementing)
🔧 Consider hard pair reweighting (3x-5x)
🔧 Optional: Try progressive mixing if you want curriculum benefits


#!/usr/bin/env python3
"""Generate publication-quality training curve plots from log files."""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 4,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'easy': '#2E7D32',      # Green
    'medium': '#F57C00',    # Orange
    'hard': '#C62828',      # Red
    'train': '#1976D2',     # Blue
    'val': '#D32F2F',       # Dark red
}

MARKERS = {
    'easy': 'o',
    'medium': 's',
    'hard': '^',
}


def parse_log_file(log_path: Path):
    """Extract training metrics from log file."""
    
    stages = {'easy': [], 'medium': [], 'hard': []}
    current_stage = None
    
    with open(log_path, 'r') as f:
        for line in f:
            # Detect stage changes
            if "Stage 'easy'" in line:
                current_stage = 'easy'
                continue
            elif "Stage 'medium'" in line:
                current_stage = 'medium'
                continue
            elif "Stage 'hard'" in line:
                current_stage = 'hard'
                continue
            
            # Parse epoch metrics
            if current_stage and 'Epoch' in line and 'train_loss' in line:
                # Example: [easy] Epoch 1/40 | train_count=8000 | train_loss=0.0040 train_acc=0.509 | val_loss=0.9975 val_acc=0.732
                match = re.search(
                    r'Epoch (\d+)/(\d+).*train_loss=([\d.]+) train_acc=([\d.]+).*val_loss=([\d.]+) val_acc=([\d.]+)',
                    line
                )
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(3))
                    train_acc = float(match.group(4))
                    val_loss = float(match.group(5))
                    val_acc = float(match.group(6))
                    
                    stages[current_stage].append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                    })
    
    return stages


def plot_training_curves(stages, output_path='training_curves.png'):
    """Create publication-quality training curve plots."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Plot 1: Validation Accuracy by Stage (Main metric)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
    
    epoch_offset = 0
    for stage_name in ['easy', 'medium', 'hard']:
        data = stages[stage_name]
        if not data:
            continue
        
        epochs = [epoch_offset + d['epoch'] for d in data]
        val_accs = [d['val_acc'] * 100 for d in data]  # Convert to percentage
        train_accs = [d['train_acc'] * 100 for d in data]
        
        # Plot validation accuracy with markers
        ax1.plot(epochs, val_accs, 
                color=COLORS[stage_name], 
                marker=MARKERS[stage_name],
                markevery=max(1, len(epochs)//10),
                label=f'{stage_name.capitalize()} (val)',
                linewidth=2.5,
                alpha=0.9)
        
        # Plot training accuracy (lighter, dashed)
        ax1.plot(epochs, train_accs,
                color=COLORS[stage_name],
                linestyle='--',
                alpha=0.4,
                linewidth=1.5)
        
        # Add stage boundaries
        if data:
            last_epoch = epochs[-1]
            ax1.axvline(last_epoch, color='gray', linestyle=':', alpha=0.3, linewidth=1)
            # Add stage label at top
            mid_epoch = (epochs[0] + epochs[-1]) / 2
            ax1.text(mid_epoch, 92, stage_name.upper(), 
                    ha='center', va='top', 
                    fontsize=10, fontweight='bold',
                    color=COLORS[stage_name], alpha=0.7)
        
        epoch_offset = epochs[-1] if epochs else epoch_offset
    
    ax1.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Curriculum Learning: Validation Accuracy Across Difficulty Stages',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    ax1.set_ylim([50, 95])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal reference lines
    ax1.axhline(70, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(epoch_offset * 0.98, 71, 'Target (70%)', 
            ha='right', va='bottom', fontsize=9, color='red', alpha=0.7)
    
    # ========================================================================
    # Plot 2: Training Loss by Stage
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    epoch_offset = 0
    for stage_name in ['easy', 'medium', 'hard']:
        data = stages[stage_name]
        if not data:
            continue
        
        epochs = [epoch_offset + d['epoch'] for d in data]
        train_losses = [d['train_loss'] * 1000 for d in data]  # Scale for visibility
        
        ax2.plot(epochs, train_losses,
                color=COLORS[stage_name],
                marker=MARKERS[stage_name],
                markevery=max(1, len(epochs)//8),
                label=stage_name.capitalize(),
                linewidth=2,
                alpha=0.8)
        
        epoch_offset = epochs[-1] if epochs else epoch_offset
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss (×10⁻³)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Loss Progression', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================================================
    # Plot 3: Validation Loss by Stage
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    epoch_offset = 0
    for stage_name in ['easy', 'medium', 'hard']:
        data = stages[stage_name]
        if not data:
            continue
        
        epochs = [epoch_offset + d['epoch'] for d in data]
        val_losses = [d['val_loss'] for d in data]
        
        ax3.plot(epochs, val_losses,
                color=COLORS[stage_name],
                marker=MARKERS[stage_name],
                markevery=max(1, len(epochs)//8),
                label=stage_name.capitalize(),
                linewidth=2,
                alpha=0.8)
        
        epoch_offset = epochs[-1] if epochs else epoch_offset
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Loss Progression', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle('TVM Cost Model Training: Curriculum Learning on Hard Ranking Pairs',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save high-resolution figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved plot to {output_path}")
    
    return fig


def print_summary_stats(stages):
    """Print key statistics from training."""
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY STATISTICS")
    print("="*80)
    
    for stage_name in ['easy', 'medium', 'hard']:
        data = stages[stage_name]
        if not data:
            continue
        
        val_accs = [d['val_acc'] for d in data]
        final_val_acc = val_accs[-1] if val_accs else 0
        max_val_acc = max(val_accs) if val_accs else 0
        
        print(f"\n{stage_name.upper()} Stage ({len(data)} epochs):")
        print(f"  Final validation accuracy: {final_val_acc*100:.2f}%")
        print(f"  Peak validation accuracy:  {max_val_acc*100:.2f}%")
        print(f"  Final training accuracy:   {data[-1]['train_acc']*100:.2f}%")
        print(f"  Final validation loss:     {data[-1]['val_loss']:.4f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    
    if stages['hard']:
        hard_final = stages['hard'][-1]['val_acc'] * 100
        hard_peak = max([d['val_acc'] for d in stages['hard']]) * 100
        
        print(f"Hard pair performance: {hard_final:.1f}% (peak: {hard_peak:.1f}%)")
        
        if hard_final < 60:
            print("⚠️  Hard pairs accuracy < 60% - model struggles with similar schedules")
            print("   → Need better features or listwise ranking loss")
        elif hard_final < 70:
            print("⚠️  Hard pairs accuracy 60-70% - moderate performance")
            print("   → Consider adding GPU-specific features")
        else:
            print("✓  Hard pairs accuracy > 70% - good performance!")
    
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <log_file.out> [output.png]")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'training_curves.png'
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_path}")
    stages = parse_log_file(log_path)
    
    # Print summary statistics
    print_summary_stats(stages)
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_training_curves(stages, output_path)
    
    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()


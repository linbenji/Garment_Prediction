"""
training_curves.py

Plots publication-quality training curves from history.json files.
Supports overlaying multiple runs for comparison.

Usage:
    # Single run
    python training_curves.py --history runs/method2_master/history.json

    # Compare two runs
    python training_curves.py \
        --history runs/method2_master/history.json runs/method3_crossattn/history.json \
        --labels "FiLM (v3)" "FiLM + CrossAttn (v4)"

    # Custom output
    python training_curves.py --history runs/method2_master/history.json --output ~/Desktop/curves.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for headless environments

# ── Color palette ─────────────────────────────────────────────────────────────

COLORS = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']


def load_history(path):
    with open(path) as f:
        return json.load(f)


def plot_metric(ax, histories, labels, metric_key, val_key=None,
                title='', ylabel='', log_scale=False):
    """Plot a single metric across one or more runs."""
    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = [r['epoch'] for r in hist]

        # Training metric
        if metric_key in hist[0]:
            values = [r[metric_key] for r in hist]
            ax.plot(epochs, values, color=COLORS[i % len(COLORS)],
                    alpha=0.3, linewidth=1, label=f'{label} (train)')

        # Validation metric
        vk = val_key or f'val_{metric_key}'
        if vk in hist[0]:
            values = [r[vk] for r in hist]
            ax.plot(epochs, values, color=COLORS[i % len(COLORS)],
                    linewidth=2, label=f'{label} (val)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')


def plot_generalization(ax, histories, labels):
    """Plot generalization gap over epochs."""
    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = []
        gaps_heavy = []
        gaps_body = []

        for r in hist:
            if 'val_mve_heavy_woven' in r and 'val_mve' in r:
                epochs.append(r['epoch'])
                gaps_heavy.append(r['val_mve_heavy_woven'] - r['val_mve'])
            if 'val_mve_unseen_body' in r and 'val_mve' in r:
                gaps_body.append(r['val_mve_unseen_body'] - r['val_mve'])

        if gaps_heavy:
            ax.plot(epochs[:len(gaps_heavy)], gaps_heavy,
                    color=COLORS[i % len(COLORS)], linewidth=2,
                    label=f'{label} (fabric gap)')
        if gaps_body:
            ax.plot(epochs[:len(gaps_body)], gaps_body,
                    color=COLORS[i % len(COLORS)], linewidth=2,
                    linestyle='--', label=f'{label} (body gap)')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=5, color='green', linewidth=0.5, linestyle=':', alpha=0.5, label='Good (<5mm)')
    ax.axhline(y=15, color='red', linewidth=0.5, linestyle=':', alpha=0.5, label='Poor (>15mm)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (mm)')
    ax.set_title('Generalization Gap (unseen - seen)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, nargs='+', required=True,
                        help='Path(s) to history.json files')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each run (default: filenames)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure (default: saves next to first history)')
    args = parser.parse_args()

    # Load histories
    histories = [load_history(p) for p in args.history]
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.history]

    if len(labels) != len(histories):
        labels = [f'Run {i + 1}' for i in range(len(histories))]

    print(f"Loaded {len(histories)} run(s): {labels}")
    for hist, label in zip(histories, labels):
        print(f"  {label}: {len(hist)} epochs")

    # ── Main figure: 2x3 grid ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')

    # Row 1: Core metrics
    plot_metric(axes[0, 0], histories, labels, 'loss',
                title='Total Loss', ylabel='Loss', log_scale=True)
    plot_metric(axes[0, 1], histories, labels, 'mve',
                title='Mean Vertex Error', ylabel='MVE (mm)')
    plot_metric(axes[0, 2], histories, labels, 'strain',
                title='Edge Strain', ylabel='Strain Loss')

    # Row 2: Generalization + component losses
    plot_metric(axes[1, 0], histories, labels, 'drape',
                title='Drape Loss (Position MSE)', ylabel='Drape Loss', log_scale=True)
    plot_metric(axes[1, 1], histories, labels, 'cls',
                title='Classification Loss', ylabel='CE Loss')
    plot_generalization(axes[1, 2], histories, labels)

    plt.tight_layout()

    # Save
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(os.path.dirname(args.history[0]), 'training_curves.png')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'Run':<25} {'Best Val MVE':>12} {'Best Epoch':>12} {'Final LR':>12}")
    print(f"{'-' * 60}")
    for hist, label in zip(histories, labels):
        val_mves = [r.get('val_mve', float('inf')) for r in hist]
        best_idx = np.argmin(val_mves)
        best_mve = val_mves[best_idx]
        best_epoch = hist[best_idx]['epoch']
        print(f"{label:<25} {best_mve:>10.2f}mm {best_epoch:>10d}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
"""
compare_results.py

Aggregates eval JSON summaries from multiple model runs into a single
comparison report:

  1. Side-by-side metrics table (console + CSV)
  2. Bar-chart comparison of all key metrics (one figure, 2 × 4 subplots)
  3. Generalisation condition breakdown per model (if by_gen data exists)
  4. Per-fabric-family MVE breakdown (if by_family data exists)
  5. Best and worst mesh predictions per model (parsed from filenames)
  6. A plain-English explanation of what every file in the meshes folder is

──────────────────────────────────────────────────────────────────────
WHAT IS IN THE MESHES FOLDER?
──────────────────────────────────────────────────────────────────────
When you run any eval script with --save-meshes it writes .obj files to:
    {run_dir}/eval_results_{split}/meshes/

Each .obj is a 3D mesh that can be opened directly in Blender or MeshLab.
Three file types are saved per sample:

  body001_light_knit_medium_pass0_mve12.3mm_pred.obj
  └─ MODEL PREDICTION — what the neural net thinks the garment looks like
     draped on body #1, wearing light-knit fabric at medium size, when
     shown a particular 2D garment photo as input.

  body001_light_knit_medium_pass0_mve12.3mm_gt.obj
  └─ GROUND TRUTH — what the physics simulation says it actually looks
     like. This is the target the model is trying to match.

  body001_light_knit_medium_template.obj   (v4 / v4.5 only, saved once)
  └─ TEMPLATE (T-pose) — the undeformed garment before any draping,
     sitting on a neutral-pose body. This is the starting point; it
     shows you how much displacement work the model had to do.

FILENAME ANATOMY:
  body001    → Body mesh ID. IDs ≥ 23 were UNSEEN during training —
               these are the hardest generalisation cases.
  light_knit → Fabric family predicted / used.
  medium     → Target garment size.
  pass0      → Which of the 3 image-input passes this sample came from.
               The same body+fabric+size is evaluated 3 times with
               different random garment photos (pass 0, 1, 2). Good
               models look similar across all passes (robust to image
               variation). Fragile models vary wildly.
  mve12.3mm  → Mean Vertex Error for this sample in millimetres.
               LOWER = BETTER. This is what this script uses to rank
               predictions. (Note: the eval scripts call this MVE, not
               MSE. MVE is the mean of per-vertex L2 distances in mm.)

WHAT TO LOOK AT:
  Lowest MVE files  → The model's best-case drape predictions. Look for
                      clean, tightly-fitting geometry that matches gt.
  Highest MVE files → Failure modes. Common patterns: wrong hem position,
                      missing sleeve puff, excessive collar gap.
  Compare pass0 / pass1 / pass2 of the same sample → Robustness check.
  Compare pred vs gt vs template → Visualise how much the model moved
                      the cloth and whether the direction was right.
──────────────────────────────────────────────────────────────────────

Usage:
    # List all available model folders (run this first to see what's available)
    python compare_results.py --list

    # Auto-discover and compare ALL runs under the default RUNS_DIR
    python compare_results.py

    # Select specific models by folder name (just the name, not the full path)
    python compare_results.py --runs method1_baseline method3_crossattn method4_patch

    # Full paths also work if you need them
    python compare_results.py --runs /workspace/runs/method1_baseline /workspace/runs/method2_master

    # Compare val split instead of test
    python compare_results.py --runs method3_crossattn method4_patch --split val

    # Change output folder
    python compare_results.py --runs method3_crossattn method4_patch --out crossattn_vs_patch
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

RUNS_DIR      = r"C:\Users\chung\Desktop\Garment_Prediction\results"
DEFAULT_SPLIT = 'test'
DEFAULT_OUT   = 'comparison_results'
TOP_N         = 5   # best and worst meshes reported per model

# Models to compare — use the exact folder names inside RUNS_DIR.
# Leave the list empty to compare every model that has eval results.
MODELS_TO_COMPARE = [
    "method1_baseline",
    "model_v3_master_bend",
    "model_v3_midpoint_collision",
    "model_v4_lora_cls_bend",
    "model_v4_5_lora_patch"
]

# Custom display names for plots and tables. 
# If a folder isn't in this list, the script will just use the folder name.
MODEL_DISPLAY_NAMES = {
    "method1_baseline": "Baseline (ViT + Concat.)",
    "model_v3_master_bend": "v3 (DINO + FiLM) + Normal & Bending Loss",
    "model_v3_midpoint_collision": "v3 + Collision Loss (w/Norm., Bend., Lapl.)",
    "model_v4_lora_cls_bend": "v4 (LoRA + CLS)",
    "model_v4_5_lora_patch": "v4.5 (LoRA + Patch)"
}

# Consistent model colours across all plots
PALETTE = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#17becf',  # cyan
]

# ── Metric definitions ────────────────────────────────────────────────────────

# Each entry: (json_key, short_label, unit_label, lower_is_better)
# Metrics shown in the main 2×4 comparison figure (8 slots)
PLOT_METRICS = [
    ('mve',        'MVE',         'mm', True),
    ('p90',        'P90',         'mm', True),
    ('chamfer',    'Chamfer',     'mm', True),
    ('hausdorff',  'Hausdorff',   'mm', True),
    ('iou',        'IoU',         '',   False),
    ('normals',    'Normal Cons', '',   False),
    ('cls_acc',    'Cls Acc',     '',   False),
    ('avg_strain', 'Avg Strain',  '',   True),
]

# All metrics written to the CSV comparison table
TABLE_METRICS = [
    ('mve',              'MVE (mm)'),
    ('p90',              'P90 (mm)'),
    ('chamfer',          'Chamfer (mm)'),
    ('hausdorff',        'Hausdorff (mm)'),
    ('iou',              'IoU'),
    ('normals',          'Normal Cons'),
    ('cls_acc',          'Cls Acc'),
    ('avg_strain',       'Avg Strain'),
    ('strain',           'Max Strain'),
    ('collision',        'Collision (mm)'),
    ('body_gen_ratio',   'Body Gen ×'),
    ('fabric_gen_ratio', 'Fabric Gen ×'),
    ('mve_std',          'MVE Std'),
    ('n_samples',        'N Samples'),
]

GEN_ORDER = [
    ('seen_body_seen_mat',        'Seen / Seen\n(train dist)'),
    ('seen_body_unseen_mat_val',  'Seen / Unseen\n(val)'),
    ('seen_body_unseen_mat_test', 'Seen / Unseen\n(test)'),
    ('unseen_body_seen_mat_val',  'Unseen / Seen\n(val)'),
    ('unseen_body_seen_mat_test', 'Unseen / Seen\n(test)'),
    ('unseen_body_unseen_mat',    'Unseen / Unseen\n(hardest)'),
]

FABRIC_ORDER = [
    'light_knit', 'medium_knit', 'heavy_knit',
    'light_woven', 'medium_woven', 'heavy_woven',
]


# ── Discovery ─────────────────────────────────────────────────────────────────

def resolve_run_path(name_or_path, runs_dir):
    """
    Accepts either a bare folder name (e.g. 'method1_baseline') or a full path.
    Bare names are looked up under runs_dir.  Full/relative paths are used as-is.
    """
    p = Path(name_or_path)
    if p.is_absolute() or p.exists():
        return p
    # Try as a name under runs_dir
    candidate = Path(runs_dir) / name_or_path
    if candidate.exists():
        return candidate
    # Return as-is so the caller can report the error clearly
    return candidate


def list_available_runs(runs_dir, split):
    """Print all folders under runs_dir that have eval results for the given split."""
    root = Path(runs_dir)
    if not root.exists():
        print(f"ERROR: RUNS_DIR not found: {runs_dir}")
        return
    print(f"\nAvailable model folders in:  {root.resolve()}")
    print(f"Showing results for split:   {split}")
    print(f"{'─'*60}")
    found_any = False
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        json_path = d / f'eval_results_{split}' / 'summary_stats.json'
        has_results  = json_path.exists()
        has_meshes   = (d / f'eval_results_{split}' / 'meshes').exists()
        status_parts = []
        if has_results:
            with open(json_path) as f:
                data = json.load(f)
            mve = data.get('mve', float('nan'))
            n   = data.get('n_samples', '?')
            status_parts.append(f"MVE={mve:.2f} mm  n={n}")
        if has_meshes:
            n_obj = len(list((d / f'eval_results_{split}' / 'meshes').glob('*_pred.obj')))
            status_parts.append(f"meshes={n_obj}")
        marker = '✓' if has_results else '✗'
        status = '  |  '.join(status_parts) if status_parts else 'no eval results yet'
        print(f"  {marker}  {d.name:<40}  {status}")
        if has_results:
            found_any = True
    if not found_any:
        print("  (no eval results found — run an eval script first)")
    print(f"{'─'*60}")
    print("Pass folder names to --runs to select a subset, e.g.:")
    print(f"  python compare_results.py --runs method1_baseline method3_crossattn\n")


def discover_runs(runs_dir, split, explicit_runs=None):
    """
    Returns dict: { run_label → { 'data': {...}, 'mesh_dir': Path } }

    explicit_runs: list of bare folder names OR full paths.
      Bare names are resolved under runs_dir automatically.
      If None, every subdirectory of runs_dir is scanned.
    """
    results = {}

    if explicit_runs:
        candidates = [resolve_run_path(r, runs_dir) for r in explicit_runs]
    else:
        root = Path(runs_dir)
        if not root.exists():
            print(f"ERROR: RUNS_DIR not found: {runs_dir}")
            return results
        candidates = sorted(d for d in root.iterdir() if d.is_dir())

    for run_dir in candidates:
        if not run_dir.exists():
            print(f"  [error] '{run_dir.name}' not found under {runs_dir}")
            continue
        json_path = run_dir / f'eval_results_{split}' / 'summary_stats.json'
        if not json_path.exists():
            print(f"  [skip]  {run_dir.name:<40} — no eval_results_{split}/summary_stats.json")
            continue
        with open(json_path) as f:
            data = json.load(f)
        
        # Get the original folder name
        raw_name = run_dir.name
        
        # Look up the custom name, fallback to the raw folder name if not found
        label = MODEL_DISPLAY_NAMES.get(raw_name, raw_name)
        
        results[label] = {
            'data':     data,
            'run_dir':  run_dir,
            'mesh_dir': run_dir / f'eval_results_{split}' / 'meshes',
        }
        n   = data.get('n_samples', '?')
        mve = data.get('mve', float('nan'))
        print(f"  [ok]    {label:<40}  MVE={mve:.2f} mm  n={n}")

    return results


# ── Table ─────────────────────────────────────────────────────────────────────

def build_dataframe(runs):
    """Build a flat pandas DataFrame, one row per model."""
    rows = []
    for label, info in runs.items():
        d = info['data']
        row = {'Model': label}

        # Core metrics
        for key, col in TABLE_METRICS:
            val = d.get(key)
            row[col] = round(val, 3) if isinstance(val, (int, float)) else '—'

        # Gen gap (absolute mm) — computed from by_gen if present
        by_gen = d.get('by_gen', {})
        seen = by_gen.get('seen_body_seen_mat')
        hard = by_gen.get('unseen_body_unseen_mat')
        if seen is not None and hard is not None:
            row['Gen Gap (mm)'] = round(hard - seen, 2)
        else:
            row['Gen Gap (mm)'] = '—'

        # Baselines
        bl = d.get('baselines', {})
        row['Zero BL (mm)'] = round(bl['zero'], 2) if 'zero' in bl else '—'
        row['Mean BL (mm)'] = round(bl['mean'], 2) if 'mean' in bl else '—'
        mve = d.get('mve')
        row['Beats Zero'] = ('✓' if (mve and 'zero' in bl and mve < bl['zero']) else '✗')
        row['Beats Mean'] = ('✓' if (mve and 'mean' in bl and mve < bl['mean']) else '—')

        rows.append(row)

    return pd.DataFrame(rows).set_index('Model')


def mark_winners(df):
    """Add a 'Best in' column listing metrics where each model wins."""
    winners = defaultdict(list)
    lower_better_cols = {
        'MVE (mm)', 'P90 (mm)', 'Chamfer (mm)', 'Hausdorff (mm)',
        'Avg Strain', 'Max Strain', 'Collision (mm)', 'Gen Gap (mm)',
        'Body Gen ×', 'Fabric Gen ×',
    }
    higher_better_cols = {'IoU', 'Normal Cons', 'Cls Acc'}

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors='coerce')
        if numeric.isna().all():
            continue
        if col in lower_better_cols:
            best_idx = numeric.idxmin()
        elif col in higher_better_cols:
            best_idx = numeric.idxmax()
        else:
            continue
        if best_idx:
            winners[best_idx].append(col.replace(' (mm)', '').replace(' ×', '×'))

    df = df.copy()
    df['★ Best in'] = [', '.join(winners.get(m, [])) for m in df.index]
    return df


def print_table(df):
    # Show only the most important columns for the console view
    priority = [
        'MVE (mm)', 'P90 (mm)', 'Gen Gap (mm)', 'IoU', 'Normal Cons',
        'Cls Acc', 'Collision (mm)', 'Beats Zero', 'Beats Mean', '★ Best in',
    ]
    cols = [c for c in priority if c in df.columns]
    print(f"\n{'='*80}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*80}")
    print(df[cols].to_string())
    print(f"{'='*80}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_metric_comparison(runs, out_dir):
    """
    2 × 4 subplot grid — one bar chart per metric.
    Bars are colour-coded by model. Best bar in each subplot gets a star.
    """
    labels  = list(runs.keys())
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    n_plots = len(PLOT_METRICS)
    ncols   = 4
    nrows   = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for ax_idx, (key, short_label, unit, lib) in enumerate(PLOT_METRICS):
        ax   = axes[ax_idx]
        vals = []
        for label in labels:
            v = runs[label]['data'].get(key)
            vals.append(v if isinstance(v, (int, float)) else np.nan)

        bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6,
                      edgecolor='white', linewidth=0.8)

        # Highlight the winning bar
        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            best_val = min(finite) if lib else max(finite)
            for bar_obj, val in zip(bars, vals):
                if not np.isnan(val) and val == best_val:
                    bar_obj.set_edgecolor('gold')
                    bar_obj.set_linewidth(2.5)
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height(),
                            ' ★', ha='left', va='bottom',
                            fontsize=10, color='goldenrod', fontweight='bold')

        ylabel = f"{short_label} ({unit})" if unit else short_label
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{'↓' if lib else '↑'}  {short_label}", fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    # Hide any unused subplots
    for ax_idx in range(n_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    # Shared legend
    # patches = [mpatches.Patch(color=colors[i], label=labels[i])
    #            for i in range(len(labels))]
    # fig.legend(handles=patches, loc='lower center',
    #            ncol=min(len(labels), 4), fontsize=8,
    #            bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Model Comparison — Key Metrics  (★ = best per metric)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = out_dir / 'comparison_metrics.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_generalization(runs, out_dir):
    """
    Grouped bar chart: x = gen condition, groups = models.
    Only drawn if at least one model has by_gen data in its JSON.
    """
    models_with_gen = {label: info for label, info in runs.items()
                       if info['data'].get('by_gen')}
    if not models_with_gen:
        print("  [skip] Gen-condition plot — no by_gen data found in any summary JSON.")
        print("         (v3/v3.5 eval scripts don't save by_gen; v4/v4.5 do.)")
        return

    labels     = list(models_with_gen.keys())
    colors     = [PALETTE[list(runs.keys()).index(l) % len(PALETTE)] for l in labels]
    n_groups   = len(GEN_ORDER)
    n_models   = len(labels)
    width      = 0.7 / n_models
    x          = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(14, 5))

    for m_idx, (label, info) in enumerate(models_with_gen.items()):
        by_gen = info['data']['by_gen']
        vals   = [by_gen.get(key, np.nan) for key, _ in GEN_ORDER]
        offset = (m_idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=colors[m_idx], label=label,
               edgecolor='white', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([disp for _, disp in GEN_ORDER], fontsize=9)
    ax.set_ylabel('MVE (mm)', fontsize=10)
    ax.set_title('Generalisation Condition Breakdown  (Body × Fabric)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Highlight the hardest bucket with a vertical shading
    ax.axvspan(n_groups - 1 - 0.5, n_groups - 0.5,
               color='red', alpha=0.06, zorder=0)
    ax.text(n_groups - 1, ax.get_ylim()[1] * 0.97,
            'hardest', ha='center', va='top', fontsize=8,
            color='red', style='italic')

    plt.tight_layout()
    path = out_dir / 'comparison_generalization.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_by_family(runs, out_dir):
    """
    Grouped bar chart: x = fabric family, groups = models.
    Only drawn if at least one model has by_family data.
    """
    models_with_fam = {label: info for label, info in runs.items()
                       if info['data'].get('by_family')}
    if not models_with_fam:
        print("  [skip] By-family plot — no by_family data found in any summary JSON.")
        return

    labels   = list(models_with_fam.keys())
    colors   = [PALETTE[list(runs.keys()).index(l) % len(PALETTE)] for l in labels]
    n_fams   = len(FABRIC_ORDER)
    n_models = len(labels)
    width    = 0.7 / n_models
    x        = np.arange(n_fams)

    fig, ax = plt.subplots(figsize=(13, 5))

    for m_idx, (label, info) in enumerate(models_with_fam.items()):
        by_fam = info['data']['by_family']
        vals   = [by_fam.get(fam, np.nan) for fam in FABRIC_ORDER]
        offset = (m_idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=colors[m_idx], label=label,
               edgecolor='white', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in FABRIC_ORDER], fontsize=9)
    ax.set_ylabel('MVE (mm)  ↓ lower = better', fontsize=10)
    ax.set_title('MVE by Fabric Family', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Mark the unseen family (heavy_woven, last bar)
    ax.axvspan(n_fams - 1 - 0.5, n_fams - 0.5,
               color='red', alpha=0.06, zorder=0)
    ax.text(n_fams - 1, ax.get_ylim()[1] * 0.97,
            'UNSEEN', ha='center', va='top', fontsize=8,
            color='red', style='italic')

    plt.tight_layout()
    path = out_dir / 'comparison_by_family.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Mesh Finder ───────────────────────────────────────────────────────────────

def find_best_worst_meshes(runs, top_n=TOP_N):
    """
    Scans each model's meshes/ directory for *_pred.obj files.
    Parses the mve value from the filename and returns sorted lists.
    Returns: { model_label: {'best': [...], 'worst': [...]} }
    """
    MVE_RE = re.compile(r'mve([\d.]+)mm', re.IGNORECASE)
    report = {}

    for label, info in runs.items():
        mesh_dir = info['mesh_dir']
        if not mesh_dir.exists():
            report[label] = None
            continue

        entries = []
        for fpath in mesh_dir.glob('*_pred.obj'):
            m = MVE_RE.search(fpath.name)
            if m:
                entries.append((float(m.group(1)), fpath.name))

        if not entries:
            report[label] = None
            continue

        entries.sort(key=lambda x: x[0])
        report[label] = {
            'best':  entries[:top_n],
            'worst': entries[-top_n:][::-1],
            'total': len(entries),
            'mean_mve': round(np.mean([e[0] for e in entries]), 2),
        }

    return report


def print_mesh_report(mesh_report):
    print(f"\n{'='*80}")
    print(f"BEST & WORST MESH PREDICTIONS  (parsed from *_pred.obj filenames)")
    print(f"Lower MVE = better prediction quality")
    print(f"{'='*80}")
    for label, data in mesh_report.items():
        print(f"\n  {label}")
        if data is None:
            print("    [no meshes folder found — run eval with --save-meshes]")
            continue
        print(f"    {data['total']} _pred.obj files  |  mean MVE across saved samples: {data['mean_mve']} mm")
        print(f"    TOP {len(data['best'])} BEST  (closest to ground truth):")
        for mve, fname in data['best']:
            print(f"      {mve:>7.2f} mm  →  {fname}")
        print(f"    TOP {len(data['worst'])} WORST  (largest error — useful for failure analysis):")
        for mve, fname in data['worst']:
            print(f"      {mve:>7.2f} mm  →  {fname}")
    print(f"{'='*80}\n")


def save_mesh_report(mesh_report, out_dir):
    lines = []
    lines.append("BEST & WORST MESH PREDICTIONS")
    lines.append("=" * 70)
    lines.append("MVE = Mean Vertex Error in millimetres.")
    lines.append("")
    for label, data in mesh_report.items():
        lines.append(f"Model: {label}")
        if data is None:
            lines.append("  No meshes folder found. Re-run eval with --save-meshes.")
            lines.append("")
            continue
        lines.append(f"  Saved samples: {data['total']}  |  Mean MVE: {data['mean_mve']} mm")
        lines.append(f"  BEST {len(data['best'])}:")
        for mve, fname in data['best']:
            lines.append(f"    {mve:>7.2f} mm  {fname}")
        lines.append(f"  WORST {len(data['worst'])}:")
        for mve, fname in data['worst']:
            lines.append(f"    {mve:>7.2f} mm  {fname}")
        lines.append("")
    path = out_dir / 'best_meshes_report.txt'
    path.write_text('\n'.join(lines))
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate model eval results into comparison plots and tables.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_results.py --list
      Show all available model folders and their eval status.

  python compare_results.py
      Compare every model that has results under RUNS_DIR.

  python compare_results.py --runs method1_baseline method3_crossattn method4_patch
      Compare three specific models by folder name.

  python compare_results.py --runs method3_crossattn method4_patch --split val --out v4_vs_v45
      Compare two models on the val split, save to a custom folder.
        """)
    parser.add_argument('--list',     action='store_true',
                        help='List all available model folders and exit.')
    parser.add_argument('--runs',     nargs='+', default=None,
                        metavar='FOLDER',
                        help='Folder names (or full paths) to compare. '
                             'Use bare names like "method1_baseline" — the script '
                             'finds them under --runs-dir automatically. '
                             'Omit to compare all available models.')
    parser.add_argument('--runs-dir', type=str,  default=RUNS_DIR,
                        help='Root directory containing model run folders '
                             f'(default: {RUNS_DIR}).')
    parser.add_argument('--split',    type=str,  default=DEFAULT_SPLIT,
                        choices=['train', 'val', 'test'],
                        help='Which eval split to read results from (default: test).')
    parser.add_argument('--out',      type=str,  default=DEFAULT_OUT,
                        help=f'Output directory for generated files (default: {DEFAULT_OUT}).')
    parser.add_argument('--top-n',    type=int,  default=TOP_N,
                        help=f'Best/worst mesh samples to report per model (default: {TOP_N}).')
    args = parser.parse_args()

    # ── --list: show available models and exit ────────────────────────────────
    if args.list:
        list_available_runs(args.runs_dir, args.split)
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve which models to compare ──────────────────────────────────────
    # Priority: --runs CLI flag > MODELS_TO_COMPARE config list > all models
    selected = args.runs or (MODELS_TO_COMPARE if MODELS_TO_COMPARE else None)

    if selected:
        print(f"\nComparing {len(selected)} model(s) from config / --runs:")
        for name in selected:
            print(f"  → {name}")
    else:
        print(f"\nMODELS_TO_COMPARE is empty — comparing all models under:  {args.runs_dir}")
    print(f"Split: {args.split}\n")
    print(f"Searching for eval_results_{args.split}/summary_stats.json ...")
    runs = discover_runs(args.runs_dir, args.split, selected)
    if not runs:
        print("No results found. Run eval scripts first.")
        sys.exit(1)
    print(f"\nFound {len(runs)} model(s): {list(runs.keys())}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\nBuilding comparison table...")
    df      = build_dataframe(runs)
    df_star = mark_winners(df)
    print_table(df_star)

    csv_path = out_dir / 'comparison_table.csv'
    df_star.to_csv(csv_path)
    print(f"  Saved: {csv_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_metric_comparison(runs, out_dir)
    plot_generalization(runs, out_dir)
    plot_by_family(runs, out_dir)

    # ── Mesh finder ───────────────────────────────────────────────────────────
    print("\nScanning mesh folders...")
    mesh_report = find_best_worst_meshes(runs, top_n=args.top_n)
    print_mesh_report(mesh_report)
    save_mesh_report(mesh_report, out_dir)

    print(f"\nAll outputs written to:  {out_dir.resolve()}")
    print("Files generated:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == '__main__':
    main()
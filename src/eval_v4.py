"""
===============================================================================
EVALUATION PIPELINE: UnfrozenCLSDrapeModel (v4)
===============================================================================

Merges the full metric suite from eval_v3 with the v4 model architecture
(UnfrozenCLSDrapeModel with optional cross-attention layers).

Step-by-Step Execution Flow:
-------------------------------------------------------------------------------
1. Initialization & Loading
   - Loads the saved checkpoint (.pt) and extracts the architecture config.
   - Instantiates UnfrozenCLSDrapeModel (restoring cross_attn_layers from cfg).
   - Initialises GarmentDataset for the requested split (usually 'test').
   - Forces front-facing camera angle (dataloader_v2.CAMERA_ANGLES = ['000']).

2. Baseline Sanity Checks (Optional — --baselines flag)
   - Zero Predictor: error if the model predicted 0 displacement (T-pose).
   - Mean Predictor: "average drape" across training set as a static prediction.
     The model must beat both to prove it has learned conditional physics.

3. The Evaluation Loop — 3 passes with deterministic seeds
   Each pass uses a different seed so the random image-pair augmentation in the
   dataloader picks different pairings; results are averaged across all passes.
   Per-sample metrics computed each pass:
   - Point-to-Point:  MVE (mean vertex error) and P90 vertex error
   - Silhouette:      Chamfer Distance, Hausdorff Distance, Voxel IoU
   - Physical:        Max/Avg Edge Strain (fabric stretch vs template)
   - Surface Fidelity: Normal Consistency (cosine similarity of face normals)
   - Vision:          Fabric Family Classification Accuracy (DINOv2 head)

4. Categorization & Bucketing
   - By Fabric Family
   - By Target Size
   - By Generalization Condition (v4 bucketing — heavy_woven is the unseen family):
       seen_body_seen_mat         | body_id <= 22, fab_group <= 5
       seen_body_unseen_mat_val   | body_id <= 12, fab_group == 6
       seen_body_unseen_mat_test  | body_id <= 22, fab_group == 6
       unseen_body_seen_mat_val   | body_id >= 23, fab_group <= 2
       unseen_body_seen_mat_test  | body_id >= 23, fab_group <= 5
       unseen_body_unseen_mat     | body_id >= 23, fab_group == 6  ← hardest

5. Mesh Export (Optional — --save-meshes flag)
   - Saves predicted, ground-truth, and template .obj files per sample.
   - Template is saved once (pass 0 only); pred/gt saved for all 3 passes.
   - SUBSET_SAVE=True caps output at 30 samples per pass to keep disk usage low.

6. Aggregation & Output
   - threshold_performance.csv  — great/good/acceptable pass-rates per metric
   - cumulative_accuracy_<metric>.png  — individual CDF plot per metric
   - summary_stats.json  — all aggregated metrics, baselines, per-family/size/gen
===============================================================================

Usage:
    # Evaluate best checkpoint on test split
    python eval_v4.py --checkpoint runs/method3_crossattn/checkpoints/best.pt

    # Evaluate on val split
    python eval_v4.py --checkpoint runs/method3_crossattn/checkpoints/best.pt --split val

    # Evaluate and save meshes for visualisation (pred + gt + template .obj files)
    python eval_v4.py --checkpoint runs/method3_crossattn/checkpoints/best.pt --save-meshes

    # Scientific run with baselines (loads full train set — slower)
    python eval_v4.py --checkpoint runs/method3_crossattn/checkpoints/best.pt --baselines
===============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.loader import DataLoader
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pandas as pd
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataloader_v2
# Force the dataloader to ONLY use the front-facing angle
dataloader_v2.CAMERA_ANGLES = ['000']
from dataloader_v2 import GarmentDataset
from models_v4 import UnfrozenCLSDrapeModel

# ── Config ────────────────────────────────────────────────────────────────────

# DATA_ROOT = '/workspace/batch_1500_lean'
# RUNS_DIR  = '/workspace/runs'

DATA_ROOT  = r'C:\Users\chung\Desktop\Garment_Prediction\dataset\batch_1500_lean'

#DATA_ROOT = r"/Users/Ben/Desktop/batch_1500_lean"
#RUNS_DIR  = r"/Users/Ben/Desktop/runs"

FABRIC_FAMILIES = [
    'light_knit', 'medium_knit', 'heavy_knit',
    'light_woven', 'medium_woven', 'heavy_woven',
]
SIZES = ['small', 'medium', 'large', 'xl', 'xxl']

# Family group mapping (mirrors step1_json_to_csv.py)
# NOTE: In v4, heavy_woven (group 6) is the UNSEEN fabric family.
FAMILY_GROUP = {
    'light_knit': 1, 'medium_knit': 2, 'heavy_knit': 3,
    'light_woven': 4, 'medium_woven': 5, 'heavy_woven': 6,
}

FABRIC_FAMILY_IDX = {f: i for i, f in enumerate(FABRIC_FAMILIES)}

# Global flag: True = cap mesh saves at 30 per pass, False = save all 180
SUBSET_SAVE = False

# Define what constitutes Great, Good, and Acceptable for each metric
THRESHOLDS = {
    'mve':       {'great': 5.0,  'good': 10.0, 'acceptable': 20.0},  # mm
    'chamfer':   {'great': 5.0,  'good': 10.0, 'acceptable': 20.0},  # mm
    'hausdorff': {'great': 20.0, 'good': 50.0, 'acceptable': 80.0},  # mm
    'iou':       {'great': 0.90, 'good': 0.80, 'acceptable': 0.70},  # Volume overlap
    'normals':   {'great': 0.95, 'good': 0.90, 'acceptable': 0.80},  # Cosine similarity
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Mesh & Geometry Helpers ───────────────────────────────────────────────────

def save_obj(path, verts, faces=None):
    """Save vertex positions (and optional faces) as a .obj file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("# Exported by eval_v4.py\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def compute_chamfer_hausdorff(pred, gt):
    """
    Symmetric Chamfer distance and one-sided Hausdorff distance.
    Chamfer: average nearest-neighbour distance in both directions.
    Hausdorff: worst-case nearest-neighbour distance — identifies outlier spikes.
    O(N log N) via KD-Trees.
    """
    tree_gt = KDTree(gt)
    dist_p_to_g, _ = tree_gt.query(pred)

    tree_pred = KDTree(pred)
    dist_g_to_p, _ = tree_pred.query(gt)

    chamfer   = (np.mean(dist_p_to_g) + np.mean(dist_g_to_p)) / 2
    hausdorff = max(np.max(dist_p_to_g), np.max(dist_g_to_p))
    return chamfer, hausdorff


def compute_iou(pred, gt, grid_res=32):
    """
    Voxel-based Intersection over Union.
    Measures whether the model correctly captured the garment's volume envelope,
    independent of specific vertex correspondence.
    """
    all_pts = np.concatenate([pred, gt], axis=0)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)

    def get_voxels(pts):
        voxels = ((pts - mins) / (maxs - mins + 1e-9) * (grid_res - 1)).astype(int)
        return set(map(tuple, np.unique(voxels, axis=0)))

    v_pred, v_gt = get_voxels(pred), get_voxels(gt)
    union = len(v_pred.union(v_gt))
    return len(v_pred.intersection(v_gt)) / union if union > 0 else 0


def compute_normals_sim(pred, gt, faces):
    """
    Mean cosine similarity of face normals between predicted and ground-truth mesh.
    High MVE with low Normal Consistency implies the garment is in the right place
    but wrinkles/folds are missing or facing the wrong direction.
    """
    def get_normals(verts):
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

    return np.mean(np.sum(get_normals(pred) * get_normals(gt), axis=1))


# ── Baselines ─────────────────────────────────────────────────────────────────

def compute_zero_baseline(loader, device):
    """
    Zero predictor: always predict zero displacement (garment stays in T-pose).
    The model must beat this to be useful at all.
    """
    errors = []
    for batch in loader:
        batch = batch.to(device)
        pred  = torch.zeros_like(batch.y)
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            errors.append((pred[mask] - batch.y[mask]).norm(dim=1).mean().item())
    return float(np.mean(errors))


def compute_mean_baseline(train_loader, eval_loader, device):
    """
    Mean predictor: predict the per-vertex mean displacement from the training set.
    The model must beat this to prove it has learned conditional physics
    (adapting to a specific image/size) rather than just memorising average shape.
    """
    print("  Computing training mean displacement...")
    sum_disp  = None
    n_samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            disp = batch.y[mask]
            if sum_disp is None:
                sum_disp = disp.clone()
            else:
                sum_disp += disp
            n_samples += 1

    mean_disp = sum_disp / n_samples   # (N_verts, 3)

    errors = []
    for batch in eval_loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            errors.append((mean_disp - batch.y[mask]).norm(dim=1).mean().item())
    return float(np.mean(errors))


# ── Main Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, results_dir,
             save_meshes=False, faces=None, template_verts=None,
             n_save=3, inputs_per_mesh=3):
    model.eval()
    os.makedirs(results_dir, exist_ok=True)
    mesh_dir = os.path.join(results_dir, 'meshes') if save_meshes else None

    # Core metrics tracker
    metrics          = defaultdict(list)

    # Categorisation trackers
    by_family        = defaultdict(list)
    by_size          = defaultdict(list)
    by_gen_condition = defaultdict(list)

    cls_correct   = 0
    total_samples = 0

    print(f"  Evaluating {len(loader.dataset)} samples  ({inputs_per_mesh} passes)...")

    for pass_idx in range(inputs_per_mesh):
        # Each pass uses a different deterministic seed so the random image-pair
        # chosen by the dataloader differs between passes but is fully reproducible.
        pass_seed = 42 + pass_idx
        set_deterministic_seed(pass_seed)
        print(f"    Pass {pass_idx + 1}/{inputs_per_mesh}  (seed={pass_seed})...")

        samples_this_pass = 0

        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_delta, fabric_logits = model(batch)

            # Classification accuracy
            preds_cls    = fabric_logits.argmax(dim=1)
            cls_correct += (preds_cls == batch.fabric_family_label).sum().item()

            # Edge strain — computed on GPU before the per-graph loop
            src, dst     = batch.edge_index
            pred_len     = torch.norm(
                (batch.pos[src] + pred_delta[src]) - (batch.pos[dst] + pred_delta[dst]), dim=1)
            gt_len       = torch.norm(
                (batch.pos[src] + batch.y[src]) - (batch.pos[dst] + batch.y[dst]), dim=1)
            batch_strain = torch.abs(pred_len - gt_len) / (gt_len + 1e-9)

            for i in range(batch.num_graphs):
                mask      = (batch.batch == i)
                edge_mask = (batch.batch[batch.edge_index[0]] == i)

                p_verts = (batch.pos[mask] + pred_delta[mask]).cpu().numpy()
                g_verts = (batch.pos[mask] + batch.y[mask]).cpu().numpy()

                # ── Core Metrics ──────────────────────────────────────────────
                per_vert_err         = np.linalg.norm(p_verts - g_verts, axis=1)
                mve_val              = per_vert_err.mean()
                p90_val              = np.quantile(per_vert_err, 0.90)
                chamfer, hausdorff   = compute_chamfer_hausdorff(p_verts, g_verts)
                iou                  = compute_iou(p_verts, g_verts)
                max_strain           = batch_strain[edge_mask].max().item()
                avg_strain           = batch_strain[edge_mask].mean().item()

                metrics['mve'].append(mve_val)
                metrics['p90'].append(p90_val)
                metrics['chamfer'].append(chamfer)
                metrics['hausdorff'].append(hausdorff)
                metrics['iou'].append(iou)
                metrics['strain'].append(max_strain)
                metrics['avg_strain'].append(avg_strain)

                if faces is not None:
                    metrics['normals'].append(
                        compute_normals_sim(p_verts, g_verts, faces))

                # ── Metadata ──────────────────────────────────────────────────
                body_id   = batch.body_id[i].item()
                fab_label = batch.fabric_family_label[i].item()
                fab_name  = (FABRIC_FAMILIES[fab_label]
                             if fab_label < len(FABRIC_FAMILIES) else 'unknown')
                fab_group = FAMILY_GROUP.get(fab_name, 0)

                size_enc  = batch.tgt_size.view(-1, 2)[i]
                width_pct = round(size_enc[0].item(), 2)
                size_name = {0.92: 'small', 1.0: 'medium', 1.08: 'large',
                             1.17: 'xl', 1.28: 'xxl'}.get(width_pct, f'w={width_pct}')

                by_family[fab_name].append(mve_val)
                by_size[size_name].append(mve_val)

                # ── v4 Generalization Condition Bucketing ─────────────────────
                # heavy_woven (group 6) is the UNSEEN fabric family in v4.
                if body_id <= 22 and fab_group <= 5:
                    gen_cond = 'seen_body_seen_mat'
                elif body_id <= 12 and fab_group == 6:
                    gen_cond = 'seen_body_unseen_mat_val'
                elif body_id <= 22 and fab_group == 6:
                    gen_cond = 'seen_body_unseen_mat_test'
                elif body_id >= 23 and fab_group <= 2:
                    gen_cond = 'unseen_body_seen_mat_val'
                elif body_id >= 23 and fab_group <= 5:
                    gen_cond = 'unseen_body_seen_mat_test'
                else:
                    gen_cond = 'unseen_body_unseen_mat'

                by_gen_condition[gen_cond].append(mve_val)

                metrics['body_id'].append(body_id)
                metrics['fab_label'].append(fab_label)

                # ── Mesh Saving ───────────────────────────────────────────────
                if save_meshes and (not SUBSET_SAVE or samples_this_pass < 30):
                    pos    = batch.pos[mask].cpu().numpy()
                    # 1. Grab the unique sample name from the batch
                    uid = batch.sample_name[i]
                    
                    # 2. Add it to the prefix!
                    prefix = f"{uid}_body{body_id:03d}_{fab_name}_{size_name}_pass{pass_idx}_mve{mve_val:.1f}mm"
                    
                    save_obj(os.path.join(mesh_dir, f"{prefix}_pred.obj"), p_verts, faces)
                    save_obj(os.path.join(mesh_dir, f"{prefix}_gt.obj"),   g_verts, faces)
                    # Template (T-pose) does not change across passes — save once
                    if pass_idx == 0:
                        tpl_prefix = f"body{body_id:03d}_{fab_name}_{size_name}_template"
                        save_obj(os.path.join(mesh_dir, f"{tpl_prefix}.obj"), pos, faces)

                total_samples     += 1
                samples_this_pass += 1

    # ── Post-Processing ───────────────────────────────────────────────────────

    # Mean across all passes
    final_stats = {k: np.mean(v) for k, v in metrics.items()
                   if k not in ['body_id', 'fab_label']}
    final_stats['mve_std']   = float(np.std(metrics['mve']))
    final_stats['n_samples'] = len(metrics['mve'])
    final_stats['cls_acc']   = cls_correct / total_samples

    # Generalisation gap ratios
    mve_arr = np.array(metrics['mve'])
    b_ids   = np.array(metrics['body_id'])
    f_lbls  = np.array(metrics['fab_label'])

    final_stats['body_gen_ratio']   = (mve_arr[b_ids >= 23].mean() /
                                       (mve_arr[b_ids < 23].mean() + 1e-9))
    final_stats['fabric_gen_ratio'] = (mve_arr[f_lbls == 5].mean() /
                                       (mve_arr[f_lbls != 5].mean() + 1e-9))

    # ── Threshold Table CSV ───────────────────────────────────────────────────
    table_data = []
    for m_name, thresh_dict in THRESHOLDS.items():
        if m_name not in metrics:
            continue
        vals = np.array(metrics[m_name])
        row  = {'Metric': m_name.upper()}
        for label, t_val in thresh_dict.items():
            acc = ((vals >= t_val).mean() * 100 if m_name in ['normals', 'iou']
                   else (vals <= t_val).mean() * 100)
            row[f"{label.capitalize()} (%)"] = f"{acc:.1f}%"
        table_data.append(row)
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(results_dir, 'threshold_performance.csv'), index=False)

    # ── Individual CDF Plots ──────────────────────────────────────────────────
    plot_metrics = ['mve', 'chamfer', 'hausdorff', 'strain', 'avg_strain', 'iou']
    if 'normals' in metrics and len(metrics['normals']) > 0:
        plot_metrics.append('normals')

    for m_name in plot_metrics:
        plt.figure(figsize=(10, 6))
        sorted_data = np.sort(metrics[m_name])
        x_frac      = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(x_frac, sorted_data, label=m_name.upper(), color='blue')

        if m_name in ['mve', 'chamfer']:
            plt.axhline(y=10,   color='red', linestyle='--', alpha=0.5,
                        label='10mm Threshold (Good)')
            y_label = 'Error Distance (mm)'
        elif m_name == 'hausdorff':
            plt.axhline(y=50,   color='red', linestyle='--', alpha=0.5,
                        label='50mm Threshold (Good)')
            y_label = 'Error Distance (mm)'
        elif m_name in ['strain', 'avg_strain']:
            plt.axhline(y=0.10, color='red', linestyle='--', alpha=0.5,
                        label='10% Strain Threshold (Good)')
            y_label = 'Strain Ratio'
        elif m_name == 'iou':
            plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.5,
                        label='80% IoU Threshold (Good)')
            y_label = 'Intersection over Union (IoU)'
        elif m_name == 'normals':
            plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5,
                        label='0.90 Cosine Sim Threshold (Good)')
            y_label = 'Cosine Similarity'

        plt.xlabel('Fraction of Test Set (Cumulative Accuracy)')
        plt.ylabel(y_label)
        plt.title(f'Cumulative Accuracy: {m_name.upper()}')
        loc = 'upper left' if m_name in ['iou', 'normals'] else 'upper right'
        plt.legend(loc=loc)
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'cumulative_accuracy_{m_name}.png'), dpi=300)
        plt.close()

    return final_stats, df, by_family, by_size, by_gen_condition


# ── Output Formatting ─────────────────────────────────────────────────────────

def print_results(stats, df, by_family, by_size, by_gen, split, baselines=None):
    mve_overall = stats['mve']

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS — {split.upper()}")
    print(f"{'='*70}")
    print(f"  Samples evaluated:       {stats['n_samples']}")
    print(f"  Classification Accuracy: {stats['cls_acc']:.1%}")
    print(f"  Overall MVE:             {mve_overall:.2f} mm  (±{stats['mve_std']:.2f})")
    print(f"  P90 Vertex Error:        {stats['p90']:.2f} mm")
    print(f"  Chamfer Distance:        {stats['chamfer']:.2f} mm")
    print(f"  Hausdorff Distance:      {stats['hausdorff']:.2f} mm")
    print(f"  Mesh Silhouette IoU:     {stats['iou']:.1%}")
    if 'normals' in stats:
        print(f"  Normal Consistency:      {stats['normals']:.3f} (cosine sim)")
    print(f"  Max Edge Strain:         {stats['strain']:.1%}")
    print(f"  Avg Edge Strain:         {stats['avg_strain']:.1%}")

    if baselines:
        print(f"\n── Baselines ──")
        z     = baselines['zero']
        arrow = '↑ beats' if mve_overall < z else '↓ worse than'
        print(f"  Zero predictor MVE:      {z:.2f} mm  ({arrow} baseline)")
        if 'mean' in baselines:
            m     = baselines['mean']
            arrow = '↑ beats' if mve_overall < m else '↓ worse than'
            print(f"  Mean predictor MVE:      {m:.2f} mm  ({arrow} baseline)")

    print(f"\n── Threshold Success Rates ──")
    print(df.to_string(index=False))

    print(f"\n── By Fabric Family ──")
    print(f"  {'Family':<20} {'MVE (mm)':>10}  {'vs overall':>12}")
    print(f"  {'-'*20} {'-'*10}  {'-'*12}")
    for fam in FABRIC_FAMILIES:
        if fam in by_family:
            val  = np.mean(by_family[fam])
            diff = val - mve_overall
            tag  = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            note = "  ← UNSEEN" if fam == 'heavy_woven' else ""
            print(f"  {fam:<20} {val:>10.2f}  {tag:>12}{note}")

    print(f"\n── By Size ──")
    print(f"  {'Size':<10} {'MVE (mm)':>10}  {'vs overall':>12}")
    print(f"  {'-'*10} {'-'*10}  {'-'*12}")
    for size in SIZES:
        if size in by_size:
            val  = np.mean(by_size[size])
            diff = val - mve_overall
            tag  = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            print(f"  {size:<10} {val:>10.2f}  {tag:>12}")

    print(f"\n── Generalisation Gaps ──")
    print(f"  Body Gen Ratio (Unseen/Seen):   {stats['body_gen_ratio']:.2f}x")
    print(f"  Fabric Gen Ratio (Unseen/Seen): {stats['fabric_gen_ratio']:.2f}x")

    print(f"\n── Specific Generalisation Conditions (MVE) ──")
    GEN_LABELS = {
        'seen_body_seen_mat':         'Seen body   / Seen material   (train dist)',
        'seen_body_unseen_mat_val':   'Seen body   / Unseen material (val)',
        'seen_body_unseen_mat_test':  'Seen body   / Unseen material (test)',
        'unseen_body_seen_mat_val':   'Unseen body / Seen material   (val)',
        'unseen_body_seen_mat_test':  'Unseen body / Seen material   (test)',
        'unseen_body_unseen_mat':     'Unseen body / Unseen material (hardest)',
    }
    for key, label in GEN_LABELS.items():
        if key in by_gen:
            val = np.mean(by_gen[key])
            print(f"  {val:>7.2f} mm  |  {label}")

    seen = np.mean(by_gen['seen_body_seen_mat'])     if 'seen_body_seen_mat'     in by_gen else None
    hard = np.mean(by_gen['unseen_body_unseen_mat']) if 'unseen_body_unseen_mat' in by_gen else None
    if seen is not None and hard is not None:
        gap     = hard - seen
        quality = ('Good generalisation'     if gap < 5  else
                   'Moderate generalisation' if gap < 15 else
                   'Poor generalisation')
        print(f"\n  Generalisation gap (hardest − seen): {gap:+.2f} mm")
        print(f"  {quality}  (< 5mm = good, < 15mm = moderate, > 15mm = poor)")

    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--data-root',   type=str, default=DATA_ROOT)
    parser.add_argument('--batch-size',  type=int, default=8)
    parser.add_argument('--save-meshes', action='store_true',
                        help='Export .obj files (pred + gt + template)')
    parser.add_argument('--baselines',   action='store_true',
                        help='Compute zero/mean baselines (loads full train set — slower)')
    args = parser.parse_args()

    # Set overall main seed before anything else
    set_deterministic_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split:      {args.split}")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    # ── Load Model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get('config', {})
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    if 'best_val_loss' in ckpt:
        print(f"Best val loss:    {ckpt['best_val_loss']:.4f}")

    model = UnfrozenCLSDrapeModel(
        gnn_layers        = cfg.get('gnn_layers', 8),
        embed_dim         = cfg.get('embed_dim',  128),
        latent_dim        = cfg.get('latent_dim', 128),
        cross_attn_layers = cfg.get('cross_attn_layers', None),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print("Model loaded.")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    eval_ds     = GarmentDataset(args.data_root, split=args.split, augment=False)
    # num_workers=0 ensures manual seed overrides apply cleanly in the main thread
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=False)
    print(f"  {len(eval_ds)} samples  ({len(eval_loader)} batches)")

    # ── Load Faces ────────────────────────────────────────────────────────────
    faces_path = os.path.join(args.data_root, 'template', 'faces.npy')
    faces = np.load(faces_path).astype(np.int32) if os.path.exists(faces_path) else None
    if faces is None:
        print("Warning: faces.npy not found — Normal Consistency skipped.")

    # ── Results Directory ─────────────────────────────────────────────────────
    run_dir     = os.path.dirname(os.path.dirname(args.checkpoint))
    results_dir = os.path.join(run_dir, f'eval_results_{args.split}')

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = {}
    if args.baselines:
        print("\nComputing baselines...")
        baselines['zero'] = compute_zero_baseline(eval_loader, device)
        print(f"  Zero predictor MVE: {baselines['zero']:.2f} mm")
        train_ds = GarmentDataset(args.data_root, split='train', augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)
        baselines['mean'] = compute_mean_baseline(train_loader, eval_loader, device)
        print(f"  Mean predictor MVE: {baselines['mean']:.2f} mm")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nStarting comprehensive evaluation...")
    stats, df, by_family, by_size, by_gen = evaluate(
        model, eval_loader, device, results_dir,
        save_meshes=args.save_meshes, faces=faces, inputs_per_mesh=3,
    )

    rounded_stats     = {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in stats.items()}
    rounded_baselines = {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in baselines.items()}

    # ── Save JSON Summary ─────────────────────────────────────────────────────
    summary = {
        'checkpoint': args.checkpoint,
        'split':      args.split,
        'epoch':      ckpt.get('epoch'),
        **rounded_stats,
        'baselines':  rounded_baselines,
        'by_family':  {k: round(float(np.mean(v)), 3) for k, v in by_family.items()},
        'by_size':    {k: round(float(np.mean(v)), 3) for k, v in by_size.items()},
        'by_gen':     {k: round(float(np.mean(v)), 3) for k, v in by_gen.items()},
    }
    with open(os.path.join(results_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print_results(stats, df, by_family, by_size, by_gen,
                  args.split, baselines if baselines else None)
    print(f"\nAll plots and tables saved to: {results_dir}")


if __name__ == '__main__':
    main()
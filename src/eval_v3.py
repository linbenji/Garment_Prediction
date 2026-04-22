"""
===============================================================================
EVALUATION PIPELINE: MasterDrapeModel (v3)
===============================================================================

This script performs a rigorous, multi-faceted evaluation of the 3D garment 
drape prediction model. It goes beyond simple average errors to measure geometric 
precision, physical validity, silhouette accuracy, and generalization capability.

Step-by-Step Execution Flow:
-------------------------------------------------------------------------------
1. Initialization & Loading
   - Loads the saved checkpoint (.pt) and extracts the architecture configuration.
   - Instantiates the MasterDrapeModel and restores its trained weights.
   - Initializes the GarmentDataset for the requested split (usually 'test').

2. Baseline Sanity Checks (Optional but Recommended)
   - Zero Predictor: Calculates the error if the model predicted 0 movement 
     (garment stays in its stiff T-pose). The model MUST beat this to be useful.
   - Mean Predictor: Calculates the "average drape" across the training set and 
     uses it as a static prediction. The model MUST beat this to prove it has 
     learned conditional physics (adjusting to specific images/sizes) rather 
     than just memorizing the average shape.

3. The Evaluation Loop (Inference & Metric Calculation)
   Iterates through the test dataset, performing a forward pass and computing:
   - Point-to-Point Metrics: Mean Vertex Error (MVE) and 90th Percentile Error (P90) 
     measure exactly how far each predicted vertex is from the ground truth.
   - Silhouette Metrics: Chamfer Distance, Hausdorff Distance (worst-case spike), 
     and Voxel IoU measure the overall "cloud shape" and volume accuracy, 
     ignoring specific vertex indices.
   - Physical Validity: Edge Strain calculates how much the fabric stretched 
     compared to the template, ensuring it behaves like cloth, not rubber.
   - Surface Fidelity: Normal Consistency (Cosine Similarity) measures if the 
     wrinkles and folds are facing the correct direction.
   - Vision Intelligence: Auxiliary Classification Accuracy checks if the DINOv2 
     backbone correctly identified the fabric family from the 2D image.

4. Categorization & "Bucketing"
   Instead of a single global average, the script slices the errors into domains:
   - By Fabric Family (e.g., Denim vs. Light Knit)
   - By Target Size (e.g., Small vs. XXL)
   - By Generalization Condition: The most critical bucket. It isolates performance 
     based on whether the model has seen the target body shape or fabric family 
     during training (e.g., 'unseen_body_unseen_mat' is the ultimate stress test).

5. Mesh Export (Optional)
   - Saves a pre-defined number of predicted and ground-truth meshes (.obj) 
     per fabric family for qualitative visual inspection in Blender/MeshLab.

6. Aggregation & Statistical Output
   - Threshold Success Rates: Converts continuous distances into a binary "Accuracy %" 
     (e.g., What percentage of test samples achieved < 5mm error?).
   - Generalization Gap: Calculates the ratio of error on unseen data vs. seen data.
   - Saves a cumulative accuracy plot (.png), a threshold table (.csv), and a 
     comprehensive metrics summary (.json) to the results directory.
===============================================================================

Usage:
    # Evaluate best checkpoint on test split (quick check)
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt

    # Evaluate on val split
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt --split val

    # Evaluate and save meshes for visualisation (RECOMMENDED RUN)
    # Outputs .obj files (3 samples per fabric family) showing the Template, Prediction, and Ground Truth
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt --save-meshes

    # Scientific run (with baselines)
    # This will take slightly longer because it has to load the entire train dataset into memory to calculate the Mean Predictor
    python eval_v3.py --checkpoint runs/method2_master/checkpoints/best.pt --baselines
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
from models_v3 import MasterDrapeModel

# ── Config ────────────────────────────────────────────────────────────────────

# DATA_ROOT = '/workspace/batch_1500_lean'
# RUNS_DIR  = '/workspace/runs'

DATA_ROOT  = r'C:\Users\chung\Desktop\Garment_Prediction\dataset\batch_1500_lean'
# DATA_ROOT  = r"/Users/Ben/Desktop/batch_1500_lean"
# RUNS_DIR   = r"/Users/Ben/Desktop/runs"

FABRIC_FAMILIES = [
    'light_knit', 'medium_knit', 'heavy_knit',
    'light_woven', 'medium_woven', 'heavy_woven',
]
SIZES = ['small', 'medium', 'large', 'xl', 'xxl']

# Family group mapping (mirrors step1_json_to_csv.py)
FAMILY_GROUP = {
    'light_knit': 1, 'medium_knit': 2, 'heavy_knit': 3,
    'light_woven': 4, 'medium_woven': 5, 'heavy_woven': 6,
}

FABRIC_FAMILY_IDX = {f: i for i, f in enumerate(FABRIC_FAMILIES)}

# Global flag to toggle saving a subset (30) vs all (180) meshes per pass
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
    
# ── Reproducibility Helper ────────────────────────────────────────────────────

def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Mesh & Geometry Helpers ───────────────────────────────────────────────────

def save_obj(path, verts, faces=None):
    """
    Save vertex positions as a .obj file.
    verts: (N, 3) numpy array in mm
    faces: (F, 3) numpy array of 0-based indices, optional
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("# Exported by eval_v3.py\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def compute_chamfer_hausdorff(pred, gt):
    """
    - Computes the maximum distance from any point in the prediction to the nearest point in the ground truth
        - Identifies the "worst-case" failure
    - O(N log N) distance calculation using KD-Trees
    """
    tree_gt = KDTree(gt)
    dist_p_to_g, _ = tree_gt.query(pred)
    
    tree_pred = KDTree(pred)
    dist_g_to_p, _ = tree_pred.query(gt)
    
    chamfer = (np.mean(dist_p_to_g) + np.mean(dist_g_to_p)) / 2
    hausdorff = max(np.max(dist_p_to_g), np.max(dist_g_to_p))
    return chamfer, hausdorff

def compute_iou(pred, gt, grid_res=32):
    """
    Voxel-based Intersection over Union for silhouette accuracy
    - It tells you if the model correctly captured the "envelope" of the garment around the body
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
    Cosine similarity of face normals
    - High MVE with low Normal Consistency means the shirt is in the right place,
            but the wrinkles and folds are missing or facing the wrong way
    """
    def get_normals(verts):
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return np.mean(np.sum(get_normals(pred) * get_normals(gt), axis=1))


# ── Baselines ─────────────────────────────────────────────────────────────────

def compute_zero_baseline(loader, device):
    """
    Zero predictor: always predict zero displacement.
    Lower bound — model must beat this.
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
    Mean predictor: predict the mean displacement vector per vertex
    computed across the training set.
    Tests whether the model learns more than the average drape shape.
    """
    print("  Computing training mean displacement...")
    sum_disp  = None
    n_samples = 0

    for batch in train_loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            disp = batch.y[mask]   # (14117, 3)
            if sum_disp is None:
                sum_disp = disp.clone()
            else:
                sum_disp += disp
            n_samples += 1

    mean_disp = sum_disp / n_samples   # (14117, 3)

    errors = []
    for batch in eval_loader:
        batch = batch.to(device)
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            errors.append((mean_disp - batch.y[mask]).norm(dim=1).mean().item())
    return float(np.mean(errors))


# ── Main Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, results_dir, save_meshes=False, faces=None, template_verts=None, n_save=3, inputs_per_mesh=3):
    model.eval()
    os.makedirs(results_dir, exist_ok=True)
    mesh_dir = os.path.join(results_dir, 'meshes') if save_meshes else None

    # Core Metrics Tracker
    metrics = defaultdict(list)
    
    # Categorization Trackers
    by_family = defaultdict(list)
    by_size = defaultdict(list)
    by_gen_condition = defaultdict(list)
    saved_per_family = defaultdict(int)
    
    cls_correct = 0
    total_samples = 0

    print(f"  Evaluating {len(loader.dataset)} samples...")

    # Multi-pass loop for 3 inputs per mesh
    for pass_idx in range(inputs_per_mesh):
        # Set a deterministic seed for this specific pass
        # This ensures the random pairs chosen in pass 0 are different from pass 1, 
        # but completely reproducible every time the script is run
        pass_seed = 42 + pass_idx
        set_deterministic_seed(pass_seed)
        
        print(f"    Pass {pass_idx + 1}/{inputs_per_mesh} (Seed: {pass_seed})...")

        # Track how many meshes we've processed in the current pass
        samples_this_pass = 0

        for batch in loader:
            batch = batch.to(device)
            pred_delta, fabric_logits = model(batch)
            
            # Classification Accuracy
            preds_cls = fabric_logits.argmax(dim=1)
            cls_correct += (preds_cls == batch.fabric_family_label).sum().item()

            # Pre-calculate Strain on the GPU (much faster)
            src, dst = batch.edge_index
            pred_len = torch.norm((batch.pos[src] + pred_delta[src]) - (batch.pos[dst] + pred_delta[dst]), dim=1)
            gt_len = torch.norm((batch.pos[src] + batch.y[src]) - (batch.pos[dst] + batch.y[dst]), dim=1)
            batch_strain = torch.abs(pred_len - gt_len) / (gt_len + 1e-9)

            for i in range(batch.num_graphs):
                # 1. Isolate the specific graph
                mask = (batch.batch == i)
                edge_mask = (batch.batch[batch.edge_index[0]] == i)
                
                p_verts = (batch.pos[mask] + pred_delta[mask]).cpu().numpy()
                g_verts = (batch.pos[mask] + batch.y[mask]).cpu().numpy()
                
                # 2. Calculate Core Metrics
                mve = np.linalg.norm(p_verts - g_verts, axis=1).mean()
                p90 = np.quantile(np.linalg.norm(p_verts - g_verts, axis=1), 0.90)
                chamfer, hausdorff = compute_chamfer_hausdorff(p_verts, g_verts)
                iou = compute_iou(p_verts, g_verts)
                max_strain = batch_strain[edge_mask].max().item()
                avg_strain = batch_strain[edge_mask].mean().item()

                metrics['mve'].append(mve)
                metrics['p90'].append(p90)
                metrics['chamfer'].append(chamfer)
                metrics['hausdorff'].append(hausdorff)
                metrics['iou'].append(iou)
                metrics['strain'].append(max_strain)
                metrics['avg_strain'].append(avg_strain)
                
                if faces is not None:
                    metrics['normals'].append(compute_normals_sim(p_verts, g_verts, faces))

                # 3. Categorization & Metadata
                body_id = batch.body_id[i].item()
                fab_label = batch.fabric_family_label[i].item()
                fab_name = FABRIC_FAMILIES[fab_label] if fab_label < len(FABRIC_FAMILIES) else 'unknown'
                fab_group = FAMILY_GROUP.get(fab_name, 0)
                
                size_enc = batch.tgt_size.view(-1, 2)[i]
                width_pct = round(size_enc[0].item(), 2)
                size_name = {0.92: 'small', 1.0: 'medium', 1.08: 'large', 1.17: 'xl', 1.28: 'xxl'}.get(width_pct, f'w={width_pct}')

                by_family[fab_name].append(mve)
                by_size[size_name].append(mve)

                # Generalization Buckets
                if body_id <= 22 and fab_group <= 5:     gen_cond = 'seen_body_seen_mat'
                elif body_id <= 12 and fab_group == 6:   gen_cond = 'seen_body_unseen_mat_val'
                elif body_id >= 23 and fab_group <= 2:   gen_cond = 'unseen_body_seen_mat_test'
                elif body_id >= 23 and fab_group == 6:   gen_cond = 'unseen_body_unseen_mat'
                elif body_id <= 22 and fab_group == 6:   gen_cond = 'seen_body_unseen_mat_test'
                elif body_id >= 23 and fab_group <= 5:   gen_cond = 'unseen_body_seen_mat_val'
                
                by_gen_condition[gen_cond].append(mve)

                metrics['body_id'].append(body_id)
                metrics['fab_label'].append(fab_label)

                # Save meshes (Check if we should save based on the SUBSET_SAVE flag)
                if save_meshes and (not SUBSET_SAVE or samples_this_pass < 30):
                    # Added 'pass_idx' to the filename so 3 different 
                    # random input pairings don't overwrite each other
                    # 1. Grab the unique sample name from the batch
                    uid = batch.sample_name[i]
                    
                    # 2. Add it to the prefix!
                    prefix = f"{uid}_body{body_id:03d}_{fab_name}_{size_name}_pass{pass_idx}_mve{mve:.1f}mm"
                    
                    # Save Prediction and Ground Truth only
                    save_obj(os.path.join(mesh_dir, f"{prefix}_pred.obj"), p_verts, faces)
                    save_obj(os.path.join(mesh_dir, f"{prefix}_gt.obj"), g_verts, faces)

                total_samples += 1
                samples_this_pass += 1


    # ── Post-Processing & Tables ──────────────────────────────────────────────
    
    # 1. Means
    final_stats = {k: np.mean(v) for k, v in metrics.items() if k not in ['body_id', 'fab_label']}
    final_stats['cls_acc'] = cls_correct / total_samples

    # 2. Generalization Gaps
    mve_arr = np.array(metrics['mve'])
    b_ids, f_lbls = np.array(metrics['body_id']), np.array(metrics['fab_label'])
    
    final_stats['body_gen_ratio'] = mve_arr[b_ids >= 23].mean() / (mve_arr[b_ids < 23].mean() + 1e-9)
    final_stats['fabric_gen_ratio'] = mve_arr[f_lbls == 5].mean() / (mve_arr[f_lbls != 5].mean() + 1e-9)

    # 3. Threshold Table CSV
    table_data = []
    for m_name, thresh_dict in THRESHOLDS.items():
        if m_name not in metrics: continue
        row = {'Metric': m_name.upper()}
        vals = np.array(metrics[m_name])
        for label, t_val in thresh_dict.items():
            acc = (vals >= t_val).mean() * 100 if m_name in ['normals', 'iou'] else (vals <= t_val).mean() * 100
            row[f"{label.capitalize()} (%)"] = f"{acc:.1f}%"
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(results_dir, 'threshold_performance.csv'), index=False)

    # ── Individual Plots ──────────────────────────────────────────────────────

    # Define all metrics to plot. 
    # (Normals is appended conditionally to prevent crashes if faces.npy is missing)
    plot_metrics = ['mve', 'chamfer', 'hausdorff', 'strain', 'avg_strain', 'iou']
    if 'normals' in metrics and len(metrics['normals']) > 0:
        plot_metrics.append('normals')
    
    for m_name in plot_metrics:
        plt.figure(figsize=(10, 6))
        # Sort the data to create a Cumulative Distribution Function (CDF)
        sorted_data = np.sort(metrics[m_name])
        # Fraction of Test Set / Accuracy
        x_frac = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        plt.plot(x_frac, sorted_data, label=f'{m_name.upper()}', color='blue')
        
        # Add visual reference lines and dynamic X-axis labels
        if m_name in ['mve', 'chamfer']:
            plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10mm Threshold (Good)')
            y_label = 'Error Distance (mm)'
        elif m_name == 'hausdorff':
            plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50mm Threshold (Good)')
            y_label = 'Error Distance (mm)'
        elif m_name in ['strain', 'avg_strain']:
            plt.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='10% Strain Threshold (Good)')
            y_label = 'Strain Ratio'
        elif m_name == 'iou':
            plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='80% IoU Threshold (Good)')
            y_label = 'Intersection over Union (IoU)'
        elif m_name == 'normals':
            plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='0.90 Cosine Sim Threshold (Good)')
            y_label = 'Cosine Similarity'

        plt.xlabel('Fraction of Test Set (Accuracy Cumulative)')
        plt.ylabel(y_label)
        plt.title(f'Cumulative Accuracy: {m_name.upper()}')
        # Adjust legend location so it doesn't block the curve
        loc = 'upper left' if m_name in ['iou', 'normals'] else 'upper right'
        plt.legend(loc=loc)
        plt.grid(True)
        
        plot_path = os.path.join(results_dir, f'cumulative_accuracy_{m_name}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return final_stats, df, by_family, by_size, by_gen_condition

# ── Output Formatting ─────────────────────────────────────────────────────────

def print_results(stats, df, by_family, by_size, by_gen, baselines):
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Classification Accuracy: {stats['cls_acc']:.1%}")
    print(f"  Overall MVE:             {stats['mve']:.2f} mm")
    print(f"  P90 Vertex Error:        {stats['p90']:.2f} mm")
    print(f"  Chamfer Distance:        {stats['chamfer']:.2f} mm")
    print(f"  Hausdorff Distance:      {stats['hausdorff']:.2f} mm")
    print(f"  Mesh Silhouette IoU:     {stats['iou']:.1%}")
    if 'normals' in stats:
        print(f"  Normal Consistency:  {stats['normals']:.3f} (cosine sim)")
    print(f"  Max Edge Strain:         {stats['strain']:.1%}")
    print(f"  Avg Edge Strain:         {stats['avg_strain']:.1%}")

    if baselines:
        print(f"\n── Baselines ──")
        print(f"  Zero Predictor MVE:      {baselines['zero']:.2f} mm")
        if 'mean' in baselines:
            print(f"  Mean Predictor MVE:  {baselines['mean']:.2f} mm")

    print(f"\n── Threshold Success Rates ──")
    print(df.to_string(index=False))

    print(f"\n── Generalization Gaps ──")
    print(f"  Body Gen Ratio (Unseen/Seen):   {stats['body_gen_ratio']:.2f}x")
    print(f"  Fabric Gen Ratio (Unseen/Seen): {stats['fabric_gen_ratio']:.2f}x")
    
    print(f"\n── Specific Generalization Conditions (MVE) ──")
    for key, val in by_gen.items():
        print(f"  {np.mean(val):>7.2f} mm | {key}")

    print(f"{'='*70}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-meshes', action='store_true', help='Export .obj files')
    parser.add_argument('--baselines', action='store_true', help='Compute zero/mean baselines (slower)')
    args = parser.parse_args()

    # Set overall main seed
    set_deterministic_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint missing.")
        return

    # Load Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = MasterDrapeModel(
        embed_dim=cfg.get('embed_dim', 128),
        latent_dim=cfg.get('latent_dim', 128),
        gnn_layers=cfg.get('gnn_layers', 8),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    
    # Load Data
    eval_ds = GarmentDataset(DATA_ROOT, split=args.split, augment=False)
    # Important: num_workers=0 ensures our manual seed overrides apply cleanly in the main thread
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load Faces
    faces_path = os.path.join(DATA_ROOT, 'template', 'faces.npy')
    faces = np.load(faces_path).astype(np.int32) if os.path.exists(faces_path) else None
    if faces is None: print("Warning: faces.npy not found. Normal Consistency skipped.")

    # Results Directory Setup
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    results_dir = os.path.join(run_dir, f'eval_results_{args.split}')

    # Baselines
    baselines = {}
    if args.baselines:
        print("\nComputing baselines...")
        baselines['zero'] = compute_zero_baseline(eval_loader, device)
        train_loader = DataLoader(GarmentDataset(DATA_ROOT, split='train', augment=False), batch_size=args.batch_size)
        baselines['mean'] = compute_mean_baseline(train_loader, eval_loader, device)

    # Evaluate
    print("\nStarting comprehensive evaluation...")
    stats, df, by_family, by_size, by_gen = evaluate(
        model, eval_loader, device, results_dir, 
        save_meshes=args.save_meshes, faces=faces, inputs_per_mesh=3
    )

    rounded_stats = {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}
    rounded_baselines = {k: round(v, 3) if isinstance(v, float) else v for k, v in baselines.items()}

    # Save and Print
    with open(os.path.join(results_dir, 'summary_stats.json'), 'w') as f:
        json.dump({
            **rounded_stats,
            'baselines': rounded_baselines,
            'by_family': {k: round(float(np.mean(v)), 3) for k, v in by_family.items()},
            'by_size': {k: round(float(np.mean(v)), 3) for k, v in by_size.items()},
            'by_gen': {k: round(float(np.mean(v)), 3) for k, v in by_gen.items()},
        }, f, indent=2, cls=NumpyEncoder)

    print_results(stats, df, by_family, by_size, by_gen, baselines)
    print(f"\nAll plots and tables saved to: {results_dir}")

if __name__ == '__main__':
    main()
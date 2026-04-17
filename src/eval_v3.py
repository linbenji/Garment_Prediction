"""
eval_v3.py

Evaluation script for MasterDrapeModel checkpoints.

Computes:
  1. Overall MVE (mean vertex error in mm)
  2. Per-fabric-family MVE breakdown
  3. Per-size MVE breakdown
  4. Generalisation condition breakdown (seen/unseen body x seen/unseen material)
  5. Zero predictor baseline (predict zero displacement)
  6. Mean predictor baseline (predict training mean displacement per vertex)
  7. Saves predicted + ground truth meshes as .obj for visualisation

Usage:
    # Evaluate best checkpoint on test split
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt

    # Evaluate on val split
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt --split val

    # Evaluate and save meshes for visualisation
    python eval_v3.py --checkpoint runs/method1_baseline/checkpoints/best.pt --save-meshes
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_v2 import GarmentDataset
from models_v3 import MasterDrapeModel

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = '/workspace/batch_1500_lean'
RUNS_DIR  = '/workspace/runs'

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


# ── OBJ writer ────────────────────────────────────────────────────────────────

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


# ── Metric helpers ────────────────────────────────────────────────────────────

def vertex_errors(pred_delta, target_delta):
    """
    Per-vertex Euclidean error in mm.
    Returns (N,) tensor.
    """
    return (pred_delta - target_delta).norm(dim=1)


def mve(pred_delta, target_delta):
    """Mean vertex error in mm — scalar."""
    return vertex_errors(pred_delta, target_delta).mean().item()


def p90_ve(pred_delta, target_delta):
    """90th percentile vertex error in mm."""
    return vertex_errors(pred_delta, target_delta).quantile(0.90).item()


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
            errors.append(mve(pred[mask], batch.y[mask]))
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
            errors.append(mve(mean_disp, batch.y[mask]))

    return float(np.mean(errors))


# ── Main evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, save_meshes=False, mesh_dir=None,
             faces=None, template_verts=None, n_save=3):
    """
    Run evaluation on a DataLoader.

    Returns dict of metrics and optionally saves predicted meshes.
    """
    model.eval()

    # Accumulators
    all_mve          = []
    all_p90          = []
    by_family        = defaultdict(list)
    by_size          = defaultdict(list)
    by_gen_condition = defaultdict(list)

    # For mesh saving — pick n_save samples per fabric family
    saved_per_family  = defaultdict(int)
    meshes_to_save    = []

    sample_idx = 0

    for batch in loader:
        batch = batch.to(device)

        predicted_delta, _ = model(batch)

        for i in range(batch.num_graphs):
            mask       = (batch.batch == i)
            pred_d     = predicted_delta[mask]   # (14117, 3)
            gt_d       = batch.y[mask]           # (14117, 3)

            err        = mve(pred_d, gt_d)
            p90_err    = p90_ve(pred_d, gt_d)

            all_mve.append(err)
            all_p90.append(p90_err)

            # Metadata from batch
            body_id   = batch.body_id[i].item()
            fab_label = batch.fabric_family_label[i].item()
            fab_name  = FABRIC_FAMILIES[fab_label] if fab_label < len(FABRIC_FAMILIES) else 'unknown'
            fab_group = FAMILY_GROUP.get(fab_name, 0)

            # Size — decode from tgt_size tensor
            # tgt_size is [width_pct, height_pct] — use it to infer size name
            size_enc  = batch.tgt_size.view(-1, 2)[i]
            width_pct = round(size_enc[0].item(), 2)
            SIZE_FROM_WIDTH = {0.92: 'small', 1.0: 'medium', 1.08: 'large',
                               1.17: 'xl', 1.28: 'xxl'}
            size_name = SIZE_FROM_WIDTH.get(width_pct, f'w={width_pct}')

            by_family[fab_name].append(err)
            by_size[size_name].append(err)

            # Generalisation condition
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
            by_gen_condition[gen_cond].append(err)

            # Collect meshes for saving
            if save_meshes and saved_per_family[fab_name] < n_save:
                pos = batch.pos[mask].cpu().numpy()   # template positions
                pred_mesh = pos + pred_d.cpu().numpy()
                gt_mesh   = pos + gt_d.cpu().numpy()
                meshes_to_save.append({
                    'idx':       sample_idx,
                    'fabric':    fab_name,
                    'size':      size_name,
                    'body_id':   body_id,
                    'err_mm':    err,
                    'pred_verts': pred_mesh,
                    'gt_verts':   gt_mesh,
                    'template_verts': pos,
                })
                saved_per_family[fab_name] += 1

            sample_idx += 1

    # ── Save meshes ───────────────────────────────────────────────────────────
    if save_meshes and mesh_dir and meshes_to_save:
        print(f"\n  Saving {len(meshes_to_save)} meshes to {mesh_dir}")
        for m in meshes_to_save:
            prefix = (f"body{m['body_id']:03d}_{m['fabric']}_"
                      f"{m['size']}_mve{m['err_mm']:.1f}mm")
            save_obj(os.path.join(mesh_dir, f"{prefix}_pred.obj"),
                     m['pred_verts'], faces)
            save_obj(os.path.join(mesh_dir, f"{prefix}_gt.obj"),
                     m['gt_verts'], faces)
            save_obj(os.path.join(mesh_dir, f"{prefix}_template.obj"),
                     m['template_verts'], faces)

    return {
        'mve':            float(np.mean(all_mve)),
        'p90_ve':         float(np.mean(all_p90)),
        'mve_std':        float(np.std(all_mve)),
        'n_samples':      len(all_mve),
        'by_family':      {k: float(np.mean(v)) for k, v in by_family.items()},
        'by_size':        {k: float(np.mean(v)) for k, v in by_size.items()},
        'by_gen':         {k: float(np.mean(v)) for k, v in by_gen_condition.items()},
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_results(results, split, baselines=None):
    print(f"\n{'='*65}")
    print(f"EVALUATION RESULTS — {split.upper()}")
    print(f"{'='*65}")
    print(f"  Samples evaluated: {results['n_samples']}")
    print(f"  Overall MVE:       {results['mve']:.2f} mm")
    print(f"  P90 vertex error:  {results['p90_ve']:.2f} mm")
    print(f"  MVE std:           {results['mve_std']:.2f} mm")

    if baselines:
        print(f"\n── Baselines ──")
        print(f"  Zero predictor:   {baselines['zero']:.2f} mm  "
              f"({'↑' if results['mve'] < baselines['zero'] else '↓'} model "
              f"{'beats' if results['mve'] < baselines['zero'] else 'worse than'} baseline)")
        if 'mean' in baselines:
            print(f"  Mean predictor:   {baselines['mean']:.2f} mm  "
                  f"({'↑' if results['mve'] < baselines['mean'] else '↓'} model "
                  f"{'beats' if results['mve'] < baselines['mean'] else 'worse than'} baseline)")

    print(f"\n── By fabric family ──")
    print(f"  {'Family':<20} {'MVE (mm)':>10}  {'vs overall':>12}")
    print(f"  {'-'*20} {'-'*10}  {'-'*12}")
    for fam in FABRIC_FAMILIES:
        if fam in results['by_family']:
            val  = results['by_family'][fam]
            diff = val - results['mve']
            tag  = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            unseen = " ← UNSEEN" if fam == 'heavy_woven' else ""
            print(f"  {fam:<20} {val:>10.2f}  {tag:>12}{unseen}")

    print(f"\n── By size ──")
    print(f"  {'Size':<10} {'MVE (mm)':>10}  {'vs overall':>12}")
    print(f"  {'-'*10} {'-'*10}  {'-'*12}")
    for size in SIZES:
        if size in results['by_size']:
            val  = results['by_size'][size]
            diff = val - results['mve']
            tag  = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            print(f"  {size:<10} {val:>10.2f}  {tag:>12}")

    print(f"\n── Generalisation conditions ──")
    GEN_LABELS = {
        'seen_body_seen_mat':         'Seen body   / Seen material   (train dist)',
        'seen_body_unseen_mat_val':   'Seen body   / Unseen material (val)',
        'seen_body_unseen_mat_test':  'Seen body   / Unseen material (test)',
        'unseen_body_seen_mat_val':   'Unseen body / Seen material   (val)',
        'unseen_body_seen_mat_test':  'Unseen body / Seen material   (test)',
        'unseen_body_unseen_mat':     'Unseen body / Unseen material (hardest)',
    }
    for key, label in GEN_LABELS.items():
        if key in results['by_gen']:
            val = results['by_gen'][key]
            print(f"  {val:>7.2f} mm  {label}")

    # Generalisation gap — key research metric
    seen  = results['by_gen'].get('seen_body_seen_mat', None)
    hard  = results['by_gen'].get('unseen_body_unseen_mat', None)
    if seen and hard:
        gap = hard - seen
        print(f"\n  Generalisation gap (hardest - seen): {gap:+.2f} mm")
        print(f"  {'Good generalisation' if gap < 5 else 'Moderate generalisation' if gap < 15 else 'Poor generalisation'} "
              f"(gap < 5mm = good, < 15mm = moderate, > 15mm = poor)")

    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--split',      type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--data-root',  type=str, default=DATA_ROOT)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-meshes', action='store_true',
                        help='Save predicted + GT meshes as .obj files')
    parser.add_argument('--baselines',  action='store_true',
                        help='Compute zero and mean predictor baselines '
                             '(requires loading train set, slower)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get('config', {})
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"Best val loss:    {ckpt.get('best_val_loss', '?'):.4f}")

    # ── Build model and load weights ──────────────────────────────────────────
    model = MasterDrapeModel(
        embed_dim  = cfg.get('embed_dim',  128),
        latent_dim = cfg.get('latent_dim', 128),
        gnn_layers = cfg.get('gnn_layers', 10),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded successfully")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.split} dataset...")
    eval_ds = GarmentDataset(args.data_root, split=args.split, augment=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )
    print(f"  {len(eval_ds)} samples  ({len(eval_loader)} batches)")

    # ── Load template faces for mesh saving ───────────────────────────────────
    faces = None
    template_verts = None
    if args.save_meshes:
        faces_path = os.path.join(args.data_root, 'template', 'faces.npy')
        tverts_path = os.path.join(args.data_root, 'template', 'template_verts.npy')
        if os.path.exists(faces_path):
            import numpy as np
            faces = np.load(faces_path).astype(np.int32)
            print(f"  Faces loaded for mesh export: {faces.shape}")
        if os.path.exists(tverts_path):
            template_verts = np.load(tverts_path).astype(np.float32)

    # ── Mesh output directory ─────────────────────────────────────────────────
    ckpt_dir  = os.path.dirname(args.checkpoint)
    run_dir   = os.path.dirname(ckpt_dir)
    mesh_dir  = os.path.join(run_dir, 'meshes', args.split) \
                if args.save_meshes else None

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\nRunning evaluation...")
    results = evaluate(
        model, eval_loader, device,
        save_meshes     = args.save_meshes,
        mesh_dir        = mesh_dir,
        faces           = faces,
        template_verts  = template_verts,
        n_save          = 3,   # save 3 samples per fabric family
    )

    # ── Compute baselines ─────────────────────────────────────────────────────
    baselines = None
    if args.baselines:
        print(f"\nComputing baselines...")
        baselines = {}

        # Zero baseline — predict zero displacement
        print("  Zero predictor...")
        baselines['zero'] = compute_zero_baseline(eval_loader, device)
        print(f"    Zero predictor MVE: {baselines['zero']:.2f} mm")

        # Mean baseline — needs training set
        print("  Mean predictor (loading train set)...")
        train_ds     = GarmentDataset(args.data_root, split='train', augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)
        baselines['mean'] = compute_mean_baseline(train_loader, eval_loader, device)
        print(f"    Mean predictor MVE: {baselines['mean']:.2f} mm")

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(results, args.split, baselines)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = os.path.join(run_dir, f'eval_{args.split}.json')
    output = {
        'checkpoint':  args.checkpoint,
        'split':       args.split,
        'epoch':       ckpt.get('epoch'),
        'results':     results,
        'baselines':   baselines,
    }
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()

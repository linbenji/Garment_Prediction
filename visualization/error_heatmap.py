"""
error_heatmap.py

Exports predicted meshes as OBJ files with per-vertex color encoding
based on prediction error. Red = high error, blue = low error.

Opens in Blender/MeshLab with vertex colors visible — immediately shows
where the model struggles (hem, sleeves, collar, seams).

Usage:
    # Export heatmaps for test split using best checkpoint
    python error_heatmap.py --checkpoint runs/method2_master/checkpoints/best.pt

    # Limit to 5 samples
    python error_heatmap.py --checkpoint runs/method2_master/checkpoints/best.pt --max-samples 5

    # Custom error cap (default 20mm)
    python error_heatmap.py --checkpoint runs/method2_master/checkpoints/best.pt --max-error 15
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataloader_v2 import GarmentDataset

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = '/Users/Ben/Desktop/batch_1500_lean'

FABRIC_FAMILIES = [
    'light_knit', 'medium_knit', 'heavy_knit',
    'light_woven', 'medium_woven', 'heavy_woven',
]


# ── Color mapping ─────────────────────────────────────────────────────────────

def error_to_rgb(errors, max_error=20.0):
    """
    Maps per-vertex errors to RGB colors.
    Blue (0,0,1) = 0 error, Green (0,1,0) = mid error, Red (1,0,0) = max error.

    Args:
        errors: (N,) numpy array of per-vertex errors in mm
        max_error: error value that maps to pure red
    Returns:
        colors: (N, 3) numpy array of RGB values in [0, 1]
    """
    t = np.clip(errors / max_error, 0.0, 1.0)

    r = np.where(t < 0.5, 0.0, (t - 0.5) * 2.0)
    g = np.where(t < 0.5, t * 2.0, 1.0 - (t - 0.5) * 2.0)
    b = np.where(t < 0.5, 1.0 - t * 2.0, 0.0)

    return np.stack([r, g, b], axis=1)


def save_colored_obj(path, verts, colors, faces=None):
    """
    Save OBJ with per-vertex colors using 'v x y z r g b' format.
    Most viewers (Blender, MeshLab) render these automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("# Error heatmap — Blue=low, Green=mid, Red=high\n")
        for v, c in zip(verts, colors):
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_ply(path, verts, colors, faces=None):
    """
    Save PLY with per-vertex colors — more reliable color support than OBJ.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_verts = len(verts)
    n_faces = len(faces) if faces is not None else 0

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if n_faces > 0:
            f.write(f"element face {n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        colors_255 = (colors * 255).astype(np.uint8)
        for v, c in zip(verts, colors_255):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

        if faces is not None:
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--max-error', type=float, default=20.0, help='Error cap for color scale (mm)')
    parser.add_argument('--format', type=str, default='both', choices=['obj', 'ply', 'both'],
                        help='Output format (PLY has more reliable color support)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    # Auto-detect model version
    if 'cross_attn_layers' in cfg:
        from models_v4 import MasterDrapeModel
        model = MasterDrapeModel(
            gnn_layers=cfg.get('gnn_layers', 8),
            embed_dim=cfg.get('embed_dim', 128),
            latent_dim=cfg.get('latent_dim', 128),
            cross_attn_layers=cfg.get('cross_attn_layers', None),
        ).to(device)
    else:
        try:
            from models_v3 import MasterDrapeModel
            model = MasterDrapeModel(
                gnn_layers=cfg.get('gnn_layers', 8),
                embed_dim=cfg.get('embed_dim', 128),
                latent_dim=cfg.get('latent_dim', 128),
            ).to(device)
        except:
            from models_v2 import HybridDrapeModel
            model = HybridDrapeModel(
                gnn_layers=cfg.get('gnn_layers', 8),
                embed_dim=cfg.get('embed_dim', 128),
                latent_dim=cfg.get('latent_dim', 128),
            ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded — epoch {ckpt.get('epoch', '?')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    dataset = GarmentDataset(DATA_ROOT, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    faces_path = os.path.join(DATA_ROOT, 'template', 'faces.npy')
    faces = np.load(faces_path).astype(np.int32) if os.path.exists(faces_path) else None

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = os.path.dirname(os.path.dirname(args.checkpoint))
    output_dir = os.path.join(run_dir, 'heatmaps', args.split)
    os.makedirs(output_dir, exist_ok=True)

    # ── Generate heatmaps ─────────────────────────────────────────────────────
    n_target = args.max_samples or len(dataset)
    sample_idx = 0
    all_errors = []

    print(f"\nGenerating error heatmaps (max_error={args.max_error}mm)...")

    for batch in loader:
        if sample_idx >= n_target:
            break

        batch = batch.to(device)
        predicted_delta, _ = model(batch)

        for i in range(batch.num_graphs):
            if sample_idx >= n_target:
                break

            mask = (batch.batch == i)
            pred_d = predicted_delta[mask].cpu().numpy()
            gt_d = batch.y[mask].cpu().numpy()
            pos = batch.pos[mask].cpu().numpy()

            # Per-vertex error
            per_vertex_err = np.linalg.norm(pred_d - gt_d, axis=1)
            mve = per_vertex_err.mean()
            p90 = np.quantile(per_vertex_err, 0.90)
            all_errors.append(mve)

            # Vertex colors
            colors = error_to_rgb(per_vertex_err, max_error=args.max_error)
            pred_verts = pos + pred_d

            # Metadata
            row = dataset.df.iloc[sample_idx]
            fab_label = batch.fabric_family_label[i].item()
            fab_name = FABRIC_FAMILIES[fab_label] if fab_label < len(FABRIC_FAMILIES) else 'unknown'
            body_id = batch.body_id[i].item()

            size_enc = batch.tgt_size.view(-1, 2)[i]
            width_pct = round(size_enc[0].item(), 2)
            size_name = {0.92: 'small', 1.0: 'medium', 1.08: 'large',
                         1.17: 'xl', 1.28: 'xxl'}.get(width_pct, 'unknown')

            prefix = f"body{body_id:03d}_{fab_name}_{size_name}_mve{mve:.1f}mm"

            if args.format in ('obj', 'both'):
                save_colored_obj(os.path.join(output_dir, f"{prefix}_heatmap.obj"),
                                 pred_verts, colors, faces)

            if args.format in ('ply', 'both'):
                save_ply(os.path.join(output_dir, f"{prefix}_heatmap.ply"),
                         pred_verts, colors, faces)

            # Also save clean GT for comparison
            if args.format in ('obj', 'both'):
                gt_verts = pos + gt_d
                gt_colors = np.tile([0.8, 0.8, 0.8], (len(gt_verts), 1))
                save_colored_obj(os.path.join(output_dir, f"{prefix}_gt.obj"),
                                 gt_verts, gt_colors, faces)

            sample_idx += 1
            if sample_idx % 10 == 0:
                print(f"  {sample_idx}/{n_target}  running MVE={np.mean(all_errors):.2f}mm")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Exported {sample_idx} heatmaps to {output_dir}")
    print(f"Mean MVE: {np.mean(all_errors):.2f}mm")
    print(f"Color scale: blue=0mm, green={args.max_error / 2:.0f}mm, red={args.max_error:.0f}mm+")
    print(f"Open in MeshLab: File > Import Mesh > select .ply file")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
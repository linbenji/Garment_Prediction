"""
predict_v3.py

Single-sample inference for MasterDrapeModel
Given a fabric type, body shape, size, and a reference image, predicts the draped shirt mesh and saves it as a .obj file

Usage:
    # Predict using a sample from the dataset by name
    python predict_final.py --checkpoint runs/method1_baseline/checkpoints/best.pt \
                            --sample body003_lean_denim_xl

    # Predict for all test samples and save meshes
    python predict_final.py --checkpoint runs/method1_baseline/checkpoints/best.pt \
                            --all-test

    # Predict with a specific image as ViT input
    python predict_final.py --checkpoint runs/method1_baseline/checkpoints/best.pt \
                            --sample body003_lean_denim_xl \
                            --image path/to/image.png
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_v2 import (
    GarmentDataset, PHYSICS_COLS, SIZE_ENCODING,
    CAMERA_ANGLES, COLOR_VARIANTS,
    FABRIC_FAMILY_LABELS, FABRIC_PRESET_LABELS,
)
from models_v3 import MasterDrapeModel

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = '/workspace/batch_1500_lean'
RUNS_DIR  = '/workspace/runs'

# ── OBJ writer ────────────────────────────────────────────────────────────────

def save_obj(path, verts, faces=None, comment=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Saved: {path}")


# ── Image transform ───────────────────────────────────────────────────────────

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Single sample prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_sample(model, sample, device):
    """
    Run inference on a single PyG Data object.
    Returns predicted vertex positions (N, 3) in mm.
    """
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([sample]).to(device)
    model.eval()
    predicted_delta, fabric_logits = model(batch)

    # Predicted mesh = template positions + predicted displacement
    pred_verts = batch.pos + predicted_delta   # (14117, 3)

    # Fabric classification confidence
    probs      = torch.softmax(fabric_logits, dim=-1)[0]
    families   = ['light_knit', 'medium_knit', 'heavy_knit',
                  'light_woven', 'medium_woven', 'heavy_woven']
    pred_fam   = families[probs.argmax().item()]
    confidence = probs.max().item()

    return pred_verts.cpu().numpy(), predicted_delta.cpu().numpy(), {
        'predicted_family':    pred_fam,
        'family_confidence':   confidence,
        'family_probs':        {f: round(probs[i].item(), 4)
                                for i, f in enumerate(families)},
    }


# ── Predict from dataset sample name ─────────────────────────────────────────

def predict_by_name(model, sample_name, dataset, device,
                    image_path=None, output_dir=None, faces=None):
    """
    Find a sample by name in the dataset and run prediction.
    """
    # Find the sample in the dataset
    idx = None
    for i in range(len(dataset)):
        if dataset.df.iloc[i]['sample_name'] == sample_name:
            idx = i
            break

    if idx is None:
        print(f"  Sample '{sample_name}' not found in {dataset.split} split")
        print(f"  Try a different split or check the sample name")
        return None

    sample = dataset[idx]
    row    = dataset.df.iloc[idx]

    # Override image if provided
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        sample.image = IMG_TRANSFORM(img).unsqueeze(0)
        print(f"  Using custom image: {image_path}")

    print(f"  Sample:  {sample_name}")
    print(f"  Fabric:  {row['fabric_preset']} ({row['fabric_family']})")
    print(f"  Size:    {row['garment_size']}")
    print(f"  Body:    {row['body_id']}")

    pred_verts, pred_delta, cls_info = predict_sample(model, sample, device)
    gt_verts = (sample.pos.numpy() + sample.y.numpy())

    # Compute error
    mve = np.linalg.norm(pred_delta - sample.y.numpy(), axis=1).mean()
    print(f"  MVE:     {mve:.2f} mm")
    print(f"  Predicted fabric family: {cls_info['predicted_family']} "
          f"(confidence: {cls_info['family_confidence']:.1%})")

    # Save meshes
    if output_dir:
        prefix = os.path.join(output_dir, sample_name)
        save_obj(f"{prefix}_predicted.obj", pred_verts, faces,
                 comment=f"Predicted drape — {sample_name} MVE={mve:.2f}mm")
        save_obj(f"{prefix}_ground_truth.obj", gt_verts, faces,
                 comment=f"Ground truth — {sample_name}")
        save_obj(f"{prefix}_template.obj", sample.pos.numpy(), faces,
                 comment=f"Template (rest pose) — {sample_name}")

        # Save prediction metadata
        meta = {
            'sample_name':       sample_name,
            'fabric_preset':     row['fabric_preset'],
            'fabric_family':     row['fabric_family'],
            'garment_size':      row['garment_size'],
            'body_id':           int(row['body_id']),
            'mve_mm':            round(float(mve), 3),
            'classification':    cls_info,
        }
        with open(f"{prefix}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved: {prefix}_meta.json")

    return pred_verts, mve, cls_info


# ── Predict all test samples ──────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model, dataset, device, output_dir, faces=None,
                max_samples=None):
    """
    Run prediction on all samples in a dataset split.
    Saves one .obj per sample — useful for batch visualisation.
    """
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=0, pin_memory=False)

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    all_mve     = []
    sample_idx  = 0
    n_target    = max_samples or len(dataset)

    print(f"  Predicting {n_target} samples...")

    for batch in loader:
        if sample_idx >= n_target:
            break
        batch = batch.to(device)
        predicted_delta, _ = model(batch)

        for i in range(batch.num_graphs):
            if sample_idx >= n_target:
                break

            mask       = (batch.batch == i)
            pred_d     = predicted_delta[mask].cpu().numpy()
            gt_d       = batch.y[mask].cpu().numpy()
            pos        = batch.pos[mask].cpu().numpy()

            pred_verts = pos + pred_d
            gt_verts   = pos + gt_d
            mve        = np.linalg.norm(pred_d - gt_d, axis=1).mean()
            all_mve.append(mve)

            # Get sample name from dataset
            row        = dataset.df.iloc[sample_idx]
            name       = row['sample_name']
            prefix     = os.path.join(output_dir, name)

            save_obj(f"{prefix}_pred.obj", pred_verts, faces)
            save_obj(f"{prefix}_gt.obj",   gt_verts,   faces)

            sample_idx += 1
            if sample_idx % 20 == 0:
                print(f"    {sample_idx}/{n_target}  "
                      f"running MVE={np.mean(all_mve):.2f}mm")

    print(f"\n  Completed {sample_idx} samples")
    print(f"  Mean MVE: {np.mean(all_mve):.2f} mm")
    print(f"  Meshes saved to: {output_dir}")
    return all_mve


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--sample',      type=str, default=None,
                        help='Sample name to predict, e.g. body003_lean_denim_xl')
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--image',       type=str, default=None,
                        help='Optional: path to a custom image for ViT input')
    parser.add_argument('--all-test',    action='store_true',
                        help='Predict all test samples and save meshes')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples when using --all-test')
    parser.add_argument('--data-root',   type=str, default=DATA_ROOT)
    parser.add_argument('--output-dir',  type=str, default=None,
                        help='Directory to save .obj files '
                             '(default: runs/{exp}/predictions/{split})')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get('config', {})

    # ── Build model ───────────────────────────────────────────────────────────
    model = MasterDrapeModel(
        embed_dim  = cfg.get('embed_dim',  128),
        latent_dim = cfg.get('latent_dim', 128),
        gnn_layers = cfg.get('gnn_layers', 10),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded — epoch {ckpt.get('epoch', '?')}")

    # ── Load template faces ───────────────────────────────────────────────────
    faces = None
    faces_path = os.path.join(args.data_root, 'template', 'faces.npy')
    if os.path.exists(faces_path):
        faces = np.load(faces_path).astype(np.int32)

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir   = os.path.dirname(args.checkpoint)
        run_dir    = os.path.dirname(ckpt_dir)
        output_dir = os.path.join(run_dir, 'predictions', args.split)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {args.split} dataset...")
    dataset = GarmentDataset(args.data_root, split=args.split, augment=False)
    print(f"  {len(dataset)} samples\n")

    # ── Run prediction ────────────────────────────────────────────────────────
    if args.all_test:
        print(f"Predicting all {args.split} samples...")
        predict_all(model, dataset, device, output_dir,
                    faces=faces, max_samples=args.max_samples)

    elif args.sample:
        print(f"Predicting sample: {args.sample}")
        result = predict_by_name(
            model, args.sample, dataset, device,
            image_path=args.image,
            output_dir=output_dir,
            faces=faces,
        )
        if result is None:
            # Try other splits
            for split in ['train', 'val', 'test']:
                if split == args.split:
                    continue
                print(f"\n  Trying {split} split...")
                other_ds = GarmentDataset(
                    args.data_root, split=split, augment=False)
                result = predict_by_name(
                    model, args.sample, other_ds, device,
                    image_path=args.image,
                    output_dir=output_dir,
                    faces=faces,
                )
                if result is not None:
                    break
    else:
        # No specific sample — predict one example per fabric family
        print("No sample specified — predicting one example per fabric family")
        seen_families = set()
        for i in range(len(dataset)):
            row    = dataset.df.iloc[i]
            family = row['fabric_family']
            if family not in seen_families:
                seen_families.add(family)
                predict_by_name(
                    model, row['sample_name'], dataset, device,
                    output_dir=output_dir,
                    faces=faces,
                )
                print()
            if len(seen_families) == 6:
                break

    print(f"\nAll predictions saved to: {output_dir}")


if __name__ == '__main__':
    main()

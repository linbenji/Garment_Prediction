"""
dataloader_v2.py

PyTorch Geometric dataset for joint ViT + GNN garment drape prediction.

Data flow:
  ViT  : rendered image -> 128-dim style embedding
  GNN  : template mesh + style embedding + SMPL + physics + size -> delta-v
  Loss : MSE(predicted delta-v, gt delta-v) weighted by per_vertex_std

Lean-only training (only lean poses)
Jersey templates excluded (is_baseline == 1) -- zero displacement, not a training target
Template images are still valid ViT inputs via _sample_input_row.

Expected directory layout:
  root_dir/
    dataset_index.csv      # single CSV, one row per sample
    per_vertex_std.npy     # (14117,) per-vertex loss weights
    meshes/                # 1495 .pt files, one per trainable sample
    images/                # 72,000 flat .png files
    template/              # edge_index.npy, uvs.npy, faces.npy
      per_body_pose/       # 75 baseline .npy files
"""

import os
import json
import random
import numpy as np
from collections import defaultdict

import torch
from torch_geometric.data import Dataset, Data
from torchvision import transforms
from PIL import Image
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# 10 physics params -- fFriction (constant 0.03) and fFurBend (constant 0) excluded
PHYSICS_COLS = [
    'fBLeftShearK', 'fBLeftShearK_v2', 'fBRightShearK_v2',
    'fBucklingStiffnessLeftShear', 'fDensity', 'fLeftShearK',
    'fLeftShearK_v2', 'fRightShearK_v2', 'fexpLeftShearK', 'fexpRightShearK',
]

SIZE_ENCODING = {
    'small':  [0.92, 0.96],
    'medium': [1.00, 1.00],
    'large':  [1.08, 1.04],
    'xl':     [1.17, 1.07],
    'xxl':    [1.28, 1.12],
}

CAMERA_ANGLES  = ['000', '045', '090', '135', '180', '225', '270', '315']
COLOR_VARIANTS = ['lightgrey', 'natural', 'navy', 'red', 'sage', 'white']

FABRIC_FAMILY_LABELS = {
    'light_knit': 0, 'medium_knit': 1, 'heavy_knit': 2,
    'light_woven': 3, 'medium_woven': 4, 'heavy_woven': 5,
}
FABRIC_PRESET_LABELS = {
    'jersey': 0, 'mesh_tulle': 1, 'french_terry': 2, 'pique': 3,
    'fleece': 4, 'neoprene_scuba': 5, 'chiffon': 6, 'organza': 7,
    'satin': 8, 'flannel': 9, 'denim': 10, 'canvas': 11,
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class GarmentDataset(Dataset):

    def __init__(self, root_dir, csv_file='dataset_index.csv',
                 split='train', augment=False):
        super().__init__(root=root_dir)
        self.root_dir = root_dir
        self.split    = split
        self.augment  = augment

        csv_path = os.path.join(root_dir, csv_file)
        df = pd.read_csv(csv_path)
        df = df[df['split_group'] == split].reset_index(drop=True)

        if 'is_baseline' in df.columns:
            n_before = len(df)
            df = df[df['is_baseline'] == 0].reset_index(drop=True)
            excluded = n_before - len(df)
            if excluded > 0:
                print(f"[GarmentDataset:{split}] Excluded {excluded} baseline rows")

        self.df = df
        print(f"[GarmentDataset:{split}] {len(self.df)} trainable samples")

        # Pre-load the 5 size templates into RAM
        self.size_templates = {}
        expected_sizes = ['small', 'medium', 'large', 'xl', 'xxl']
        
        for size in expected_sizes:
            # Matches the exact naming convention: body000_lean_jersey_small.pt
            file_name = f"body000_lean_jersey_{size}.pt"
            t_path = os.path.join(root_dir, 'template_sizes', file_name)
            if os.path.exists(t_path):
                # Load the full PyTorch dictionary into RAM
                self.size_templates[size] = torch.load(t_path, weights_only=True)
            else:
                print(f"  WARNING: Missing template for {size} at {t_path}")

        # Full split df (including baselines) for ViT image sampling
        full_df = pd.read_csv(csv_path)
        full_df = full_df[full_df['split_group'] == split].reset_index(drop=True)
        self._full_df = full_df

        # Precompute family index over full df for O(1) ViT input sampling
        self.family_index = defaultdict(list)
        for i, row in full_df.iterrows():
            self.family_index[row['fabric_family']].append(i)

        # Log-normalisation stats for physics params
        self._physics_log_stats = self._compute_physics_stats()

        # Image transforms
        aug_transforms = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.img_transform = transforms.Compose(
            aug_transforms if augment else base_transforms)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]

        # Load preprocessed .pt mesh file
        pt_path = os.path.join(
            self.root_dir, 'meshes', f"{row['sample_name']}.pt")
        mesh = torch.load(pt_path, weights_only=True)

        # Grab the correct base geometry for this shirt size
        size_str = row['garment_size']
        template = self.size_templates[size_str]

        # Target conditioning
        tgt_smpl    = torch.tensor(
            json.loads(row['smpl_betas']), dtype=torch.float32)       # (10,)
        tgt_pose    = torch.zeros(72, dtype=torch.float32)            # (72,) zeros for lean
        tgt_physics = self._get_physics(row)                          # (10,)
        tgt_size    = torch.tensor(
            SIZE_ENCODING[row['garment_size']], dtype=torch.float32)  # (2,)

        # ViT input image -- same fabric family, possibly different body/size
        input_row    = self._sample_input_row(row)
        angle        = random.choice(CAMERA_ANGLES)
        color        = random.choice(COLOR_VARIANTS)
        image_tensor = self._load_image(
            input_row['base_image_name'], angle, color)
        
        # Extract the input state for the ViT
        in_smpl = torch.tensor(
            json.loads(input_row['smpl_betas']), dtype=torch.float32)
        in_size = torch.tensor(
            SIZE_ENCODING[input_row['garment_size']], dtype=torch.float32)
        in_physics = self._get_physics(input_row)

        return Data(
            # Graph structure
            pos        = template['pos'],           # (14117, 3)
            edge_index = template['edge_index'],    # (2, 82988)
            edge_attr  = template['edge_attr'],     # (82988, 4)

            # Node features
            uvs     = template['uvs'],              # (14117, 2)
            normals = template['normals'],          # (14117, 3)

            # Target
            y           = mesh['displacement'], # (14117, 3)  delta-v
            loss_weight = mesh['loss_weight'],  # (14117,)

            # Global conditioning
            tgt_smpl    = tgt_smpl,             # (10,)
            tgt_pose    = tgt_pose,             # (72,)
            tgt_physics = tgt_physics,          # (10,)
            tgt_size    = tgt_size,             # (2,)

            # ViT input (Image + Context)
            image      = image_tensor.unsqueeze(0), # (1, 3, 224, 224)
            in_smpl    = in_smpl,
            in_size    = in_size,
            in_physics = in_physics,

            # Labels
            fabric_family_label = torch.tensor(
                FABRIC_FAMILY_LABELS.get(row['fabric_family'], -1),
                dtype=torch.long),
            fabric_preset_label = torch.tensor(
                FABRIC_PRESET_LABELS.get(row['fabric_preset'], -1),
                dtype=torch.long),

            # Metadata
            body_id     = torch.tensor(int(row['body_id']), dtype=torch.long),
            sample_name = row['sample_name'],
        )

    def _sample_input_row(self, target_row):
        family     = target_row['fabric_family']
        candidates = [
            i for i in self.family_index[family]
            if self._full_df.iloc[i]['sample_name'] != target_row['sample_name']
        ]
        if not candidates:
            return target_row
        return self._full_df.iloc[random.choice(candidates)]

    def _load_image(self, base_name, angle, color):
        fname    = f"{base_name}_{color}_{angle}.png"
        img_path = os.path.join(self.root_dir, 'images', fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return self.img_transform(Image.open(img_path).convert('RGB'))

    def _get_physics(self, row):
        raw = torch.tensor(
            [float(row[col]) for col in PHYSICS_COLS], dtype=torch.float32)
        if self._physics_log_stats is not None:
            log_mean, log_std = self._physics_log_stats
            log_raw = torch.log(raw.clamp(min=1e-10))
            raw     = (log_raw - log_mean) / log_std.clamp(min=1e-8)
        return raw

    def _compute_physics_stats(self):
        missing = [c for c in PHYSICS_COLS if c not in self.df.columns]
        if missing:
            print(f"  WARNING: Physics columns missing: {missing}")
            return None
        vals     = self.df[PHYSICS_COLS].values.astype(np.float32)
        log_vals = np.log(np.clip(vals, 1e-10, None))
        return (torch.from_numpy(log_vals.mean(axis=0)),
                torch.from_numpy(log_vals.std(axis=0)))


# ── Factory ───────────────────────────────────────────────────────────────────

def make_dataloaders(root_dir, batch_size=16, num_workers=4,
                     csv_file='dataset_index.csv'):
    from torch_geometric.loader import DataLoader
    train = GarmentDataset(root_dir, csv_file, split='train', augment=True)
    val   = GarmentDataset(root_dir, csv_file, split='val',   augment=False)
    test  = GarmentDataset(root_dir, csv_file, split='test',  augment=False)
    kw    = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train, shuffle=True,  **kw),
            DataLoader(val,   shuffle=False, **kw),
            DataLoader(test,  shuffle=False, **kw))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 \
           else r'C:\Dev\Clothing_Project\batches\batch_1500_lean'

    print("=" * 65)
    print("DATALOADER SMOKE TEST")
    print("=" * 65)
    print(f"Root: {root}\n")

    # ── Per-split sample count verification ──────────────────────────────────
    print("── Split counts ──")
    EXPECTED = {'train': 1127, 'val': 168, 'test': 180}
    datasets  = {}
    all_ok    = True
    for split, expected in EXPECTED.items():
        ds  = GarmentDataset(root, split=split)
        ok  = len(ds) == expected
        all_ok &= ok
        status = "✓" if ok else "✗"
        print(f"  {status} {split:6s}: {len(ds):4d} samples  (expected {expected})")
        datasets[split] = ds

    # ── Two-axis split leakage check ──────────────────────────────────────────
    print("\n── Split leakage check ──")
    import pandas as pd
    df_train = datasets['train'].df
    df_val   = datasets['val'].df
    df_test  = datasets['test'].df

    # heavy_woven must not appear in train
    hw_in_train = (df_train['fabric_family'] == 'heavy_woven').sum()
    ok = hw_in_train == 0
    all_ok &= ok
    print(f"  {'✓' if ok else '✗'} heavy_woven in train: {hw_in_train} (expected 0)")

    # bodies 23-24 must not appear in train
    unseen_in_train = df_train['body_id'].isin([23, 24]).sum()
    ok = unseen_in_train == 0
    all_ok &= ok
    print(f"  {'✓' if ok else '✗'} bodies 23-24 in train: {unseen_in_train} (expected 0)")

    # test must contain all three generalisation conditions
    FAMILY_GROUP = {
        'light_knit': 1, 'medium_knit': 2, 'heavy_knit': 3,
        'light_woven': 4, 'medium_woven': 5, 'heavy_woven': 6,
    }
    df_test['fg'] = df_test['fabric_family'].map(FAMILY_GROUP)
    seen_body_unseen_mat   = ((df_test['body_id'] <= 22) & (df_test['fg'] == 6)).sum()
    unseen_body_seen_mat   = ((df_test['body_id'] >= 23) & (df_test['fg'] <= 5)).sum()
    unseen_body_unseen_mat = ((df_test['body_id'] >= 23) & (df_test['fg'] == 6)).sum()

    for label, val, exp in [
        ("test seen body/unseen mat",   seen_body_unseen_mat,   100),
        ("test unseen body/seen mat",   unseen_body_seen_mat,    60),
        ("test unseen body/unseen mat", unseen_body_unseen_mat,  20),
    ]:
        ok = val == exp
        all_ok &= ok
        print(f"  {'✓' if ok else '✗'} {label}: {val} (expected {exp})")

    # ── Single sample field check ─────────────────────────────────────────────
    print("\n── Sample field check (train[0]) ──")
    sample = datasets['train'][0]
    EXPECTED_SHAPES = {
        'pos':         (14117, 3),
        'edge_index':  (2, 82988),
        'edge_attr':   (82988, 4),
        'uvs':         (14117, 2),
        'normals':     (14117, 3),
        'y':           (14117, 3),
        'loss_weight': (14117,),
        'tgt_smpl':    (10,),
        'tgt_pose':    (72,),
        'tgt_physics': (10,),
        'tgt_size':    (2,),
        'image':       (1, 3, 224, 224),
    }
    for key, expected_shape in EXPECTED_SHAPES.items():
        val = getattr(sample, key)
        ok  = tuple(val.shape) == expected_shape
        all_ok &= ok
        status = "✓" if ok else "✗"
        print(f"  {status} {key:<25} {str(tuple(val.shape)):<20} "
              f"{'OK' if ok else f'expected {expected_shape}'}")

    # ── Displacement sanity ───────────────────────────────────────────────────
    print("\n── Displacement sanity ──")
    disp = sample.y.norm(dim=1)
    print(f"  mean={disp.mean():.2f}mm  max={disp.max():.2f}mm  "
          f"p99={disp.quantile(0.99):.2f}mm")

    # ── Physics sanity ────────────────────────────────────────────────────────
    print("\n── Physics vector (log-normalised, should be ~N(0,1)) ──")
    p = sample.tgt_physics
    print(f"  mean={p.mean():.3f}  std={p.std():.3f}  "
          f"min={p.min():.3f}  max={p.max():.3f}")

    # ── Image sanity ──────────────────────────────────────────────────────────
    print("\n── Image tensor ──")
    img = sample.image
    print(f"  shape={tuple(img.shape)}  "
          f"min={img.min():.3f}  max={img.max():.3f}  mean={img.mean():.3f}")

    # ── Final result ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_ok:
        print("✓ All smoke tests passed — dataloader ready for training")
    else:
        print("✗ Some checks failed — review output above")
    print("=" * 65)

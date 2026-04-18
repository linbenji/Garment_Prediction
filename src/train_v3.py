"""
train_v3.py

Training loop for MasterDrapeModel (DINO + GNN garment drape prediction)
Method 2 final: DINOv2 + FiLM-Modulated MeshGraphNet

Usage:
    # Debug run (50 samples, 2 epochs — verify pipeline)
    python train_v3.py

    # Full training run
    python train_v3.py --no-debug

    # Resume from checkpoint
    python train_v3.py --no-debug --resume runs/exp_001/best.pt
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import gc
gc.collect()
torch.cuda.empty_cache()

# ── Add project root to path so imports work regardless of working directory ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_v2 import GarmentDataset
from models_v3 import MasterDrapeModel, AutomaticLossWeighter, drape_loss

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = '/workspace/batch_1500_lean'
RUNS_DIR  = '/workspace/runs'

#DATA_ROOT  = r"/Users/Ben/Desktop/batch_1500_lean"
#RUNS_DIR   = r"/Users/Ben/Desktop/runs"

#DATA_ROOT  = r"C:\Dev\Clothing_Project\batches\batch_1500_lean"
#RUNS_DIR   = r"C:\Dev\Clothing_Project\batches\runs"

# ── Debug flag — set True to verify pipeline on small subset ──────────────────
# Runs 2 epochs on 50 samples, no multiprocessing, easier to debug errors.
# Set --no-debug on command line for full training.
DEBUG = False

# ── Hyperparameters ───────────────────────────────────────────────────────────

CONFIG = {
    # Data
    'batch_size':      2  if DEBUG else 4,
    'num_workers':     0  if DEBUG else 4,
    'pin_memory':      False if DEBUG else True,
    'subset_size':     50 if DEBUG else None,

    # Training
    'max_epochs':      2   if DEBUG else 100,
    'early_stop_patience': 15,
    'grad_clip':       1.0,

    # Optimiser (AdamW)
    'lr':              1e-4,
    'weight_decay':    1e-4,

    # Scheduler (ReduceLROnPlateau)
    'lr_patience':     5,
    'lr_factor':       0.5,
    'lr_min':          1e-6,

    # Model
    'embed_dim':       128,
    'latent_dim':      128,
    'gnn_layers':      8,

    # Logging
    'log_every':       10,
    'use_wandb':       True,
    'experiment_name': 'method2_master',
}


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Subset sampler for debug mode ─────────────────────────────────────────────

def make_subset(dataset, n):
    """Randomly sample n indices from a dataset for debug runs."""
    from torch.utils.data import Subset
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return Subset(dataset, indices)


# ── Metric helpers ────────────────────────────────────────────────────────────

def mean_vertex_error(pred_delta, target_delta):
    """
    Mean per-vertex Euclidean error in mm.
    pred_delta, target_delta: (total_nodes, 3)
    Returns scalar.
    """
    return (pred_delta - target_delta).norm(dim=1).mean().item()


def per_condition_error(pred_delta, target_delta, batch, metadata):
    """
    Break down mean vertex error by fabric family and size.
    Used during validation to understand where the model struggles.

    Returns dict: {'canvas': float, 'denim': float, ...}
    """
    errors = {}
    for i, meta in enumerate(metadata):
        mask  = (batch == i)
        err   = (pred_delta[mask] - target_delta[mask]).norm(dim=1).mean().item()
        key   = f"{meta['fabric_family']}_{meta['garment_size']}"
        if key not in errors:
            errors[key] = []
        errors[key].append(err)
    return {k: np.mean(v) for k, v in errors.items()}


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device, config, epoch, logger, amp_dtype, use_scaler, scaler, loss_weighter):
    torch.cuda.empty_cache()
    model.train()

    total_loss   = 0.0
    total_drape  = 0.0
    total_cls    = 0.0
    total_mve    = 0.0
    total_strain = 0.0
    n_batches    = 0
    start_time   = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimiser.zero_grad()

        # Run the forward pass in Mixed Precision (for computational efficiency)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            predicted_delta, fabric_logits = model(batch)

            # Get the raw unweighted losses
            _, d_loss, e_loss, col_loss, c_loss = drape_loss(
                predicted_delta, batch.y, batch.pos, batch.edge_index, 
                batch.loss_weight, fabric_logits, batch.fabric_family_label,
                cls_weight=1.0, strain_weight=1.0 # Set to 1.0 so they are raw values
            )

            # Let the AutomaticWeighter calculate the total loss
            loss = loss_weighter(d_loss, e_loss, c_loss)

        # Check for NaN before backprop
        if torch.isnan(loss):
            print(f"  WARNING: NaN loss at epoch {epoch} batch {batch_idx} — skipping")
            continue

        # Backward pass (conditional)
        if use_scaler:
            # Fallback for older GPUs (uses float16 + Scaler)
            scaler.scale(loss).backward()
            # Unscale the gradients before clipping so the threshold remains accurate
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            # Step the optimizer and update the scaler
            scaler.step(optimiser)
            scaler.update()
        else:
            # Native path for A100 / RTX 5090 (uses bfloat16, no Scaler needed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimiser.step()

        mve = mean_vertex_error(predicted_delta.detach(), batch.y)

        total_loss   += loss.item()
        total_drape  += d_loss.item()
        total_cls    += c_loss.item()
        total_strain += e_loss.item()
        total_mve    += mve
        n_batches    += 1

        # Per-batch logging
        if (batch_idx + 1) % config['log_every'] == 0:
            avg_loss  = total_loss  / n_batches
            avg_mve   = total_mve   / n_batches
            elapsed   = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss={avg_loss:.4f}  mve={avg_mve:.2f}mm  "
                  f"t={elapsed:.1f}s")

            if logger:
                step = (epoch - 1) * len(loader) + batch_idx
                logger.log_train(step, avg_loss, total_drape/n_batches,
                                 total_cls/n_batches, total_strain/n_batches, avg_mve)

    if n_batches == 0:
        return {'loss': float('nan'), 'drape': float('nan'),
                'cls': float('nan'), 'strain': float('nan'), 'mve': float('nan')}

    return {
        'loss':   total_loss   / n_batches,
        'drape':  total_drape  / n_batches,
        'cls':    total_cls    / n_batches,
        'strain': total_strain / n_batches,
        'mve':    total_mve    / n_batches,
    }


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device, config, epoch, logger, loss_weighter, split='val'):
    torch.cuda.empty_cache()
    model.eval()

    total_loss  = 0.0
    total_drape = 0.0
    total_cls   = 0.0
    total_mve   = 0.0
    total_strain = 0.0
    n_batches   = 0

    # Track errors by generalisation condition
    heavy_woven_errs   = []   # unseen material family
    unseen_body_errs   = []   # unseen body shapes
    seen_both_errs     = []   # seen body + seen material (sanity check)

    for batch in loader:
        batch = batch.to(device)

        predicted_delta, fabric_logits = model(batch)

        # Get the raw unweighted losses
        _, d_loss, e_loss, col_loss, c_loss = drape_loss(
            predicted_delta, batch.y, batch.pos, batch.edge_index, 
            batch.loss_weight, fabric_logits, batch.fabric_family_label,
            cls_weight=1.0, strain_weight=1.0 # Set to 1.0 so they are raw values
        )

        # Let the AutomaticWeighter calculate the total loss
        loss = loss_weighter(d_loss, e_loss, c_loss)

        if torch.isnan(loss):
            continue

        mve = mean_vertex_error(predicted_delta, batch.y)

        total_loss   += loss.item()
        total_drape  += d_loss.item()
        total_cls    += c_loss.item()
        total_strain += e_loss.item()
        total_mve    += mve
        n_batches    += 1

        # Break down errors by generalisation condition
        # We access metadata from the batch using the batch index tensor
        for i in range(batch.num_graphs):
            mask       = (batch.batch == i)
            err        = (predicted_delta[mask] - batch.y[mask]).norm(dim=1).mean().item()
            body_id    = batch.body_id[i].item()
            fab_label  = batch.fabric_family_label[i].item()

            # fabric_family_label 5 = heavy_woven (denim, canvas) — unseen
            if fab_label == 5:
                heavy_woven_errs.append(err)
            # bodies 23-24 are unseen
            elif body_id >= 23:
                unseen_body_errs.append(err)
            else:
                seen_both_errs.append(err)

    if n_batches == 0:
        return {'loss': float('nan'), 'drape': float('nan'),
                'cls': float('nan'), 'strain': float('nan'), 'mve': float('nan')}

    results = {
        'loss':   total_loss   / n_batches,
        'drape':  total_drape  / n_batches,
        'cls':    total_cls    / n_batches,
        'strain': total_strain / n_batches,
        'mve':    total_mve    / n_batches,
    }

    if heavy_woven_errs:
        results['mve_heavy_woven'] = np.mean(heavy_woven_errs)
    if unseen_body_errs:
        results['mve_unseen_body'] = np.mean(unseen_body_errs)
    if seen_both_errs:
        results['mve_seen_both']   = np.mean(seen_both_errs)

    if logger:
        logger.log_val(epoch, results, split)

    return results


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    """
    Unified logger supporting TensorBoard (local) and wandb (cloud)
    TensorBoard: open a terminal and run:
        tensorboard --logdir C:/Dev/Clothing_Project/runs
    Then open http://localhost:6006 in browser
    """

    def __init__(self, run_dir, config, use_wandb=False):
        self.use_wandb = use_wandb
        self.run_dir   = run_dir
        self.tb_writer = None

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=run_dir)
            print(f"  TensorBoard logging → {run_dir}")
            print(f"  Run: tensorboard --logdir {os.path.dirname(run_dir)}")
        except ImportError:
            print("  TensorBoard not installed — pip install tensorboard")

        # Weights & Biases
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="garment-drape",
                    name=config['experiment_name'],
                    config=config,
                    dir=run_dir,
                )
                print(f"  wandb logging enabled")
            except ImportError:
                print("  wandb not installed — pip install wandb")
                self.use_wandb = False

    def log_train(self, step, loss, drape, cls, strain, mve):
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss',        loss,  step)
            self.tb_writer.add_scalar('train/drape_loss',  drape, step)
            self.tb_writer.add_scalar('train/cls_loss',    cls,   step)
            self.tb_writer.add_scalar('train/strain_loss', strain, step)
            self.tb_writer.add_scalar('train/mve_mm',      mve,   step)
        if self.use_wandb:
            import wandb
            wandb.log({'train/loss': loss, 'train/drape': drape, 'train/cls': cls,
                       'train/strain': strain, 'train/mve': mve, 'step': step})

    def log_val(self, epoch, results, split='val'):
        if self.tb_writer:
            for k, v in results.items():
                self.tb_writer.add_scalar(f'{split}/{k}', v, epoch)
        if self.use_wandb:
            import wandb
            wandb.log({f'{split}/{k}': v for k, v in results.items()} |
                      {'epoch': epoch})

    def log_lr(self, epoch, lr):
        if self.tb_writer:
            self.tb_writer.add_scalar('train/lr', lr, epoch)
        if self.use_wandb:
            import wandb
            wandb.log({'train/lr': lr, 'epoch': epoch})

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimiser, scheduler, epoch, best_val_loss,
                    config, metrics):
    torch.save({
        'epoch':         epoch,
        'model_state':   model.state_dict(),
        'optim_state':   optimiser.state_dict(),
        'sched_state':   scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config':        config,
        'metrics':       metrics,
    }, path)


def load_checkpoint(path, model, optimiser, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimiser.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])
    print(f"  Resumed from epoch {ckpt['epoch']}  "
          f"best_val_loss={ckpt['best_val_loss']:.6f}")
    return ckpt['epoch'], ckpt['best_val_loss']


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-debug',  action='store_true',
                        help='Run full training (default: debug mode)')
    parser.add_argument('--resume',    type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    args = parser.parse_args()

    debug = DEBUG and not args.no_debug
    if debug:
        print("=" * 65)
        print("DEBUG MODE — 50 samples, 2 epochs")
        print("Run with --no-debug for full training")
        print("=" * 65)

    cfg = CONFIG.copy()
    if debug:
        cfg['batch_size']  = 2
        cfg['num_workers'] = 0
        cfg['subset_size'] = 50
        cfg['max_epochs']  = 2

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
    
    # Determine the best available mixed precision format
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("  Hardware supports bfloat16: Enabled (GradScaler disabled)")
        use_scaler = False
    else:
        amp_dtype = torch.float16
        print("  Hardware requires float16: Enabled (GradScaler enabled)")
        use_scaler = True

    # ── Experiment directory ──────────────────────────────────────────────────
    exp_name  = cfg['experiment_name'] + ('_debug' if debug else '')
    run_dir   = os.path.join(RUNS_DIR, exp_name)
    ckpt_dir  = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\nRun dir: {run_dir}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = GarmentDataset(args.data_root, split='train', augment=not debug)
    val_ds   = GarmentDataset(args.data_root, split='val',   augment=False)

    if cfg['subset_size']:
        train_ds = make_subset(train_ds, cfg['subset_size'])
        val_ds   = make_subset(val_ds,   min(20, len(val_ds)))
        print(f"  Debug subset — train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        persistent_workers=cfg['num_workers'] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory'],
        persistent_workers=cfg['num_workers'] > 0,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = MasterDrapeModel(
        gnn_layers = cfg['gnn_layers'],
        embed_dim  = cfg['embed_dim'],
        latent_dim = cfg['latent_dim'],
    ).to(device)

    # Set the priorities: [Drape, Strain, Classification]
    # e.g. Drape is 100% priority, Strain is a 20% penalty, and Classification is a 10% nudge
    loss_weighter = AutomaticLossWeighter(num_tasks=3, priors=[1.0, 0.2, 0.1]).to(device)

    # Count parameters
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parameters: {total:,} total  {trainable:,} trainable  "
          f"{frozen:,} frozen (DINOv2)")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = AdamW([
        {'params': filter(lambda p: p.requires_grad, model.parameters())},
        {'params': loss_weighter.parameters(), 'weight_decay': 0.0} # No weight decay on loss params
    ], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode     = 'min',
        patience = cfg['lr_patience'],
        factor   = cfg['lr_factor'],
        min_lr   = cfg['lr_min'],
        threshold = 0.01,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = Logger(run_dir, cfg, use_wandb=cfg['use_wandb'])

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float('inf')
    no_improve     = 0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimiser, scheduler, device)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"TRAINING — {cfg['experiment_name']}")
    print(f"  Epochs:     {cfg['max_epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  LR:         {cfg['lr']}")
    print(f"  Device:     {device}")
    print(f"{'='*65}\n")

    history = []

    # Initialize the AMP Scaler (This prevents gradients from rounding down to zero when using 16-bit math)
    # Only used for older GPUs
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, cfg['max_epochs'] + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        train_metrics = train_epoch(
            model, train_loader, optimiser, device, cfg, epoch, logger, amp_dtype, use_scaler, scaler, loss_weighter)

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = val_epoch(
            model, val_loader, device, cfg, epoch, logger, loss_weighter, split='val')

        # ── Scheduler step ────────────────────────────────────────────────────
        scheduler.step(val_metrics['loss'])
        current_lr = optimiser.param_groups[0]['lr']
        logger.log_lr(epoch, current_lr)

        epoch_time = time.time() - epoch_start

        # ── Print epoch summary ───────────────────────────────────────────────
        improved = val_metrics['loss'] < best_val_loss
        marker   = " ← best" if improved else ""

        print(f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_mve={train_metrics['mve']:.2f}mm | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_mve={val_metrics['mve']:.2f}mm | "
              f"lr={current_lr:.2e} | "
              f"t={epoch_time:.1f}s{marker}")

        # Print generalisation breakdown every 10 epochs
        if epoch % 10 == 0 or epoch == cfg['max_epochs']:
            if 'mve_heavy_woven' in val_metrics:
                print(f"  Generalisation:")
                print(f"    heavy_woven (unseen mat):  "
                      f"{val_metrics.get('mve_heavy_woven', float('nan')):.2f}mm")
                print(f"    unseen body:               "
                      f"{val_metrics.get('mve_unseen_body', float('nan')):.2f}mm")
                print(f"    seen body+mat:             "
                      f"{val_metrics.get('mve_seen_both',   float('nan')):.2f}mm")

        # ── Checkpoint ────────────────────────────────────────────────────────
        record = {'epoch': epoch, **train_metrics,
                  **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(record)

        # Save best checkpoint
        if improved:
            best_val_loss = val_metrics['loss']
            no_improve    = 0
            save_checkpoint(
                os.path.join(ckpt_dir, 'best.pt'),
                model, optimiser, scheduler, epoch,
                best_val_loss, cfg, val_metrics,
            )
        else:
            no_improve += 1

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'),
                model, optimiser, scheduler, epoch,
                best_val_loss, cfg, val_metrics,
            )

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= cfg['early_stop_patience'] and not debug:
            print(f"\nEarly stopping — no improvement for "
                  f"{cfg['early_stop_patience']} epochs")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Best checkpoint: {os.path.join(ckpt_dir, 'best.pt')}")

    # Save training history
    history_path = os.path.join(run_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")

    logger.close()
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
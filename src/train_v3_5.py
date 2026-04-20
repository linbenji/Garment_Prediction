"""
train_v3_5.py — PATCHED

Training loop for NonbelieverDrapeModel (DINO + Hierarchical GNN).
Supports normal consistency and Laplacian losses via config flags.

Patches relative to the original:
  - gnn_layers: 8 -> 12 (makes skip indices [3, 7] genuinely mid-stack,
    so the hierarchical fuser receives distinct features rather than
    duplicating the final layer).
  - Sentinel return dicts in train_epoch / val_epoch now match the
    real return schema exactly (no missing 'collision', no phantom
    'bending').
  - build_loss_weighter and weighted_loss share a single source of
    truth for task ordering (_active_task_names + _PRIOR_KEY) so they
    can never drift out of sync.
  - torch.compile switched from mode="reduce-overhead" (CUDA-graphs,
    incompatible with dynamic shapes + pyKeOps + our Python control
    flow) to mode="default".
  - mean_vertex_error now returns a tensor; train-loop running sums
    accumulate on-device and sync once per logging window.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_v2 import GarmentDataset
from models_v3_5 import (
    NonbelieverDrapeModel, AutomaticLossWeighter, drape_loss
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT = '/workspace/batch_1500_lean'
RUNS_DIR  = '/workspace/runs'

#DATA_ROOT  = r"/Users/Ben/Desktop/batch_1500_lean"
#RUNS_DIR   = r"/Users/Ben/Desktop/runs"

#DATA_ROOT  = r"C:\Dev\Clothing_Project\batches\batch_1500_lean"
#RUNS_DIR   = r"C:\Dev\Clothing_Project\batches\runs"

DEBUG = False

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

    # Optimiser
    'lr':              1e-4,
    'weight_decay':    1e-4,

    # Scheduler
    'lr_patience':     5,
    'lr_factor':       0.5,
    'lr_min':          1e-6,

    # Model
    'embed_dim':       128,
    'latent_dim':      128,
    'gnn_layers':      12,   # bumped from 8 — skip indices [3, 7] now mid-stack

    # Surface quality losses (toggle on/off)
    'use_normal_consistency': True,
    'use_laplacian':          True,

    # Loss priors for AutomaticLossWeighter (active-set order lives in
    # _active_task_names below — these are the values it looks up).
    'prior_drape':     1.0,
    'prior_strain':    1.5,
    'prior_normal':    8.0,
    'prior_laplacian': 1.0,
    'prior_collision': 2.0,
    'prior_cls':       0.1,

    # Freeze the weighter so priors are the *final* effective weights
    # (no learned uncertainty adaptation).
    'freeze_weighter': True,

    # Logging
    'log_every':       10,
    'use_wandb':       True,
    'experiment_name': 'model_v3_5',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_subset(dataset, n):
    from torch.utils.data import Subset
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return Subset(dataset, indices)

def mean_vertex_error(pred_delta, target_delta):
    """Return a scalar tensor (no host sync). Call .item() at log time."""
    return (pred_delta - target_delta).norm(dim=1).mean()


# ── Loss-task ordering: single source of truth ───────────────────────────────
#
# Both build_loss_weighter (which registers priors) and weighted_loss (which
# packs per-batch loss tensors) consult _active_task_names(cfg). This makes it
# impossible for the two orderings to drift apart when a new task is added.

_PRIOR_KEY = {
    'drape':     'prior_drape',
    'strain':    'prior_strain',
    'cls':       'prior_cls',
    'collision': 'prior_collision',
    'normal':    'prior_normal',
    'laplacian': 'prior_laplacian',
}

def _active_task_names(cfg):
    names = ['drape', 'strain', 'cls', 'collision']
    if cfg['use_normal_consistency']:
        names.append('normal')
    if cfg.get('use_laplacian', False):
        names.append('laplacian')
    return names

def build_loss_weighter(cfg, device):
    """Build AutomaticLossWeighter with priors matching the active task set.

    Task order matches weighted_loss(): see _active_task_names().
    """
    names  = _active_task_names(cfg)
    priors = [cfg[_PRIOR_KEY[n]] for n in names]
    print(f"  Loss weighter: {len(priors)} tasks — {dict(zip(names, priors))}")
    return AutomaticLossWeighter(num_tasks=len(priors), priors=priors).to(device)

def weighted_loss(loss_weighter, cfg, d_loss, e_loss, c_loss, col_loss, n_loss, lap_loss):
    """Pack active losses in the same order build_loss_weighter used."""
    loss_map = {
        'drape':     d_loss,
        'strain':    e_loss,
        'cls':       c_loss,
        'collision': col_loss,
        'normal':    n_loss,
        'laplacian': lap_loss,
    }
    names = _active_task_names(cfg)
    return loss_weighter(*(loss_map[n] for n in names))


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device, config, epoch, logger,
                amp_dtype, use_scaler, scaler, loss_weighter,
                faces_t, get_body_data):
    torch.cuda.empty_cache()
    model.train()

    # Accumulate running sums on-device; sync once per log window / epoch end.
    total_loss   = torch.zeros((), device=device)
    total_drape  = torch.zeros((), device=device)
    total_cls    = torch.zeros((), device=device)
    total_strain = torch.zeros((), device=device)
    total_normal = torch.zeros((), device=device)
    total_lap    = torch.zeros((), device=device)
    total_col    = torch.zeros((), device=device)
    total_mve    = torch.zeros((), device=device)
    n_batches    = 0
    start_time   = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimiser.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            predicted_delta, fabric_logits = model(batch)

            d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss = drape_loss(
                predicted_delta, batch.y, batch.pos, batch.edge_index,
                batch.loss_weight, fabric_logits, batch.fabric_family_label,
                batch_idx=batch.batch, faces=faces_t,
                body_ids=batch.body_id, get_body_data=get_body_data,
                use_normal_consistency=config['use_normal_consistency'],
                use_laplacian=config.get('use_laplacian', False),
            )

            loss = weighted_loss(loss_weighter, config,
                                 d_loss, e_loss, c_loss, col_loss, n_loss, lap_loss)

        # NaN guard still forces one sync; that's unavoidable.
        if torch.isnan(loss):
            print(f"  WARNING: NaN loss at epoch {epoch} batch {batch_idx} — skipping")
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimiser.step()

        with torch.no_grad():
            mve = mean_vertex_error(predicted_delta.detach(), batch.y)
            total_loss   += loss.detach()
            total_drape  += d_loss.detach()
            total_cls    += c_loss.detach()
            total_strain += e_loss.detach()
            total_normal += n_loss.detach()
            total_lap    += lap_loss.detach()
            total_col    += col_loss.detach()
            total_mve    += mve.detach()
        n_batches += 1

        if (batch_idx + 1) % config['log_every'] == 0:
            # Single sync for all running averages.
            avg_loss   = (total_loss   / n_batches).item()
            avg_drape  = (total_drape  / n_batches).item()
            avg_cls    = (total_cls    / n_batches).item()
            avg_strain = (total_strain / n_batches).item()
            avg_normal = (total_normal / n_batches).item()
            avg_lap    = (total_lap    / n_batches).item()
            avg_col    = (total_col    / n_batches).item()
            avg_mve    = (total_mve    / n_batches).item()
            elapsed    = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss={avg_loss:.4f}  mve={avg_mve:.2f}mm  col={avg_col:.4f}mm t={elapsed:.1f}s")

            if logger:
                step = (epoch - 1) * len(loader) + batch_idx
                logger.log_train(step, avg_loss, avg_drape, avg_cls, avg_strain,
                                 avg_mve, avg_normal, avg_lap, avg_col)

    # Sentinel fallback MUST match the regular return schema below.
    empty_keys = ['loss', 'drape', 'cls', 'strain', 'normal',
                  'laplacian', 'collision', 'mve']
    if n_batches == 0:
        return {k: float('nan') for k in empty_keys}

    return {
        'loss':      (total_loss   / n_batches).item(),
        'drape':     (total_drape  / n_batches).item(),
        'cls':       (total_cls    / n_batches).item(),
        'strain':    (total_strain / n_batches).item(),
        'normal':    (total_normal / n_batches).item(),
        'laplacian': (total_lap    / n_batches).item(),
        'collision': (total_col    / n_batches).item(),
        'mve':       (total_mve    / n_batches).item(),
    }


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device, config, epoch, logger, loss_weighter,
              faces_t, get_body_data, amp_dtype, split='val'):
    torch.cuda.empty_cache()
    model.eval()

    total_loss   = torch.zeros((), device=device)
    total_drape  = torch.zeros((), device=device)
    total_cls    = torch.zeros((), device=device)
    total_strain = torch.zeros((), device=device)
    total_normal = torch.zeros((), device=device)
    total_lap    = torch.zeros((), device=device)
    total_col    = torch.zeros((), device=device)
    total_mve    = torch.zeros((), device=device)
    n_batches    = 0

    heavy_woven_errs = []
    unseen_body_errs = []
    seen_both_errs   = []

    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            predicted_delta, fabric_logits = model(batch)

        d_loss, e_loss, col_loss, n_loss, lap_loss, c_loss = drape_loss(
            predicted_delta, batch.y, batch.pos, batch.edge_index,
            batch.loss_weight, fabric_logits, batch.fabric_family_label,
            batch_idx=batch.batch, faces=faces_t,
            body_ids=batch.body_id, get_body_data=get_body_data,
            use_normal_consistency=config['use_normal_consistency'],
            use_laplacian=config.get('use_laplacian', False),
        )

        loss = weighted_loss(loss_weighter, config,
                             d_loss, e_loss, c_loss, col_loss, n_loss, lap_loss)

        if torch.isnan(loss):
            continue

        mve = mean_vertex_error(predicted_delta, batch.y)

        total_loss   += loss.detach()
        total_drape  += d_loss.detach()
        total_cls    += c_loss.detach()
        total_strain += e_loss.detach()
        total_normal += n_loss.detach()
        total_lap    += lap_loss.detach()
        total_col    += col_loss.detach()
        total_mve    += mve.detach()
        n_batches    += 1

        # Per-subset diagnostic errors (small, so per-sample .item() is fine)
        for i in range(batch.num_graphs):
            mask      = (batch.batch == i)
            err       = (predicted_delta[mask] - batch.y[mask]).norm(dim=1).mean().item()
            body_id   = batch.body_id[i].item()
            fab_label = batch.fabric_family_label[i].item()
            if fab_label == 5:
                heavy_woven_errs.append(err)
            elif body_id >= 23:
                unseen_body_errs.append(err)
            else:
                seen_both_errs.append(err)

    # Sentinel fallback MUST match the regular return schema below.
    empty_keys = ['loss', 'drape', 'cls', 'strain', 'normal',
                  'laplacian', 'collision', 'mve']
    if n_batches == 0:
        return {k: float('nan') for k in empty_keys}

    results = {
        'loss':      (total_loss   / n_batches).item(),
        'drape':     (total_drape  / n_batches).item(),
        'cls':       (total_cls    / n_batches).item(),
        'strain':    (total_strain / n_batches).item(),
        'normal':    (total_normal / n_batches).item(),
        'laplacian': (total_lap    / n_batches).item(),
        'collision': (total_col    / n_batches).item(),
        'mve':       (total_mve    / n_batches).item(),
    }

    if heavy_woven_errs:
        results['mve_heavy_woven'] = float(np.mean(heavy_woven_errs))
    if unseen_body_errs:
        results['mve_unseen_body'] = float(np.mean(unseen_body_errs))
    if seen_both_errs:
        results['mve_seen_both']   = float(np.mean(seen_both_errs))

    if logger:
        logger.log_val(epoch, results, split)

    return results


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, run_dir, config, use_wandb=False):
        self.use_wandb = use_wandb
        self.run_dir   = run_dir
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=run_dir)
            print(f"  TensorBoard logging → {run_dir}")
        except ImportError:
            print("  TensorBoard not installed")
        if use_wandb:
            try:
                import wandb
                wandb.init(project="garment-drape", name=config['experiment_name'],
                           config=config, dir=run_dir)
                print(f"  wandb logging enabled")
            except ImportError:
                print("  wandb not installed")
                self.use_wandb = False

    def log_train(self, step, loss, drape, cls, strain, mve,
                  normal=0, laplacian=0, collision=0):
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss',           loss,      step)
            self.tb_writer.add_scalar('train/drape_loss',     drape,     step)
            self.tb_writer.add_scalar('train/cls_loss',       cls,       step)
            self.tb_writer.add_scalar('train/strain_loss',    strain,    step)
            self.tb_writer.add_scalar('train/normal_loss',    normal,    step)
            self.tb_writer.add_scalar('train/laplacian_loss', laplacian, step)
            self.tb_writer.add_scalar('train/collision_loss', collision, step)
            self.tb_writer.add_scalar('train/mve_mm',         mve,       step)
        if self.use_wandb:
            import wandb
            wandb.log({'train/loss': loss, 'train/drape': drape, 'train/cls': cls,
                       'train/strain': strain, 'train/normal': normal,
                       'train/laplacian': laplacian, 'train/collision': collision,
                       'train/mve': mve, 'step': step})

    def log_val(self, epoch, results, split='val'):
        if self.tb_writer:
            for k, v in results.items():
                self.tb_writer.add_scalar(f'{split}/{k}', v, epoch)
        if self.use_wandb:
            import wandb
            wandb.log({f'{split}/{k}': v for k, v in results.items()} | {'epoch': epoch})

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

def save_checkpoint(path, model, optimiser, scheduler, loss_weighter, epoch, best_val_mve, config, metrics):
    torch.save({
        'epoch': epoch, 'model_state': model.state_dict(),
        'optim_state': optimiser.state_dict(), 'sched_state': scheduler.state_dict(),
        'loss_weighter_state': loss_weighter.state_dict(),
        'best_val_mve': best_val_mve,
        # keep legacy key for any downstream tooling that still reads it
        'best_val_loss': best_val_mve,
        'config': config, 'metrics': metrics,
    }, path)

def load_checkpoint(path, model, optimiser, scheduler, loss_weighter, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimiser.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])

    if 'loss_weighter_state' in ckpt:
        loss_weighter.load_state_dict(ckpt['loss_weighter_state'])

    best = ckpt.get('best_val_mve', ckpt.get('best_val_loss', float('inf')))
    print(f"  Resumed from epoch {ckpt['epoch']}  best_val_mve={best:.6f}")
    return ckpt['epoch'], best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-debug',  action='store_true')
    parser.add_argument('--resume',    type=str, default=None)
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    args = parser.parse_args()

    debug = DEBUG and not args.no_debug
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
        torch.backends.cudnn.benchmark = True

    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True

    exp_name = cfg['experiment_name'] + ('_debug' if debug else '')
    run_dir  = os.path.join(RUNS_DIR, exp_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── Load Template Faces ───────────────────────────────────────────────────
    print("\nLoading template faces...")
    faces_np = np.load(os.path.join(args.data_root, 'template', 'faces.npy'))
    faces_t  = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    print(f"  Faces: {faces_t.shape}")

    # ── Precompute Body Cache ─────────────────────────────────────────────────
    print("\nInitializing Body Cache for Collision...")
    body_cache = {}
    def get_body_data(body_id, device):
        if body_id not in body_cache:
            path = os.path.join(args.data_root, 'template', 'bodies', f'body_{body_id:03d}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing mannequin file for collision: {path}")
            data = torch.load(path, map_location=device, weights_only=True)
            body_cache[body_id] = (data['pos'], data['normals'])
        return body_cache[body_id]

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = GarmentDataset(args.data_root, split='train', augment=not debug)
    val_ds   = GarmentDataset(args.data_root, split='val',   augment=False)

    if cfg['subset_size']:
        train_ds = make_subset(train_ds, cfg['subset_size'])
        val_ds   = make_subset(val_ds,   min(20, len(val_ds)))

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                              persistent_workers=cfg['num_workers'] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory'],
                            persistent_workers=cfg['num_workers'] > 0)
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = NonbelieverDrapeModel(
        gnn_layers=cfg['gnn_layers'], embed_dim=cfg['embed_dim'], latent_dim=cfg['latent_dim'],
    ).to(device)

    # mode="default" — reduce-overhead uses CUDA graphs, which don't
    # capture dynamic shapes, pyKeOps kernels, or .item() control flow.
    model = torch.compile(model, mode="default")

    loss_weighter = build_loss_weighter(cfg, device)

    if cfg.get('freeze_weighter', False):
        for p in loss_weighter.parameters():
            p.requires_grad = False
        print(f"  Loss weighter FROZEN — priors are now static weights")
    else:
        print(f"  Loss weighter trainable — priors are initial values only "
              f"(weighter will adapt toward ~1/loss equilibrium)")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parameters: {total:,} total  {trainable:,} trainable  {frozen:,} frozen")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optim_groups = [
        {'params': list(filter(lambda p: p.requires_grad, model.parameters()))},
    ]
    if not cfg.get('freeze_weighter', False):
        optim_groups.append({'params': loss_weighter.parameters(), 'weight_decay': 0.0})
    optimiser = AdamW(optim_groups, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # Scheduler monitors val_mve (mm-space) instead of val_loss — val_loss is
    # inflated by val_cls climbing on held-out heavy_woven fabrics.
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=cfg['lr_patience'],
                                  factor=cfg['lr_factor'], min_lr=cfg['lr_min'], threshold=0.01)

    logger = Logger(run_dir, cfg, use_wandb=cfg['use_wandb'])

    start_epoch  = 1
    best_val_mve = float('inf')
    no_improve   = 0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_mve = load_checkpoint(args.resume, model, optimiser, scheduler, loss_weighter, device)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"TRAINING — {cfg['experiment_name']}{'  [DEBUG]' if debug else ''}")
    print(f"  GNN layers:         {cfg['gnn_layers']}")
    print(f"  Normal consistency: {'ON' if cfg['use_normal_consistency'] else 'OFF'}")
    print(f"  Laplacian:          {'ON' if cfg.get('use_laplacian', False) else 'OFF'}")
    print(f"  Early-stop + LR scheduler monitor: val_mve (mm)")
    print(f"{'='*65}\n")

    history = []
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, cfg['max_epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimiser, device, cfg, epoch,
                                    logger, amp_dtype, use_scaler, scaler, loss_weighter,
                                    faces_t, get_body_data)

        val_metrics = val_epoch(model, val_loader, device, cfg, epoch, logger, loss_weighter,
                                faces_t, get_body_data, amp_dtype, split='val')

        scheduler.step(val_metrics['mve'])
        current_lr = optimiser.param_groups[0]['lr']
        logger.log_lr(epoch, current_lr)
        epoch_time = time.time() - epoch_start

        improved = val_metrics['mve'] < best_val_mve
        marker   = " ← best" if improved else ""

        print(f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
              f"train_loss={train_metrics['loss']:.4f} mve={train_metrics['mve']:.2f}mm | "
              f"val_loss={val_metrics['loss']:.4f} mve={val_metrics['mve']:.2f}mm | "
              f"lr={current_lr:.2e} t={epoch_time:.1f}s{marker}")

        if epoch % 10 == 0 or epoch == cfg['max_epochs']:
            print(f"           train d={train_metrics['drape']:.3f} s={train_metrics['strain']:.3f} "
                  f"n={train_metrics['normal']:.4f} "
                  f"lap={train_metrics['laplacian']:.4f} cls={train_metrics['cls']:.3f} | "
                  f"val d={val_metrics['drape']:.3f} s={val_metrics['strain']:.3f} "
                  f"n={val_metrics['normal']:.4f} "
                  f"lap={val_metrics['laplacian']:.4f} cls={val_metrics['cls']:.3f}")
            if 'mve_heavy_woven' in val_metrics:
                print(f"  heavy_woven: {val_metrics.get('mve_heavy_woven', float('nan')):.2f}mm  "
                      f"unseen_body: {val_metrics.get('mve_unseen_body', float('nan')):.2f}mm  "
                      f"seen_both: {val_metrics.get('mve_seen_both', float('nan')):.2f}mm")

        record = {'epoch': epoch, **train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(record)

        if improved:
            best_val_mve = val_metrics['mve']
            no_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), model, optimiser, scheduler, loss_weighter,
                            epoch, best_val_mve, cfg, val_metrics)
        else:
            no_improve += 1

        if epoch % 10 == 0:
            save_checkpoint(os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'), model, optimiser,
                            scheduler, loss_weighter, epoch, best_val_mve, cfg, val_metrics)

        if no_improve >= cfg['early_stop_patience'] and not debug:
            print(f"\nEarly stopping — no improvement in val_mve for {cfg['early_stop_patience']} epochs")
            break

    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE — Best val_mve: {best_val_mve:.4f} mm")
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    logger.close()
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
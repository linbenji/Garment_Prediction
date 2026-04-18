"""
train_v4.py

Training loop for MasterDrapeModel v4 (DINO + FiLM + CLS Cross-Attention)
Supports normal consistency and bending energy losses via config flags.

Usage:
    python train_v4.py --no-debug
    python train_v4.py --no-debug --resume runs/method3_crossattn/checkpoints/best.pt
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
from models_v4 import MasterDrapeModel, AutomaticLossWeighter, drape_loss, build_face_adjacency

# ── Config ────────────────────────────────────────────────────────────────────

#DATA_ROOT = '/workspace/batch_1500_lean'
#RUNS_DIR  = '/workspace/runs'

DATA_ROOT  = r"/Users/Ben/Desktop/batch_1500_lean"
RUNS_DIR   = r"/Users/Ben/Desktop/runs"

#DATA_ROOT  = r"C:\Dev\Clothing_Project\batches\batch_1500_lean"
#RUNS_DIR   = r"C:\Dev\Clothing_Project\batches\runs"

DEBUG = True

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
    'embed_dim':          128,
    'latent_dim':         128,
    'gnn_layers':         8,
    'cross_attn_layers':  [3, 7],

    # Surface quality losses (toggle on/off)
    'use_normal_consistency': True,
    'use_bending_energy':     False,

    # Loss priors for AutomaticLossWeighter
    'prior_drape':    1.0,
    'prior_strain':   0.2,
    'prior_cls':      0.1,
    'prior_normal':   0.5,
    'prior_bending':  0.3,

    # Logging
    'log_every':       10,
    'use_wandb':       True,
    'experiment_name': 'method3_crossattn',
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
    return (pred_delta - target_delta).norm(dim=1).mean().item()

def build_loss_weighter(cfg, device):
    priors = [cfg['prior_drape'], cfg['prior_strain'], cfg['prior_cls']]
    names  = ['drape', 'strain', 'cls']
    if cfg['use_normal_consistency']:
        priors.append(cfg['prior_normal'])
        names.append('normal')
    if cfg['use_bending_energy']:
        priors.append(cfg['prior_bending'])
        names.append('bending')
    print(f"  Loss weighter: {len(priors)} tasks — {dict(zip(names, priors))}")
    return AutomaticLossWeighter(num_tasks=len(priors), priors=priors).to(device)

def weighted_loss(loss_weighter, cfg, d_loss, e_loss, c_loss, n_loss, b_loss):
    losses = [d_loss, e_loss, c_loss]
    if cfg['use_normal_consistency']:
        losses.append(n_loss)
    if cfg['use_bending_energy']:
        losses.append(b_loss)
    return loss_weighter(*losses)


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device, config, epoch, logger,
                amp_dtype, use_scaler, scaler, loss_weighter,
                faces_t, face_adj, shared_edges):
    torch.cuda.empty_cache()
    model.train()

    total_loss   = 0.0
    total_drape  = 0.0
    total_cls    = 0.0
    total_strain = 0.0
    total_normal = 0.0
    total_bend   = 0.0
    total_mve    = 0.0
    n_batches    = 0
    start_time   = time.time()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimiser.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            predicted_delta, fabric_logits = model(batch)

            _, d_loss, e_loss, col_loss, n_loss, b_loss, c_loss = drape_loss(
                predicted_delta, batch.y, batch.pos, batch.edge_index,
                batch.loss_weight, fabric_logits, batch.fabric_family_label,
                batch_idx=batch.batch, faces=faces_t, face_adj=face_adj,
                shared_edges=shared_edges,
                use_normal_consistency=config['use_normal_consistency'],
                use_bending_energy=config['use_bending_energy'],
                cls_weight=1.0, strain_weight=1.0, normal_weight=1.0, bending_weight=1.0,
            )

            loss = weighted_loss(loss_weighter, config, d_loss, e_loss, c_loss, n_loss, b_loss)

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

        mve = mean_vertex_error(predicted_delta.detach(), batch.y)

        total_loss   += loss.item()
        total_drape  += d_loss.item()
        total_cls    += c_loss.item()
        total_strain += e_loss.item()
        total_normal += n_loss.item()
        total_bend   += b_loss.item()
        total_mve    += mve
        n_batches    += 1

        if (batch_idx + 1) % config['log_every'] == 0:
            avg_loss = total_loss / n_batches
            avg_mve  = total_mve  / n_batches
            elapsed  = time.time() - start_time
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(loader)} | "
                  f"loss={avg_loss:.4f}  mve={avg_mve:.2f}mm  t={elapsed:.1f}s")

            if logger:
                step = (epoch - 1) * len(loader) + batch_idx
                logger.log_train(step, avg_loss, total_drape/n_batches,
                                 total_cls/n_batches, total_strain/n_batches,
                                 avg_mve, total_normal/n_batches, total_bend/n_batches)

    if n_batches == 0:
        return {k: float('nan') for k in ['loss','drape','cls','strain','normal','bending','mve']}

    return {
        'loss':    total_loss   / n_batches,
        'drape':   total_drape  / n_batches,
        'cls':     total_cls    / n_batches,
        'strain':  total_strain / n_batches,
        'normal':  total_normal / n_batches,
        'bending': total_bend   / n_batches,
        'mve':     total_mve    / n_batches,
    }


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device, config, epoch, logger, loss_weighter,
              faces_t, face_adj, shared_edges, split='val'):
    torch.cuda.empty_cache()
    model.eval()

    total_loss   = 0.0
    total_drape  = 0.0
    total_cls    = 0.0
    total_strain = 0.0
    total_normal = 0.0
    total_bend   = 0.0
    total_mve    = 0.0
    n_batches    = 0

    heavy_woven_errs = []
    unseen_body_errs = []
    seen_both_errs   = []

    for batch in loader:
        batch = batch.to(device)
        predicted_delta, fabric_logits = model(batch)

        _, d_loss, e_loss, col_loss, n_loss, b_loss, c_loss = drape_loss(
            predicted_delta, batch.y, batch.pos, batch.edge_index,
            batch.loss_weight, fabric_logits, batch.fabric_family_label,
            batch_idx=batch.batch, faces=faces_t, face_adj=face_adj,
            shared_edges=shared_edges,
            use_normal_consistency=config['use_normal_consistency'],
            use_bending_energy=config['use_bending_energy'],
            cls_weight=1.0, strain_weight=1.0, normal_weight=1.0, bending_weight=1.0,
        )

        loss = weighted_loss(loss_weighter, config, d_loss, e_loss, c_loss, n_loss, b_loss)

        if torch.isnan(loss):
            continue

        mve = mean_vertex_error(predicted_delta, batch.y)

        total_loss   += loss.item()
        total_drape  += d_loss.item()
        total_cls    += c_loss.item()
        total_strain += e_loss.item()
        total_normal += n_loss.item()
        total_bend   += b_loss.item()
        total_mve    += mve
        n_batches    += 1

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

    if n_batches == 0:
        return {k: float('nan') for k in ['loss','drape','cls','strain','normal','bending','mve']}

    results = {
        'loss':    total_loss   / n_batches,
        'drape':   total_drape  / n_batches,
        'cls':     total_cls    / n_batches,
        'strain':  total_strain / n_batches,
        'normal':  total_normal / n_batches,
        'bending': total_bend   / n_batches,
        'mve':     total_mve    / n_batches,
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

    def log_train(self, step, loss, drape, cls, strain, mve, normal=0, bending=0):
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss',         loss,    step)
            self.tb_writer.add_scalar('train/drape_loss',   drape,   step)
            self.tb_writer.add_scalar('train/cls_loss',     cls,     step)
            self.tb_writer.add_scalar('train/strain_loss',  strain,  step)
            self.tb_writer.add_scalar('train/normal_loss',  normal,  step)
            self.tb_writer.add_scalar('train/bending_loss', bending, step)
            self.tb_writer.add_scalar('train/mve_mm',       mve,     step)
        if self.use_wandb:
            import wandb
            wandb.log({'train/loss': loss, 'train/drape': drape, 'train/cls': cls,
                       'train/strain': strain, 'train/normal': normal,
                       'train/bending': bending, 'train/mve': mve, 'step': step})

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

def save_checkpoint(path, model, optimiser, scheduler, loss_weighter, epoch, best_val_loss, config, metrics):
    torch.save({
        'epoch': epoch, 'model_state': model.state_dict(),
        'optim_state': optimiser.state_dict(), 'sched_state': scheduler.state_dict(),
        'loss_weighter_state': loss_weighter.state_dict(),
        'best_val_loss': best_val_loss, 'config': config, 'metrics': metrics,
    }, path)

def load_checkpoint(path, model, optimiser, scheduler, loss_weighter, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimiser.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])

    # Load the loss weighter state if it exists (for backward compatibility)
    if 'loss_weighter_state' in ckpt:
        loss_weighter.load_state_dict(ckpt['loss_weighter_state'])

    print(f"  Resumed from epoch {ckpt['epoch']}  best_val_loss={ckpt['best_val_loss']:.6f}")
    return ckpt['epoch'], ckpt['best_val_loss']


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

    # ── Precompute face adjacency ─────────────────────────────────────────────
    print("\nPrecomputing face adjacency...")
    faces_np = np.load(os.path.join(args.data_root, 'template', 'faces.npy'))
    faces_t  = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    face_adj, shared_edges = build_face_adjacency(faces_np)
    face_adj     = face_adj.to(device)
    shared_edges = shared_edges.to(device)
    print(f"  Faces: {faces_t.shape}  Adjacent pairs: {face_adj.shape[0]}")

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
    model = MasterDrapeModel(
        gnn_layers=cfg['gnn_layers'], embed_dim=cfg['embed_dim'], latent_dim=cfg['latent_dim'],
        cross_attn_layers=cfg['cross_attn_layers'],
    ).to(device)

    loss_weighter = build_loss_weighter(cfg, device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"  Parameters: {total:,} total  {trainable:,} trainable  {frozen:,} frozen")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimiser = AdamW([
        {'params': filter(lambda p: p.requires_grad, model.parameters())},
        {'params': loss_weighter.parameters(), 'weight_decay': 0.0}
    ], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=cfg['lr_patience'],
                                  factor=cfg['lr_factor'], min_lr=cfg['lr_min'], threshold=0.01)

    logger = Logger(run_dir, cfg, use_wandb=cfg['use_wandb'])

    start_epoch   = 1
    best_val_loss = float('inf')
    no_improve    = 0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimiser, scheduler, loss_weighter, device)
        start_epoch += 1

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"TRAINING — {cfg['experiment_name']}")
    print(f"  Normal consistency: {'ON' if cfg['use_normal_consistency'] else 'OFF'}")
    print(f"  Bending energy:     {'ON' if cfg['use_bending_energy'] else 'OFF'}")
    print(f"{'='*65}\n")

    history = []
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, cfg['max_epochs'] + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimiser, device, cfg, epoch,
                                    logger, amp_dtype, use_scaler, scaler, loss_weighter,
                                    faces_t, face_adj, shared_edges)

        val_metrics = val_epoch(model, val_loader, device, cfg, epoch, logger, loss_weighter,
                                faces_t, face_adj, shared_edges, split='val')

        scheduler.step(val_metrics['loss'])
        current_lr = optimiser.param_groups[0]['lr']
        logger.log_lr(epoch, current_lr)
        epoch_time = time.time() - epoch_start

        improved = val_metrics['loss'] < best_val_loss
        marker   = " ← best" if improved else ""

        print(f"Epoch {epoch:3d}/{cfg['max_epochs']} | "
              f"train_loss={train_metrics['loss']:.4f} mve={train_metrics['mve']:.2f}mm | "
              f"val_loss={val_metrics['loss']:.4f} mve={val_metrics['mve']:.2f}mm | "
              f"lr={current_lr:.2e} t={epoch_time:.1f}s{marker}")

        if epoch % 10 == 0 or epoch == cfg['max_epochs']:
            if 'mve_heavy_woven' in val_metrics:
                print(f"  heavy_woven: {val_metrics.get('mve_heavy_woven', float('nan')):.2f}mm  "
                      f"unseen_body: {val_metrics.get('mve_unseen_body', float('nan')):.2f}mm  "
                      f"seen_both: {val_metrics.get('mve_seen_both', float('nan')):.2f}mm")

        record = {'epoch': epoch, **train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(record)

        if improved:
            best_val_loss = val_metrics['loss']
            no_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), model, optimiser, scheduler, loss_weighter,
                           epoch, best_val_loss, cfg, val_metrics)
        else:
            no_improve += 1

        if epoch % 10 == 0:
            save_checkpoint(os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'), model, optimiser,
                           scheduler, loss_weighter, epoch, best_val_loss, cfg, val_metrics)

        if no_improve >= cfg['early_stop_patience'] and not debug:
            print(f"\nEarly stopping — no improvement for {cfg['early_stop_patience']} epochs")
            break

    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE — Best val loss: {best_val_loss:.6f}")
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    logger.close()
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
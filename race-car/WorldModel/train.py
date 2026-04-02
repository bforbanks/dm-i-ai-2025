#!/usr/bin/env python3
"""
WorldModel training script.

Run from project root (dm-i-ai-2025/):

    python race-car/WorldModel/train.py \
        --model model_a \
        --data  laneshift_dataset.npz \
        --project laneshift-worldmodel \
        --run-name model_a_run1

All models (model_a, model_a_lg, model_b, model_b_lg, model_c, model_d)
are supported via --model.  See race-car/WorldModel/model.py for descriptions.

Architecture summary:
    A/A_lg  GRU belief propagator (CNN on prev_pos, GRU, FFN)
    B/B_lg  Chunked-history transformer (no recurrent state; uses sensor history)
    C       Hybrid GRU + inter-lane transformer
    D       Feedforward with output feedback (no GRU)

Training details:
    - Segments of T_seg ticks are processed per batch (stateless TBPTT)
    - State is reset to zero at the start of each segment (simplifies batching)
    - ModelB processes all T ticks in one forward pass (no recurrence)
    - Models A/C/D process tick-by-tick with autoregressive state propagation
    - Loss = CrossEntropy(pos) + lambda_vel * Huber(vel, masked to cars present)
    - Early stopping on val loss with patience = 5 epochs
    - Best checkpoint saved to --out-dir / {run_name}_best.pt
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make sure race-car/ is importable when run from project root
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RACECAR_DIR  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RACECAR_DIR)

from WorldModel.model   import build_model, car_x_to_bin, MODEL_CONFIGS, count_params, BINS, N_LANES
from WorldModel.dataset import LaneShiftDataset

# ── optional wandb ────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    print("wandb not installed — logging to stdout only.", flush=True)


# ── loss functions ────────────────────────────────────────────────────────────

def position_loss(pos_logits: torch.Tensor, car_x: torch.Tensor) -> torch.Tensor:
    """
    CrossEntropy over BINS+1 classes (BINS = no-car class).
    pos_logits : [B, T, 5, BINS+1]
    car_x      : [B, T, 5]  float (NaN = no car)
    """
    B, T, _, _ = pos_logits.shape
    logits_flat = pos_logits.reshape(B * T * N_LANES, BINS + 1)
    targets     = car_x_to_bin(car_x.reshape(B * T * N_LANES))
    return F.cross_entropy(logits_flat, targets)


def velocity_loss(vel_pred: torch.Tensor, car_vx: torch.Tensor) -> torch.Tensor:
    """
    Huber loss on lanes where a car is present (car_vx not NaN).
    vel_pred : [B, T, 5]
    car_vx   : [B, T, 5]  (NaN = no car)
    """
    mask = ~torch.isnan(car_vx)
    if not mask.any():
        return vel_pred.new_zeros(())
    return F.huber_loss(vel_pred[mask], car_vx[mask], delta=5.0)


# ── forward pass helpers ──────────────────────────────────────────────────────

def run_recurrent_segment(model, sensors_seg, ego_y_seg, state, device):
    """
    Process T ticks tick-by-tick (for models with recurrent state).

    sensors_seg : [B, T, 16]
    ego_y_seg   : [B, T, 1]

    Returns:
        pos_logits : [B, T, 5, BINS+1]
        vel_pred   : [B, T, 5]
        state      : updated (still in computation graph for BPTT)
    """
    B, T, _ = sensors_seg.shape
    pos_list, vel_list = [], []

    for t in range(T):
        pl, vp, state = model(
            sensors_seg[:, t, :].to(device),
            ego_y_seg[:, t, :].to(device),
            state,
        )
        pos_list.append(pl)
        vel_list.append(vp)

    pos_logits = torch.stack(pos_list, dim=1)   # [B, T, 5, 65]
    vel_pred   = torch.stack(vel_list, dim=1)   # [B, T, 5]
    return pos_logits, vel_pred, state



# ── training / eval loops ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, lambda_vel):
    model.train()
    total_loss = total_pos = total_vel = 0.0
    n_batches  = 0

    for batch in loader:
        sensors = batch['sensors']  # [B, T, 16]
        car_x   = batch['car_x']    # [B, T, 5]
        car_vx  = batch['car_vx']   # [B, T, 5]
        ego_y   = batch['ego_y']    # [B, T, 1]
        B, T, _ = sensors.shape

        state     = model.init_state(B, device)
        sensors_d = sensors.to(device)
        ego_y_d   = ego_y.to(device)
        car_x_d   = car_x.to(device)
        car_vx_d  = car_vx.to(device)
        pos_logits, vel_pred, _ = run_recurrent_segment(
            model, sensors_d, ego_y_d, state, device)

        l_pos = position_loss(pos_logits, car_x_d)
        l_vel = velocity_loss(vel_pred, car_vx_d)
        loss  = l_pos + lambda_vel * l_vel

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos  += l_pos.item()
        total_vel  += l_vel.item()
        n_batches  += 1

    if scheduler is not None:
        scheduler.step()

    return {
        'loss': total_loss / n_batches,
        'pos':  total_pos  / n_batches,
        'vel':  total_vel  / n_batches,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, lambda_vel):
    model.eval()
    total_loss = total_pos = total_vel = 0.0
    n_batches  = 0

    for batch in loader:
        sensors = batch['sensors']
        car_x   = batch['car_x']
        car_vx  = batch['car_vx']
        ego_y   = batch['ego_y']
        B, T, _ = sensors.shape

        state     = model.init_state(B, device)
        sensors_d = sensors.to(device)
        ego_y_d   = ego_y.to(device)
        car_x_d   = car_x.to(device)
        car_vx_d  = car_vx.to(device)
        pos_logits, vel_pred, _ = run_recurrent_segment(
            model, sensors_d, ego_y_d, state, device)

        l_pos = position_loss(pos_logits, car_x_d)
        l_vel = velocity_loss(vel_pred, car_vx_d)
        loss  = l_pos + lambda_vel * l_vel

        total_loss += loss.item()
        total_pos  += l_pos.item()
        total_vel  += l_vel.item()
        n_batches  += 1

    return {
        'loss': total_loss / n_batches,
        'pos':  total_pos  / n_batches,
        'vel':  total_vel  / n_batches,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',      required=True,
                    choices=list(MODEL_CONFIGS.keys()),
                    help='Model architecture name')
    ap.add_argument('--data',       default='laneshift_dataset.npz',
                    help='Path to .npz dataset (default: laneshift_dataset.npz)')
    ap.add_argument('--out-dir',    default='race-car/WorldModel/checkpoints',
                    help='Directory to save best checkpoint')
    ap.add_argument('--run-name',   default=None,
                    help='W&B run name (defaults to --model)')
    ap.add_argument('--project',    default='laneshift-worldmodel',
                    help='W&B project name')
    ap.add_argument('--epochs',     type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--T-seg',      type=int, default=120,
                    help='Ticks per training segment (default: 120)')
    ap.add_argument('--lr',         type=float, default=3e-4)
    ap.add_argument('--lambda-vel', type=float, default=0.1,
                    help='Velocity loss weight (default: 0.1)')
    ap.add_argument('--patience',   type=int, default=5,
                    help='Early stopping patience in epochs (default: 5)')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--no-wandb',   action='store_true')
    args = ap.parse_args()

    run_name = args.run_name or args.model
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # ── datasets ─────────────────────────────────────────────────────────────
    train_ds = LaneShiftDataset(args.data, split='train', T_seg=args.T_seg,
                                seed=args.seed)
    val_ds   = LaneShiftDataset(args.data, split='val',   T_seg=args.T_seg,
                                seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model).to(device)
    n_params = count_params(model)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}", flush=True)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config={
                'model': args.model,
                'n_params': n_params,
                'T_seg': args.T_seg,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'lambda_vel': args.lambda_vel,
                'epochs': args.epochs,
                'patience': args.patience,
                **MODEL_CONFIGS[args.model],
            }
        )

    # ── training loop with early stopping ────────────────────────────────────
    best_val_loss  = float('inf')
    epochs_no_impr = 0
    best_ckpt_path = os.path.join(args.out_dir, f'{run_name}_best.pt')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler,
                                    device, args.lambda_vel)
        val_metrics   = eval_epoch(model, val_loader, device, args.lambda_vel)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(pos={train_metrics['pos']:.4f} vel={train_metrics['vel']:.4f})  "
            f"val_loss={val_metrics['loss']:.4f} "
            f"(pos={val_metrics['pos']:.4f} vel={val_metrics['vel']:.4f})  "
            f"lr={lr_now:.2e}  t={elapsed:.0f}s",
            flush=True,
        )

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'lr': lr_now,
                'train/loss': train_metrics['loss'],
                'train/pos':  train_metrics['pos'],
                'train/vel':  train_metrics['vel'],
                'val/loss':   val_metrics['loss'],
                'val/pos':    val_metrics['pos'],
                'val/vel':    val_metrics['vel'],
            })

        # Early stopping
        if val_metrics['loss'] < best_val_loss - 1e-5:
            best_val_loss  = val_metrics['loss']
            epochs_no_impr = 0
            torch.save({
                'epoch': epoch,
                'model_name': args.model,
                'model_state': model.state_dict(),
                'val_loss': best_val_loss,
                'config': MODEL_CONFIGS[args.model],
            }, best_ckpt_path)
            print(f"  ✓ New best val_loss={best_val_loss:.4f} saved to {best_ckpt_path}",
                  flush=True)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).",
                      flush=True)
                break

    print(f"\nTraining complete.  Best val_loss={best_val_loss:.4f}", flush=True)

    if use_wandb:
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.finish()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Experiment 7 — full-game training (length-sorted batches, padding, masked loss).

One batch row = one game from start to end; RNN state resets only at t=0 per row.
Independent of segment-based train.py and submit_model_*.sh (except shared model code).

Run from repo root:

    python race-car/WorldModel/train_run_1/train_fullgame.py \\
        --model model_a --data laneshift_dataset.npz \\
        --run-name model_a_fullgame --project laneshift-worldmodel
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RACECAR_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, RACECAR_DIR)

from WorldModel.model import (
    build_model,
    car_x_to_bin,
    MODEL_CONFIGS,
    count_params,
    BINS,
    N_LANES,
)
from WorldModel.dataset import LaneShiftGameDataset

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def collate_padded_games(batch: list[dict]) -> dict:
    max_t = max(x['sensors'].shape[0] for x in batch)
    b = len(batch)
    nan = float('nan')
    sensors = torch.full((b, max_t, 16), nan, dtype=torch.float32)
    car_x = torch.full((b, max_t, 5), nan, dtype=torch.float32)
    car_vx = torch.full((b, max_t, 5), nan, dtype=torch.float32)
    ego_y = torch.full((b, max_t, 1), nan, dtype=torch.float32)
    valid = torch.zeros(b, max_t, dtype=torch.bool)
    for i, item in enumerate(batch):
        t = item['sensors'].shape[0]
        sensors[i, :t] = item['sensors']
        car_x[i, :t] = item['car_x']
        car_vx[i, :t] = item['car_vx']
        ego_y[i, :t] = item['ego_y']
        valid[i, :t] = True
    out = {
        'sensors': sensors,
        'car_x': car_x,
        'car_vx': car_vx,
        'ego_y': ego_y,
        'valid': valid,
    }
    if batch and 'parser' in batch[0]:
        from parser_features import PARSER_DIM
        parser = torch.zeros(b, max_t, PARSER_DIM, dtype=torch.float32)
        for i, item in enumerate(batch):
            t = item['sensors'].shape[0]
            parser[i, :t] = item['parser']
        out['parser'] = parser
    return out


class LengthSortedBatchSampler(Sampler[list[int]]):
    """Sort games by tick count, chunk into batches, optionally shuffle batch order."""

    def __init__(
        self,
        dataset: LaneShiftGameDataset,
        batch_size: int,
        shuffle_batch_order: bool,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_batch_order = shuffle_batch_order
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        lengths = np.array([self.dataset.tick_len(i) for i in range(n)], dtype=np.int64)
        order = np.argsort(lengths)
        bs = self.batch_size
        batches = [order[i : i + bs].tolist() for i in range(0, n, bs)]
        if self.shuffle_batch_order:
            rng = np.random.RandomState(self.seed + 100_003 * self.epoch)
            rng.shuffle(batches)
        yield from batches


def gather_state(state: dict, idx: torch.Tensor) -> dict:
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def scatter_state(state_full: dict, new_sub: dict, idx: torch.Tensor) -> None:
    for k in new_sub:
        state_full[k][idx] = new_sub[k]


def detach_state(state: dict) -> dict:
    return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}


def run_recurrent_padded_game(
    model: nn.Module,
    sensors: torch.Tensor,
    ego_y: torch.Tensor,
    valid: torch.Tensor,
    device: torch.device,
    tbptt_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    b, t_max, _ = sensors.shape
    state = model.init_state(b, device)
    sensors = sensors.to(device)
    ego_y = ego_y.to(device)
    valid = valid.to(device)

    pos_list: list[torch.Tensor] = []
    vel_list: list[torch.Tensor] = []
    zref = torch.zeros(1, device=device, dtype=sensors.dtype)

    for t in range(t_max):
        active = valid[:, t]
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            pl, vp, new_sub = model(sensors[idx, t], ego_y[idx, t], gather_state(state, idx))
            full_pl = pl.new_zeros(b, pl.size(1), pl.size(2))
            full_vp = vp.new_zeros(b, vp.size(1))
            full_pl[idx] = pl
            full_vp[idx] = vp
            scatter_state(state, new_sub, idx)
        else:
            c = BINS + 1
            full_pl = zref.new_zeros(b, N_LANES, c)
            full_vp = zref.new_zeros(b, N_LANES)

        pos_list.append(full_pl)
        vel_list.append(full_vp)

        if tbptt_chunk and (t + 1) % tbptt_chunk == 0:
            state = detach_state(state)

    pos_logits = torch.stack(pos_list, dim=1)
    vel_pred = torch.stack(vel_list, dim=1)
    return pos_logits, vel_pred, state


def masked_position_loss(
    pos_logits: torch.Tensor,
    car_x: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    b, tt, l, c = pos_logits.shape
    m = valid.unsqueeze(-1).expand(b, tt, l).reshape(-1)
    logits = pos_logits.reshape(b * tt * l, c)
    targets = car_x_to_bin(car_x.reshape(b * tt * l))
    if not m.any():
        return pos_logits.new_zeros(())
    return F.cross_entropy(logits[m], targets[m])


def masked_velocity_loss(
    vel_pred: torch.Tensor,
    car_vx: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    m = valid.unsqueeze(-1).expand_as(car_vx) & ~torch.isnan(car_vx)
    if not m.any():
        return vel_pred.new_zeros(())
    return F.huber_loss(vel_pred[m], car_vx[m], delta=5.0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    lambda_vel: float,
    tbptt_chunk: int,
) -> dict[str, float]:
    model.train()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0

    for batch in loader:
        sensors = batch['sensors']
        car_x = batch['car_x']
        car_vx = batch['car_vx']
        ego_y = batch['ego_y']
        valid = batch['valid']

        pos_logits, vel_pred, _ = run_recurrent_padded_game(
            model, sensors, ego_y, valid, device, tbptt_chunk,
        )
        car_x_d = car_x.to(device)
        car_vx_d = car_vx.to(device)
        valid_d = valid.to(device)

        l_pos = masked_position_loss(pos_logits, car_x_d, valid_d)
        l_vel = masked_velocity_loss(vel_pred, car_vx_d, valid_d)
        loss = l_pos + lambda_vel * l_vel

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tot_loss += loss.item()
        tot_pos += l_pos.item()
        tot_vel += l_vel.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return {
        'loss': tot_loss / max(n_batches, 1),
        'pos': tot_pos / max(n_batches, 1),
        'vel': tot_vel / max(n_batches, 1),
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_vel: float,
    tbptt_chunk: int,
) -> dict[str, float]:
    model.eval()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0

    for batch in loader:
        sensors = batch['sensors']
        car_x = batch['car_x']
        car_vx = batch['car_vx']
        ego_y = batch['ego_y']
        valid = batch['valid']

        pos_logits, vel_pred, _ = run_recurrent_padded_game(
            model, sensors, ego_y, valid, device, tbptt_chunk,
        )
        car_x_d = car_x.to(device)
        car_vx_d = car_vx.to(device)
        valid_d = valid.to(device)

        l_pos = masked_position_loss(pos_logits, car_x_d, valid_d)
        l_vel = masked_velocity_loss(vel_pred, car_vx_d, valid_d)
        loss = l_pos + lambda_vel * l_vel

        tot_loss += loss.item()
        tot_pos += l_pos.item()
        tot_vel += l_vel.item()
        n_batches += 1

    return {
        'loss': tot_loss / max(n_batches, 1),
        'pos': tot_pos / max(n_batches, 1),
        'vel': tot_vel / max(n_batches, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--data', default='laneshift_dataset.npz')
    ap.add_argument('--out-dir', default='race-car/WorldModel/checkpoints')
    ap.add_argument('--run-name', default=None,
                    help='defaults to {model}_fullgame')
    ap.add_argument('--project', default='laneshift-worldmodel')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=8,
                    help='games per batch (padded to max length in batch)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda-vel', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='0 recommended (custom batch sampler)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-ticks', type=int, default=None,
                    help='truncate each game to first N ticks (default: full game)')
    ap.add_argument('--tbptt-chunk', type=int, default=200,
                    help='detach RNN state every N ticks (0 = full backprop; may OOM)')
    ap.add_argument('--no-shuffle-batches', action='store_true',
                    help='keep deterministic batch grouping (sorted order only)')
    ap.add_argument('--no-wandb', action='store_true')
    args = ap.parse_args()

    run_name = args.run_name or f'{args.model}_fullgame'
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    train_ds = LaneShiftGameDataset(
        args.data, split='train', seed=args.seed, max_ticks=args.max_ticks,
    )
    val_ds = LaneShiftGameDataset(
        args.data, split='val', seed=args.seed, max_ticks=args.max_ticks,
    )

    train_sampler = LengthSortedBatchSampler(
        train_ds, args.batch_size,
        shuffle_batch_order=not args.no_shuffle_batches,
        seed=args.seed,
    )
    val_sampler = LengthSortedBatchSampler(
        val_ds, args.batch_size, shuffle_batch_order=False, seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_padded_games,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate_padded_games,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    model = build_model(args.model).to(device)
    n_params = count_params(model)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    tbptt = args.tbptt_chunk

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config={
                'train_mode': 'fullgame',
                'model': args.model,
                'n_params': n_params,
                'batch_size': args.batch_size,
                'max_ticks': args.max_ticks,
                'tbptt_chunk': tbptt,
                'lr': args.lr,
                'lambda_vel': args.lambda_vel,
                'epochs': args.epochs,
                'patience': args.patience,
                **MODEL_CONFIGS[args.model],
            },
        )

    best_val_loss = float('inf')
    epochs_no_impr = 0
    best_ckpt_path = os.path.join(args.out_dir, f'{run_name}_best.pt')

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(0)

        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.lambda_vel, tbptt,
        )
        val_metrics = eval_epoch(
            model, val_loader, device, args.lambda_vel, tbptt,
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"lr={lr_now:.2e}  t={elapsed:.0f}s",
            flush=True,
        )

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'lr': lr_now,
                'train/loss': train_metrics['loss'],
                'train/pos': train_metrics['pos'],
                'train/vel': train_metrics['vel'],
                'val/loss': val_metrics['loss'],
                'val/pos': val_metrics['pos'],
                'val/vel': val_metrics['vel'],
            })

        if val_metrics['loss'] < best_val_loss - 1e-5:
            best_val_loss = val_metrics['loss']
            epochs_no_impr = 0
            torch.save({
                'epoch': epoch,
                'model_name': args.model,
                'model_state': model.state_dict(),
                'val_loss': best_val_loss,
                'config': MODEL_CONFIGS[args.model],
                'train_mode': 'fullgame',
                'max_ticks': args.max_ticks,
                'tbptt_chunk': tbptt,
            }, best_ckpt_path)
            print(f"  ✓ New best val_loss={best_val_loss:.4f} → {best_ckpt_path}",
                  flush=True)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    print(f"\nTraining complete.  Best val_loss={best_val_loss:.4f}", flush=True)
    if use_wandb:
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.finish()


if __name__ == '__main__':
    main()
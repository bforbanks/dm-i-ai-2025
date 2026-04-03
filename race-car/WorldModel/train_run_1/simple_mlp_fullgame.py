#!/usr/bin/env python3
"""
Experiment 9 — single hidden-layer MLP on full games with output feedback.

Each tick: preprocessed sensors + ego_y + SensorParser (20-D) + **previous tick’s
softmax position belief (flat) + previous tick’s predicted velocities** →
Linear → GELU → Linear → logits. Same padded full-game batching / masked loss as exp7.

Run from repo root:

    python race-car/WorldModel/train_run_1/simple_mlp_fullgame.py \\
        --data laneshift_dataset.npz --hidden 1024
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
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RACECAR_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, RACECAR_DIR)
sys.path.insert(0, SCRIPT_DIR)

from WorldModel.model import BINS, D_SENSOR, EGO_Y_MAX, N_LANES, preprocess_sensors

from parser_dataset import LaneShiftGameDatasetWithParser
from parser_features import PARSER_DIM

import train_fullgame as tfg

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class SimpleRecurrentFullGameMLP(nn.Module):
    """One hidden layer; recurrence via previous softmax positions + prev velocities."""

    def __init__(self, hidden: int = 1024, bins: int = BINS, parser_dim: int = PARSER_DIM):
        super().__init__()
        self.bins = bins
        self.parser_dim = parser_dim
        nb = bins + 1
        d_in = D_SENSOR + 1 + parser_dim + N_LANES * nb + N_LANES
        n_pos = N_LANES * nb
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_pos + N_LANES)

    def init_state(self, b: int, device: torch.device) -> dict:
        nb = self.bins + 1
        pos = torch.zeros(b, N_LANES, nb, device=device)
        pos[..., self.bins] = 1.0
        return {
            "prev_pos": pos,
            "prev_vel": torch.zeros(b, N_LANES, device=device),
        }

    def forward(
        self,
        sensors_raw: torch.Tensor,
        ego_y_raw: torch.Tensor,
        parser_raw: torch.Tensor,
        state: dict,
    ):
        """
        One tick, batch size B arbitrary.
        sensors_raw [B, 16], ego_y_raw [B, 1], parser_raw [B, PARSER_DIM]
        """
        prev_pos = state["prev_pos"]
        prev_vel = state["prev_vel"]

        s = preprocess_sensors(sensors_raw)
        e = ego_y_raw / EGO_Y_MAX
        flat_p = prev_pos.reshape(prev_pos.size(0), -1)
        x = torch.cat([s, e, parser_raw, flat_p, prev_vel], dim=-1)
        h = F.gelu(self.fc1(x))
        o = self.fc2(h)
        nb = self.bins + 1
        n_pos = N_LANES * nb
        pos_logits = o[:, :n_pos].view(-1, N_LANES, nb)
        vel_pred = o[:, n_pos:].view(-1, N_LANES)

        pos_prob = F.softmax(pos_logits, dim=-1)
        new_state = {
            "prev_pos": pos_prob,
            "prev_vel": vel_pred.detach(),
        }
        return pos_logits, vel_pred, new_state


def run_recurrent_padded_simple_mlp(
    model: nn.Module,
    sensors: torch.Tensor,
    ego_y: torch.Tensor,
    parser: torch.Tensor,
    valid: torch.Tensor,
    device: torch.device,
    tbptt_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    b, t_max, _ = sensors.shape
    state = model.init_state(b, device)
    sensors = sensors.to(device)
    ego_y = ego_y.to(device)
    parser = parser.to(device)
    valid = valid.to(device)

    pos_list: list[torch.Tensor] = []
    vel_list: list[torch.Tensor] = []
    zref = torch.zeros(1, device=device, dtype=sensors.dtype)
    nb = model.bins + 1

    for t in range(t_max):
        active = valid[:, t]
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            pl, vp, new_sub = model(
                sensors[idx, t],
                ego_y[idx, t],
                parser[idx, t],
                tfg.gather_state(state, idx),
            )
            full_pl = pl.new_zeros(b, N_LANES, nb)
            full_vp = vp.new_zeros(b, N_LANES)
            full_pl[idx] = pl
            full_vp[idx] = vp
            tfg.scatter_state(state, new_sub, idx)
        else:
            full_pl = zref.new_zeros(b, N_LANES, nb)
            full_vp = zref.new_zeros(b, N_LANES)

        pos_list.append(full_pl)
        vel_list.append(full_vp)

        if tbptt_chunk and (t + 1) % tbptt_chunk == 0:
            state = tfg.detach_state(state)

    return torch.stack(pos_list, dim=1), torch.stack(vel_list, dim=1), state


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train_epoch(model, loader, optimizer, scheduler, device, lambda_vel, tbptt_chunk: int):
    model.train()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0
    for batch in loader:
        sensors = batch["sensors"]
        ego_y = batch["ego_y"]
        car_x = batch["car_x"].to(device)
        car_vx = batch["car_vx"].to(device)
        valid = batch["valid"].to(device)
        parser = batch["parser"]

        pos_logits, vel_pred, _ = run_recurrent_padded_simple_mlp(
            model, sensors, ego_y, parser, valid, device, tbptt_chunk,
        )

        l_pos = tfg.masked_position_loss(pos_logits, car_x, valid)
        l_vel = tfg.masked_velocity_loss(vel_pred, car_vx, valid)
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
        "loss": tot_loss / max(n_batches, 1),
        "pos": tot_pos / max(n_batches, 1),
        "vel": tot_vel / max(n_batches, 1),
    }


@torch.no_grad()
def eval_epoch(model, loader, device, lambda_vel, tbptt_chunk: int):
    model.eval()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0
    for batch in loader:
        sensors = batch["sensors"]
        ego_y = batch["ego_y"]
        car_x = batch["car_x"].to(device)
        car_vx = batch["car_vx"].to(device)
        valid = batch["valid"].to(device)
        parser = batch["parser"]

        pos_logits, vel_pred, _ = run_recurrent_padded_simple_mlp(
            model, sensors, ego_y, parser, valid, device, tbptt_chunk,
        )

        l_pos = tfg.masked_position_loss(pos_logits, car_x, valid)
        l_vel = tfg.masked_velocity_loss(vel_pred, car_vx, valid)
        loss = l_pos + lambda_vel * l_vel

        tot_loss += loss.item()
        tot_pos += l_pos.item()
        tot_vel += l_vel.item()
        n_batches += 1

    return {
        "loss": tot_loss / max(n_batches, 1),
        "pos": tot_pos / max(n_batches, 1),
        "vel": tot_vel / max(n_batches, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="laneshift_dataset.npz")
    ap.add_argument("--out-dir", default="race-car/WorldModel/checkpoints")
    ap.add_argument("--run-name", default="simple_mlp_fullgame_parser")
    ap.add_argument("--project", default="laneshift-worldmodel")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="games per batch (memory grows with max game length × batch)")
    ap.add_argument("--tbptt-chunk", type=int, default=200,
                    help="detach recurrent state every N ticks (0 = full BPTT)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-vel", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=None,
                    help="truncate games to first N ticks (memory / speed)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle-batches", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tbptt = args.tbptt_chunk

    train_ds = LaneShiftGameDatasetWithParser(
        args.data, split="train", seed=args.seed, max_ticks=args.max_ticks,
    )
    val_ds = LaneShiftGameDatasetWithParser(
        args.data, split="val", seed=args.seed, max_ticks=args.max_ticks,
    )

    train_sampler = tfg.LengthSortedBatchSampler(
        train_ds, args.batch_size,
        shuffle_batch_order=not args.no_shuffle_batches,
        seed=args.seed,
    )
    val_sampler = tfg.LengthSortedBatchSampler(
        val_ds, args.batch_size, shuffle_batch_order=False, seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=tfg.collate_padded_games,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=tfg.collate_padded_games,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SimpleRecurrentFullGameMLP(hidden=args.hidden, parser_dim=PARSER_DIM).to(device)
    n_params = count_params(model)
    print(
        f"SimpleRecurrentFullGameMLP(hidden={args.hidden}, parser_dim={PARSER_DIM}, "
        f"tbptt_chunk={tbptt})  |  {n_params:,} params  |  {device}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "train_mode": "simple_mlp_fullgame_parser_recurrent",
                "hidden": args.hidden,
                "parser_dim": PARSER_DIM,
                "tbptt_chunk": tbptt,
                "n_params": n_params,
                "batch_size": args.batch_size,
                "max_ticks": args.max_ticks,
                "lr": args.lr,
                "lambda_vel": args.lambda_vel,
                "epochs": args.epochs,
            },
        )

    best_val = float("inf")
    bad = 0
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(0)
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, args.lambda_vel, tbptt)
        va = eval_epoch(model, val_loader, device, args.lambda_vel, tbptt)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  train={tr['loss']:.4f}  val={va['loss']:.4f}  "
            f"lr={lr:.2e}  {dt:.0f}s",
            flush=True,
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "lr": lr,
                "train/loss": tr["loss"],
                "val/loss": va["loss"],
            })

        if va["loss"] < best_val - 1e-5:
            best_val = va["loss"]
            bad = 0
            torch.save({
                "epoch": epoch,
                "model_name": "simple_recurrent_mlp_fullgame_parser",
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "hidden": args.hidden,
                "parser_dim": PARSER_DIM,
                "tbptt_chunk": tbptt,
            }, best_path)
            print(f"  ✓ best val={best_val:.4f} → {best_path}", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.", flush=True)
                break

    print(f"\nDone. Best val_loss={best_val:.4f}", flush=True)
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val
        wandb.finish()


if __name__ == "__main__":
    main()

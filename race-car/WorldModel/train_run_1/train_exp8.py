#!/usr/bin/env python3
"""
Experiment 8 — model_a_lg + SensorParser features (value + mask per lane).

Raw sensors unchanged; each tick also gets a 20-D vector from SensorParser
(x, x_mask, v, v_mask per lane — zeros when absent).

Run from repo root:

    python race-car/WorldModel/train_run_1/train_exp8.py \\
        --data laneshift_dataset.npz --project laneshift-worldmodel
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RACECAR_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, RACECAR_DIR)
sys.path.insert(0, SCRIPT_DIR)

from WorldModel.train import position_loss, velocity_loss

from model_a_lg_parser import ModelALgWithParser, PARSER_DIM
from parser_dataset import LaneShiftDatasetWithParser

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


def run_recurrent_segment_parser(model, sensors_seg, ego_y_seg, parser_seg, state, device):
    b, t, _ = sensors_seg.shape
    pos_list, vel_list = [], []
    for ti in range(t):
        pl, vp, state = model(
            sensors_seg[:, ti, :].to(device),
            ego_y_seg[:, ti, :].to(device),
            parser_seg[:, ti, :].to(device),
            state,
        )
        pos_list.append(pl)
        vel_list.append(vp)
    pos_logits = torch.stack(pos_list, dim=1)
    vel_pred = torch.stack(vel_list, dim=1)
    return pos_logits, vel_pred, state


def train_epoch(model, loader, optimizer, scheduler, device, lambda_vel):
    model.train()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0
    for batch in loader:
        sensors = batch["sensors"]
        car_x = batch["car_x"]
        car_vx = batch["car_vx"]
        ego_y = batch["ego_y"]
        parser = batch["parser"]
        b, t, _ = sensors.shape
        state = model.init_state(b, device)
        pos_logits, vel_pred, _ = run_recurrent_segment_parser(
            model, sensors.to(device), ego_y.to(device), parser.to(device), state, device,
        )
        car_x_d = car_x.to(device)
        car_vx_d = car_vx.to(device)
        l_pos = position_loss(pos_logits, car_x_d)
        l_vel = velocity_loss(vel_pred, car_vx_d)
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
def eval_epoch(model, loader, device, lambda_vel):
    model.eval()
    tot_loss = tot_pos = tot_vel = 0.0
    n_batches = 0
    for batch in loader:
        sensors = batch["sensors"]
        car_x = batch["car_x"]
        car_vx = batch["car_vx"]
        ego_y = batch["ego_y"]
        parser = batch["parser"]
        b, t, _ = sensors.shape
        state = model.init_state(b, device)
        pos_logits, vel_pred, _ = run_recurrent_segment_parser(
            model, sensors.to(device), ego_y.to(device), parser.to(device), state, device,
        )
        car_x_d = car_x.to(device)
        car_vx_d = car_vx.to(device)
        l_pos = position_loss(pos_logits, car_x_d)
        l_vel = velocity_loss(vel_pred, car_vx_d)
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="laneshift_dataset.npz")
    ap.add_argument("--out-dir", default="race-car/WorldModel/checkpoints")
    ap.add_argument("--run-name", default="model_a_lg_parser")
    ap.add_argument("--project", default="laneshift-worldmodel")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--T-seg", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-vel", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  parser_dim={PARSER_DIM}", flush=True)

    train_ds = LaneShiftDatasetWithParser(
        args.data, split="train", T_seg=args.T_seg, seed=args.seed,
    )
    val_ds = LaneShiftDatasetWithParser(
        args.data, split="val", T_seg=args.T_seg, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = ModelALgWithParser().to(device)
    n_params = count_params(model)
    print(f"Model: ModelALgWithParser  |  Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    use_wandb = _WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "train_mode": "exp8_parser",
                "model": "model_a_lg_parser",
                "parser_dim": PARSER_DIM,
                "n_params": n_params,
                "T_seg": args.T_seg,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_vel": args.lambda_vel,
                "epochs": args.epochs,
                "patience": args.patience,
            },
        )

    best_val_loss = float("inf")
    epochs_no_impr = 0
    best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, scheduler, device, args.lambda_vel)
        val_m = eval_epoch(model, val_loader, device, args.lambda_vel)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}  train_loss={train_m['loss']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  lr={lr_now:.2e}  t={elapsed:.0f}s",
            flush=True,
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "lr": lr_now,
                "train/loss": train_m["loss"],
                "train/pos": train_m["pos"],
                "train/vel": train_m["vel"],
                "val/loss": val_m["loss"],
                "val/pos": val_m["pos"],
                "val/vel": val_m["vel"],
            })

        if val_m["loss"] < best_val_loss - 1e-5:
            best_val_loss = val_m["loss"]
            epochs_no_impr = 0
            torch.save({
                "epoch": epoch,
                "model_name": "model_a_lg_parser",
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "parser_dim": PARSER_DIM,
            }, best_path)
            print(f"  ✓ best val_loss={best_val_loss:.4f} → {best_path}", flush=True)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= args.patience:
                print("Early stopping.", flush=True)
                break

    print(f"\nDone. Best val_loss={best_val_loss:.4f}", flush=True)
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()


if __name__ == "__main__":
    main()

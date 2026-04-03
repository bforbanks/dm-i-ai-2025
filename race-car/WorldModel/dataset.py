#!/usr/bin/env python3
"""
LaneShift dataset for WorldModel training.

Splits are performed at the game level (not tick level) to avoid leakage.

Split sizes (with seed=42):
    train :  81% of games  (8100 / 10000)
    val   :  10% of games  (1000 / 10000)  — used for early stopping
    test  :   9% of games  ( 900 / 10000)  — final holdout, never used during training

Each dataset item is a fixed-length segment of T_seg consecutive ticks from
a single game.  Segments are taken non-overlapping from each game.  Any
partial tail shorter than T_seg // 2 ticks is dropped.

The numpy arrays are loaded into memory at construction time (dataset ≈ 1.5 GB).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LaneShiftDataset(Dataset):
    """
    Returns dicts with keys:
        sensors  : [T_seg, 16]  float32   (raw; NaN = no detection)
        car_x    : [T_seg, 5]   float32   (raw; NaN = lane empty)
        car_vx   : [T_seg, 5]   float32   (NaN = lane empty)
        ego_y    : [T_seg, 1]   float32   (raw pixels)
        valid    : [T_seg]      bool      (False for padding; all True here since
                                           segments have uniform length)
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',           # 'train' | 'val' | 'test'
        T_seg: int = 120,
        seed: int = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.T_seg = T_seg

        # ── load dataset ──────────────────────────────────────────────────────
        print(f"Loading {data_path} …", flush=True)
        d = np.load(data_path)
        sensors   = d['sensors'].astype(np.float32)    # [total_ticks, 16]
        car_x     = d['car_x'].astype(np.float32)      # [total_ticks, 5]
        car_vx    = d['car_vx'].astype(np.float32)     # [total_ticks, 5]
        ego_xy    = d['ego_xy'].astype(np.float32)     # [total_ticks, 2]
        lengths   = d['game_lengths'].astype(np.int64) # [n_games]

        # ── game-level split ─────────────────────────────────────────────────
        n_games = len(lengths)
        rng     = np.random.RandomState(seed)
        perm    = rng.permutation(n_games)

        n_val   = int(round(0.10 * n_games))    # 1000
        n_test  = int(round(0.09 * n_games))    #  900
        n_train = n_games - n_val - n_test       # 8100

        split_idx = {
            'train': perm[:n_train],
            'val':   perm[n_train:n_train + n_val],
            'test':  perm[n_train + n_val:],
        }[split]

        # ── build flat-start index ────────────────────────────────────────────
        starts_all = np.concatenate([[0], np.cumsum(lengths[:-1])])

        # ── build segment list ────────────────────────────────────────────────
        segments = []   # list of (flat_start, seg_len)
        for gi in split_idx:
            T         = int(lengths[gi])
            flat_base = int(starts_all[gi])
            t0        = 0
            while t0 + T_seg <= T:
                segments.append((flat_base + t0, T_seg))
                t0 += T_seg
            # include last partial segment if long enough
            tail = T - t0
            if tail >= T_seg // 2 and tail > 0:
                # backward-pad by repeating first tick (handled below)
                pass    # skip for now — all segments are exactly T_seg long

        self.segments = segments

        # ── store arrays in memory ────────────────────────────────────────────
        self.sensors = sensors
        self.car_x   = car_x
        self.car_vx  = car_vx
        self.ego_y   = ego_xy[:, 1:2]    # only y coordinate

        print(f"  split={split}  games={len(split_idx)}  segments={len(segments)}", flush=True)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict:
        flat_start, seg_len = self.segments[idx]
        sl = slice(flat_start, flat_start + seg_len)
        return {
            'sensors': torch.from_numpy(self.sensors[sl].copy()),  # [T, 16]
            'car_x':   torch.from_numpy(self.car_x[sl].copy()),    # [T, 5]
            'car_vx':  torch.from_numpy(self.car_vx[sl].copy()),   # [T, 5]
            'ego_y':   torch.from_numpy(self.ego_y[sl].copy()),    # [T, 1]
        }


class LaneShiftGameDataset(Dataset):
    """
    One item = one full game (variable length).

    Used by train_run_1/train_fullgame.py: batches are built by sorting games
    by length and padding within each batch.  RNN state resets only at game
    start (per row), not at artificial segment cuts.

    Args:
        max_ticks: if set, each game is truncated to its first ``max_ticks``
            ticks (still one contiguous prefix per game).
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        seed: int = 42,
        max_ticks: int | None = None,
    ):
        assert split in ('train', 'val', 'test')
        self.max_ticks = max_ticks

        print(f"Loading {data_path} (full-game mode) …", flush=True)
        d = np.load(data_path)
        sensors = d['sensors'].astype(np.float32)
        car_x   = d['car_x'].astype(np.float32)
        car_vx  = d['car_vx'].astype(np.float32)
        ego_xy  = d['ego_xy'].astype(np.float32)
        lengths = d['game_lengths'].astype(np.int64)

        n_games = len(lengths)
        rng     = np.random.RandomState(seed)
        perm    = rng.permutation(n_games)

        n_val   = int(round(0.10 * n_games))
        n_test  = int(round(0.09 * n_games))
        n_train = n_games - n_val - n_test

        split_idx = {
            'train': perm[:n_train],
            'val':   perm[n_train:n_train + n_val],
            'test':  perm[n_train + n_val:],
        }[split]

        starts_all = np.concatenate([[0], np.cumsum(lengths[:-1])])

        self.sensors = sensors
        self.car_x   = car_x
        self.car_vx  = car_vx
        self.ego_y   = ego_xy[:, 1:2]
        self.starts_all = starts_all
        self.lengths    = lengths

        self.game_ids: list[int] = []
        self.tick_lens: list[int] = []
        for gi in split_idx:
            T = int(lengths[gi])
            if max_ticks is not None:
                T = min(T, int(max_ticks))
            if T < 1:
                continue
            self.game_ids.append(int(gi))
            self.tick_lens.append(T)

        print(
            f"  split={split}  games={len(self.game_ids)}  "
            f"(full-game items; max_ticks={max_ticks})",
            flush=True,
        )

    def tick_len(self, idx: int) -> int:
        return self.tick_lens[idx]

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> dict:
        gi   = self.game_ids[idx]
        T    = self.tick_lens[idx]
        base = int(self.starts_all[gi])
        sl   = slice(base, base + T)
        return {
            'sensors': torch.from_numpy(self.sensors[sl].copy()),
            'car_x':   torch.from_numpy(self.car_x[sl].copy()),
            'car_vx':  torch.from_numpy(self.car_vx[sl].copy()),
            'ego_y':   torch.from_numpy(self.ego_y[sl].copy()),
        }


def build_sensor_windows(
    sensors: torch.Tensor,
    n_chunks: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Build chunked sensor windows for ModelB (no-state transformer).

    sensors : [T, 32]  (preprocessed — values + mask)
    Returns : [T, n_chunks, chunk_size, 32]
        chunk index 0 = most recent chunk (ending at current tick t)
        chunk index n_chunks-1 = oldest chunk

    The first H-1 ticks are backward-filled with sensors[0].
    H = n_chunks * chunk_size.
    """
    T   = sensors.size(0)
    H   = n_chunks * chunk_size
    D   = sensors.size(1)

    # Pad: repeat first sensor tick H-1 times at the front
    pad = sensors[0:1].expand(H - 1, -1)               # [H-1, 32]
    padded = torch.cat([pad, sensors], dim=0)           # [T+H-1, 32]

    # Build index tensor: for each tick t (0..T-1), grab padded[t : t+H]
    t_idx = torch.arange(T, device=sensors.device)     # [T]
    h_idx = torch.arange(H, device=sensors.device)     # [H]
    idx   = t_idx.unsqueeze(1) + h_idx.unsqueeze(0)    # [T, H]
    windows = padded[idx]                               # [T, H, 32]

    # Reshape to chunks, flip so index 0 = most recent
    chunks = windows.view(T, n_chunks, chunk_size, D)   # [T, n_chunks, cs, 32]
    chunks = chunks.flip(1)                             # flip: 0 = newest chunk
    return chunks

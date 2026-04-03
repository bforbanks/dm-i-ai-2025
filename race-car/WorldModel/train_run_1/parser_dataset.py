#!/usr/bin/env python3
"""Segment dataset + precomputed SensorParser vectors (experiment 8)."""

from __future__ import annotations

import numpy as np
import torch

from WorldModel.dataset import LaneShiftDataset, LaneShiftGameDataset

from parser_features import load_or_precompute_parser_features


class LaneShiftDatasetWithParser(LaneShiftDataset):
    """
    Same segments as LaneShiftDataset; adds ``parser`` [T_seg, PARSER_DIM]
    from SensorParser (zeros + masks when parser has no value).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        T_seg: int = 120,
        seed: int = 42,
        parser_use_cache: bool = True,
        parser_force_recompute: bool = False,
    ):
        super().__init__(data_path, split=split, T_seg=T_seg, seed=seed)
        d = np.load(data_path)
        lengths = d["game_lengths"].astype(np.int64)
        ego_y = d["ego_xy"][:, 1].astype(np.float32)
        self.parser_feats = load_or_precompute_parser_features(
            data_path,
            self.sensors,
            ego_y,
            lengths,
            use_cache=parser_use_cache,
            force_recompute=parser_force_recompute,
        )

    def __getitem__(self, idx: int) -> dict:
        out = super().__getitem__(idx)
        flat_start, seg_len = self.segments[idx]
        sl = slice(flat_start, flat_start + seg_len)
        out["parser"] = torch.from_numpy(self.parser_feats[sl].copy())
        return out


class LaneShiftGameDatasetWithParser(LaneShiftGameDataset):
    """
    Full games (like ``LaneShiftGameDataset``) + ``parser`` [T, PARSER_DIM]
    per item (SensorParser, reset each game).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        seed: int = 42,
        max_ticks: int | None = None,
        parser_use_cache: bool = True,
        parser_force_recompute: bool = False,
    ):
        super().__init__(data_path, split=split, seed=seed, max_ticks=max_ticks)
        d = np.load(data_path)
        lengths = d["game_lengths"].astype(np.int64)
        ego_y = d["ego_xy"][:, 1].astype(np.float32)
        self.parser_feats = load_or_precompute_parser_features(
            data_path,
            self.sensors,
            ego_y,
            lengths,
            use_cache=parser_use_cache,
            force_recompute=parser_force_recompute,
        )

    def __getitem__(self, idx: int) -> dict:
        out = super().__getitem__(idx)
        gi = self.game_ids[idx]
        t = self.tick_lens[idx]
        base = int(self.starts_all[gi])
        sl = slice(base, base + t)
        out["parser"] = torch.from_numpy(self.parser_feats[sl].copy())
        return out

#!/usr/bin/env python3
"""model_a_lg + SensorParser feature vector (experiment 8 only)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from WorldModel.model import (
    BINS,
    D_SENSOR,
    EGO_Y_MAX,
    N_LANES,
    PrevPosCNN,
    PerLaneHead,
    preprocess_sensors,
)

# Must match parser_features.PARSER_DIM (5 lanes × 4)
PARSER_DIM = 20


class ModelALgWithParser(nn.Module):
    """Same capacity as model_a_lg, with extra parser inputs [*, PARSER_DIM]."""

    def __init__(
        self,
        bins: int = BINS,
        gru_hidden: int = 256,
        fuse_dim: int = 512,
        cnn_dim: int = 64,
        lane_emb_dim: int = 16,
        parser_dim: int = PARSER_DIM,
        parser_enc_dim: int = 32,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.bins = bins
        self.parser_dim = parser_dim

        self.sensor_enc = nn.Sequential(
            nn.Linear(D_SENSOR, 64), nn.GELU(), nn.Linear(64, 64))
        self.parser_enc = nn.Sequential(
            nn.Linear(parser_dim, parser_enc_dim),
            nn.GELU(),
            nn.Linear(parser_enc_dim, parser_enc_dim),
        )
        self.prev_pos_cnn = PrevPosCNN(bins=bins, out_dim=cnn_dim)
        self.belief_proj = nn.Linear(cnn_dim + N_LANES, 64)
        self.gru = nn.GRUCell(64, gru_hidden)

        fuse_in = 64 + parser_enc_dim + gru_hidden + 1
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, 128), nn.LayerNorm(128), nn.GELU(),
        )
        self.pos_head = PerLaneHead(128, bins + 1, lane_emb_dim)
        self.vel_head = PerLaneHead(128, 1, lane_emb_dim)

    def init_state(self, b: int, device):
        pos = torch.zeros(b, N_LANES, self.bins + 1, device=device)
        pos[..., self.bins] = 1.0
        return {
            "prev_pos": pos,
            "prev_vel": torch.zeros(b, N_LANES, device=device),
            "hidden": torch.zeros(b, self.gru_hidden, device=device),
        }

    def forward(self, sensors_raw, ego_y_raw, parser_raw, state):
        prev_pos = state["prev_pos"]
        prev_vel = state["prev_vel"]
        hidden = state["hidden"]

        s = preprocess_sensors(sensors_raw)
        ego_y = ego_y_raw / EGO_Y_MAX
        p = self.parser_enc(parser_raw)

        s_enc = self.sensor_enc(s)
        cnn_out = self.prev_pos_cnn(prev_pos)
        h = self.gru(
            F.gelu(self.belief_proj(torch.cat([cnn_out, prev_vel], dim=1))),
            hidden,
        )
        fused = self.fusion(torch.cat([s_enc, p, h, ego_y], dim=1))
        pos_logits = self.pos_head(fused)
        vel_pred = self.vel_head(fused).squeeze(-1)

        pos_prob = F.softmax(pos_logits, dim=-1)
        new_state = {
            "prev_pos": pos_prob,
            "prev_vel": vel_pred.detach(),
            "hidden": h,
        }
        return pos_logits, vel_pred, new_state

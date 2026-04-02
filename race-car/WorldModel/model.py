#!/usr/bin/env python3
"""
WorldModel — four candidate architectures for NPC position/velocity prediction.

All models expose the same tick-level interface:

    pos_logits, vel_pred, new_state = model(sensors_raw, ego_y_raw, state)

    sensors_raw : [B, 16]      raw readings, NaN where sensor sees nothing
    ego_y_raw   : [B, 1]       ego y-coordinate in raw pixels (0–1200)
    state       : dict         model-specific recurrent state (see init_state)

    pos_logits  : [B, 5, BINS+1]   unnormalized; softmax → P(bin or no-car | obs)
    vel_pred    : [B, 5]            relative velocity (car.vx − ego.vx), px/tick
    new_state   : dict

Use model.init_state(B, device) to get the zero-state at game start.

Architecture summary
--------------------
A  – GRU belief propagator
     CNN over prev_pos belief → GRU hidden state → fusion FFN → per-lane heads.
B  – Variable-length chunked-history transformer
     Sensor history is chunked into CHUNK_SIZE-tick blocks (default 4), each
     embedded into a d_model token. The number of tokens grows with game time
     up to max_ticks // chunk_size (default 125). The transformer attends over
     ALL available history via a padding mask — no fixed window. No GRU state.
     No prev_pos/vel input.
C  – Hybrid (GRU + inter-lane transformer)
     Like A but the fusion layer is a one-layer transformer over 5 lane tokens
     produced from the GRU output.
D  – Feedforward with feedback (no GRU)
     Same inputs as A but no GRU hidden state. Relies entirely on the prev_pos
     CNN and prev_vel for temporal memory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants ─────────────────────────────────────────────────────────────────
BINS        = 64
N_LANES     = 5
OUT_BINS    = BINS + 1      # 64 position bins + 1 no-car class
X_MIN       = -1000.0
X_MAX       = 2600.0
SENSOR_MAX  = 1000.0
EGO_Y_MAX   = 1200.0
VEL_SCALE   = 28.0          # robust normalisation for velocity (covers ~p99.9)
D_SENSOR    = 32            # 16 values + 16 NaN mask

# ── preprocessing helpers (called inside model.forward) ──────────────────────

def preprocess_sensors(raw: torch.Tensor) -> torch.Tensor:
    """[*, 16] with NaN → [*, 32]  (values/1000 ‖ binary mask)"""
    mask = (~torch.isnan(raw)).float()
    vals = torch.nan_to_num(raw, nan=0.0) / SENSOR_MAX
    return torch.cat([vals, mask], dim=-1)


def car_x_to_bin(car_x: torch.Tensor) -> torch.LongTensor:
    """[*, 5] float (NaN = no car) → [*, 5] long  (0–BINS-1 pos, BINS = no car)"""
    no_car = torch.isnan(car_x)
    b = ((car_x - X_MIN) / (X_MAX - X_MIN) * BINS).long().clamp(0, BINS - 1)
    b = b.masked_fill(no_car, BINS)
    return b


# ── shared sub-modules ────────────────────────────────────────────────────────

class PrevPosCNN(nn.Module):
    """
    Compress previous position belief [B, 5, BINS+1] → [B, out_dim].
    Treats lanes as channels, bins as sequence length → 1-D convolutions.
    """
    def __init__(self, bins: int = BINS, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_LANES, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),    # [B, 16, 8]
        )
        self.fc = nn.Linear(16 * 8, out_dim)

    def forward(self, prev_pos: torch.Tensor) -> torch.Tensor:
        # prev_pos: [B, 5, BINS+1]
        h = self.net(prev_pos)          # [B, 16, 8]
        return self.fc(h.flatten(1))    # [B, out_dim]


class PerLaneHead(nn.Module):
    """
    Projects a global representation → per-lane outputs, using a lane embedding
    so each lane gets distinct predictions from a shared weight head.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 lane_emb_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.lane_emb = nn.Embedding(N_LANES, lane_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + lane_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        B = x.size(0)
        dev = x.device
        l   = torch.arange(N_LANES, device=dev)
        emb = self.lane_emb(l).unsqueeze(0).expand(B, -1, -1)   # [B, 5, le]
        xp  = x.unsqueeze(1).expand(-1, N_LANES, -1)            # [B, 5, in_dim]
        return self.net(torch.cat([xp, emb], dim=-1))            # [B, 5, out_dim]



# ── Model A: GRU belief propagator ───────────────────────────────────────────

class ModelA(nn.Module):
    """
    CNN over previous position belief + GRU hidden state → fusion FFN → heads.

    State keys: prev_pos [B,5,BINS+1], prev_vel [B,5], hidden [B,gru_hidden]
    """
    def __init__(self, bins: int = BINS, gru_hidden: int = 128,
                 fuse_dim: int = 256, cnn_dim: int = 64,
                 lane_emb_dim: int = 16):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.bins       = bins

        self.sensor_enc = nn.Sequential(
            nn.Linear(D_SENSOR, 64), nn.GELU(), nn.Linear(64, 64))

        self.prev_pos_cnn = PrevPosCNN(bins=bins, out_dim=cnn_dim)

        self.belief_proj = nn.Linear(cnn_dim + N_LANES, 64)
        self.gru         = nn.GRUCell(64, gru_hidden)

        fuse_in = 64 + gru_hidden + 1
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, 128),     nn.LayerNorm(128),       nn.GELU(),
        )
        self.pos_head = PerLaneHead(128, bins + 1, lane_emb_dim)
        self.vel_head = PerLaneHead(128, 1, lane_emb_dim)

    def init_state(self, B: int, device):
        pos = torch.zeros(B, N_LANES, self.bins + 1, device=device)
        pos[..., self.bins] = 1.0           # all mass on no-car class
        return {
            'prev_pos': pos,
            'prev_vel': torch.zeros(B, N_LANES, device=device),
            'hidden':   torch.zeros(B, self.gru_hidden, device=device),
        }

    def forward(self, sensors_raw, ego_y_raw, state):
        prev_pos = state['prev_pos']
        prev_vel = state['prev_vel']
        hidden   = state['hidden']

        s     = preprocess_sensors(sensors_raw)         # [B, 32]
        ego_y = ego_y_raw / EGO_Y_MAX                  # [B, 1]

        s_enc   = self.sensor_enc(s)                    # [B, 64]
        cnn_out = self.prev_pos_cnn(prev_pos)           # [B, cnn_dim]
        h = self.gru(F.gelu(self.belief_proj(
            torch.cat([cnn_out, prev_vel], dim=1))), hidden)

        fused      = self.fusion(torch.cat([s_enc, h, ego_y], dim=1))  # [B, 128]
        pos_logits = self.pos_head(fused)                               # [B, 5, 65]
        vel_pred   = self.vel_head(fused).squeeze(-1)                   # [B, 5]

        pos_prob  = F.softmax(pos_logits, dim=-1)
        new_state = {
            'prev_pos': pos_prob,
            'prev_vel': vel_pred.detach(),
            'hidden':   h,
        }
        return pos_logits, vel_pred, new_state


# ── Model B: variable-length chunked-history transformer ─────────────────────

class ModelB(nn.Module):
    """
    Sensor history is chunked into CHUNK_SIZE-tick blocks, embedded, and fed to
    a transformer.  The number of tokens grows with game time up to MAX_TOKENS
    (= max_ticks // chunk_size), so the transformer always sees ALL available
    history — there is no fixed window.  Unfilled positions in the buffer are
    masked out via src_key_padding_mask so they contribute nothing to attention.

    MAX_TOKENS = max_ticks // chunk_size
        e.g. max_ticks=500, chunk_size=4  →  max_tokens=125

    State keys:
        chunk_buf  [B, max_tokens, d_model]  — rolling buffer of embedded tokens
        tick_ctr   [B]  int                  — how many raw ticks seen so far
        tick_accum [B, chunk_size, D_SENSOR] — partial chunk being accumulated

    A new token is pushed every CHUNK_SIZE ticks.  Between pushes the
    transformer still runs on the same token set (current tick always uses the
    latest complete chunk plus any partial chunk encoded as zeros — effectively
    the model just reuses the last complete token for the most recent partial
    window, which is a minor approximation acceptable at chunk_size=4).
    """
    def __init__(self, bins: int = BINS, chunk_size: int = 4, max_ticks: int = 500,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 lane_emb_dim: int = 16):
        super().__init__()
        self.bins       = bins
        self.chunk_size = chunk_size
        self.max_tokens = max_ticks // chunk_size   # e.g. 125
        self.d_model    = d_model

        # Chunk embedder: chunk_size raw sensor ticks → d_model
        self.chunk_emb = nn.Sequential(
            nn.Linear(chunk_size * D_SENSOR, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Positional encoding over token index (0 = most recent, max_tokens-1 = oldest)
        self.pos_emb  = nn.Embedding(self.max_tokens, d_model)
        self.ego_proj = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pos_head = PerLaneHead(d_model, bins + 1, lane_emb_dim)
        self.vel_head = PerLaneHead(d_model, 1, lane_emb_dim)

    def init_state(self, B: int, device):
        return {
            'chunk_buf':  torch.zeros(B, self.max_tokens, self.d_model, device=device),
            'tick_ctr':   torch.zeros(B, dtype=torch.long, device=device),
            'tick_accum': torch.zeros(B, self.chunk_size, D_SENSOR, device=device),
        }

    def _embed_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: [B, chunk_size, D_SENSOR] → [B, d_model]"""
        return self.chunk_emb(chunk.flatten(1))

    def forward(self, sensors_raw, ego_y_raw, state):
        B   = sensors_raw.size(0)
        dev = sensors_raw.device

        s     = preprocess_sensors(sensors_raw)         # [B, 32]
        ego_y = ego_y_raw / EGO_Y_MAX                   # [B, 1]

        chunk_buf  = state['chunk_buf']                 # [B, max_tokens, d_model]
        tick_ctr   = state['tick_ctr']                  # [B]
        tick_accum = state['tick_accum']                # [B, chunk_size, D_SENSOR]

        # ── accumulate current tick into partial chunk ────────────────────────
        pos_in_chunk = (tick_ctr % self.chunk_size).long()   # [B], 0..chunk_size-1

        # Write s into tick_accum at position pos_in_chunk for each sample.
        # We do this with a scatter: create a one-hot write mask.
        new_accum = tick_accum.clone()
        for b in range(B):
            new_accum[b, pos_in_chunk[b]] = s[b]

        tick_ctr = tick_ctr + 1

        # ── when a chunk is complete, push it into the buffer ─────────────────
        chunk_complete = (tick_ctr % self.chunk_size == 0)   # [B] bool

        new_chunk_emb = self._embed_chunk(new_accum)          # [B, d_model]

        # Shift buffer left by 1 (oldest token drops off) and write new token at end
        # Only do this for samples where chunk_complete is True
        new_buf = chunk_buf.clone()
        if chunk_complete.any():
            # Shift: [B, max_tokens-1, d] with new token appended
            shifted = torch.cat([chunk_buf[:, 1:, :], new_chunk_emb.unsqueeze(1)], dim=1)
            mask = chunk_complete.view(B, 1, 1).expand_as(new_buf)
            new_buf = torch.where(mask, shifted, new_buf)

        # Reset accum for samples that just completed a chunk
        new_accum = torch.where(
            chunk_complete.view(B, 1, 1).expand_as(new_accum),
            torch.zeros_like(new_accum),
            new_accum,
        )

        # ── how many tokens are filled (capped at max_tokens) ─────────────────
        n_filled = (tick_ctr // self.chunk_size).clamp(max=self.max_tokens)
        # For the transformer we use the maximum across the batch so we can
        # pad uniformly. Padding mask handles the rest.
        max_filled = int(n_filled.max().item())
        max_filled = max(max_filled, 1)             # at least 1 token

        # Slice the filled portion of the buffer (most recent tokens are at the end)
        tokens = new_buf[:, -max_filled:, :]        # [B, max_filled, d_model]

        # Positional encoding: index 0 = most recent token (rightmost in buffer)
        pos_idx = torch.arange(max_filled - 1, -1, -1, device=dev)   # [max_filled] desc
        pos_idx = pos_idx.clamp(max=self.max_tokens - 1)
        tokens  = tokens + self.pos_emb(pos_idx).unsqueeze(0)

        # Ego conditioning
        tokens = tokens + self.ego_proj(ego_y).unsqueeze(1)

        # Padding mask: True = this position should be IGNORED by attention
        # Shape [B, max_filled]; positions beyond n_filled[b] are padding
        pad_mask = torch.arange(max_filled, device=dev).unsqueeze(0) < (
            max_filled - n_filled.unsqueeze(1))     # [B, max_filled]

        # Transformer
        out = self.transformer(tokens, src_key_padding_mask=pad_mask)  # [B, max_filled, d]

        # Mean-pool over non-padding tokens
        valid = (~pad_mask).float().unsqueeze(-1)   # [B, max_filled, 1]
        ctx   = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B, d_model]

        pos_logits = self.pos_head(ctx)             # [B, 5, 65]
        vel_pred   = self.vel_head(ctx).squeeze(-1) # [B, 5]

        new_state = {
            'chunk_buf':  new_buf,
            'tick_ctr':   tick_ctr,
            'tick_accum': new_accum,
        }
        return pos_logits, vel_pred, new_state


# ── Model C: GRU + inter-lane transformer ────────────────────────────────────

class ModelC(nn.Module):
    """
    Like A but the final representation is a transformer over 5 per-lane tokens
    constructed from the GRU output, allowing explicit inter-lane reasoning.

    State keys: prev_pos, prev_vel, hidden
    """
    def __init__(self, bins: int = BINS, gru_hidden: int = 64,
                 d_model: int = 64, n_heads: int = 4,
                 cnn_dim: int = 64, lane_emb_dim: int = 16):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.bins       = bins
        self.d_model    = d_model

        self.sensor_enc  = nn.Sequential(
            nn.Linear(D_SENSOR, 64), nn.GELU(), nn.Linear(64, 64))
        self.prev_pos_cnn = PrevPosCNN(bins=bins, out_dim=cnn_dim)

        self.belief_proj = nn.Linear(cnn_dim + N_LANES, 64)
        self.gru         = nn.GRUCell(64, gru_hidden)

        # Project GRU output + sensor + ego_y into per-lane tokens
        token_in = gru_hidden + 64 + 1             # gru + sensor + ego_y
        self.token_proj = nn.Linear(token_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.pos_head = nn.Linear(d_model, bins + 1)
        self.vel_head = nn.Linear(d_model, 1)

    def init_state(self, B: int, device):
        pos = torch.zeros(B, N_LANES, self.bins + 1, device=device)
        pos[..., self.bins] = 1.0
        return {
            'prev_pos': pos,
            'prev_vel': torch.zeros(B, N_LANES, device=device),
            'hidden':   torch.zeros(B, self.gru_hidden, device=device),
        }

    def forward(self, sensors_raw, ego_y_raw, state):
        B = sensors_raw.size(0)
        prev_pos = state['prev_pos']
        prev_vel = state['prev_vel']
        hidden   = state['hidden']

        s     = preprocess_sensors(sensors_raw)
        ego_y = ego_y_raw / EGO_Y_MAX

        s_enc   = self.sensor_enc(s)
        cnn_out = self.prev_pos_cnn(prev_pos)
        h = self.gru(F.gelu(self.belief_proj(
            torch.cat([cnn_out, prev_vel], dim=1))), hidden)

        # Build per-lane tokens: broadcast global state to 5 lane vectors
        ctx = torch.cat([h, s_enc, ego_y], dim=1)                       # [B, gru_h+64+1]
        tokens = self.token_proj(ctx).unsqueeze(1).expand(-1, N_LANES, -1)  # [B, 5, d]
        tokens = self.transformer(tokens)                                 # [B, 5, d]

        pos_logits = self.pos_head(tokens)                               # [B, 5, 65]
        vel_pred   = self.vel_head(tokens).squeeze(-1)                   # [B, 5]

        pos_prob  = F.softmax(pos_logits, dim=-1)
        new_state = {
            'prev_pos': pos_prob,
            'prev_vel': vel_pred.detach(),
            'hidden':   h,
        }
        return pos_logits, vel_pred, new_state


# ── Model D: feedforward with output feedback (no GRU) ───────────────────────

class ModelD(nn.Module):
    """
    Same inputs as A but no GRU. Temporal memory lives entirely in the
    prev_pos belief (compressed by CNN) and prev_vel.

    State keys: prev_pos [B,5,BINS+1], prev_vel [B,5]
    """
    def __init__(self, bins: int = BINS, fuse_dim: int = 256,
                 cnn_dim: int = 64, lane_emb_dim: int = 16):
        super().__init__()
        self.bins = bins

        self.sensor_enc  = nn.Sequential(
            nn.Linear(D_SENSOR, 64), nn.GELU(), nn.Linear(64, 64))
        self.prev_pos_cnn = PrevPosCNN(bins=bins, out_dim=cnn_dim)

        fuse_in = 64 + cnn_dim + N_LANES + 1     # sensor + cnn + prev_vel + ego_y
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim), nn.GELU(),
            nn.Linear(fuse_dim, 128),      nn.LayerNorm(128),       nn.GELU(),
        )
        self.pos_head = PerLaneHead(128, bins + 1, lane_emb_dim)
        self.vel_head = PerLaneHead(128, 1, lane_emb_dim)

    def init_state(self, B: int, device):
        pos = torch.zeros(B, N_LANES, self.bins + 1, device=device)
        pos[..., self.bins] = 1.0
        return {
            'prev_pos': pos,
            'prev_vel': torch.zeros(B, N_LANES, device=device),
        }

    def forward(self, sensors_raw, ego_y_raw, state):
        prev_pos = state['prev_pos']
        prev_vel = state['prev_vel']

        s     = preprocess_sensors(sensors_raw)
        ego_y = ego_y_raw / EGO_Y_MAX

        s_enc   = self.sensor_enc(s)
        cnn_out = self.prev_pos_cnn(prev_pos)

        fused      = self.fusion(torch.cat([s_enc, cnn_out, prev_vel, ego_y], dim=1))
        pos_logits = self.pos_head(fused)
        vel_pred   = self.vel_head(fused).squeeze(-1)

        pos_prob  = F.softmax(pos_logits, dim=-1)
        new_state = {
            'prev_pos': pos_prob,
            'prev_vel': vel_pred.detach(),
        }
        return pos_logits, vel_pred, new_state


# ── factory ───────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'model_a': dict(
        cls='ModelA', bins=64, gru_hidden=128, fuse_dim=256, cnn_dim=64),
    'model_a_lg': dict(
        cls='ModelA', bins=64, gru_hidden=256, fuse_dim=512, cnn_dim=64),
    'model_b': dict(
        cls='ModelB', bins=64, chunk_size=4, max_ticks=500,
        d_model=64, n_heads=4, n_layers=2),
    'model_b_lg': dict(
        cls='ModelB', bins=64, chunk_size=4, max_ticks=500,
        d_model=128, n_heads=4, n_layers=3),
    'model_c': dict(
        cls='ModelC', bins=64, gru_hidden=64, d_model=64,
        n_heads=4, cnn_dim=64),
    'model_d': dict(
        cls='ModelD', bins=64, fuse_dim=256, cnn_dim=64),
}

_CLS_MAP = {
    'ModelA': ModelA, 'ModelB': ModelB, 'ModelC': ModelC, 'ModelD': ModelD}


def build_model(name: str) -> nn.Module:
    cfg = dict(MODEL_CONFIGS[name])
    cls = _CLS_MAP[cfg.pop('cls')]
    return cls(**cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

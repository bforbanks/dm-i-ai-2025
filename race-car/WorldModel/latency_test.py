#!/usr/bin/env python3
"""
Latency benchmark for WorldModel candidate architectures.

Run from project root:
    python race-car/WorldModel/latency_test.py

Tests CPU inference time for a single tick (batch=1) across several
architecture configurations, to check whether they fit the 30ms budget
on a ThinkPad T16 Gen 4.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared building blocks ────────────────────────────────────────────────────

class SensorEncoder(nn.Module):
    """1D conv over a window of sensor readings [window, 16] → [d_out]."""
    def __init__(self, window: int = 5, d_sensor: int = 16, d_out: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_sensor, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, d_out, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.d_out = d_out

    def forward(self, x):
        # x: [B, window, 16] → transpose → [B, 16, window]
        x = x.transpose(1, 2)
        x = self.net(x)            # [B, d_out, window]
        return x.mean(dim=2)       # [B, d_out]


# ── Option A: GRU belief propagator ──────────────────────────────────────────

class OptionA(nn.Module):
    """GRU belief propagator with shared position head."""
    def __init__(self, bins: int = 32, gru_hidden: int = 128,
                 d_sensor_enc: int = 32, d_fuse: int = 256):
        super().__init__()
        self.bins   = bins
        self.n_lanes = 5
        out_per_lane = bins + 1

        self.sensor_enc = SensorEncoder(d_out=d_sensor_enc)

        belief_in = self.n_lanes * out_per_lane + self.n_lanes  # pos_flat + vel_flat
        self.belief_proj = nn.Linear(belief_in, 64)
        self.gru = nn.GRUCell(64, gru_hidden)

        fuse_in = d_sensor_enc + gru_hidden + 1   # + ego_y
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, d_fuse), nn.LayerNorm(d_fuse), nn.GELU(),
            nn.Linear(d_fuse, d_fuse), nn.LayerNorm(d_fuse), nn.GELU(),
            nn.Linear(d_fuse, 128),    nn.LayerNorm(128),    nn.GELU(),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, out_per_lane),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, self.n_lanes),
        )

    def forward(self, sensor_window, ego_y, prev_pos, prev_vel, hidden):
        # sensor_window: [B, 5, 16]
        # ego_y:         [B, 1]
        # prev_pos:      [B, 5, bins+1]
        # prev_vel:      [B, 5]
        # hidden:        [B, gru_hidden]
        B = sensor_window.size(0)

        s_enc = self.sensor_enc(sensor_window)                       # [B, 32]
        belief = torch.cat([prev_pos.flatten(1), prev_vel], dim=1)   # [B, 5*(B+1)+5]
        h = self.gru(F.gelu(self.belief_proj(belief)), hidden)       # [B, gru_hidden]

        fused = self.fusion(torch.cat([s_enc, h, ego_y], dim=1))     # [B, 128]

        pos_logits = self.pos_head(fused).unsqueeze(1).expand(-1, 5, -1)  # [B, 5, bins+1]
        vel_pred   = self.vel_head(fused)                                  # [B, 5]

        return pos_logits, vel_pred, h

    def init_state(self, batch_size: int, device):
        bins = self.bins
        pos = torch.zeros(batch_size, self.n_lanes, bins + 1, device=device)
        pos[..., -1] = 1.0   # all mass on "no car" class (last bin)
        vel    = torch.zeros(batch_size, self.n_lanes, device=device)
        hidden = torch.zeros(batch_size, self.gru.hidden_size, device=device)
        return pos, vel, hidden


# ── Option B: Transformer over lane tokens ───────────────────────────────────

class LaneTransformer(nn.Module):
    """Small transformer with lane tokens."""
    def __init__(self, bins: int = 32, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.bins    = bins
        self.n_lanes = 5
        out_per_lane = bins + 1

        self.sensor_enc  = SensorEncoder(d_out=32)
        # Each lane token: sensor_enc(32, shared) + prev_pos_lane(B+1) + prev_vel_lane(1) + ego_y(1)
        token_in = 32 + out_per_lane + 1 + 1
        self.token_proj  = nn.Linear(token_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                               batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.pos_head = nn.Linear(d_model, out_per_lane)
        self.vel_head = nn.Linear(d_model, 1)

    def forward(self, sensor_window, ego_y, prev_pos, prev_vel):
        # sensor_window: [B, 5, 16]
        # ego_y:         [B, 1]
        # prev_pos:      [B, 5, bins+1]
        # prev_vel:      [B, 5]
        B = sensor_window.size(0)

        s_enc = self.sensor_enc(sensor_window)                       # [B, 32]
        s_enc = s_enc.unsqueeze(1).expand(-1, 5, -1)                # [B, 5, 32]
        ego_e = ego_y.unsqueeze(1).expand(-1, 5, -1)                # [B, 5, 1]
        vel_e = prev_vel.unsqueeze(-1)                               # [B, 5, 1]

        tokens = torch.cat([s_enc, prev_pos, vel_e, ego_e], dim=-1) # [B, 5, token_in]
        tokens = self.token_proj(tokens)                             # [B, 5, d_model]
        tokens = self.transformer(tokens)                            # [B, 5, d_model]

        pos_logits = self.pos_head(tokens)                           # [B, 5, bins+1]
        vel_pred   = self.vel_head(tokens).squeeze(-1)               # [B, 5]
        return pos_logits, vel_pred

    def init_state(self, batch_size: int, device):
        bins = self.bins
        pos = torch.zeros(batch_size, self.n_lanes, bins + 1, device=device)
        pos[..., -1] = 1.0
        vel = torch.zeros(batch_size, self.n_lanes, device=device)
        return pos, vel


# ── Option C: GRU + single transformer layer ─────────────────────────────────

class OptionC(nn.Module):
    """Hybrid: GRU for temporal memory, then one transformer over lane tokens."""
    def __init__(self, bins: int = 32, gru_hidden: int = 64, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.bins    = bins
        self.n_lanes = 5
        out_per_lane = bins + 1

        self.sensor_enc  = SensorEncoder(d_out=32)
        belief_in = self.n_lanes * out_per_lane + self.n_lanes
        self.belief_proj = nn.Linear(belief_in, 64)
        self.gru         = nn.GRUCell(64, gru_hidden)

        # per-lane token: gru_out_slice(gru_hidden//5→~12) + sensor(32) + prev_pos(B+1) + ego_y(1)
        lane_slice = gru_hidden // self.n_lanes  # 12 or 13
        token_in   = lane_slice + 32 + out_per_lane + 1 + 1
        self.token_proj  = nn.Linear(token_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                               batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pos_head    = nn.Linear(d_model, out_per_lane)
        self.vel_head    = nn.Linear(d_model, 1)
        # project gru hidden → 5 × lane_slice via linear (avoids divisibility constraint)
        self.h_to_lanes  = nn.Linear(gru_hidden, self.n_lanes * lane_slice)

        self.lane_slice = lane_slice
        self.gru_hidden = gru_hidden

    def forward(self, sensor_window, ego_y, prev_pos, prev_vel, hidden):
        B = sensor_window.size(0)
        s_enc   = self.sensor_enc(sensor_window)                      # [B, 32]
        belief  = torch.cat([prev_pos.flatten(1), prev_vel], dim=1)
        h       = self.gru(F.gelu(self.belief_proj(belief)), hidden)  # [B, gru_hidden]

        h_lanes = self.h_to_lanes(h).view(B, self.n_lanes, self.lane_slice)  # [B, 5, lane_slice]
        s_e     = s_enc.unsqueeze(1).expand(-1, 5, -1)                # [B, 5, 32]
        ego_e   = ego_y.unsqueeze(1).expand(-1, 5, -1)                # [B, 5, 1]
        vel_e   = prev_vel.unsqueeze(-1)                               # [B, 5, 1]

        tokens  = torch.cat([h_lanes, s_e, prev_pos, vel_e, ego_e], dim=-1)
        tokens  = self.token_proj(tokens)                              # [B, 5, d_model]
        tokens  = self.transformer(tokens)                             # [B, 5, d_model]

        pos_logits = self.pos_head(tokens)                             # [B, 5, bins+1]
        vel_pred   = self.vel_head(tokens).squeeze(-1)                 # [B, 5]
        return pos_logits, vel_pred, h

    def init_state(self, batch_size: int, device):
        bins = self.bins
        pos    = torch.zeros(batch_size, self.n_lanes, bins + 1, device=device)
        pos[..., -1] = 1.0
        vel    = torch.zeros(batch_size, self.n_lanes, device=device)
        hidden = torch.zeros(batch_size, self.gru_hidden, device=device)
        return pos, vel, hidden


# ── benchmark harness ─────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(name, model, make_inputs, n_warmup=50, n_runs=500):
    model.eval()
    device = next(model.parameters()).device

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            inputs = make_inputs(device)
            _ = model(*inputs)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            inputs = make_inputs(device)
            t0 = time.perf_counter()
            _ = model(*inputs)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    p50 = times[n_runs // 2]
    p95 = times[int(n_runs * 0.95)]
    p99 = times[int(n_runs * 0.99)]
    print(f"  {name:<40} params={count_params(model):>7,}  "
          f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  "
          f"{'OK' if p99 < 30 else 'SLOW'}")
    return p50, p99


def make_a_inputs(model):
    def fn(device):
        pos, vel, hidden = model.init_state(1, device)
        sw   = torch.rand(1, 5, 16, device=device)
        eg_y = torch.rand(1, 1, device=device)
        return sw, eg_y, pos, vel, hidden
    return fn


def make_b_inputs(model):
    def fn(device):
        pos, vel = model.init_state(1, device)
        sw   = torch.rand(1, 5, 16, device=device)
        eg_y = torch.rand(1, 1, device=device)
        return sw, eg_y, pos, vel
    return fn


def make_c_inputs(model):
    def fn(device):
        pos, vel, hidden = model.init_state(1, device)
        sw   = torch.rand(1, 5, 16, device=device)
        eg_y = torch.rand(1, 1, device=device)
        return sw, eg_y, pos, vel, hidden
    return fn


if __name__ == "__main__":
    torch.set_num_threads(1)   # single-threaded: simulates real-time inference
    device = torch.device("cpu")

    print("=" * 80)
    print("  WorldModel latency benchmark  (CPU, batch=1, single-threaded)")
    print("  Target: p99 < 30 ms")
    print("=" * 80)

    configs = []

    # Option A variants
    for bins in [32, 64]:
        for gru_h in [64, 128]:
            for d_fuse in [128, 256]:
                m = OptionA(bins=bins, gru_hidden=gru_h, d_fuse=d_fuse).to(device)
                configs.append((f"A  bins={bins} gru={gru_h} fuse={d_fuse}", m, make_a_inputs(m)))

    # Option B variants
    for bins in [32, 64]:
        for d_model in [32, 64, 128]:
            for n_layers in [1, 2]:
                m = LaneTransformer(bins=bins, d_model=d_model, n_layers=n_layers).to(device)
                configs.append((f"B  bins={bins} d={d_model} L={n_layers}", m, make_b_inputs(m)))

    # Option C variants
    for bins in [32, 64]:
        for gru_h in [64, 128]:
            for d_model in [64, 128]:
                m = OptionC(bins=bins, gru_hidden=gru_h, d_model=d_model).to(device)
                configs.append((f"C  bins={bins} gru={gru_h} d={d_model}", m, make_c_inputs(m)))

    print()
    for name, model, make_inputs in configs:
        benchmark(name, model, make_inputs)

    print()
    print("Note: torch.set_num_threads(1) simulates single-threaded inference.")
    print("Production inference with all threads available will be faster.")

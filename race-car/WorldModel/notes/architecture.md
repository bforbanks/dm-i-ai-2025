# WorldModel — Architecture Notes

## Task
Given raw sensor readings and ego lane position, predict at each tick:
1. **Position**: For each of the 5 lanes, a probability distribution over where the NPC car is (or that the lane is empty). Represented as a histogram: `bins` position bins + 1 "no car" class.
2. **Velocity**: For each lane, the relative velocity of the NPC car (car.vx − ego.vx). Only meaningful when a car is present.

---

## Input/Output spec

### Inputs (per tick)
| Input | Dim | Notes |
|-------|-----|-------|
| Sensor values (last 5 ticks) | 5 × 16 | NaN → 0; include separate NaN mask 5×16 → total 5×32 or just 5×16 with 0-fill |
| `ego.y` | 1 | Only meaningful ego coordinate; x is always 620 |
| Previous position distribution | 5 × (B+1) | B=bins, +1 for "no car". Initialized to 100% "no car". |
| Previous velocity prediction | 5 | Initialized to 0.0 (or dataset mean ≈ −7.6 px/tick) |

### Outputs (per tick)
| Output | Dim | Notes |
|--------|-----|-------|
| Position logits | 5 × (B+1) | Softmax per lane → probability distribution |
| Velocity predictions | 5 | Regression; MSE/Huber loss masked to lanes with cars |

With B=32: output dim = 5×33 + 5 = **170**  
With B=64: output dim = 5×65 + 5 = **330**

---

## Architecture Option A — GRU Belief Propagator (your proposal, refined)

```
Inputs at tick t:
  sensor_window   : [5, 16]   ← last 5 ticks of sensor readings (NaN→0)
  ego_y           : [1]
  prev_pos_dist   : [5, B+1]  ← own output from tick t-1
  prev_vel        : [5]

Step 1 — Sensor temporal encoder:
  sensor_window [5, 16]
    → Conv1D(16→32, kernel=3, pad=1) + ReLU      [5, 32]
    → Conv1D(32→32, kernel=3, pad=1) + ReLU      [5, 32]
    → mean-pool over time dim                    [32]

Step 2 — Belief encoder (GRU hidden state carries cross-tick memory):
  prev_pos_flat  = flatten(prev_pos_dist)        [5*(B+1)]
  prev_vel_flat  = prev_vel                      [5]
  belief_input   = concat(prev_pos_flat, prev_vel_flat)  [5*(B+1)+5]
    → Linear → [64] → GRU(hidden=128)
  gru_out        : [128]       ← carries recurrent state across ticks

Step 3 — Fusion FFN:
  fused = concat(sensor_enc[32], gru_out[128], ego_y[1])   [161]
    → Linear(161→256) + LayerNorm + GELU
    → Linear(256→256) + LayerNorm + GELU
    → Linear(256→128) + LayerNorm + GELU
  repr : [128]

Step 4 — Position head (one per lane, shared weights):
  repr [128]
    → Linear(128→64) + GELU                    [64]
    → Linear(64→B+1)                            [B+1]   ← logits
    → Softmax                                   [B+1]   ← probability distribution
  Applied 5 times (lane-wise, shared weights).

Step 5 — Velocity head:
  repr [128]
    → Linear(128→64) + GELU
    → Linear(64→5)                              [5]     ← one scalar per lane
```

**Parameters (B=32):**
- Sensor encoder: ~3K
- Belief input projection: (5×33+5)×64 ≈ 11K; GRU: ~100K
- Fusion FFN: ~200K
- Position head (shared): ~10K
- Velocity head: ~10K
- **Total: ~330K**

**Parameters (B=64):**
- Belief input projection grows: (5×65+5)×64 ≈ 21K; rest similar
- **Total: ~350K**

---

## Architecture Option B — Transformer (lane tokens)

Treat the **5 lanes** as tokens in a small transformer. Sensors are projected to a per-lane embedding, then lanes attend to each other.

```
Per-lane input construction:
  For lane l:
    sensor_enc_l = temporal_sensor_enc(sensor_window)    [32]  ← shared
    lane_belief_l = prev_pos_dist[l]                     [B+1]
    lane_vel_l    = prev_vel[l]                          [1]
    token_l       = concat(sensor_enc_l, lane_belief_l, lane_vel_l, ego_y)
                    [32 + B+1 + 1 + 1 = B+35]
    → Linear(B+35 → D)                                  [D]   D = model_dim

Transformer:
  tokens : [5, D]
  → 2–3 layers of multi-head self-attention (heads=4) + FFN
  → output : [5, D]

Heads:
  Position: token[l] → Linear(D→B+1) → softmax
  Velocity: token[l] → Linear(D→1)
```

With D=64, 2 layers, 4 heads:
- ~150K parameters total (B=32)
- Only 5 tokens → attention is O(25), trivially fast

**Advantage over A**: lanes can explicitly attend to each other — important because the ego's decision and sensor readings mix information from multiple lanes.

---

## Architecture Option C — Hybrid (recommended)

Combine A and B: use the GRU belief state for temporal continuity, then run one transformer layer over 5 lane tokens that each receive a slice of the GRU output.

```
[temporal sensor enc → GRU hidden] × tick
→ GRU output [128] → split to 5 lane vectors [5, 25]
→ concat each with ego_y, lane-specific prev belief
→ 1 transformer layer over 5 tokens
→ per-lane heads
```

~200K params, very fast.

---

## Latency budget (ThinkPad T16 Gen 4)

Target: **≤ 30 ms** per tick (game runs at 60 fps → 16.7 ms/tick, but we can double-buffer and process one tick behind).

Rough estimates for CPU inference:
| Model | Params | Expected latency (CPU, batch=1) |
|-------|--------|---------------------------------|
| Option A (B=32) | ~330K | ~2–5 ms |
| Option A (B=64) | ~350K | ~3–6 ms |
| Option B (B=32, D=64) | ~150K | ~1–3 ms |
| Option C (B=32) | ~200K | ~2–4 ms |

All options should be well within 30 ms. The bottleneck will be Python/PyTorch overhead (per-call overhead ~0.5–2 ms on CPU), not compute.

→ **Run latency_test.py before finalizing architecture.**

---

## Normalization / preprocessing

| Feature | Normalization |
|---------|--------------|
| sensors (non-NaN) | Divide by 1000 → [0, 1] (NaN → 0) |
| NaN mask | 1.0 = present, 0.0 = absent (no separate mask needed if 0-fill is used) |
| ego.y | Divide by 1200 → [0, 1] |
| car_x labels (histogram bins) | Range [-1000, 2600]; bin width = 3600/B |
| car_vx labels | Divide by 28 → roughly [-1, 1] (covers p99.9 of range) |
| prev_vel input | Same normalization as velocity labels |

---

## Loss

```
L = L_pos + λ_vel * L_vel

L_pos = mean over lanes and ticks of:
    CrossEntropy(pos_logits[lane], target_bin[lane])
    where target = "no car" class if NaN, else the bin containing car_x

L_vel = mean over (lane, tick) where car is present of:
    Huber(vel_pred[lane], car_vx[lane] / 28.0)
```

λ_vel ≈ 0.1–0.5 (tune; velocity is secondary).

---

## Bins

B=32 → 3600/32 = 112.5 px resolution  
B=64 → 56 px resolution  
B=128 → 28 px resolution

The ego car is ~360 px wide. B=32 gives ~3× car-width resolution — coarse but sufficient for the LaneShift agent's needs. B=64 is a good default.

---

## Open design questions

1. **NaN handling**: 0-fill vs learned NaN embedding. 0-fill is simplest (sensor = 0 means "at distance 0" which is physically impossible, so the model can learn it means "no detection"). A separate binary mask doubles sensor input size but may train faster.
2. **Shared vs per-lane position head weights**: Lanes are geometrically identical, so sharing weights is natural. But the agent has asymmetric sensor geometry relative to lanes — per-lane may be worth trying.
3. **Bins = B**: Start at 32 or 64. Can increase after latency is confirmed acceptable.
4. **GRU hidden dim**: 64 or 128. Test latency first.
5. **Initializing hidden state**: At game start, prev_pos_dist = [0, 0, ..., 0, 1] (all mass on "no car"), prev_vel = 0. For the sensor window, backward-fill with the first actual observation (tile tick-0 sensors back 4 steps).
6. **Training sequence length**: BPTT through T=60–120 ticks. Truncated BPTT with detach to manage memory.

# Train Run 1 — WorldModel

Six models trained concurrently on the DTU HPC (gpuv100 queue).

## Dataset split (seed=42, by game)


| Split | Games | Ticks (approx) | Purpose                          |
| ----- | ----- | -------------- | -------------------------------- |
| train | 8100  | ~19.0M         | gradient updates                 |
| val   | 1000  | ~2.3M          | early stopping (best checkpoint) |
| test  | 900   | ~2.1M          | final holdout, reported once     |


Games are split at the game level to avoid tick-level leakage.

---

## Models


| Job | Name       | Architecture                        | Params (approx) | Key hyperparams         |
| --- | ---------- | ----------------------------------- | --------------- | ----------------------- |
| A   | model_a    | GRU belief propagator               | ~270K           | gru=128, fuse=256       |
| B   | model_a_lg | GRU belief propagator (large)       | ~590K           | gru=256, fuse=512       |
| C   | model_b    | Chunked-history transformer         | ~115K           | d=64, L=2, N_chunks=16  |
| D   | model_b_lg | Chunked-history transformer (large) | ~400K           | d=128, L=3, N_chunks=16 |
| E   | model_c    | Hybrid GRU + inter-lane attn        | ~160K           | gru=64, d=64            |
| F   | model_d    | Feedforward + output feedback       | ~330K           | fuse=256 (no GRU)       |


### Architecture brief

**Model A / A_lg  (GRU belief propagator)**

- Input per tick: sensors [32] (values/1000 + NaN mask) + ego_y [1]
- CNN over previous position distribution [5, 65] → compressed belief [64]
- Compressed belief + previous velocity → GRU cell → hidden state [128/256]
- Fusion FFN(sensor_enc + GRU_hidden + ego_y) → 128-d representation
- Per-lane position head (with lane embedding) → 5 × 65 logits
- Per-lane velocity head → 5 scalars

**Model B / B_lg  (Chunked-history transformer)**

- No recurrent state; all temporal context from sensor history
- Sensor history partitioned into N_chunks=16 chunks of chunk_size=4 ticks each
(= 64 ticks of context going backwards from current tick)
- Each chunk: [4, 32] → MLP embedder → [d_model]
- Chunk tokens + positional encoding + ego_y → TransformerEncoder → mean pool
- Per-lane heads (with lane embedding)

**Model C  (Hybrid)**

- CNN over prev_pos + GRU (hidden=64) for temporal memory
- GRU output projected to 5 lane tokens → 1-layer TransformerEncoder for inter-lane attention
- Per-lane position and velocity heads

**Model D  (Feedforward with output feedback)**

- Identical inputs to A but NO GRU hidden state
- Relies entirely on prev_pos CNN + prev_vel for temporal context
- Deeper FFN (4 layers) to compensate

---

## Training hyperparameters (all models)


| Parameter               | Value                                                    |
| ----------------------- | -------------------------------------------------------- |
| T_seg                   | 120 ticks per training segment                           |
| Batch size              | 1024 segments (A, A_lg, C, D); 32 (B, B_lg)             |
| Optimizer               | AdamW (weight_decay=1e-4)                                |
| Learning rate           | 3e-4 with CosineAnnealingLR (eta_min=1.5e-5)             |
| Loss                    | CrossEntropy (position) + 0.1 × Huber(velocity, delta=5) |
| Early stopping patience | 10 epochs of no val improvement                          |
| Max epochs              | 5000                                                     |
| Grad clip               | 1.0                                                      |


Note on stateless TBPTT: model state is reset to zero at the start of each
training segment. The model therefore learns to propagate state within a
120-tick window. At inference time, state is carried across the full game
(up to 3600 ticks). This train-inference gap may favour architectures that
learn compact, informative beliefs quickly (A and C).

---

## 7. Experiment 7 — full-game training (separate from jobs A–F)

`train_fullgame.py` + `submit_exp7_fullgame.sh` train on **whole games** (one batch row = one game from start to end). Games are **sorted by length** and batched so padding within a batch is minimized; loss ignores padded timesteps. RNN state resets **only at the start of each game** (row), not every 120 ticks.

- **Not** included in `submit_all.sh` — submit manually when you want this run.
- **`--tbptt-chunk`**: default `200` detaches hidden state every *N* ticks to limit backprop memory through time (`0` = full BPTT over the whole padded length; can OOM on long games).
- **`--max-ticks`**: optional cap on game length (prefix of each game).
- **`--num-workers`**: default `0` (safe with the custom batch sampler).
- Checkpoint is `{run_name}_best.pt` with extra keys `train_mode`, `max_ticks`, `tbptt_chunk`.

```bash
cd ~/Desktop/dm-i-ai-2025
bsub < race-car/WorldModel/train_run_1/submit_exp7_fullgame.sh
# or locally:
python race-car/WorldModel/train_run_1/train_fullgame.py --model model_a --data laneshift_dataset.npz
```

All six architectures in `MODEL_CONFIGS` are supported (same tick-wise `forward` as `train.py`).

---

## 8. Experiment 8 — `model_a_lg` + SensorParser vector

Separate from segment jobs A–F and from experiment 7. Trains **`ModelALgWithParser`**: same GRU size as `model_a_lg`, plus a **20-D** tick input from `LaneShift/sensor_parser.py` (per lane: `x/1000`, `x_mask`, `v/28`, `v_mask`; **0** when the parser has no value).

- Script: `train_exp8.py` · submit: `submit_exp8_parser.sh`
- Dataset: `parser_dataset.py` precomputes parser features in **one pass** over the `.npz` (parser reset each game).
- First epoch startup includes parser precompute (can take a few minutes).

```bash
cd ~/Desktop/dm-i-ai-2025
bsub < race-car/WorldModel/train_run_1/submit_exp8_parser.sh
```

---

## 9. Experiment 9 — minimal MLP on full games

**One GELU hidden layer** (default **1024**), then a linear head. Per-tick input = **preprocessed sensors + ego_y + SensorParser (20-D) + previous tick’s softmax position belief (flattened) + previous tick’s predicted velocities** — same feedback idea as Model A/D, without a GRU/CNN.

Training runs a **tick loop** over padded games (like exp7), with optional **`--tbptt-chunk`** (default **200**) to cap backprop through time.

- Script: `simple_mlp_fullgame.py` · submit: `submit_exp9_simple_mlp.sh`
- Dataset: `LaneShiftGameDatasetWithParser` · collate pads `parser` when present.
- Default **`--batch-size 4`** — raise only if VRAM allows (cost scales with `T_max × B`).
- Optional **`--max-ticks`** to cap sequence length for speed / memory.

```bash
cd ~/Desktop/dm-i-ai-2025
bsub < race-car/WorldModel/train_run_1/submit_exp9_simple_mlp.sh
```

---

## Expected outputs

Each job writes:

- `checkpoints/{model}_best.pt`  — best checkpoint by val loss
- `gpu_{JOB_ID}.out` / `.err`  — LSF stdout/stderr
- W&B run at project `laneshift-worldmodel`

---

## Submission

```bash
cd ~/Desktop/dm-i-ai-2025
bash race-car/WorldModel/train_run_1/submit_all.sh
```

Monitor:

```bash
bstat                     # all running jobs
bpeek <JOB_ID>            # tail live output
```


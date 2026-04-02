# LaneShift Development Notes

## Project goal
Build a **world model** for the car environment: given raw sensor readings, predict where each of the 5 NPC cars is (or whether there is no car in a given lane).

Output representation: 5 histograms (one per lane), each with `bins` bins spanning [x_min, x_max] plus one extra "no car" probability. Softmax per histogram enforces the simplex constraint. Total output size = `bins * 5 + 5`.

---

## Data collection (`collect_data.py`)

### What is saved (ragged-as-flat format)
| Array          | Shape              | Dtype   | Meaning |
|----------------|--------------------|---------|---------|
| `sensors`      | [total_ticks, 16]  | float32 | Raw sensor distances; NaN = no detection |
| `car_x`        | [total_ticks, 5]   | float32 | NPC car x-position per lane; NaN = lane empty |
| `car_vx`       | [total_ticks, 5]   | float32 | NPC car x-velocity relative to ego per lane; NaN = lane empty |
| `game_lengths` | [N_games]          | int32   | Number of ticks in each game |

Games are concatenated along axis 0. Use `game_lengths` to split them back:
```python
data = np.load("laneshift_dataset.npz")
starts = np.concatenate([[0], np.cumsum(data["game_lengths"][:-1])])
# game i:
s = data["sensors"][starts[i] : starts[i] + data["game_lengths"][i]]
```

### Why ragged-as-flat
Games end at different ticks (crash vs timeout). Padding to max_length wastes ~10× memory for short games. Storing as Python object arrays prevents memory-mapping. The flat+lengths approach is efficient, memory-mappable, and trivial to batch in a DataLoader.

### Sensor ordering (columns)
Matches `sensor_options` in `src/game/core.py`:
```
front, right_front, right_side, right_back, back,
left_back, left_side, left_front,
left_side_front, front_left_front, front_right_front,
right_side_front, right_side_back, back_right_back,
back_left_back, left_side_back
```

### Ground truth (car_x)
- `car_x[i, lane_idx]` = `car.x` (screen x-coordinate) for the NPC car in that lane
- x ranges roughly from **-1000** (despawn/spawn behind) to **2600** (despawn/spawn ahead)
  - Ego car is centered at ~800 on a 1600-wide screen
  - Cars with `x < -1000` or `x > 2600` are removed from the game
- At most 4 of the 5 lanes have an NPC car simultaneously (only 4 NPC cars in bucket)
- Early ticks of each game have all-NaN ground truth (cars haven't spawned yet)

### Issues / design choices
- **Broken imports in LaneShift.py**: `LaneShift.py` imports from `models.utilities.sensor_parser` and `models.laneshift_config`, which are aliases for the actual files in `LaneShift/`. The script patches `sys.modules` before import to resolve this without touching the original files.
- **Headless pygame**: `SDL_VIDEODRIVER=dummy` + `SDL_AUDIODRIVER=dummy` env vars before `pygame.init()`. A `1×1` display mode is still required because `Road.__init__` creates a `pygame.Surface` and `Car.load_sprite` calls `pygame.image.load`.
- **Tick management**: `update_game()` from `core.py` does not increment `STATE.ticks`; the script does it manually to match the game_loop's semantics.
- **Collision detection**: Not wired into `update_game()`; the script calls `_check_collisions()` after each step, matching what `game_loop` does.
- **Agent**: Uses the actual `LaneShiftAgent` so data reflects realistic gameplay scenarios (challenging overtakes, lane changes, emergency braking). A random policy would also work but produce less representative data.
- **Observation order**: Sample is collected *before* the action is applied. Sensors at time t were updated at the end of tick t-1, so they are consistent with car positions at time t.

### How to run
```bash
cd race-car
python LaneShift/collect_data.py                        # 10 000 games, seed 0
python LaneShift/collect_data.py --n-games 50000        # more games
python LaneShift/collect_data.py --save-path data/train.npz
```

---

## Model design (planned, not yet implemented)
- Input: 16 sensor values (NaN → 0 + mask, or learned NaN embedding)
- Shared encoder → two heads:
  - **Position head**: `bins * 5 + 5` logits → 5 independent softmaxes (one per lane)
    - Each softmax: `bins` probability mass positions + 1 "no car" class
    - x range: [-1000, 2600] (despawn bounds); `bins` = 200 (hyperparameter)
    - Loss: cross-entropy against the bin containing true x, or the no-car class
  - **Velocity head**: 5 scalar outputs, one per lane (relative vx of NPC vs ego)
    - Loss: masked MSE / Huber, only computed on lanes where `car_vx` is not NaN
    - λ-weighted addition to total loss

### Why cross-entropy converges to a proper distribution
Minimising expected cross-entropy H(p, q) over q is equivalent to minimising KL(p‖q) = H(p,q) - H(p), since H(p) is fixed. The information inequality KL(p‖q) ≥ 0 with equality iff p = q (proved via Jensen on -log) guarantees the minimiser is the true conditional distribution. In this setting: the same sensor pattern can arise from many game histories (NPC speeds are random, spawning is random), so the model is forced to learn P(bin | sensors) properly — a real distribution, not a delta.

### Velocity: save relative (car.vx - ego.vx), not absolute
- Relative velocity is what sensors encode (rate of change of detected distance)
- Can replace the current sensor-delta estimator in `LaneShift.determine_velocity_front/side`
- NPC y-velocity is always 0; only x matters

---

## Open questions
- Should we also predict NPC car velocity? That might be useful for the LaneShift agent to replace `sensor_parser.py`'s velocity estimates.
- How many game seeds to use for train/val/test split?
- Should early (all-NaN) ticks be excluded from training?

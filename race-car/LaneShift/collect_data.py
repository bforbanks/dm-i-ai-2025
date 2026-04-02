#!/usr/bin/env python3
"""
Collect full games of (raw_sensor, ground_truth_car_x) data from LaneShift.

Run from the project root (dm-i-ai-2025/):
    python race-car/LaneShift/collect_data.py [--n-games 10000] [--save-path laneshift_dataset.npz]

Output .npz (ragged games stored as flat arrays + length index):
    sensors      float32 [total_ticks, 16]  – raw sensor distances; NaN = no detection
    car_x        float32 [total_ticks, 5]   – NPC car x-pos per lane; NaN = lane empty
    car_vx       float32 [total_ticks, 5]   – NPC car relative x-velocity (car - ego) per lane; NaN = lane empty
    ego_xy       float32 [total_ticks, 2]   – ego car screen position [x, y]
    game_lengths int32   [N_games]          – ticks per game; use np.split() to recover games

Recover game i:
    starts = np.concatenate([[0], np.cumsum(game_lengths[:-1])])
    sl = slice(starts[i], starts[i] + game_lengths[i])
    game_sensors = sensors[sl]
    game_car_x   = car_x  [sl]
    game_car_vx  = car_vx [sl]
    game_ego_xy  = ego_xy [sl]
"""

import sys
import os
import multiprocessing as mp
import numpy as np
import importlib.util
import types
from tqdm import tqdm
import argparse

# ── path / import setup ──────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))  # …/race-car/LaneShift/
RACECAR_DIR = os.path.dirname(SCRIPT_DIR)                  # …/race-car/

sys.path.insert(0, RACECAR_DIR)
sys.path.insert(0, SCRIPT_DIR)

# LaneShift.py uses "models.X" aliases that don't exist on disk.
# Patch the module registry before importing so it resolves correctly.
def _load_file(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sys.modules.setdefault("models",           types.ModuleType("models"))
sys.modules.setdefault("models.utilities", types.ModuleType("models.utilities"))
sys.modules["models.utilities.sensor_parser"] = _load_file(
    "sensor_parser",  os.path.join(SCRIPT_DIR, "sensor_parser.py"))
sys.modules["models.laneshift_config"] = _load_file(
    "laneshift_config", os.path.join(SCRIPT_DIR, "laneshift_config.py"))

# Headless pygame – must be set before pygame.init()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()
pygame.display.set_mode((1, 1))   # minimal surface; required by Road/Car sprite loading

# SDL_VIDEODRIVER=dummy skips SDL_image initialisation, so pygame.image.load
# can only read BMP files.  Patch it to use Pillow instead, which decodes the
# PNG independently and hands raw bytes to pygame.image.frombuffer.
from PIL import Image as _PILImage

def _pil_load(path: str) -> pygame.Surface:
    # Car.load_sprite passes a path relative to race-car/; make it absolute
    # so the script works regardless of the current working directory.
    if not os.path.isabs(path):
        path = os.path.join(RACECAR_DIR, path)
    img  = _PILImage.open(path).convert("RGBA")
    surf = pygame.image.frombuffer(img.tobytes(), img.size, "RGBA")
    return surf

pygame.image.load = _pil_load

from src.game.core import initialize_game_state, update_game, intersects
import src.game.core as _core
from LaneShift import LaneShift as LaneShiftAgent

# ── constants ────────────────────────────────────────────────────────────────
N_LANES   = 5
MAX_TICKS = 60 * 60   # mirror of core.py

# Canonical sensor order (matches core.py sensor_options list)
SENSOR_NAMES = [
    "front",
    "right_front",
    "right_side",
    "right_back",
    "back",
    "left_back",
    "left_side",
    "left_front",
    "left_side_front",
    "front_left_front",
    "front_right_front",
    "right_side_front",
    "right_side_back",
    "back_right_back",
    "back_left_back",
    "left_side_back",
]


# ── per-tick helpers ─────────────────────────────────────────────────────────
def _sensor_row() -> np.ndarray:
    """16-element float32 array; NaN where sensor has no reading."""
    readings = {s.name: s.reading for s in _core.STATE.sensors}
    return np.array(
        [readings.get(n) if readings.get(n) is not None else np.nan
         for n in SENSOR_NAMES],
        dtype=np.float32,
    )


def _ground_truth_row() -> np.ndarray:
    """5-element float32 array; NaN where lane has no NPC car."""
    gt = [np.nan] * N_LANES
    for car in _core.STATE.cars:
        if car is _core.STATE.ego or car.lane is None:
            continue
        idx = _core.STATE.road.lanes.index(car.lane)
        gt[idx] = float(car.x)
    return np.array(gt, dtype=np.float32)


def _velocity_row() -> np.ndarray:
    """5-element float32 array; NPC car x-velocity relative to ego per lane; NaN if lane empty.

    Relative velocity (car.vx - ego.vx) is what the sensors physically encode and what the
    world model should ultimately predict to replace the current sensor-delta estimator.
    """
    vx = [np.nan] * N_LANES
    ego_vx = _core.STATE.ego.velocity.x
    for car in _core.STATE.cars:
        if car is _core.STATE.ego or car.lane is None:
            continue
        idx = _core.STATE.road.lanes.index(car.lane)
        vx[idx] = float(car.velocity.x - ego_vx)
    return np.array(vx, dtype=np.float32)


def _ego_xy_row() -> np.ndarray:
    """2-element float32 array [x, y] — ego car screen position."""
    ego = _core.STATE.ego
    return np.array([float(ego.x), float(ego.y)], dtype=np.float32)


def _build_state_dict() -> dict:
    """State dict that LaneShiftAgent.return_action() expects."""
    S = _core.STATE
    sensors = {
        s.name: (round(s.reading) if s.reading is not None else None)
        for s in S.sensors
    }
    return {
        "distance":      S.distance,
        "velocity":      {"x": S.ego.velocity.x, "y": S.ego.velocity.y},
        "sensors":       sensors,
        "elapsed_ticks": S.ticks,
    }


def _check_collisions():
    S = _core.STATE
    for car in S.cars:
        if car is not S.ego and intersects(S.ego.rect, car.rect):
            S.crashed = True
    for wall in S.road.walls:
        if intersects(S.ego.rect, wall.rect):
            S.crashed = True


def _is_done() -> bool:
    return _core.STATE.crashed or _core.STATE.ticks >= MAX_TICKS


# ── single game ───────────────────────────────────────────────────────────────
def _run_game(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Play one full game and return all its ticks.

    Returns:
        sensors  float32 [T, 16]
        car_x    float32 [T, 5]
    """
    initialize_game_state("http://dummy", seed)
    agent = LaneShiftAgent()

    game_sensors: list[np.ndarray] = []
    game_car_x:   list[np.ndarray] = []
    game_car_vx:  list[np.ndarray] = []
    game_ego_xy:  list[np.ndarray] = []

    while not _is_done():
        # observe
        game_sensors.append(_sensor_row())
        game_car_x.append(_ground_truth_row())
        game_car_vx.append(_velocity_row())
        game_ego_xy.append(_ego_xy_row())

        # act
        state_dict  = _build_state_dict()
        action_list = agent.return_action(state_dict)
        action      = action_list[0] if action_list else "NOTHING"

        # step
        update_game(action)
        _core.STATE.ticks += 1
        _check_collisions()

    return np.stack(game_sensors), np.stack(game_car_x), np.stack(game_car_vx), np.stack(game_ego_xy)


# ── main collection loop ─────────────────────────────────────────────────────
def collect(n_games: int = 10_000, seed_start: int = 0, save_path: str = "laneshift_dataset.npz"):
    """
    Play n_games full games and save the data.

    Storage format (ragged-as-flat):
        sensors      [total_ticks, 16]  – all ticks from all games concatenated
        car_x        [total_ticks, 5]   – matching ground truths
        game_lengths [n_games]          – number of ticks in each game

    This avoids padding while keeping everything in one efficient .npz file.
    """
    n_workers = max(1, mp.cpu_count() * 15 // 16)
    tqdm.write(f"Using {n_workers}/{mp.cpu_count()} workers")

    all_sensors:      list[np.ndarray] = []
    all_car_x:        list[np.ndarray] = []
    all_car_vx:       list[np.ndarray] = []
    all_ego_xy:       list[np.ndarray] = []
    all_game_lengths: list[int]        = []

    seeds = range(seed_start, seed_start + n_games)
    with mp.Pool(n_workers) as pool:
        for sensors, car_x, car_vx, ego_xy in tqdm(
            pool.imap_unordered(_run_game, seeds),
            total=n_games, desc="collecting", unit="game",
        ):
            all_sensors.append(sensors)
            all_car_x.append(car_x)
            all_car_vx.append(car_vx)
            all_ego_xy.append(ego_xy)
            all_game_lengths.append(len(sensors))

    sensors_arr      = np.concatenate(all_sensors, axis=0)   # [total_ticks, 16]
    car_x_arr        = np.concatenate(all_car_x,   axis=0)   # [total_ticks, 5]
    car_vx_arr       = np.concatenate(all_car_vx,  axis=0)   # [total_ticks, 5]
    ego_xy_arr       = np.concatenate(all_ego_xy,  axis=0)   # [total_ticks, 2]
    game_lengths_arr = np.array(all_game_lengths, dtype=np.int32)  # [n_games]

    out = save_path if os.path.isabs(save_path) else os.path.abspath(save_path)
    np.savez_compressed(
        out,
        sensors=sensors_arr,
        car_x=car_x_arr,
        car_vx=car_vx_arr,
        ego_xy=ego_xy_arr,
        game_lengths=game_lengths_arr,
    )

    total_ticks = int(game_lengths_arr.sum())
    tqdm.write("")
    tqdm.write(f"Saved → {out}")
    tqdm.write(f"  games        : {n_games}")
    tqdm.write(f"  total ticks  : {total_ticks:,}")
    tqdm.write(f"  sensors      : {sensors_arr.shape}  {sensors_arr.dtype}")
    tqdm.write(f"  car_x        : {car_x_arr.shape}   {car_x_arr.dtype}")
    tqdm.write(f"  car_vx       : {car_vx_arr.shape}  {car_vx_arr.dtype}")
    tqdm.write(f"  ego_xy       : {ego_xy_arr.shape}  {ego_xy_arr.dtype}")
    tqdm.write(
        f"  game_lengths : {game_lengths_arr.shape}  min={game_lengths_arr.min()}"
        f"  max={game_lengths_arr.max()}  mean={game_lengths_arr.mean():.1f}"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-games",    type=int, default=10_000,
                    help="Number of full games to play (default: 10000)")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="Seed for the first game; incremented by 1 per game (default: 0)")
    ap.add_argument("--save-path",  type=str, default="laneshift_dataset.npz",
                    help="Output file; relative paths are anchored to race-car/ (default: laneshift_dataset.npz)")
    args = ap.parse_args()

    collect(args.n_games, args.seed_start, args.save_path)
    pygame.quit()

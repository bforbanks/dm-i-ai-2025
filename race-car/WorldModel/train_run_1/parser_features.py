#!/usr/bin/env python3
"""Vectorise SensorParser output for experiment 8 (fixed dim, zeros + masks)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np

# Same order as collect_data.py / core sensor list
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

VEL_SCALE = 28.0
PARSER_DIM = 5 * 4  # per lane: x, x_mask, v, v_mask

# Bump when `parsed_list_to_vec` / SensorParser wiring changes (invalidates caches).
PARSER_CACHE_VERSION = 1


def parser_cache_path(data_path: str) -> str:
    """Sidecar cache next to the dataset: ``dataset.npz`` → ``dataset.sensor_parser_feats.npz``."""
    base, _ = os.path.splitext(os.path.abspath(data_path))
    return base + ".sensor_parser_feats.npz"


def _source_stat_fingerprint(path: str) -> tuple[int, int]:
    st = os.stat(path)
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return int(st.st_size), int(mtime_ns)


def load_or_precompute_parser_features(
    data_path: str,
    sensors: np.ndarray,
    ego_y: np.ndarray,
    lengths: np.ndarray,
    *,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Return ``[total_ticks, PARSER_DIM]``. Uses a compressed ``.sensor_parser_feats.npz``
    beside the dataset when ``use_cache`` and the source file size/mtime still match.
    """
    data_path = os.path.abspath(data_path)
    n = int(sensors.shape[0])

    if use_cache and not force_recompute:
        cache_path = parser_cache_path(data_path)
        if os.path.isfile(cache_path):
            try:
                z = np.load(cache_path)
                sz, mt = _source_stat_fingerprint(data_path)
                if (
                    int(z["cache_version"]) == PARSER_CACHE_VERSION
                    and int(z["source_size"]) == sz
                    and int(z["source_mtime_ns"]) == mt
                    and int(z["n_ticks"]) == n
                ):
                    print(f"Loaded SensorParser cache ({cache_path})", flush=True)
                    return np.asarray(z["parser_feats"], dtype=np.float32)
            except Exception as exc:
                print(f"Parser cache unreadable ({exc}); recomputing …", flush=True)

    print("Precomputing SensorParser features (one pass over all games) …", flush=True)
    out = precompute_parser_features(sensors, ego_y, lengths).astype(np.float32)

    if use_cache:
        cache_path = parser_cache_path(data_path)
        sz, mt = _source_stat_fingerprint(data_path)
        np.savez_compressed(
            cache_path,
            parser_feats=out,
            cache_version=np.int32(PARSER_CACHE_VERSION),
            source_size=np.int64(sz),
            source_mtime_ns=np.int64(mt),
            n_ticks=np.int64(n),
        )
        print(f"  Wrote SensorParser cache → {cache_path}", flush=True)

    print(f"  parser_feats shape={out.shape}", flush=True)
    return out


def _load_sensor_parser_class():
    # …/race-car/WorldModel/train_run_1/this_file → …/race-car/LaneShift/
    _train_run = os.path.dirname(os.path.abspath(__file__))
    _racecar = os.path.dirname(os.path.dirname(_train_run))
    path = os.path.join(_racecar, "LaneShift", "sensor_parser.py")
    spec = importlib.util.spec_from_file_location("sensor_parser", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SensorParser


SensorParser = _load_sensor_parser_class()


def sensor_row_to_state_dict(row: np.ndarray) -> dict:
    sensors = {}
    for i, name in enumerate(SENSOR_NAMES):
        v = float(row[i])
        sensors[name] = None if np.isnan(v) else v
    return {"sensors": sensors}


def parsed_list_to_vec(parsed: list) -> np.ndarray:
    """5 lanes → 20 floats per lane: x/1000, x_mask, v/VEL_SCALE, v_mask."""
    out = np.zeros(PARSER_DIM, dtype=np.float32)
    for lane in range(5):
        k = lane * 4
        p = parsed[lane]
        if p.get("type") is None:
            continue
        x = p.get("x")
        if x is not None:
            out[k] = float(x) / 1000.0
            out[k + 1] = 1.0
        v = p.get("velocity")
        if v is not None:
            out[k + 2] = float(v) / VEL_SCALE
            out[k + 3] = 1.0
    return out


def precompute_parser_features(
    sensors: np.ndarray,
    ego_y: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """
    One row per global tick. Resets SensorParser at each game boundary.
    y_pos for the parser = ego_y offset from start of that game (simple proxy).
    """
    n = sensors.shape[0]
    out = np.zeros((n, PARSER_DIM), dtype=np.float32)
    starts = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    n_games = len(lengths)

    for gi in range(n_games):
        parser = SensorParser()
        base = int(starts[gi])
        t_end = int(lengths[gi])
        y0 = float(ego_y[base])
        for t in range(t_end):
            idx = base + t
            state = sensor_row_to_state_dict(sensors[idx])
            y_pos = float(ego_y[idx] - y0)
            parsed = parser.parse_sensors(state, y_pos)
            out[idx] = parsed_list_to_vec(parsed)
    return out

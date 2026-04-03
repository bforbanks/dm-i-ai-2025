#!/usr/bin/env python3
"""
WorldModel prediction visualizer.

Shows the game side-by-side with the model's live predictions:
  Left   — game reconstruction (road, NPC cars, ego, sensor beams)
  Right  — per-lane histograms: predicted distribution (blue bars),
            true bin (red line), and predicted vs actual relative velocity

The model runs tick-by-tick with full recurrent state — exactly as it would
at inference time.  When you scrub or jump to a different tick the model
is replayed from tick 0 of the current game to rebuild its state.

Usage (from project root):
    python race-car/WorldModel/visualize_predictions.py \\
        --model  race-car/WorldModel/FirstModel.pt \\
        --data   laneshift_dataset.npz

Controls:
  Space            play / pause
  Left / Right     step one tick (hold to scrub)
  Up / Down        next / previous game
  Shift+Up/Down    jump ±100 games
  + / -            double / halve playback speed
  G                type a game number to jump to (Enter confirm, Esc cancel)
  Q / Escape       quit
"""

import argparse
import os
import sys
import numpy as np
import pygame
import torch
import torch.nn.functional as F

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RACECAR_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RACECAR_DIR)

from WorldModel.model import build_model, car_x_to_bin, preprocess_sensors, BINS, X_MIN, X_MAX, N_LANES

# ── game-coordinate constants (must match engine) ─────────────────────────────
GAME_W, GAME_H = 1600, 1200
MARGIN         = 40
LANE_COUNT     = 5
LANE_H         = (GAME_H - 2 * MARGIN) / LANE_COUNT
LANE_CY        = [MARGIN + (i + 0.5) * LANE_H for i in range(LANE_COUNT)]
CAR_W, CAR_H   = 360, 179

SENSOR_DEGS = [90, 135, 180, 225, 270, 315, 0, 45,
               22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

# ── layout ────────────────────────────────────────────────────────────────────
WIN_W     = 1600
WIN_H     = 900
HEADER_H  = 40
PLAYBAR_H = 44
STATUS_H  = 26
VIEW_H    = WIN_H - HEADER_H - PLAYBAR_H - STATUS_H

GAME_PANEL_W = WIN_W // 2       # 800
PRED_PANEL_W = WIN_W - GAME_PANEL_W   # 800

# ── colours ───────────────────────────────────────────────────────────────────
C_BG      = ( 22,  24,  32)
C_ROAD    = ( 38,  42,  52)
C_WALL    = ( 70,  30,  30)
C_LANE    = (160, 160, 160)
C_EGO     = (255, 215,  50)
C_NPC     = [(100, 160, 255), (255,  90,  90), ( 60, 210,  80),
             (255, 155,  50), (195,  90, 255)]
C_BEAM    = (255,  55,  55)
C_TXT     = (210, 215, 225)
C_DIM     = (105, 110, 128)
C_HIST    = ( 70, 130, 200)         # predicted histogram bars
C_TRUE    = (255,  60,  60)         # true bin marker
C_VEL_P   = ( 90, 220, 130)         # predicted velocity
C_VEL_T   = (255, 200,  50)         # true velocity
C_PB_BG   = ( 40,  44,  58)
C_PB_FG   = ( 90, 165, 255)


def _font(size: int) -> pygame.font.Font:
    for name in ("DejaVu Sans", "dejavusans", "Liberation Sans", "Noto Sans"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(path: str):
    d       = np.load(path)
    lengths = d["game_lengths"].astype(int)
    starts  = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(int)
    return d["sensors"], d["car_x"], d["car_vx"], d["ego_xy"], lengths, starts


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print(f"Loaded {ckpt['model_name']}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model


# ── run model from tick 0 → target_tick (replays state) ──────────────────────

@torch.no_grad()
def replay_to_tick(model, sensors_game, ego_xy_game, target_tick: int, device):
    """
    Run the model from tick 0 up to and including target_tick.
    Returns (all_pos_prob, all_vel_pred) arrays for every tick 0..target_tick.
      all_pos_prob : [target_tick+1, 5, BINS+1]  numpy float32
      all_vel_pred : [target_tick+1, 5]           numpy float32
    """
    T = target_tick + 1
    state = model.init_state(1, device)

    all_pos  = np.zeros((T, N_LANES, BINS + 1), dtype=np.float32)
    all_vel  = np.zeros((T, N_LANES),           dtype=np.float32)

    for t in range(T):
        s_raw = torch.from_numpy(sensors_game[t:t+1]).to(device)     # [1, 16]
        e_raw = torch.tensor([[float(ego_xy_game[t, 1])]], device=device)  # [1, 1] ego_y only

        pos_logits, vel_pred, state = model(s_raw, e_raw, state)

        all_pos[t] = F.softmax(pos_logits[0], dim=-1).cpu().numpy()
        all_vel[t] = vel_pred[0].cpu().numpy()

    return all_pos, all_vel


@torch.no_grad()
def step_model(model, state, sensors_t, ego_y_t, device):
    """Single tick forward; returns (pos_prob [5,BINS+1], vel [5], new_state)."""
    s_raw = torch.from_numpy(sensors_t[np.newaxis]).to(device)
    e_raw = torch.tensor([[float(ego_y_t)]], device=device)
    pos_logits, vel_pred, new_state = model(s_raw, e_raw, state)
    pos_prob = F.softmax(pos_logits[0], dim=-1).cpu().numpy()
    vel      = vel_pred[0].cpu().numpy()
    return pos_prob, vel, new_state


# ── draw: game panel (left) ───────────────────────────────────────────────────

def draw_game_panel(surf, rect, sensors_t, car_x_t, ego_xy_t):
    scx = rect.width  / GAME_W
    scy = rect.height / GAME_H

    def gp(gx, gy): return (int(rect.x + gx * scx), int(rect.y + gy * scy))
    def gsz(w, h):  return (max(1, int(w * scx)), max(1, int(h * scy)))

    pygame.draw.rect(surf, C_ROAD, rect)

    wh = max(2, int(MARGIN * scy))
    pygame.draw.rect(surf, C_WALL, (rect.x, rect.y, rect.width, wh))
    pygame.draw.rect(surf, C_WALL, (rect.x, rect.bottom - wh, rect.width, wh))

    for i in range(LANE_COUNT + 1):
        y = MARGIN + i * LANE_H
        pygame.draw.line(surf, C_LANE, gp(0, y), gp(GAME_W, y), 1)

    beam = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    ecx  = float(ego_xy_t[0]) + CAR_W / 2
    ecy  = float(ego_xy_t[1]) + CAR_H / 2
    for deg, val in zip(SENSOR_DEGS, sensors_t):
        if np.isnan(val):
            continue
        rad = np.radians(deg - 90)
        ex_ = ecx + np.cos(rad) * float(val)
        ey_ = ecy + np.sin(rad) * float(val)
        p0  = (int(ecx * scx), int(ecy * scy))
        p1  = (int(ex_ * scx), int(ey_ * scy))
        pygame.draw.line(beam, (*C_BEAM, 55), p0, p1, 1)
        pygame.draw.circle(beam, (*C_BEAM, 200), p1, 3)
    surf.blit(beam, (rect.x, rect.y))

    for lane in range(LANE_COUNT):
        cx = car_x_t[lane]
        if np.isnan(cx):
            continue
        cy = LANE_CY[lane] - CAR_H / 2
        r  = pygame.Rect(*gp(float(cx), cy), *gsz(CAR_W, CAR_H))
        pygame.draw.rect(surf, C_NPC[lane], r, border_radius=3)
        pygame.draw.rect(surf, (255, 255, 255), r, 1, border_radius=3)

    r = pygame.Rect(*gp(float(ego_xy_t[0]), float(ego_xy_t[1])), *gsz(CAR_W, CAR_H))
    pygame.draw.rect(surf, C_EGO, r, border_radius=3)
    pygame.draw.rect(surf, (255, 255, 255), r, 2, border_radius=3)


# ── draw: prediction panel (right) ───────────────────────────────────────────

def draw_pred_panel(surf, rect, pos_prob, vel_pred, car_x_t, car_vx_t):
    """
    pos_prob  : [5, BINS+1]  predicted probability distribution per lane
    vel_pred  : [5]           predicted relative velocity per lane
    car_x_t   : [5]           true NPC x (NaN if no car)
    car_vx_t  : [5]           true relative velocity (NaN if no car)
    """
    pygame.draw.rect(surf, (16, 18, 28), rect)

    fs   = _font(13)
    fb   = _font(14)

    # Lane panel height: split VIEW_H into 5 equal rows
    lane_h  = rect.height // N_LANES
    VEL_H   = 36     # height reserved for velocity readout at bottom of each lane row
    HIST_H  = lane_h - VEL_H - 4

    for lane in range(N_LANES):
        lx  = rect.x
        ly  = rect.y + lane * lane_h
        lw  = rect.width
        col = C_NPC[lane]

        # Lane label background strip
        pygame.draw.rect(surf, (28, 30, 42), (lx, ly, lw, lane_h))
        pygame.draw.line(surf, (50, 55, 70), (lx, ly), (lx + lw, ly), 1)

        # ── histogram ────────────────────────────────────────────────────────
        HIST_PAD_L = 6
        HIST_PAD_R = 6
        hist_rect  = pygame.Rect(lx + HIST_PAD_L, ly + 2,
                                 lw - HIST_PAD_L - HIST_PAD_R, HIST_H)

        prob     = pos_prob[lane]                       # [BINS+1]
        pos_part = prob[:BINS]                          # first BINS = position probs
        no_car   = prob[BINS]                           # last entry = no-car prob

        bar_w    = max(1, hist_rect.width // BINS)
        max_prob = float(pos_part.max()) if pos_part.max() > 1e-6 else 1e-6

        for b in range(BINS):
            bh = int((pos_part[b] / max_prob) * (HIST_H - 2))
            bx = hist_rect.x + b * bar_w
            by = hist_rect.bottom - bh
            if bh > 0:
                pygame.draw.rect(surf, col, (bx, by, max(1, bar_w - 1), bh))

        # True bin marker (red vertical line)
        true_x = car_x_t[lane]
        if not np.isnan(true_x):
            true_bin = int(((true_x - X_MIN) / (X_MAX - X_MIN)) * BINS)
            true_bin = max(0, min(BINS - 1, true_bin))
            tx = hist_rect.x + true_bin * bar_w + bar_w // 2
            pygame.draw.line(surf, C_TRUE, (tx, hist_rect.y), (tx, hist_rect.bottom), 2)

        # Axis labels: x_min and x_max
        lbl_min = fs.render(f"{int(X_MIN)}", True, C_DIM)
        lbl_max = fs.render(f"{int(X_MAX)}", True, C_DIM)
        surf.blit(lbl_min, (hist_rect.x, hist_rect.bottom - lbl_min.get_height()))
        surf.blit(lbl_max, (hist_rect.right - lbl_max.get_width(),
                            hist_rect.bottom - lbl_max.get_height()))

        # Lane label + no-car probability
        label = fb.render(f"Lane {lane+1}  P(no car)={no_car*100:.0f}%", True, col)
        surf.blit(label, (lx + HIST_PAD_L, ly + 2))

        # ── velocity readout ─────────────────────────────────────────────────
        vy = ly + HIST_H + 4
        vx = lx + HIST_PAD_L

        vp = vel_pred[lane]
        vt = car_vx_t[lane]

        # Draw a small horizontal bar chart: pred (green) and true (yellow)
        BAR_MAX   = 30.0    # px/tick scale
        BAR_W_MAX = lw - HIST_PAD_L - HIST_PAD_R - 120
        BAR_H_EACH = 10
        bar_cx    = lx + lw // 2   # centre of bar (zero point)

        def draw_vel_bar(val, color, y_off):
            if np.isnan(val):
                return
            frac = float(val) / BAR_MAX
            bar_px = int(np.clip(frac * (BAR_W_MAX // 2), -BAR_W_MAX // 2, BAR_W_MAX // 2))
            x0 = bar_cx
            x1 = bar_cx + bar_px
            pygame.draw.rect(surf, color,
                             (min(x0, x1), vy + y_off, max(1, abs(bar_px)), BAR_H_EACH))

        # Zero line
        pygame.draw.line(surf, C_DIM,
                         (bar_cx, vy), (bar_cx, vy + 2 * BAR_H_EACH + 2), 1)

        draw_vel_bar(vp, C_VEL_P, 0)
        draw_vel_bar(vt, C_VEL_T, BAR_H_EACH + 2)

        # Numeric labels
        vp_txt = f"pred={vp:+.1f}" if not np.isnan(vp) else "pred=---"
        vt_txt = f"true={vt:+.1f}" if not np.isnan(vt) else "true=---"
        surf.blit(fs.render(vp_txt, True, C_VEL_P), (vx, vy))
        surf.blit(fs.render(vt_txt, True, C_VEL_T), (vx, vy + BAR_H_EACH + 2))

    # Legend at very top-right corner
    leg_y = rect.y + 3
    for txt, col in [("-- pred vel", C_VEL_P), ("-- true vel", C_VEL_T),
                     ("| true bin", C_TRUE)]:
        lbl = fs.render(txt, True, col)
        surf.blit(lbl, (rect.right - lbl.get_width() - 4, leg_y))
        leg_y += lbl.get_height() + 1


# ── chrome ────────────────────────────────────────────────────────────────────

def draw_header(surf, font, game_idx, n_games, tick, game_len, speed, model_name,
                typing_game, typed_str):
    pygame.draw.rect(surf, (26, 28, 42), (0, 0, WIN_W, HEADER_H))
    pygame.draw.line(surf, C_DIM, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 1)
    if typing_game:
        info = f"Jump to game: {typed_str}|  (Enter confirm, Esc cancel)"
    else:
        info = (f"Game {game_idx+1}/{n_games}  |  Tick {tick}/{game_len-1}  |  "
                f"{speed:.3g}x  |  {model_name}")
    surf.blit(font.render(info, True, C_TXT), (12, (HEADER_H - font.get_height()) // 2))
    hint = _font(12).render(
        "SPC play   ←→ tick   ↑↓ game (Shift ±100)   +/- speed   G jump   Q quit",
        True, C_DIM)
    surf.blit(hint, (WIN_W - hint.get_width() - 8, (HEADER_H - hint.get_height()) // 2))


def draw_playbar(surf, tick, game_len, is_playing):
    y = HEADER_H + VIEW_H
    pygame.draw.rect(surf, C_PB_BG, (0, y, WIN_W, PLAYBAR_H))
    pygame.draw.line(surf, C_DIM, (0, y), (WIN_W, y), 1)

    PAD   = 52
    track = pygame.Rect(PAD, y + 15, WIN_W - PAD * 2, 12)
    frac  = tick / max(game_len - 1, 1)
    fill  = max(0, int(frac * track.width))

    pygame.draw.rect(surf, (34, 38, 55), track, border_radius=6)
    if fill:
        pygame.draw.rect(surf, C_PB_FG,
                         pygame.Rect(track.x, track.y, fill, track.height),
                         border_radius=6)
    kx = track.x + fill
    pygame.draw.circle(surf, (255, 255, 255), (kx, track.centery), 8)
    pygame.draw.circle(surf, C_PB_FG,         (kx, track.centery), 6)

    sym = _font(20).render("||" if is_playing else " >", True, C_TXT)
    surf.blit(sym, (8, y + (PLAYBAR_H - sym.get_height()) // 2))
    return track


def draw_status(surf, font, tick, pos_prob_t, vel_pred_t, car_x_t, car_vx_t):
    y = WIN_H - STATUS_H
    pygame.draw.rect(surf, (18, 20, 32), (0, y, WIN_W, STATUS_H))
    # Show per-lane most-likely bin and whether model is more confident about car/no-car
    parts = []
    for lane in range(N_LANES):
        prob = pos_prob_t[lane]
        if prob[BINS] > 0.5:
            parts.append(f"L{lane+1}:empty({prob[BINS]*100:.0f}%)")
        else:
            best_bin  = int(np.argmax(prob[:BINS]))
            best_x    = X_MIN + (best_bin + 0.5) * (X_MAX - X_MIN) / BINS
            parts.append(f"L{lane+1}:x~{best_x:.0f}({prob[:BINS].max()*100:.0f}%)")
    msg = "  ".join(parts)
    surf.blit(font.render(msg, True, C_DIM), (8, y + (STATUS_H - font.get_height()) // 2))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="race-car/WorldModel/FirstModel.pt",
                    help="Path to .pt checkpoint")
    ap.add_argument("--data",  default="laneshift_dataset.npz",
                    help="Path to dataset .npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    print("Loading data…")
    sensors, car_x, car_vx, ego_xy, lengths, starts = load_data(args.data)
    n_games = len(lengths)
    print(f"  {n_games} games, {len(sensors):,} ticks")

    print("Loading model…")
    model      = load_model(args.model, device)
    model_name = os.path.basename(args.model)

    pygame.init()
    pygame.key.set_repeat(160, 35)
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"WorldModel Predictions — {model_name}")
    clock = pygame.time.Clock()
    font  = _font(15)

    game_rect = pygame.Rect(0,             HEADER_H, GAME_PANEL_W, VIEW_H)
    pred_rect = pygame.Rect(GAME_PANEL_W,  HEADER_H, PRED_PANEL_W, VIEW_H)
    pb_track  = pygame.Rect(52, HEADER_H + VIEW_H + 15, WIN_W - 104, 12)

    # ── state ─────────────────────────────────────────────────────────────────
    game_idx    = 0
    tick        = 0
    is_playing  = False
    speed       = 1.0
    play_accum  = 0.0
    dragging_pb = False
    typing_game = False
    typed_str   = ""

    # Per-tick cached predictions for the current game up to current tick.
    # Reset when game changes or we scrub backwards.
    pred_cache_pos = None   # [tick+1, 5, BINS+1]
    pred_cache_vel = None   # [tick+1, 5]
    model_state    = None   # current recurrent state at `tick`

    def rebuild_cache_to(tgt_tick):
        """(Re)run model from 0 → tgt_tick, cache all predictions."""
        nonlocal pred_cache_pos, pred_cache_vel, model_state
        sl = slice(starts[game_idx], starts[game_idx] + lengths[game_idx])
        print(f"Replaying game {game_idx+1} to tick {tgt_tick}…", flush=True)
        pred_cache_pos, pred_cache_vel = replay_to_tick(
            model, sensors[sl], ego_xy[sl], tgt_tick, device)
        # rebuild current state by running to tgt_tick
        model_state = None   # not needed; we read from cache

    def switch_game(idx):
        nonlocal game_idx, tick, pred_cache_pos, pred_cache_vel, model_state
        game_idx = idx
        tick     = 0
        pred_cache_pos = None
        pred_cache_vel = None
        model_state    = None
        rebuild_cache_to(min(lengths[game_idx] - 1, 0))

    switch_game(0)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        game_len = lengths[game_idx]

        # Ensure cache covers current tick
        if pred_cache_pos is None or tick >= len(pred_cache_pos):
            rebuild_cache_to(tick)

        # Slice data for current tick
        base        = starts[game_idx] + tick
        sensors_t   = sensors[base]
        car_x_t     = car_x[base]
        car_vx_t    = car_vx[base]
        ego_xy_t    = ego_xy[base]
        pos_prob_t  = pred_cache_pos[tick]    # [5, BINS+1]
        vel_pred_t  = pred_cache_vel[tick]    # [5]

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if typing_game:
                    if event.key == pygame.K_RETURN:
                        if typed_str:
                            idx = int(typed_str) - 1
                            switch_game(max(0, min(idx, n_games - 1)))
                        typing_game = False; typed_str = ""
                    elif event.key == pygame.K_ESCAPE:
                        typing_game = False; typed_str = ""
                    elif event.key == pygame.K_BACKSPACE:
                        typed_str = typed_str[:-1]
                    elif event.unicode.isdigit():
                        typed_str += event.unicode
                    continue

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_playing = not is_playing
                    play_accum = 0.0
                elif event.key == pygame.K_RIGHT:
                    new_tick = min(tick + 1, game_len - 1)
                    if new_tick > tick:
                        if new_tick >= len(pred_cache_pos):
                            rebuild_cache_to(new_tick)
                    tick = new_tick
                    is_playing = False
                elif event.key == pygame.K_LEFT:
                    tick = max(tick - 1, 0)
                    is_playing = False
                    # scrubbing backwards is fine; cache already covers it
                elif event.key == pygame.K_UP:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    switch_game((game_idx + (100 if shift else 1)) % n_games)
                elif event.key == pygame.K_DOWN:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    switch_game((game_idx - (100 if shift else 1)) % n_games)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 2, 64.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 2, 0.125)
                elif event.key == pygame.K_g:
                    typing_game = True; typed_str = ""

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pb_track.collidepoint(event.pos):
                    dragging_pb = True
                    frac = (event.pos[0] - pb_track.x) / pb_track.width
                    new_tick = int(np.clip(frac, 0, 1) * (game_len - 1))
                    if new_tick >= len(pred_cache_pos):
                        rebuild_cache_to(new_tick)
                    tick = new_tick

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_pb = False

            elif event.type == pygame.MOUSEMOTION and dragging_pb:
                frac = (event.pos[0] - pb_track.x) / pb_track.width
                new_tick = int(np.clip(frac, 0, 1) * (game_len - 1))
                if new_tick >= len(pred_cache_pos):
                    rebuild_cache_to(new_tick)
                tick = new_tick

        # ── playback advance ──────────────────────────────────────────────────
        if is_playing:
            play_accum += speed * dt * 60
            steps       = int(play_accum)
            play_accum -= steps
            new_tick = min(tick + steps, game_len - 1)
            if new_tick > tick:
                if new_tick >= len(pred_cache_pos):
                    rebuild_cache_to(new_tick)
                tick = new_tick
            if tick >= game_len - 1:
                is_playing = False

        # ── render ────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        draw_game_panel(screen, game_rect, sensors_t, car_x_t, ego_xy_t)
        draw_pred_panel(screen, pred_rect, pos_prob_t, vel_pred_t, car_x_t, car_vx_t)

        # vertical divider
        pygame.draw.line(screen, (60, 65, 85),
                         (GAME_PANEL_W, HEADER_H), (GAME_PANEL_W, HEADER_H + VIEW_H), 2)

        draw_header(screen, font, game_idx, n_games, tick, game_len,
                    speed, model_name, typing_game, typed_str)
        draw_playbar(screen, tick, game_len, is_playing)
        draw_status(screen, font, tick, pos_prob_t, vel_pred_t, car_x_t, car_vx_t)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LaneShift dataset visualizer.

Usage (from project root):
    python race-car/LaneShift/visualize_data.py [path/to/laneshift_dataset.npz]

Controls:
  Space          play / pause
  Left / Right   step one tick (hold to scrub)
  Up / Down      next / previous game
  + / -          double / halve playback speed
  Tab              cycle Game view → Eagle-eye → Stats
  Click playbar    jump to tick
  G                type a game number to jump to
  Shift + Up/Down  jump ±100 games
  Q / Escape       quit
"""

import argparse
import sys
import numpy as np
import pygame


def _font(size: int) -> pygame.font.Font:
    """Return a font with good Unicode coverage. DejaVu Sans ships with most
    Linux distros and includes arrows, box-drawing chars, and common symbols.
    Falls back to the pygame default if not found."""
    for name in ("DejaVu Sans", "dejavusans", "Noto Sans", "notosans",
                 "FreeSans", "Liberation Sans"):
        f = pygame.font.SysFont(name, size)
        if f is not None:
            return f
    return pygame.font.SysFont(None, size)

# ── game coordinate constants (must match the engine) ────────────────────────
GAME_W, GAME_H = 1600, 1200
MARGIN         = 40
LANE_COUNT     = 5
LANE_H         = (GAME_H - 2 * MARGIN) / LANE_COUNT      # 224 px
LANE_CY        = [MARGIN + (i + 0.5) * LANE_H for i in range(LANE_COUNT)]
CAR_W, CAR_H   = 360, 179
X_MIN, X_MAX   = -1000, 2600                              # NPC despawn bounds

# Sensor angles: (deg - 90) * π/180 gives screen-space angle where 0 = right (+x = forward)
SENSOR_DEGS = [90, 135, 180, 225, 270, 315, 0, 45,
               22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5]
SENSOR_LABELS = [
    "front",    "r_front",  "r_side",   "r_back",
    "back",     "l_back",   "l_side",   "l_front",
    "l_s_frt",  "fl_frt",   "fr_frt",   "r_s_frt",
    "r_s_bck",  "br_bck",   "bl_bck",   "l_s_bck",
]

# ── window layout ─────────────────────────────────────────────────────────────
WIN_W     = 1300
HEADER_H  = 44
PLAYBAR_H = 50
STATUS_H  = 28
VIEW_H    = 700
WIN_H     = HEADER_H + VIEW_H + PLAYBAR_H + STATUS_H

# ── colours ───────────────────────────────────────────────────────────────────
C_BG      = ( 22,  24,  32)
C_ROAD    = ( 38,  42,  52)
C_WALL    = ( 70,  30,  30)
C_LANE    = (160, 160, 160)
C_EGO     = (255, 215,  50)
C_NPC     = [(100, 160, 255), (255,  90,  90), ( 60, 210,  80),
             (255, 155,  50), (195,  90, 255)]
C_BEAM    = (255,  55,  55)
C_PB_BG   = ( 40,  44,  58)
C_PB_FG   = ( 90, 165, 255)
C_TXT     = (210, 215, 225)
C_DIM     = (105, 110, 128)


# ── data ─────────────────────────────────────────────────────────────────────
def load_data(path: str):
    d       = np.load(path)
    lengths = d["game_lengths"].astype(int)
    starts  = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(int)
    return (d["sensors"], d["car_x"], d["car_vx"], d["ego_xy"], lengths, starts)


# ── view 0: game reconstruction ──────────────────────────────────────────────
def draw_game_view(surf: pygame.Surface, rect: pygame.Rect,
                   sensors_t, car_x_t, ego_xy_t):
    scx = rect.width  / GAME_W
    scy = rect.height / GAME_H

    def gp(gx, gy): return (int(rect.x + gx * scx), int(rect.y + gy * scy))
    def gsz(w, h):  return (max(1, int(w * scx)), max(1, int(h * scy)))

    pygame.draw.rect(surf, C_ROAD, rect)

    # margin walls
    wh = max(2, int(MARGIN * scy))
    pygame.draw.rect(surf, C_WALL, (rect.x, rect.y, rect.width, wh))
    pygame.draw.rect(surf, C_WALL, (rect.x, rect.bottom - wh, rect.width, wh))

    # lane dividers
    for i in range(LANE_COUNT + 1):
        y = MARGIN + i * LANE_H
        pygame.draw.line(surf, C_LANE, gp(0, y), gp(GAME_W, y), 1)

    # sensor beams (alpha layer so they don't overdraw cars)
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

    # NPC cars
    for lane in range(LANE_COUNT):
        cx = car_x_t[lane]
        if np.isnan(cx):
            continue
        cy = LANE_CY[lane] - CAR_H / 2
        r  = pygame.Rect(*gp(float(cx), cy), *gsz(CAR_W, CAR_H))
        pygame.draw.rect(surf, C_NPC[lane], r, border_radius=3)
        pygame.draw.rect(surf, (255, 255, 255), r, 1, border_radius=3)

    # ego car (drawn last so it's always on top)
    r = pygame.Rect(*gp(float(ego_xy_t[0]), float(ego_xy_t[1])), *gsz(CAR_W, CAR_H))
    pygame.draw.rect(surf, C_EGO, r, border_radius=3)
    pygame.draw.rect(surf, (255, 255, 255), r, 2, border_radius=3)


# ── view 1: eagle-eye ────────────────────────────────────────────────────────
#
# Left panel  (62 % of width): x-position-over-time for all cars.
#   X axis = tick index, Y axis = screen x-position of car.
#   The ego car stays nearly constant (~800); NPC cars drift left/right.
#
# Right panel (38 % of width): sensor bar chart + per-lane relative velocity.

def build_trajectory_surface(car_x_game, ego_xy_game, w, h) -> pygame.Surface:
    """Pre-render trajectory lines once per game onto a (w × h) surface."""
    T  = len(car_x_game)
    s  = pygame.Surface((w, h))
    s.fill((26, 28, 40))

    def tx(i): return int(i / max(T - 1, 1) * w)
    def py_(x): return int(h - (x - X_MIN) / (X_MAX - X_MIN) * h)

    # subtle grid
    for gval in [X_MIN, 0, 800, 1600, X_MAX]:
        pygame.draw.line(s, (36, 40, 56), (0, py_(gval)), (w, py_(gval)), 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        pygame.draw.line(s, (36, 40, 56), (int(frac * w), 0), (int(frac * w), h), 1)

    # ego x trajectory (nearly constant at ~800)
    ego_xs = ego_xy_game[:, 0] + CAR_W / 2
    pts = [(tx(i), py_(float(ego_xs[i]))) for i in range(T)]
    if len(pts) > 1:
        pygame.draw.lines(s, C_EGO, False, pts, 2)

    # NPC trajectories, with gaps at NaN
    for lane in range(LANE_COUNT):
        seg = []
        for i in range(T):
            v = car_x_game[i, lane]
            if not np.isnan(v):
                seg.append((tx(i), py_(float(v))))
            else:
                if len(seg) > 1:
                    pygame.draw.lines(s, C_NPC[lane], False, seg, 2)
                seg = []
        if len(seg) > 1:
            pygame.draw.lines(s, C_NPC[lane], False, seg, 2)

    return s


def draw_eagle_view(surf: pygame.Surface, rect: pygame.Rect,
                    traj_surf: pygame.Surface,
                    sensors_t, car_vx_t, tick: int, T: int):
    pygame.draw.rect(surf, (18, 20, 30), rect)

    traj_w = int(rect.width * 0.62)
    side_w = rect.width - traj_w
    traj_r = pygame.Rect(rect.x,           rect.y, traj_w, rect.height)
    side_r = pygame.Rect(rect.x + traj_w,  rect.y, side_w, rect.height)

    # ── left: trajectory + tick cursor ────────────────────────────────────
    PADL, PADT, PADB = 54, 34, 26
    plot = pygame.Rect(traj_r.x + PADL, traj_r.y + PADT,
                       traj_r.width - PADL - 6, traj_r.height - PADT - PADB)

    # blit pre-rendered trajectories (scaled if needed)
    if traj_surf.get_size() != (plot.width, plot.height):
        ts = pygame.transform.smoothscale(traj_surf, (plot.width, plot.height))
    else:
        ts = traj_surf
    surf.blit(ts, (plot.x, plot.y))
    pygame.draw.rect(surf, C_DIM, plot, 1)

    fs = _font(14)
    fm = _font(16)

    # Y-axis labels
    for gval in [X_MIN, 0, 800, 1600, X_MAX]:
        yy = plot.bottom - int((gval - X_MIN) / (X_MAX - X_MIN) * plot.height)
        lbl = fs.render(str(gval), True, C_DIM)
        surf.blit(lbl, (traj_r.x + PADL - lbl.get_width() - 4, yy - 6))

    # X-axis labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        xx  = plot.x + int(frac * plot.width)
        lbl = fs.render(str(int(frac * (T - 1))), True, C_DIM)
        surf.blit(lbl, (xx - lbl.get_width() // 2, plot.bottom + 4))

    # tick cursor
    if 0 <= tick < T:
        cx = plot.x + int(tick / max(T - 1, 1) * plot.width)
        pygame.draw.line(surf, (255, 255, 255), (cx, plot.y), (cx, plot.bottom), 1)

    # legend
    for lane in range(LANE_COUNT):
        lbl = fs.render(f"Lane {lane+1}", True, C_NPC[lane])
        surf.blit(lbl, (plot.right - 52, plot.y + 4 + lane * 14))
    surf.blit(fs.render("Ego", True, C_EGO),
              (plot.right - 52, plot.y + 4 + LANE_COUNT * 14))

    surf.blit(fm.render("car x-position over time", True, C_DIM),
              (plot.x, traj_r.y + 6))
    surf.blit(fm.render("tick →", True, C_DIM),
              (plot.right - 46, plot.bottom + 4))
    surf.blit(fs.render("x pos →", True, C_DIM),
              (traj_r.x + 2, plot.y + plot.height // 2))

    # ── right: sensor bars + velocity table ───────────────────────────────
    pygame.draw.line(surf, C_DIM, (side_r.x, side_r.y), (side_r.x, side_r.bottom), 1)

    sx = side_r.x + 10
    sy = side_r.y + 10
    bar_w = side_r.width - 10 - 70

    surf.blit(fm.render("Sensors @ tick", True, C_DIM), (sx, sy)); sy += 20

    for label, val in zip(SENSOR_LABELS, sensors_t):
        lbl = fs.render(f"{label:<8}", True, C_TXT)
        surf.blit(lbl, (sx, sy))
        bx = sx + 68
        pygame.draw.rect(surf, (32, 36, 52), (bx, sy + 1, bar_w, 10))
        if not np.isnan(val):
            fill = min(max(0, int(float(val) / 1000 * bar_w)), bar_w)
            pygame.draw.rect(surf, C_BEAM, (bx, sy + 1, fill, 10))
            vl = fs.render(f"{int(val):4d}", True, C_TXT)
        else:
            vl = fs.render(" ---", True, C_DIM)
        surf.blit(vl, (bx + bar_w + 3, sy))
        sy += 14
        if sy > side_r.bottom - 105:
            break

    sy += 10
    surf.blit(fm.render("Lane rel-vx (cur tick)", True, C_DIM), (sx, sy)); sy += 18
    for lane in range(LANE_COUNT):
        v = car_vx_t[lane]
        s_str = f"L{lane+1}: {v:+.2f}" if not np.isnan(v) else f"L{lane+1}:  ---"
        surf.blit(fs.render(s_str, True, C_NPC[lane]), (sx, sy)); sy += 15


# ── view 2: statistics ───────────────────────────────────────────────────────
MAX_TICKS_GAME = 60 * 60   # engine max — used to classify timeout vs crash

def precompute_stats(sensors, car_x, car_vx, lengths, starts):
    """Compute dataset-wide stats once at startup."""
    n_games    = len(lengths)
    crashed    = lengths < MAX_TICKS_GAME
    return {
        "n_games":        n_games,
        "total_ticks":    int(lengths.sum()),
        "n_crashed":      int(crashed.sum()),
        "len_mean":       float(lengths.mean()),
        "len_median":     float(np.median(lengths)),
        "len_std":        float(lengths.std()),
        "len_min":        int(lengths.min()),
        "len_max":        int(lengths.max()),
        # per-lane occupancy across entire dataset
        "lane_occ":       [float(np.mean(~np.isnan(car_x[:, l]))) for l in range(LANE_COUNT)],
        # per-sensor NaN rate across entire dataset
        "sensor_nan":     [float(np.mean(np.isnan(sensors[:, s]))) for s in range(16)],
        # per-lane mean absolute relative velocity (when present)
        "lane_vx_mean":   [float(np.nanmean(np.abs(car_vx[:, l]))) for l in range(LANE_COUNT)],
    }


def game_stats(sensors, car_x, car_vx, lengths, starts, game_idx):
    """Compute per-game stats for the current game."""
    sl = slice(starts[game_idx], starts[game_idx] + lengths[game_idx])
    sx = sensors[sl]
    cx = car_x[sl]
    vx = car_vx[sl]
    T  = lengths[game_idx]
    return {
        "ticks":       T,
        "crashed":     T < MAX_TICKS_GAME,
        "lane_occ":    [float(np.mean(~np.isnan(cx[:, l]))) for l in range(LANE_COUNT)],
        "lane_x_mean": [float(np.nanmean(cx[:, l])) if not np.all(np.isnan(cx[:, l])) else float("nan")
                        for l in range(LANE_COUNT)],
        "lane_vx_mean":[float(np.nanmean(np.abs(vx[:, l]))) if not np.all(np.isnan(vx[:, l])) else float("nan")
                        for l in range(LANE_COUNT)],
        "sensor_fire": [float(np.mean(~np.isnan(sx[:, s]))) for s in range(16)],
    }


def draw_stats_view(surf: pygame.Surface, rect: pygame.Rect,
                    ds_stats: dict, gm_stats: dict, game_idx: int, n_games: int,
                    scroll: int = 0):
    # Render to an oversized off-screen surface then blit a scrolled slice
    VIRTUAL_H = 2000
    virt = pygame.Surface((rect.width, VIRTUAL_H))
    virt.fill((18, 20, 30))

    fs  = _font(15)
    fm  = _font(17)
    fb  = _font(18)

    def text(s, x, y, col=C_TXT, f=fs):
        virt.blit(f.render(s, True, col), (x, y))
        return y + f.get_height() + 3

    col_w = rect.width // 2
    lx, rx = 30, col_w + 20
    y_l = y_r = 18

    # ── left column: current game ─────────────────────────────────────────
    y_l = text(f"── Game {game_idx+1}/{n_games} ──", lx, y_l, C_EGO, fb); y_l += 4

    end = "CRASH" if gm_stats["crashed"] else "timeout"
    y_l = text(f"Ticks   : {gm_stats['ticks']}  [{end}]", lx, y_l,
               (255, 100, 80) if gm_stats["crashed"] else (100, 200, 100))

    y_l = text("", lx, y_l)
    y_l = text("Lane occupancy (% of ticks car present):", lx, y_l, C_DIM)
    for lane in range(LANE_COUNT):
        occ = gm_stats["lane_occ"][lane]
        bar = int(occ * 20)
        y_l = text(f"  L{lane+1}  {'█'*bar}{'░'*(20-bar)}  {occ*100:5.1f}%",
                   lx, y_l, C_NPC[lane])

    y_l = text("", lx, y_l)
    y_l = text("Mean car x-pos (when present):", lx, y_l, C_DIM)
    for lane in range(LANE_COUNT):
        v = gm_stats["lane_x_mean"][lane]
        s = f"  L{lane+1}  {v:8.1f} px" if not np.isnan(v) else f"  L{lane+1}  ---"
        y_l = text(s, lx, y_l, C_NPC[lane])

    y_l = text("", lx, y_l)
    y_l = text("Mean |rel-vx| per lane:", lx, y_l, C_DIM)
    for lane in range(LANE_COUNT):
        v = gm_stats["lane_vx_mean"][lane]
        s = f"  L{lane+1}  {v:6.3f} px/tick" if not np.isnan(v) else f"  L{lane+1}  ---"
        y_l = text(s, lx, y_l, C_NPC[lane])

    y_l = text("", lx, y_l)
    y_l = text("Sensor fire-rate (% ticks):", lx, y_l, C_DIM)
    for i, (label, fr) in enumerate(zip(SENSOR_LABELS, gm_stats["sensor_fire"])):
        bar = int(fr * 16)
        y_l = text(f"  {label:<8} {'█'*bar}{'░'*(16-bar)}  {fr*100:4.0f}%",
                   lx, y_l, C_TXT)

    # ── right column: dataset-wide ────────────────────────────────────────
    y_r = text("── Dataset ──", rx, y_r, C_EGO, fb); y_r += 4

    y_r = text(f"Games       : {ds_stats['n_games']:,}", rx, y_r)
    y_r = text(f"Total ticks : {ds_stats['total_ticks']:,}", rx, y_r)
    cr  = ds_stats['n_crashed']
    y_r = text(f"Crashed     : {cr:,}  ({cr/ds_stats['n_games']*100:.1f}%)",
               rx, y_r, (255, 120, 80))
    y_r = text(f"Timeout     : {ds_stats['n_games']-cr:,}", rx, y_r, (100, 200, 100))

    y_r = text("", rx, y_r)
    y_r = text("Game length (ticks):", rx, y_r, C_DIM)
    y_r = text(f"  mean   {ds_stats['len_mean']:7.1f}", rx, y_r)
    y_r = text(f"  median {ds_stats['len_median']:7.1f}", rx, y_r)
    y_r = text(f"  std    {ds_stats['len_std']:7.1f}", rx, y_r)
    y_r = text(f"  min    {ds_stats['len_min']:7d}", rx, y_r)
    y_r = text(f"  max    {ds_stats['len_max']:7d}", rx, y_r)

    y_r = text("", rx, y_r)
    y_r = text("Lane occupancy (dataset-wide):", rx, y_r, C_DIM)
    for lane in range(LANE_COUNT):
        occ = ds_stats["lane_occ"][lane]
        bar = int(occ * 20)
        y_r = text(f"  L{lane+1}  {'█'*bar}{'░'*(20-bar)}  {occ*100:5.1f}%",
                   rx, y_r, C_NPC[lane])

    y_r = text("", rx, y_r)
    y_r = text("Mean |rel-vx| per lane (dataset):", rx, y_r, C_DIM)
    for lane in range(LANE_COUNT):
        v = ds_stats["lane_vx_mean"][lane]
        y_r = text(f"  L{lane+1}  {v:6.3f} px/tick", rx, y_r, C_NPC[lane])

    y_r = text("", rx, y_r)
    y_r = text("Sensor NaN-rate (dataset-wide):", rx, y_r, C_DIM)
    for label, nr in zip(SENSOR_LABELS, ds_stats["sensor_nan"]):
        bar = int(nr * 16)
        y_r = text(f"  {label:<8} {'█'*bar}{'░'*(16-bar)}  {nr*100:4.0f}% NaN",
                   rx, y_r, C_TXT)

    content_h = max(y_l, y_r) + 10
    max_scroll = max(0, content_h - rect.height)
    scroll = max(0, min(scroll, max_scroll))

    pygame.draw.rect(surf, (18, 20, 30), rect)
    surf.blit(virt, (rect.x, rect.y), pygame.Rect(0, scroll, rect.width, rect.height))

    # Scrollbar
    if max_scroll > 0:
        sb_h = max(20, int(rect.height * rect.height / content_h))
        sb_y = rect.y + int(scroll / max_scroll * (rect.height - sb_h))
        pygame.draw.rect(surf, (60, 65, 90), (rect.right - 6, sb_y, 4, sb_h), border_radius=2)

    return scroll, max_scroll


# ── chrome: header / playbar / status ────────────────────────────────────────
def draw_header(surf, font, game_idx, n_games, tick, game_len, speed, view_name):
    pygame.draw.rect(surf, (26, 28, 42), (0, 0, WIN_W, HEADER_H))
    pygame.draw.line(surf, C_DIM, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 1)
    info = (f"Game {game_idx+1}/{n_games}  |  "
            f"Tick {tick}/{game_len-1}  |  "
            f"{speed:.3g}x speed  |  {view_name}")
    surf.blit(font.render(info, True, C_TXT),
              (12, (HEADER_H - font.get_height()) // 2))
    hint = _font(14).render(
        "↑↓ game (Shift ±100)   ←→ tick   G jump   Space play   +/- speed   Tab view   Q quit",
        True, C_DIM)
    surf.blit(hint, (WIN_W - hint.get_width() - 8,
                     (HEADER_H - hint.get_height()) // 2))


def draw_playbar(surf, tick, game_len, is_playing) -> pygame.Rect:
    y = HEADER_H + VIEW_H
    pygame.draw.rect(surf, C_PB_BG, (0, y, WIN_W, PLAYBAR_H))
    pygame.draw.line(surf, C_DIM, (0, y), (WIN_W, y), 1)

    PAD   = 56
    track = pygame.Rect(PAD, y + 17, WIN_W - PAD * 2, 14)
    frac  = tick / max(game_len - 1, 1)
    fill  = max(0, int(frac * track.width))

    pygame.draw.rect(surf, (34, 38, 55), track, border_radius=7)
    if fill:
        pygame.draw.rect(surf, C_PB_FG,
                         pygame.Rect(track.x, track.y, fill, track.height),
                         border_radius=7)
    kx = track.x + fill
    pygame.draw.circle(surf, (255, 255, 255), (kx, track.centery), 9)
    pygame.draw.circle(surf, C_PB_FG,          (kx, track.centery), 7)

    sym = _font(23).render(
        "||" if is_playing else " >", True, C_TXT)
    surf.blit(sym, (10, y + (PLAYBAR_H - sym.get_height()) // 2))

    return track


def draw_status(surf, font, sensors_t, car_x_t, ego_xy_t):
    y = WIN_H - STATUS_H
    pygame.draw.rect(surf, (18, 20, 32), (0, y, WIN_W, STATUS_H))
    cx  = [f"{v:.0f}" if not np.isnan(v) else "---" for v in car_x_t]
    msg = (f"car_x: [{', '.join(cx)}]   "
           f"ego: ({ego_xy_t[0]:.0f}, {ego_xy_t[1]:.0f})   "
           f"sensors firing: {int(np.sum(~np.isnan(sensors_t)))}/16")
    surf.blit(font.render(msg, True, C_DIM),
              (12, y + (STATUS_H - font.get_height()) // 2))


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="laneshift_dataset.npz")
    args = ap.parse_args()

    sensors, car_x, car_vx, ego_xy, lengths, starts = load_data(args.path)
    n_games = len(lengths)
    print(f"Loaded {n_games} games, {len(sensors):,} total ticks")

    pygame.init()
    pygame.key.set_repeat(180, 40)   # hold arrow key → auto-repeat scrubbing
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("LaneShift Dataset Viewer")
    clock  = pygame.time.Clock()
    font   = _font(17)

    view_rect = pygame.Rect(0, HEADER_H, WIN_W, VIEW_H)

    # ── viewer state ─────────────────────────────────────────────────────
    game_idx    = 0
    tick        = 0
    is_playing  = False
    speed       = 1.0
    view_mode   = 0           # 0 = game view, 1 = eagle-eye, 2 = stats
    traj_surf   = None
    play_accum  = 0.0
    dragging_pb = False
    typing_game = False       # True while user is typing a game number
    typed_str   = ""          # digits typed so far
    stats_scroll     = 0
    stats_max_scroll = 0

    view_names  = {0: "Game view", 1: "Eagle-eye", 2: "Stats"}
    print("Computing dataset statistics…")
    ds_stats = precompute_stats(sensors, car_x, car_vx, lengths, starts)

    # traj_surf layout (kept in sync with eagle-eye draw code)
    PADL, PADT, PADB = 54, 34, 26
    TRAJ_W = int(WIN_W * 0.62) - PADL - 6
    TRAJ_H = VIEW_H - PADT - PADB

    def switch_game(idx):
        nonlocal traj_surf, tick
        tick      = 0
        sl        = slice(starts[idx], starts[idx] + lengths[idx])
        traj_surf = build_trajectory_surface(
            car_x[sl], ego_xy[sl], TRAJ_W, TRAJ_H)

    switch_game(game_idx)

    # playbar track rect (used for hit-testing; matches draw_playbar geometry)
    pb_track = pygame.Rect(56, HEADER_H + VIEW_H + 17, WIN_W - 112, 14)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        game_len   = lengths[game_idx]
        base       = starts[game_idx] + tick
        sensors_t  = sensors[base]
        car_x_t    = car_x  [base]
        car_vx_t   = car_vx [base]
        ego_xy_t   = ego_xy [base]

        # ── events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # ── game-number input mode ────────────────────────────────
                if typing_game:
                    if event.key == pygame.K_RETURN:
                        if typed_str:
                            idx = int(typed_str) - 1          # user enters 1-based
                            game_idx = max(0, min(idx, n_games - 1))
                            switch_game(game_idx)
                        typing_game = False
                        typed_str   = ""
                    elif event.key == pygame.K_ESCAPE:
                        typing_game = False
                        typed_str   = ""
                    elif event.key == pygame.K_BACKSPACE:
                        typed_str = typed_str[:-1]
                    elif event.unicode.isdigit():
                        typed_str += event.unicode
                    continue   # swallow all other keys while typing

                # ── normal mode ───────────────────────────────────────────
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    is_playing = not is_playing
                    play_accum = 0.0
                elif event.key == pygame.K_RIGHT:
                    tick = min(tick + 1, game_len - 1)
                    is_playing = False
                elif event.key == pygame.K_LEFT:
                    tick = max(tick - 1, 0)
                    is_playing = False
                elif event.key == pygame.K_UP:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    game_idx = (game_idx + (100 if shift else 1)) % n_games
                    switch_game(game_idx)
                elif event.key == pygame.K_DOWN:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    game_idx = (game_idx - (100 if shift else 1)) % n_games
                    switch_game(game_idx)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 2, 64.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 2, 0.125)
                elif event.key == pygame.K_TAB:
                    view_mode = (view_mode + 1) % 3
                    stats_scroll = 0
                elif event.key == pygame.K_g:
                    typing_game = True
                    typed_str   = ""

            elif event.type == pygame.MOUSEWHEEL:
                if view_mode == 2:
                    stats_scroll = max(0, min(stats_scroll - event.y * 20, stats_max_scroll))

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pb_track.collidepoint(event.pos):
                    dragging_pb = True
                    frac = (event.pos[0] - pb_track.x) / pb_track.width
                    tick = int(np.clip(frac, 0, 1) * (game_len - 1))

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_pb = False

            elif event.type == pygame.MOUSEMOTION and dragging_pb:
                frac = (event.pos[0] - pb_track.x) / pb_track.width
                tick = int(np.clip(frac, 0, 1) * (game_len - 1))

        # ── playback advance ──────────────────────────────────────────────
        if is_playing:
            play_accum += speed * dt * 60   # ticks per second = speed * 60
            steps       = int(play_accum)
            play_accum -= steps
            tick       += steps
            if tick >= game_len:
                tick       = game_len - 1
                is_playing = False

        # ── render ────────────────────────────────────────────────────────
        screen.fill(C_BG)

        if view_mode == 0:
            draw_game_view(screen, view_rect, sensors_t, car_x_t, ego_xy_t)
        elif view_mode == 1:
            draw_eagle_view(screen, view_rect, traj_surf,
                            sensors_t, car_vx_t, tick, game_len)
        else:
            gm_stats = game_stats(sensors, car_x, car_vx, lengths, starts, game_idx)
            stats_scroll, stats_max_scroll = draw_stats_view(
                screen, view_rect, ds_stats, gm_stats, game_idx, n_games, stats_scroll)

        header_view = (f"Jump to game: {typed_str}▌  (Enter confirm, Esc cancel)"
                       if typing_game else view_names[view_mode])
        draw_header(screen, font, game_idx, n_games, tick, game_len,
                    speed, header_view)
        draw_playbar(screen, tick, game_len, is_playing)
        draw_status(screen, font, sensors_t, car_x_t, ego_xy_t)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

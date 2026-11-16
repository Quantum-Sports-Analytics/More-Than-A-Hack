"""Reading the Metrica match (BAR-ATH) for the ball-in-the-air model, at 10 frames per second.

Copies of read_atd / read_slots / merge_ball / drop_jumps / resample from
"Imported xgb dead ball model/run_dead_ball_xgb.py", so this folder runs on its own:
1. players from the tracking file; ball = the new detections (detections_balle_BAR-ATH.csv) where they exist;
2. positions faster than 45 m/s from both neighbours are dropped;
3. despike: a position more than DESPIKE_M from the median of its 7 neighbours (0.28 s) is dropped
   (the same filter is applied to the simulated SkillCorner ball, see project.py);
4. gaps filled with straight lines inside each half; 25 -> 10 frames per second.
Axes: metres, centre spot at (0, 0), y towards the camera.
"""
import re

import numpy as np
import pandas as pd

from common import DATA

TACT = DATA / "tactical"
LENGTH, WIDTH = 105.3, 68.0
VIDEO_FPS, FPS = 25, 10
HALVES = {1: (11185, 90651), 2: (110808, 190197)}
MAX_SPEED = 45.0
DESPIKE_M, DESPIKE_W = 3.0, 7


def read_atd(path=TACT / "BAR-ATH_tactical_atd.txt"):
    rows = []
    with open(path) as f:
        for line in f:
            _, players, ball = line.rstrip("\n").split(":")
            rows.append(np.array((players + ";" + ball).replace(";", ",").split(","), dtype=float))
    a = np.vstack(rows)
    return a[:, :-3].reshape(len(a), -1, 3)[..., :2], a[:, -3:-1]


def read_slots(path=TACT / "BAR-ATH_tactical_atd.xml"):
    xml = open(path, encoding="utf-8").read()
    teams = dict(re.findall(r'<Team id="([^"]+)">\s*<Name>([^<]+)</Name>', xml))
    slots = []
    for block in re.findall(r"<Player .*?</Player>", xml, flags=re.S):
        team_id = re.search(r'teamId="([^"]+)"', block).group(1)
        slots.append({"id": int(re.search(r'id="(\d+)"', block).group(1)),
                      "team": 0 if teams[team_id] == "Barcelona" else 1,
                      "gk": bool(re.search(r"<Name>position_type</Name>\s*<Value>Goalkeeper</Value>", block))})
    return slots


def merged_ball(old_ball):
    det = pd.read_csv(TACT / "detections_balle_BAR-ATH.csv")
    ball = old_ball.copy()
    source = np.full(len(ball), "tracking", dtype=object)
    f = det["frame"].to_numpy()
    has = det["source"].ne("aucune").to_numpy() & det["x_norm"].notna().to_numpy()
    ball[f] = np.where(has[:, None], det[["x_norm", "y_norm"]].to_numpy(), np.nan)
    source[f] = det["source"].to_numpy()
    conf = np.full(len(ball), np.nan)
    conf[f] = det["conf_video"].to_numpy()
    return ball, source, conf


def to_m(a):
    return np.stack([(a[..., 0] - 0.5) * LENGTH, (a[..., 1] - 0.5) * WIDTH], -1)


def drop_jumps(ball, fps, passes=2):
    out = ball.copy()
    for _ in range(passes):
        idx = np.flatnonzero(~np.isnan(out[:, 0]))
        if len(idx) < 3:
            break
        p = out[idx]
        step = np.hypot(*(p[1:] - p[:-1]).T) / (np.diff(idx) / fps)
        v_prev, v_next = np.r_[0, step], np.r_[step, 0]
        out[idx[(v_prev > MAX_SPEED) & (v_next > MAX_SPEED)]] = np.nan
    return out


def despike(ball, max_dev=DESPIKE_M, w=DESPIKE_W):
    """Drop positions farther than max_dev from the median of the w positions around them."""
    out = ball.copy()
    med = pd.DataFrame(out).rolling(w, center=True, min_periods=3).median().to_numpy()
    bad = np.hypot(*(out - med).T) > max_dev
    out[bad] = np.nan
    return out, int(bad.sum())


def fill(ball):
    out = ball.copy()
    for c in (0, 1):
        out[:, c] = pd.Series(out[:, c]).interpolate(limit_area="inside").to_numpy()
    return out


def resample(arr, lo, hi):
    pos = np.arange(lo, hi + 1e-9, VIDEO_FPS / FPS)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, hi)
    w = (pos - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = arr[i0] * (1 - w) + arr[i1] * w
    exact = (w == 0).reshape(-1)
    out[exact] = arr[i0[exact]]
    return out, pos


def load():
    """Per half: dict(period, video_pos (10 fps sample -> video frame, float), bx, by, X, Y, observed (bool, a real
    ball position before filling), source), plus team and goalkeeper arrays for the 46 slots."""
    players, old = read_atd()
    slots = read_slots()
    team = np.array([s["team"] for s in slots])
    is_gk = np.array([s["gk"] for s in slots])
    ball, source, conf = merged_ball(old)
    halves = []
    n_spikes = 0
    for p, (lo, hi) in HALVES.items():
        b = to_m(ball[lo:hi + 1])
        b = drop_jumps(b, VIDEO_FPS)
        b, k = despike(b)
        n_spikes += k
        observed = ~np.isnan(b[:, 0])
        bf = fill(b)
        pl = to_m(players[lo:hi + 1])
        b10, pos = resample(bf, 0, hi - lo)
        p10, _ = resample(pl, 0, hi - lo)
        obs10, _ = resample(observed.astype(float), 0, hi - lo)
        halves.append(dict(period=p, video_pos=pos + lo, bx=b10[:, 0], by=b10[:, 1], X=p10[..., 0], Y=p10[..., 1],
                           observed=obs10 > 0.99, source_25=source[lo:hi + 1], conf_25=conf[lo:hi + 1],
                           ball25=b, ball25_filled=bf))
    return halves, team, is_gk, n_spikes

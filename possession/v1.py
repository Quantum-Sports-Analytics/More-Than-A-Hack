"""Possession v1: soft team evidence + dead-ball evidence, decoded by a 3-state Viterbi.

Design (see docs): the ground truth flips at the touch of the other team, possessions of
1-3 s exist, and the tracked ball is noisy (~2 m) and missing on 41 % of graded frames in long
gaps. So instead of hard contact rules we accumulate per-frame evidence and let a Viterbi decoder
with a switching cost place each change where the evidence turns. No window smoothing.

Inputs: the tactical ATD (xml + txt). Output: frame,label CSV with 199,319 rows.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

PITCH = (105.3, 68.0)                     # metres, from games.py
HALVES = ((11185, 90652), (110808, 190198))  # graded video frames, half-open
N_FRAMES = 199319
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D", "X"])


# ----------------------------------------------------------------------------------- loading
def load_tactical(data_dir, cache_dir=None):
    """Players (N, S, 3) and ball (N, 3) in normalised coords, plus slot team (0 home, 1 away)."""
    meta_path = os.path.join(data_dir, "tactical", "BAR-ATH_tactical_atd.xml")
    raw_path = os.path.join(data_dir, "tactical", "BAR-ATH_tactical_atd.txt")
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    slot_team = np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                         dtype=np.int8)
    cache = os.path.join(cache_dir, "tactical_atd.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["P"], z["B"], slot_team
    n_slots = len(slot_team)
    P = np.full((N_FRAMES, n_slots, 3), np.nan, dtype=np.float32)
    B = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            fid, players, ball = line.rstrip("\n").split(":")
            i = int(fid) - 1                      # ATD frame_id 1.. -> video frame 0..
            P[i] = np.array([g.split(",") for g in players.replace("NaN", "nan").split(";")],
                            dtype=np.float32)
            B[i] = np.array(ball.replace("NaN", "nan").split(","), dtype=np.float32)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, P=P, B=B)
    return P, B, slot_team


# ---------------------------------------------------------------------------------- features
def smooth_ball(bx, by, win=7):
    """Savitzky-Golay (2nd order, 7 frames) position and velocity inside each visible run."""
    xs, ys = np.full_like(bx, np.nan), np.full_like(by, np.nan)
    vx, vy = xs.copy(), ys.copy()
    vis = np.isfinite(bx)
    edges = np.diff(np.r_[0, vis.astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < win:
            xs[a:b], ys[a:b] = bx[a:b], by[a:b]
            continue
        xs[a:b], ys[a:b] = savgol_filter(bx[a:b], win, 2), savgol_filter(by[a:b], win, 2)
        vx[a:b] = savgol_filter(bx[a:b], win, 2, deriv=1) * 25
        vy[a:b] = savgol_filter(by[a:b], win, 2, deriv=1) * 25
    return xs, ys, vx, vy


def features(P, B, slot_team, pitch=PITCH):
    L, W = pitch
    px, py = P[:, :, 0].astype(float) * L, P[:, :, 1].astype(float) * W
    bx, by = B[:, 0].astype(float) * L, B[:, 1].astype(float) * W
    xs, ys, vx, vy = smooth_ball(bx, by)
    vis = np.isfinite(xs)
    dist = np.hypot(px - xs[:, None], py - ys[:, None])
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(slot_team == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(slot_team == 1, dist, np.nan), axis=1)
        spd = np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * 25
        med_speed = np.nanmedian(spd, axis=1)
    med_speed[~np.isfinite(med_speed)] = np.nanmedian(med_speed)
    return dict(vis=vis, bx=xs, by=ys, ball_speed=np.hypot(vx, vy), d_home=d_home, d_away=d_away,
                med_speed=med_speed, L=L, W=W)


def team_evidence(f, scale=1.5, reach=2.5, clip=3.0):
    """Log-odds that H (positive) rather than A controls the ball, per frame. 0 without ball."""
    dh, da = f["d_home"], f["d_away"]
    e = np.clip((da - dh) / scale, -clip, clip)
    w = np.exp(-np.fmin(dh, da) / reach)
    ghost = f["vis"] & (np.nan_to_num(f["ball_speed"], nan=0) < 2.0) & (np.fmin(dh, da) > 4.0)
    out = np.where(f["vis"] & ~ghost, w * e, 0.0)
    return np.nan_to_num(out)


def dead_evidence(f, speed_ref=1.3):
    """Log-odds of a dead ball, per frame, from collective kinematics and ball state."""
    ps2 = median_filter(f["med_speed"], size=51, mode="nearest")      # 2 s rolling median
    L, W = f["L"], f["W"]
    vis = f["vis"]
    bx, by = np.nan_to_num(f["bx"], nan=L / 2), np.nan_to_num(f["by"], nan=W / 2)
    out_of_pitch = vis & ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1))
    slow_ball = vis & (np.nan_to_num(f["ball_speed"], nan=99) < 1.0)
    logit = (2.0 * (speed_ref - ps2) + 1.5 * (~vis) + 4.0 * out_of_pitch + 0.5 * slow_ball - 1.5)
    return logit


# ---------------------------------------------------------------------------------- decoding
def viterbi(em, switch_cost, dead_cost):
    """em: (T, 3) per-frame log-scores for H, A, D. Returns the best state path."""
    T = len(em)
    trans = np.array([[0, -switch_cost, -dead_cost],
                      [-switch_cost, 0, -dead_cost],
                      [-dead_cost, -dead_cost, 0]], dtype=float)
    score = em[0].copy()
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans            # from i (rows) to j (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + em[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def drop_short_dead(path, min_frames):
    """Dead-ball runs shorter than min_frames are absorbed (a false D splits a possession)."""
    out = path.copy()
    edges = np.diff(np.r_[0, (out == D).astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_frames:
            continue
        before = out[a - 1] if a > 0 else None
        after = out[b] if b < len(out) else None
        if before is None and after is None:
            continue
        if before is None or before == after:
            out[a:b] = after if before is None else before
        else:                                     # different teams either side: switch at the start
            out[a:b] = after
    return out


def predict(P, B, slot_team, switch_cost=8.0, dead_cost=6.0, dead_gain=0.25,
            min_dead_s=3.0, team_kw=None, dead_kw=None):
    f = features(P, B, slot_team)
    e = team_evidence(f, **(team_kw or {}))
    dl = dead_evidence(f, **(dead_kw or {}))
    em = np.stack([e / 2, -e / 2, dead_gain * dl], axis=1)
    labels = np.full(N_FRAMES, 3, dtype=np.int8)          # X outside graded halves
    for a, b in HALVES:
        path = viterbi(em[a:b], switch_cost, dead_cost)
        labels[a:b] = drop_short_dead(path, int(min_dead_s * 25))
    return labels, dict(features=f, team_evidence=e, dead_logit=dl)


def write_csv(labels, path):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        for i, k in enumerate(labels):
            fh.write(f"{i},{LABELS[k]}\n")

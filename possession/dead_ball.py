"""Step 2: dead-ball (D) detector trained on the Sportec Open DFL Dataset, applied to BAR-ATH.

Idea (Lang et al. 2022; Borghesi et al. 2023): a stoppage is visible in the collective kinematics
of the 22 players (walking, frozen shape, slow centroids) with a few seconds of past AND future
context, plus weak ball cues (ball missing for a long time, out of pitch, slow). Every feature is
identity-free and computed by the SAME function on Sportec (TRACAB, clean) and on our tactical
ATD (noisy, 46 anonymous slots, ball missing on 41 % of graded frames in long gaps).

Supervision: Sportec per-frame BallStatus (1 alive / 0 dead). To make ball features transfer, the
Sportec ball is degraded at training time: long gaps drawn from our empirical ATD gap lengths, with
a state-dependent gap rate calibrated on the FULL tactical match by a ball-free model (never on
the public 5 minutes), plus ~2 m slowly varying position noise.

Decoding: centred 1 s mean of the probability, hysteresis (enter > p_hi, stay > p_lo), short live
holes (< 1 s) between two D merged, D runs shorter than min_dur dropped. Operating point
(p_hi 0.9, p_lo 0.6, min_dur 3 s) chosen on Sportec leave-one-match-out with a spellF1 proxy
(oracle Sportec teams + our D), never on the public 5 minutes. Smart Tagging SET PIECES starts are
used only to EXTEND a D that the detector closed <= 20 s before the tag (end-of-D evidence).

Team labels (H/A) come from the v1 last-touch decoder (impulse peaks of the smoothed ball
velocity, height 3 m/s, nearest player <= 3 m, last touching team carried forward).

Usage:  python possession/dead_ball.py --data <data_dir> --sportec <dir_with_npz> --out v2.csv
The Sportec npz files are produced by possession/fetch_sportec.py (CC-BY data fetched from the
Hugging Face mirror pysport/idsse-data; keep the npz OUTSIDE the repo).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1  # noqa: E402

FPS = 25
SPORTEC_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SHIFTS_S = (1, 2, 4, 8, 15, 20)            # time-shifted copies, seconds (Lang et al. used up to 80 s)
SET_PIECE_TYPES = ("Throw-in", "Corner", "Goal Kick", "Free Kick", "Kick Off")


# ------------------------------------------------------------------------------- base features
def _central_speed(p, k):
    """|p(t+k) - p(t-k)| / (2k / FPS) along axis 0, NaN at the edges."""
    out = np.full(p.shape, np.nan, dtype=np.float32)
    out[k:-k] = (p[2 * k:] - p[:-2 * k]) / (2 * k / FPS)
    return out


def _run_lengths(mask):
    """For a boolean mask: length of the run containing t, time since run start, time to run end."""
    T = len(mask)
    length = np.zeros(T, np.float32)
    since = np.zeros(T, np.float32)
    until = np.zeros(T, np.float32)
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        n = b - a
        length[a:b] = n
        since[a:b] = np.arange(n)
        until[a:b] = np.arange(n)[::-1]
    return length / FPS, since / FPS, until / FPS


def base_features(px, py, team, bx, by, L, W):
    """Per-frame identity-free features for ONE half.

    px, py: (T, S) player positions in metres, origin at a pitch corner, NaN when unseen.
    team: (S,) 0/1. bx, by: (T,) ball in metres, NaN when unseen (None: players only).
    """
    px = px.astype(np.float32)
    py = py.astype(np.float32)
    f = {}
    with np.errstate(all="ignore"):
        # player speeds over +-2 frames (0.16 s); larger values are tracking jumps
        spd = np.hypot(_central_speed(px, 2), _central_speed(py, 2))
        spd[spd > 12] = np.nan
        nvis = np.isfinite(spd).sum(1).astype(np.float32)
        nvis[nvis == 0] = np.nan
        q = np.nanpercentile(spd, [25, 50, 75, 90], axis=1)
        f["spd_q25"], f["spd_med"], f["spd_q75"], f["spd_q90"] = q
        f["spd_mean"] = np.nanmean(spd, axis=1)
        f["frac_slow"] = (spd < 1.0).sum(1) / nvis
        f["frac_walk"] = (spd < 2.0).sum(1) / nvis
        f["frac_fast"] = (spd > 4.0).sum(1) / nvis
        # displacement over 2 s (robust to jitter): frozen configurations
        disp = np.hypot(_central_speed(px, FPS), _central_speed(py, FPS))
        disp[disp > 10] = np.nan
        f["disp2_med"] = np.nanmedian(disp, axis=1)
        f["disp2_q90"] = np.nanpercentile(disp, 90, axis=1)
        # team centroids (median: robust to dropouts), speeds over 1 s, spreads
        cs, spread, cxs = [], [], []
        for k in (0, 1):
            m = team == k
            cx = np.nanmedian(px[:, m], axis=1)
            cy = np.nanmedian(py[:, m], axis=1)
            cs.append(np.hypot(_central_speed(cx, 12), _central_speed(cy, 12)))
            spread.append(np.nanmean(np.hypot(px[:, m] - cx[:, None], py[:, m] - cy[:, None]), axis=1))
            cxs.append((cx, cy))
        f["cspd_min"], f["cspd_max"] = np.fmin(*cs), np.fmax(*cs)
        f["spread_min"], f["spread_max"] = np.fmin(*spread), np.fmax(*spread)
        f["team_gap"] = np.hypot(cxs[0][0] - cxs[1][0], cxs[0][1] - cxs[1][1])
        f["ext_x"] = np.nanpercentile(px, 90, axis=1) - np.nanpercentile(px, 10, axis=1)
        f["ext_y"] = np.nanpercentile(py, 90, axis=1) - np.nanpercentile(py, 10, axis=1)
        allx = np.nanmedian(px, axis=1)
        f["block_x"] = np.abs(allx - L / 2)                   # mirrored: side-independent
    if bx is not None:
        f.update(ball_features(px, py, bx, by, L, W))
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def ball_features(px, py, bx, by, L, W):
    """Per-frame ball cues for ONE half (the tracked ball may be missing or offset)."""
    f = {}
    with np.errstate(all="ignore"):
        # Savitzky-Golay as in v1
        xs, ys, vx, vy = v1.smooth_ball(bx.astype(float), by.astype(float))
        vis = np.isfinite(xs)
        f["ball_vis"] = vis.astype(np.float32)
        f["ball_spd"] = np.hypot(vx, vy)
        edge = np.fmin(np.fmin(xs, L - xs), np.fmin(ys, W - ys))
        f["ball_edge"] = edge                                  # < 0 outside the pitch
        f["ball_out"] = np.where(vis, (edge < -1.0).astype(np.float32), np.nan)
        f["ball_goal_dist"] = np.fmin(xs, L - xs)
        f["ball_touch_dist"] = np.fmin(ys, W - ys)
        d = np.hypot(px - xs[:, None], py - ys[:, None])
        f["ball_near"] = np.nanmin(d, axis=1)
        f["ball_n5"] = np.where(vis, (d < 5).sum(1), np.nan)
        glen, gsince, guntil = _run_lengths(~vis)
        f["gap_len"], f["gap_since"], f["gap_until"] = glen, gsince, guntil
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def context_features(base, players_only=False):
    """Rolling statistics and time-shifted copies (per half; no cross-half leakage)."""
    out = {}
    core = ["spd_med", "spd_q75", "spd_q90", "frac_slow", "frac_fast", "disp2_med", "cspd_max",
            "cspd_min", "spread_max", "ext_x"]
    for c in base.columns:
        if players_only and (c.startswith("ball") or c.startswith("gap")):
            continue
        out[c] = base[c]
    for c in core:
        s = base[c]
        for w in (1, 4, 10):
            out[f"{c}_m{w}"] = s.rolling(w * FPS, center=True, min_periods=1).mean()
    ref = {c: base[c].rolling(2 * FPS, center=True, min_periods=1).median()
           for c in ("spd_med", "frac_slow", "disp2_med", "cspd_max")}
    for c, s in ref.items():
        for sh in SHIFTS_S:
            out[f"{c}_p{sh}"] = s.shift(sh * FPS).bfill()      # past
            out[f"{c}_f{sh}"] = s.shift(-sh * FPS).ffill()     # future
        out[f"{c}_pastmin5"] = s.rolling(5 * FPS, min_periods=1).min()
        out[f"{c}_futmin5"] = s[::-1].rolling(5 * FPS, min_periods=1).min()[::-1]
    if not players_only:
        vis = base["ball_vis"]
        for w in (2, 6, 20):
            out[f"ball_vis_m{w}"] = vis.rolling(w * FPS, center=True, min_periods=1).mean()
        bo = base["ball_out"].fillna(0)
        out["ball_out_past3"] = bo.rolling(3 * FPS, min_periods=1).max()
        out["ball_out_past10"] = bo.rolling(10 * FPS, min_periods=1).max()
        out["ball_spd_m1"] = base["ball_spd"].rolling(FPS, center=True, min_periods=1).median()
        out["ball_edge_m2"] = base["ball_edge"].rolling(2 * FPS, center=True, min_periods=1).median()
    return pd.DataFrame(out).astype(np.float32)


# ------------------------------------------------------------------------------------- Sportec
def load_sportec(path):
    """Yield one dict per half, trimmed to [first alive frame, last alive frame]."""
    z = np.load(path)
    L, W = float(z["L"]), float(z["W"])
    for h in ("h1", "h2"):
        bs = z[f"{h}_BS"]
        alive = np.flatnonzero(bs == 1)
        a, b = alive[0], alive[-1] + 1
        yield dict(px=z[f"{h}_PX"][a:b] + L / 2, py=z[f"{h}_PY"][a:b] + W / 2, team=z[f"{h}_team"],
                   bx=z[f"{h}_BX"][a:b] + L / 2, by=z[f"{h}_BY"][a:b] + W / 2,
                   dead=(bs[a:b] == 0), poss=z[f"{h}_BP"][a:b], L=L, W=W)


def degrade_ball(bx, by, dead, rng, gap_frames, q_dead, miss_live, noise_sd=1.5, tau_s=1.0):
    """Drop the ball like the tactical ATD does and add slowly varying ~2 m position noise.

    On the full tactical match a ball-free model says the ball is missing on ~72 % of dead frames
    and ~13 % of live frames, in long gaps. Gaps that start at random cannot reproduce that (they
    spill into live play), so the simulation has two parts:
      * stoppage-anchored gaps: each true dead run is hidden with probability q_dead, from a start
        jittered by U(-1 s, 0.3 len) to an end jittered by U(-2 s, 0.3 len) before the restart;
      * random live gaps: lengths drawn from the empirical ATD gap lengths, started with a
        per-frame hazard set from miss_live.
    """
    T = len(bx)
    miss = np.zeros(T, bool)
    for a, b in runs_of(dead):
        if rng.random() >= q_dead:
            continue
        n = b - a
        s = a + int(rng.uniform(-1 * FPS, 0.3 * n))
        e = b - int(rng.uniform(-2 * FPS, 0.3 * n))
        if e > s:
            miss[max(s, 0):min(e, T)] = True
    G = float(np.mean(gap_frames))
    h_live = miss_live / ((1 - miss_live) * G)
    u = rng.random(T)
    t = 0
    while t < T:
        if not dead[t] and u[t] < h_live:
            n = int(rng.choice(gap_frames))
            miss[t:t + n] = True
            t += n
        else:
            t += 1
    a_ = np.exp(-1.0 / (tau_s * FPS))
    e = np.empty((T, 2))
    z = rng.normal(size=(T, 2)) * noise_sd * np.sqrt(1 - a_ * a_)
    e[0] = rng.normal(size=2) * noise_sd
    for i in range(1, T):
        e[i] = a_ * e[i - 1] + z[i]
    bx2 = np.where(miss, np.nan, bx + e[:, 0])
    by2 = np.where(miss, np.nan, by + e[:, 1])
    return bx2.astype(np.float32), by2.astype(np.float32)


# ------------------------------------------------------------------------------------ our data
def tactical_halves(P, B, slot_team, pitch=v1.PITCH):
    """Graded halves of the tactical ATD as dicts in metres."""
    L, W = pitch
    for a, b in v1.HALVES:
        yield dict(px=P[a:b, :, 0] * L, py=P[a:b, :, 1] * W, team=slot_team,
                   bx=B[a:b, 0] * L, by=B[a:b, 1] * W, L=L, W=W, start=a, stop=b)


def ball_gap_lengths(B):
    """Lengths (frames) of the ball-missing runs inside the graded halves."""
    out = []
    for a, b in v1.HALVES:
        miss = ~np.isfinite(B[a:b, 0])
        e = np.diff(np.r_[0, miss.astype(np.int8), 0])
        out += list(np.flatnonzero(e == -1) - np.flatnonzero(e == 1))
    return np.array(out)


def set_piece_starts(data_dir):
    """Video frames at which Smart Tagging opens a SET PIECES instance."""
    root = ET.parse(os.path.join(data_dir, "tactical", "BAR-ATH_tactical_smart_tagging.xml")).getroot()
    frames = []
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        if code != "SET PIECES":
            continue
        frames.append(int(round(float(inst.findtext("start")) * FPS)))
    return np.array(sorted(frames))


# ------------------------------------------------------------------------------------ decoding
def runs_of(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def decode(p, p_hi=0.7, p_lo=0.5, min_dur_s=3.0, merge_s=1.0, smooth_s=1.0):
    """Hysteresis + minimum duration on the dead-ball probability of ONE half -> bool mask."""
    ps = pd.Series(p).rolling(max(1, int(smooth_s * FPS)), center=True, min_periods=1).mean().values
    cand = ps > p_lo
    out = np.zeros(len(p), bool)
    for a, b in runs_of(cand):                      # keep a low-threshold run only if it peaks high
        if ps[a:b].max() > p_hi:
            out[a:b] = True
    for a, b in runs_of(~out):                      # merge short live holes between two D
        if a > 0 and b < len(out) and (b - a) < merge_s * FPS:
            out[a:b] = True
    for a, b in runs_of(out):
        if (b - a) < min_dur_s * FPS:
            out[a:b] = False
    return out


def bridge_set_pieces(dead, tags, max_gap_s=20.0):
    """Smart Tagging opens a SET PIECES instance while the ball is still dead (before the restart).

    If a tag starts on a frame decoded as live and a decoded D ended at most max_gap_s earlier,
    the D is extended up to the tag start: the detector ended the stoppage too early (typical of
    goal kicks, where players jog back into shape). dead: bool mask of ONE half; tags: frame
    indices relative to that half. Tags never create a D on their own.
    """
    out = dead.copy()
    for t in tags:
        if t < 0 or t >= len(out) or out[t]:
            continue
        lo = max(0, t - int(max_gap_s * FPS))
        prev = np.flatnonzero(out[lo:t])
        if len(prev):
            out[lo + prev[-1] + 1:t + 1] = True
    return out


# --------------------------------------------------------------------------------- team logic
def last_touch_team(P, B, slot_team, h=3.0, dist_max=3.0, k=4):
    """v1 last-touch decoder (reproduces the v1 artifact's H/A logic). Returns codes 0/1/3."""
    f = v1.features(P, B, slot_team)
    bxs, bys = f["bx"], f["by"]
    vx, vy = np.gradient(bxs) * FPS, np.gradient(bys) * FPS
    ok = np.isfinite(vx).astype(float)
    vx, vy = np.nan_to_num(vx), np.nan_to_num(vy)
    T = len(vx)
    cs = lambda a: np.r_[0, np.cumsum(a)]                                   # noqa: E731
    Sx, Sy, So = cs(vx), cs(vy), cs(ok)
    t = np.arange(k, T - k - 1)
    nb, na = So[t] - So[t - k], So[t + 1 + k] - So[t + 1]
    bxm, bym = (Sx[t] - Sx[t - k]) / np.maximum(nb, 1), (Sy[t] - Sy[t - k]) / np.maximum(nb, 1)
    axm, aym = (Sx[t + 1 + k] - Sx[t + 1]) / np.maximum(na, 1), (Sy[t + 1 + k] - Sy[t + 1]) / np.maximum(na, 1)
    imp = np.full(T, np.nan)
    imp[t] = np.where((nb == k) & (na == k), np.hypot(axm - bxm, aym - bym), np.nan)
    dmin = np.fmin(f["d_home"], f["d_away"])
    team_near = np.where(f["d_home"] <= f["d_away"], 0, 1)
    pk, _ = find_peaks(np.nan_to_num(imp), height=h, distance=5)
    pk = pk[dmin[pk] <= dist_max]
    lab = np.full(T, 3, np.int8)
    for a, b in v1.HALVES:
        cur = 3
        touches = pk[(pk >= a) & (pk < b)]
        j = 0
        for i in range(a, b):
            while j < len(touches) and touches[j] <= i:
                cur = team_near[touches[j]]
                j += 1
            lab[i] = cur
    return lab


# ------------------------------------------------------------------------------------- model
def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sportec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    # degradation rates calibrated on the full tactical match with the ball-free model (see report)
    ap.add_argument("--q-dead", type=float, default=0.965)
    ap.add_argument("--miss-live", type=float, default=0.165)
    ap.add_argument("--p-hi", type=float, default=0.9)     # chosen on Sportec LOMO (spellF1 proxy)
    ap.add_argument("--p-lo", type=float, default=0.6)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=5, help="training subsample (frames)")
    ap.add_argument("--no-set-pieces", action="store_true")
    ap.add_argument("--save-proba", default=None, help="optional .npy for the per-frame P(dead)")
    a = ap.parse_args()
    P, B, slot_team = v1.load_tactical(a.data, a.cache)
    gaps = ball_gap_lengths(B)
    rng = np.random.default_rng(0)
    X, y = [], []
    for mid in SPORTEC_IDS:
        path = os.path.join(a.sportec, f"{mid}.npz")
        if not os.path.exists(path):
            continue
        for hf in load_sportec(path):
            bx, by = degrade_ball(hf["bx"], hf["by"], hf["dead"], rng, gaps, a.q_dead, a.miss_live)
            F = context_features(base_features(hf["px"], hf["py"], hf["team"], bx, by, hf["L"], hf["W"]))
            X.append(F.iloc[::a.step])
            y.append(hf["dead"][::a.step])
    model = make_model().fit(pd.concat(X), np.concatenate(y))
    cols = list(X[0].columns)
    del X, y
    team = last_touch_team(P, B, slot_team)
    tags = set_piece_starts(a.data)
    labels = np.full(v1.N_FRAMES, 3, np.int8)
    proba = np.full(v1.N_FRAMES, np.nan, np.float32)
    for hf in tactical_halves(P, B, slot_team):
        F = context_features(base_features(hf["px"], hf["py"], hf["team"], hf["bx"], hf["by"], hf["L"], hf["W"]))
        p = model.predict_proba(F[cols])[:, 1]
        dead = decode(p, a.p_hi, a.p_lo, a.min_dur)
        s, e = hf["start"], hf["stop"]
        if not a.no_set_pieces:
            dead = bridge_set_pieces(dead, tags[(tags >= s) & (tags < e)] - s)
        labels[s:e] = np.where(dead, v1.D, team[s:e])
        proba[s:e] = p
    v1.write_csv(labels, a.out)
    if a.save_proba:
        np.save(a.save_proba, proba)


if __name__ == "__main__":
    main()

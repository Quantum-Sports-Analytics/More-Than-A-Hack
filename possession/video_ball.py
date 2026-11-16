"""Video ball track for the tactical video, fused with the tracked (ATD) ball.

Pipeline
1. Load the GPU detection parts (frame, src, u, v, conf); src 0 = full frame, 1 = x2 crop on the
   ATD ball, 2 = x2 crop on a full-frame candidate.
2. Project every candidate to pitch metres with the inverse of that frame's homography
   (the csv maps normalised pitch (x, y, 1) -> pixels).
3. Drop candidates more than OUT_MARGIN metres outside the pitch; merge duplicates of the same
   object seen by several sources (<= DEDUP_PX pixels apart in the same frame).
4. Static-object blacklist learnt from the detections themselves: 0.5 m pitch cells hit in an
   abnormal number of distinct minutes (>= 99.9th percentile over all cells) whose detections are
   farther than 3 m from every player more often than the average detection; dilated by one cell.
   The three spots (centre, penalty spots) are always included.
5. Offline dynamic programming over candidates plus a 'no ball' state (min-cost path, one ball per
   frame). Transition cost = Laplace on the mean speed over the gap (scale = mean speed of
   confident frame-to-frame pairs), forbidden above v_max (99th percentile of the same speeds) plus
   a 1 m allowance; gaps up to MAX_GAP frames are bridged, longer ones go through 'no ball' and pay
   track end + start costs. Emission reward = log-odds that a candidate of that confidence /
   multi-source bin has a plausible continuation in the next frame (unsupervised), plus a
   log-likelihood-ratio term for being within 3 m of the ATD ball (estimated by contrasting strong
   and weak candidates), halved because that contrast is itself a proxy.
6. Fusion per frame: video (detected or bridged) > Savitzky-Golay-smoothed ATD > NaN, with the
   source recorded. Smoothed position / velocity are computed inside runs of constant source so a
   source switch never creates a fake impulse.

No parameter is fitted on the public ground truth.
CLI: python possession/video_ball.py --parts <dir> --data <data dir> --out <npz>
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1  # noqa: E402

L, W = v1.PITCH
FPS = 25
OUT_MARGIN = 2.0          # m outside the pitch beyond which a candidate is dropped
DEDUP_PX = 8.0            # px: same object seen by full frame and crops
CELL = 0.5                # m, blacklist grid
MAX_GAP = 12              # frames bridged inside one track (0.5 s)
SPOTS = ((L / 2, W / 2), (11.0, W / 2), (L - 11.0, W / 2))   # centre and penalty spots, metres
SRC_NONE, SRC_VIDEO, SRC_BRIDGE, SRC_ATD = 0, 1, 2, 3


# ----------------------------------------------------------------------------------- loading
def load_parts(parts_dir):
    cols = {k: [] for k in ("frame", "src", "u", "v", "conf")}
    files = sorted(glob.glob(os.path.join(parts_dir, "*.npz")))
    cover = 0
    for p in files:
        z = np.load(p)
        cover += int(z["last"]) - int(z["first"]) + 1     # 'last' is inclusive
        for k in cols:
            cols[k].append(z[k])
    det = {k: np.concatenate(v) for k, v in cols.items()}
    o = np.lexsort((-det["conf"], det["frame"]))
    return {k: v[o] for k, v in det.items()}, len(files), cover


def load_inverse_homographies(data_dir):
    cols = [f"h{i}{j}" for i in range(3) for j in range(3)]
    h = pd.read_csv(os.path.join(data_dir, "tactical", "BAR-ATH_tactical_homography.csv"),
                    usecols=["frame", "valid"] + cols)
    assert (h["frame"].values == np.arange(len(h))).all()
    return np.linalg.inv(h[cols].values.reshape(-1, 3, 3)), h["valid"].values.astype(bool)


def project(det, Hinv):
    q = np.stack([det["u"], det["v"], np.ones_like(det["u"])], 1).astype(float)
    r = np.einsum("nij,nj->ni", Hinv[det["frame"]], q)
    return r[:, 0] / r[:, 2] * L, r[:, 1] / r[:, 2] * W


def nearest_player(frames, x, y, PX, PY, chunk=200_000):
    out = np.empty(len(frames))
    for s in range(0, len(frames), chunk):
        e = s + chunk
        d = np.hypot(PX[frames[s:e]] - x[s:e, None], PY[frames[s:e]] - y[s:e, None])
        out[s:e] = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
    return out


def dedup(frame, u, v, src):
    """Greedy merge (input sorted by frame, conf desc). Returns representative mask, srcmask, nsrc."""
    rep = np.arange(len(frame))
    starts = np.flatnonzero(np.r_[True, frame[1:] != frame[:-1]])
    ends = np.r_[starts[1:], len(frame)]
    for a, b in zip(starts, ends):
        for i in range(a + 1, b):
            d = np.hypot(u[a:i] - u[i], v[a:i] - v[i])
            j = np.flatnonzero((d <= DEDUP_PX) & (rep[a:i] == np.arange(a, i)))
            if len(j):
                rep[i] = a + j[0]
    srcmask = np.zeros(len(frame), np.int8)
    np.bitwise_or.at(srcmask, rep, (1 << src.astype(np.int8)).astype(np.int8))
    nsrc = np.zeros(len(frame), np.int8)
    np.add.at(nsrc, rep, 1)
    return rep == np.arange(len(frame)), srcmask, nsrc


# --------------------------------------------------------------------------------- blacklist
def cell_key(x, y):
    return (np.floor(x / CELL).astype(np.int64) + 1000) * 10000 + np.floor(y / CELL).astype(np.int64) + 1000


def build_blacklist(frame, x, y, dpl, pct=99.9):
    """Static hotspots learnt from the data. Returns (blacklisted cell keys, hotspot table, stats)."""
    key = cell_key(x, y)
    g = pd.DataFrame({"k": key, "m": frame // (60 * FPS), "far": dpl > 3.0}).groupby("k").agg(
        n=("m", "size"), n_min=("m", "nunique"), far=("far", "mean"))
    t_min, far_ref = float(np.percentile(g["n_min"], pct)), float((dpl > 3.0).mean())
    hot = g[(g["n_min"] >= t_min) & (g["far"] > far_ref)].copy()
    seeds = set(hot.index) | set(cell_key(np.array([s[0] for s in SPOTS]), np.array([s[1] for s in SPOTS])))
    bl = {k + dx * 10000 + dy for k in seeds for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    hot["x"] = (hot.index // 10000 - 1000 + 0.5) * CELL
    hot["y"] = (hot.index % 10000 - 1000 + 0.5) * CELL
    return np.array(sorted(bl)), hot, dict(t_min=t_min, far_ref=far_ref, n_hot=len(hot), n_cells=len(bl))


# ----------------------------------------------------------------------------- emission stats
CONF_BINS = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01])


def continuation(frame, x, y, v_max):
    """True if another candidate lies within plausible reach in the next frame."""
    lo = np.searchsorted(frame, frame + 1, "left")
    hi = np.searchsorted(frame, frame + 1, "right")
    reach = v_max / FPS + 0.5
    out = np.zeros(len(frame), bool)
    for i in np.flatnonzero(hi > lo):
        a, b = lo[i], hi[i]
        out[i] = (np.hypot(x[a:b] - x[i], y[a:b] - y[i]) <= reach).any()
    return out


def speed_stats(frame, x, y, conf):
    """Frame-to-frame speeds of the single confident (>= 0.5) candidate of consecutive frames."""
    m = conf >= 0.5
    u, c = np.unique(frame[m], return_counts=True)
    sel = m & np.isin(frame, u[c == 1])
    fs, xs, ys = frame[sel], x[sel], y[sel]
    ok = np.diff(fs) == 1
    sp = np.hypot(np.diff(xs), np.diff(ys))[ok] * FPS
    return sp


def emission_reward(C, atd_x, atd_y, v_max, atd_weight=0.5):
    cont = continuation(C["frame"], C["x"], C["y"], v_max)
    b = np.clip(np.digitize(C["conf"], CONF_BINS) - 1, 0, len(CONF_BINS) - 2)
    multi = (C["nsrc"] > 1).astype(int)
    rate = np.full((len(CONF_BINS) - 1, 2), 0.5)
    for i in range(rate.shape[0]):
        for j in (0, 1):
            m = (b == i) & (multi == j)
            if m.sum() >= 50:
                rate[i, j] = cont[m].mean()
    rate = np.clip(rate, 0.02, 0.98)
    reward = np.log(rate / (1 - rate))[b, multi]
    # ATD proximity log-likelihood ratio: strong (conf >= .7, multi-source) vs weak (< .2) candidates
    da = np.hypot(C["x"] - atd_x[C["frame"]], C["y"] - atd_y[C["frame"]])
    has = np.isfinite(da)
    strong = has & (C["conf"] >= 0.7) & (C["nsrc"] > 1)
    weak = has & (C["conf"] < 0.2)
    p_s, p_w = (da[strong] <= 3).mean(), (da[weak] <= 3).mean()
    llr_in, llr_out = np.log(p_s / p_w), np.log((1 - p_s) / (1 - p_w))
    reward = reward + atd_weight * np.where(~has, 0.0, np.where(da <= 3, llr_in, llr_out))
    return reward, dict(rate=rate.round(3).tolist(), atd_llr_in=float(llr_in), atd_llr_out=float(llr_out),
                        p_strong=float(p_s), p_weak=float(p_w))


# ----------------------------------------------------------------------------------- tracker
def track_half(frame, x, y, reward, a, b, v_scale, v_max, c_start, c_end, max_gap=MAX_GAP):
    """Min-cost path over candidates of frames [a, b) + a 'no ball' state.

    Returns the chosen candidate index per frame (-1 = none) and a bridged flag per frame.
    """
    lo_f = np.searchsorted(frame, np.arange(a, b + 1), "left")      # first cand of frame t
    S = np.full(len(frame), np.inf)
    pred = np.full(len(frame), -2, np.int64)                         # -1 = from 'none'
    T = b - a
    N = np.empty(T + 1)
    Npred = np.full(T + 1, -1, np.int64)                             # -1 = from N(t-1)
    N[0] = 0.0                                                       # N[k] = none at frame a+k-1
    for k in range(T):
        t = a + k
        i0, i1 = lo_f[k], lo_f[k + 1]
        # 'none' at t: from none at t-1 or end of a track whose last frame is t-1
        best_n, arg_n = N[k], -1
        if k > 0:
            j0, j1 = lo_f[k - 1], lo_f[k]
            if j1 > j0:
                jj = j0 + np.argmin(S[j0:j1])
                if S[jj] + c_end < best_n:
                    best_n, arg_n = S[jj] + c_end, jj
        N[k + 1], Npred[k + 1] = best_n, arg_n
        if i1 == i0:
            continue
        w0 = lo_f[max(0, k - max_gap)]
        base = np.full(i1 - i0, N[k] + c_start)                    # start from none at t-1
        arg = np.full(i1 - i0, -1, np.int64)
        if i0 > w0:
            dt = (t - frame[w0:i0]) / FPS
            d = np.hypot(x[i0:i1, None] - x[None, w0:i0], y[i0:i1, None] - y[None, w0:i0])
            cost = S[None, w0:i0] + (d / dt) / v_scale
            cost[d > v_max * dt + 1.0] = np.inf
            j = np.argmin(cost, 1)
            cj = cost[np.arange(i1 - i0), j]
            better = cj < base
            base = np.where(better, cj, base)
            arg = np.where(better, w0 + j, -1)
        S[i0:i1] = base - reward[i0:i1]
        pred[i0:i1] = arg
    # backtrack: state is a candidate index (>= 0) or 'none at frame t' (cand = -1)
    choice = np.full(T, -1, np.int64)
    cand, t = -1, b - 1
    j0, j1 = lo_f[T - 1], lo_f[T]
    if j1 > j0:
        jj = j0 + np.argmin(S[j0:j1])
        if S[jj] < N[T]:
            cand = jj
    while True:
        if cand >= 0:
            choice[frame[cand] - a] = cand
            t = frame[cand] - 1
            cand = pred[cand]                      # -1 -> 'none at frame[cand]-1'
        else:
            if t < a:
                break
            p = Npred[t - a + 1]
            if p >= 0:
                cand = p                           # a track ended at frame t-1
            else:
                t -= 1
    # mark bridged frames between consecutive chosen candidates of one track
    bridged = np.zeros(T, bool)
    idx = np.flatnonzero(choice >= 0)
    for i, j in zip(idx[:-1], idx[1:]):
        if j - i > 1 and pred[choice[j]] == choice[i]:
            bridged[i + 1:j] = True
    return choice, bridged


# ------------------------------------------------------------------------------------ fusion
def runs(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def smooth_runs(x, y, group, win=7):
    """SG (2nd order) position/velocity inside runs of constant non-zero group."""
    xs, ys, vx, vy = (np.full(len(x), np.nan) for _ in range(4))
    edges = np.flatnonzero(np.r_[True, group[1:] != group[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if group[a] == 0:
            continue
        if b - a < win:
            xs[a:b], ys[a:b] = x[a:b], y[a:b]
            continue
        xs[a:b], ys[a:b] = savgol_filter(x[a:b], win, 2), savgol_filter(y[a:b], win, 2)
        vx[a:b] = savgol_filter(x[a:b], win, 2, deriv=1) * FPS
        vy[a:b] = savgol_filter(y[a:b], win, 2, deriv=1) * FPS
    return xs, ys, vx, vy


def run(parts_dir, data_dir, cache=None, log=print):
    t0 = time.time()
    det, n_files, cover = load_parts(parts_dir)
    log(f"{n_files} parts, {cover} frames covered, {len(det['frame'])} raw candidates")
    Hinv, valid = load_inverse_homographies(data_dir)
    x, y = project(det, Hinv)
    ok = valid[det["frame"]] & (x >= -OUT_MARGIN) & (x <= L + OUT_MARGIN) & (y >= -OUT_MARGIN) & (y <= W + OUT_MARGIN)
    n_out = int((~ok).sum())
    det = {k: v[ok] for k, v in det.items()}
    x, y = x[ok], y[ok]
    P, Bt, slot_team = v1.load_tactical(data_dir, cache)
    PX, PY = P[:, :, 0].astype(float) * L, P[:, :, 1].astype(float) * W
    dpl = nearest_player(det["frame"], x, y, PX, PY)
    bl_keys, hot, bl_stats = build_blacklist(det["frame"], x, y, dpl)
    inbl = np.isin(cell_key(x, y), bl_keys)
    log(f"dropped {n_out} outside/invalid; blacklist {bl_stats} removes {int(inbl.sum())} "
        f"({inbl.mean():.1%}) of in-pitch candidates  [{time.time() - t0:.0f}s]")
    keep = ~inbl
    det = {k: v[keep] for k, v in det.items()}
    x, y = x[keep], y[keep]
    isrep, srcmask, nsrc = dedup(det["frame"], det["u"], det["v"], det["src"])
    C = dict(frame=det["frame"][isrep], x=x[isrep], y=y[isrep], u=det["u"][isrep], v=det["v"][isrep],
             conf=det["conf"][isrep], srcmask=srcmask[isrep], nsrc=nsrc[isrep])
    sp = speed_stats(C["frame"], C["x"], C["y"], C["conf"])
    v_scale, v_max = float(sp.mean()), float(max(35.0, np.percentile(sp, 99)))
    atd_x, atd_y = Bt[:, 0].astype(float) * L, Bt[:, 1].astype(float) * W
    reward, em_stats = emission_reward(C, atd_x, atd_y, v_max)
    c_start = c_end = float(np.median(reward[reward > 0]))     # one typical good detection
    log(f"{len(C['frame'])} candidates after dedup; v_scale {v_scale:.2f} m/s, v_max {v_max:.1f} m/s, "
        f"c_start=c_end {c_start:.2f}  [{time.time() - t0:.0f}s]")
    n = v1.N_FRAMES
    vid_x, vid_y, vid_conf = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    vid_src = np.zeros(n, np.int8)
    for a, b in v1.HALVES:
        choice, bridged = track_half(C["frame"], C["x"], C["y"], reward, a, b, v_scale, v_max, c_start, c_end)
        sel = choice >= 0
        vid_x[a:b][sel], vid_y[a:b][sel] = C["x"][choice[sel]], C["y"][choice[sel]]
        vid_conf[a:b][sel] = C["conf"][choice[sel]]
        vid_src[a:b][sel] = SRC_VIDEO
        # linear bridge of the undetected frames inside one track
        fr = np.arange(a, b)
        for s, e in runs(bridged):
            i0, i1 = a + s - 1, a + e
            for arr in (vid_x, vid_y):
                arr[a + s:a + e] = np.interp(fr[s:e], [i0, i1], [arr[i0], arr[i1]])
        vid_src[a:b][bridged] = SRC_BRIDGE
        log(f"half {a}-{b}: video {sel.mean():.1%} detected + {bridged.mean():.1%} bridged  [{time.time() - t0:.0f}s]")
    # ATD smoothed
    ax, ay, _, _ = v1.smooth_ball(atd_x, atd_y)
    src = np.where(vid_src > 0, vid_src, np.where(np.isfinite(ax), SRC_ATD, SRC_NONE)).astype(np.int8)
    fx = np.where(vid_src > 0, vid_x, ax)
    fy = np.where(vid_src > 0, vid_y, ay)
    grp = np.where(src == SRC_ATD, 2, np.where(src > 0, 1, 0))
    graded = np.zeros(n, bool)
    for a, b in v1.HALVES:
        graded[a:b] = True
    grp[~graded] = 0
    sx, sy, svx, svy = smooth_runs(np.nan_to_num(fx), np.nan_to_num(fy), grp)
    params = dict(OUT_MARGIN=OUT_MARGIN, DEDUP_PX=DEDUP_PX, CELL=CELL, MAX_GAP=MAX_GAP, v_scale=v_scale,
                  v_max=v_max, c_start=c_start, c_end=c_end, n_raw_out=n_out, **bl_stats, **em_stats,
                  speed_pct=dict(zip(["p10", "p25", "p50", "p75", "p90", "p99"],
                                     np.percentile(sp, [10, 25, 50, 75, 90, 99]).round(2).tolist())))
    out = dict(video_x=vid_x.astype(np.float32), video_y=vid_y.astype(np.float32),
               video_conf=vid_conf.astype(np.float32), atd_x=ax.astype(np.float32), atd_y=ay.astype(np.float32),
               x=fx.astype(np.float32), y=fy.astype(np.float32), source=src,
               xs=sx.astype(np.float32), ys=sy.astype(np.float32), vx=svx.astype(np.float32),
               vy=svy.astype(np.float32), hotspots=hot[["x", "y", "n", "n_min", "far"]].values.astype(np.float32),
               blacklist_cells=bl_keys, params=np.array(json.dumps(params)))
    log(f"done in {time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="folder with the part_*.npz detection files")
    ap.add_argument("--data", required=True, help="challenge data dir (contains tactical/)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--cache", default=None, help="optional cache dir for the parsed ATD")
    a = ap.parse_args()
    out = run(a.parts, a.data, a.cache)
    np.savez_compressed(a.out, **out)
    src = out["source"]
    for a_, b_ in v1.HALVES:
        s = src[a_:b_]
        print(f"[{a_},{b_}) video {np.isin(s, (1, 2)).mean():.1%}  atd-only {(s == 3).mean():.1%}  none {(s == 0).mean():.1%}")


if __name__ == "__main__":
    main()

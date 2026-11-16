"""Dead-ball detector v3 ("bornes D"): sharper start / end of stoppages with the video ball.

Same frame-wise HistGradientBoosting idea as dead_ball.py (v2), trained on the 7 Sportec Open DFL
matches (BallStatus, leave-one-match-out), with two changes:

1. A second ball input, the VIDEO ball (balle_fusionnee.npz, sources 1/2 = detected/bridged),
   next to the ATD ball used by v2. On Sportec it is simulated:
   * Sportec freezes the ball at its exit point during every stoppage and teleports it to the
     restart spot (verified: speed 0 on 100 % of dead frames, median jump 6 m at the restart), so
     dead-ball trajectories are re-synthesised (synth_dead_ball): the ball keeps rolling after
     the whistle / line, waits, is carried or passed to the taker, is placed, then the real
     Sportec restart kick follows;
   * live aerial balls are projected onto the ground from the tactical camera (about 28 m high,
     ~40 m behind the far touchline, from the homography decomposition), so lofted balls near a
     line can look "out";
   * visibility as the tactical video tracker: dropped > 2 m outside the pitch, on the learnt
     static blacklist (centre / penalty spots, line marks), plus random gaps whose lengths and
     rates come from the real BAR-ATH video track inside / outside v2 stoppages (label-free);
   * 0.1 m AR(1) position noise (tau 4 s; a still video ball has ~0.05-0.1 m/s jitter).
   The ATD ball is degraded as in v2 (dead_ball.degrade_ball) but on the re-synthesised dead ball.
2. Video-ball features aimed at the boundaries: signed distance to the lines (out of pitch with
   0.5 / 1.5 m margins, past / future windows), aerial suspicion (projected speed spikes), ball
   still (time since / until the still run), placement spots (corner, 6 m box, penalty, centre,
   touchline), keeper proxy (deepest player of each team) holding the ball near his goal line,
   and "kick from a still ball" events (time since / until), which mark the restart.

Decoding as v2 (centred mean, hysteresis, merge, minimum duration) with the operating point and
an optional end-snap to the next "kick from still" chosen on Sportec LOMO with the spellF1 proxy,
weighting false D more than missed D. Smart Tagging SET PIECES can only EXTEND a D (v2 rule).

CLI (BAR-ATH, writes dead_v3.npz with p_dead float32 and dead bool for the 199,319 frames):
  python possession/dead_bounds.py --data data --sportec <dir with J03W*.npz> \
      --ball balle_fusionnee.npz --v2 v2_dead_ball.csv --out dead_v3.npz [--cache <dir>]
No parameter is fitted on the public ground truth.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1  # noqa: E402
import dead_ball as db  # noqa: E402

FPS = 25
SPORTEC_IDS = db.SPORTEC_IDS
OUT_MARGIN = 2.0                 # video tracker drops candidates beyond this (video_ball.py)
CELL = 0.5
CAM_BACK, CAM_H = 40.0, 28.0     # camera ~40 m behind the y=W touchline, ~28 m high (homography)
KICK_V = 5.0                     # m/s: speed reached by a kick from a still ball
STILL_V = 0.7                    # m/s: smoothed speed of a "still" ball


def runs_of(mask):
    return db.runs_of(mask)


# --------------------------------------------------------------------------------- Sportec
def load_sportec(path):
    """dead_ball.load_sportec plus ball height bz."""
    z = np.load(path)
    L, W = float(z["L"]), float(z["W"])
    out = []
    for h in ("h1", "h2"):
        bs = z[f"{h}_BS"]
        alive = np.flatnonzero(bs == 1)
        a, b = alive[0], alive[-1] + 1
        out.append(dict(px=(z[f"{h}_PX"][a:b] + L / 2).astype(np.float32),
                        py=(z[f"{h}_PY"][a:b] + W / 2).astype(np.float32), team=z[f"{h}_team"],
                        bx=(z[f"{h}_BX"][a:b] + L / 2).astype(np.float32),
                        by=(z[f"{h}_BY"][a:b] + W / 2).astype(np.float32),
                        bz=z[f"{h}_BZ"][a:b].astype(np.float32), dead=(bs[a:b] == 0),
                        poss=z[f"{h}_BP"][a:b], L=L, W=W))
    return out


def stoppage_types(hh):
    """Coarse type of every Sportec stoppage from the exit point and the restart spot."""
    L, W, bx, by = hh["L"], hh["W"], hh["bx"], hh["by"]
    T = len(bx)
    out = []
    for a, b in runs_of(hh["dead"]):
        xo, yo = bx[a], by[a]
        xr, yr = (bx[b], by[b]) if b < T else (np.nan, np.nan)
        out_x, out_y = (xo < 0) | (xo > L), (yo < 0) | (yo > W)
        gx = min(xr, L - xr)
        centre = abs(xr - L / 2) < 1.5 and abs(yr - W / 2) < 1.5
        if out_y and not out_x:
            t = "touche"
        elif out_x:
            if abs(yo - W / 2) < 3.66 and centre:
                t = "but"
            elif gx <= 6.0 and abs(yr - W / 2) <= 9.2:
                t = "6m"
            elif gx < 1.5 and min(yr, W - yr) < 1.5:
                t = "corner"
            else:
                t = "sortie_but_autre"
        elif centre:
            t = "coup_envoi_autre"
        elif abs(gx - 11) < 0.7 and abs(yr - W / 2) < 0.7:
            t = "penalty"
        else:
            t = "faute_autre"
        out.append(t)
    return np.array(out)


# ------------------------------------------------------------------------------ simulation
def _ar1(T, sd, tau_s, rng):
    a = np.exp(-1.0 / (tau_s * FPS))
    z = rng.normal(size=(T, 2)) * sd * np.sqrt(1 - a * a)
    e = np.empty((T, 2))
    e[0] = rng.normal(size=2) * sd
    for i in range(1, T):
        e[i] = a * e[i - 1] + z[i]
    return e


def synth_dead_ball(bx, by, bz, dead, px, py, L, W, rng, p_pass=0.5, p_dribble=0.5, place_s=(1.0, 8.0)):
    """Replace Sportec's frozen dead-ball position by a physical trajectory.

    Per stoppage [a, b): roll on from the last live velocity with deceleration U(2, 5) m/s^2 for
    at most U(0.4, 1.5) s; wait near the stop point, clipped to 0.2-1.0 m outside the lines
    (retrieved / replacement ball; 0.1 m slow jitter); move to the restart spot
    (Sportec ball at b) either carried (U(1.5, 4) m/s) or passed (U(8, 15) m/s, probability
    p_pass); stay placed U(1, 8) s; the real Sportec restart kick follows at b. Heights: a held
    ball (wait, probability 0.5; carry) is ~1 m up, a throw-in ball ~2 m (overhead), so the
    ground projection moves it like the real video ball.
    """
    bx, by, bz = bx.astype(float).copy(), by.astype(float).copy(), bz.astype(float).copy()
    T = len(bx)
    for a, b in runs_of(dead):
        n = b - a
        if a < 3:
            continue
        p0 = np.array([bx[a - 1], by[a - 1]])
        v0 = (p0 - np.array([bx[a - 3], by[a - 3]])) * FPS / 2
        s0 = float(np.hypot(*v0))
        dec = rng.uniform(2, 5)
        k_roll = int(min(s0 / dec, rng.uniform(0.4, 1.5), n / FPS) * FPS)
        t = np.arange(1, k_roll + 1) / FPS
        u = v0 / s0 if s0 > 0 else np.zeros(2)
        roll = p0 + np.outer(np.maximum(s0 * t - 0.5 * dec * t ** 2, 0), u)
        p1 = roll[-1] if k_roll else p0
        # a ball that ran far out is retrieved (or replaced) near the line: wait 0.5-1.5 m out
        lim = rng.uniform(0.2, 1.0)
        pw = np.array([np.clip(p1[0], -lim, L + lim), np.clip(p1[1], -lim, W + lim)])
        pr = np.array([bx[b], by[b]]) if b < T else p1
        rem = n - k_roll
        k_place = min(int(rng.uniform(*place_s) * FPS), rem)
        speed = rng.uniform(8, 15) if rng.random() < p_pass else rng.uniform(1.5, 4)
        k_move = min(int(np.hypot(*(pr - pw)) / speed * FPS) + 1, rem - k_place)
        k_wait = rem - k_place - k_move
        held = rng.random() < 0.5                       # ball picked up by hand while waiting
        passed = speed > 5
        throw = b < T and min(pr[1], W - pr[1]) < 1.5   # restart on the touchline: throw-in
        seg, hz = [roll], [np.zeros(len(roll))]
        if k_wait > 0:
            w0 = pw + _ar1(k_wait, 0.1, 4.0, rng)
            z0 = np.full(k_wait, rng.uniform(0.8, 1.2) if held else 0.0)
            if k_wait > 2 * FPS and rng.random() < p_dribble:
                # part of the wait the ball is at a player's feet (dribbled / juggled / brought)
                s1 = int(rng.uniform(0, 0.5) * k_wait)
                e1 = s1 + int(rng.uniform(0.2, 0.5) * k_wait)
                i0 = a + k_roll + s1
                j = int(np.nanargmin(np.hypot(px[i0] - w0[s1, 0], py[i0] - w0[s1, 1])))
                off = rng.normal(size=2) * 0.4
                fol = np.stack([px[i0:a + k_roll + e1, j], py[i0:a + k_roll + e1, j]], 1) + off
                w0[s1:e1] = fol
                w0[e1:] = fol[-1] + (w0[e1:] - w0[e1])     # left where the player dropped it
                z0[s1:e1] = 0.0
                pw = w0[-1]
            seg.append(w0)
            hz.append(z0)
        if k_move > 0:
            w = (np.arange(1, k_move + 1) / k_move)[:, None]
            seg.append(pw + w * (pr - pw))
            hz.append(np.full(k_move, 0.0 if passed else rng.uniform(0.8, 1.2)))
        if k_place > 0:
            seg.append(np.repeat(pr[None], k_place, 0) + rng.normal(size=(k_place, 2)) * 0.01)
            hz.append(np.full(k_place, rng.uniform(1.8, 2.3) if throw else 0.0))
        traj = np.concatenate([s for s in seg if len(s)])[:n]
        bx[a:b], by[a:b] = traj[:, 0], traj[:, 1]
        bz[a:b] = np.concatenate(hz)[:n]
    return bx, by, bz


def project_ground(bx, by, bz, L, W):
    """Ground point of the camera ray through a ball at height bz (camera at (L/2, W+40, 28))."""
    cx, cy = L / 2, W + CAM_BACK
    z = np.clip(np.nan_to_num(bz), 0, CAM_H - 2)
    k = CAM_H / (CAM_H - z)
    return cx + (bx - cx) * k, cy + (by - cy) * k


def blacklist_mask(x, y, L, W, cells):
    """True where (x, y) falls on a BAR-ATH blacklisted 0.5 m cell (coordinates rescaled)."""
    if cells is None or len(cells) == 0:
        return np.zeros(len(x), bool)
    Lb, Wb = v1.PITCH
    xs, ys = np.nan_to_num(x * Lb / L, nan=-999), np.nan_to_num(y * Wb / W, nan=-999)
    key = (np.floor(xs / CELL).astype(np.int64) + 1000) * 10000 + np.floor(ys / CELL).astype(np.int64) + 1000
    return np.isin(key, cells)


def _random_gaps(cand, target, gap_frames, rng, existing):
    """Add gaps (lengths drawn from gap_frames, clipped to cand) at random cand frames until the
    missed fraction of cand frames reaches target (existing misses count)."""
    miss = existing.copy()
    idx = np.flatnonzero(cand)
    if len(idx) == 0 or len(gap_frames) == 0:
        return miss
    need = int(target * len(idx)) - int((miss & cand).sum())
    while need > 0:
        s = int(rng.choice(idx))
        n = int(rng.choice(gap_frames))
        seg = cand[s:s + n] & ~miss[s:s + n]
        miss[s:s + n] |= cand[s:s + n]
        need -= int(seg.sum())
    return miss


def simulate_video_ball(bx, by, bz, dead, L, W, rng, gaps_live, gaps_dead, miss_live, miss_dead,
                        cells=None, noise_sd=0.1, aerial=True, noise_tau=4.0, ghost=None,
                        hide_start=0.0, hide_end=0.0):
    """Video-like ball from a (re-synthesised) Sportec ball. Returns x, y with NaN when unseen.

    ghost: (px, py, fraction) -- that fraction of the dead-ball gap runs is filled by a false
    track sitting on the nearest player (a video false positive).
    hide_start / hide_end: probability that a stoppage's exit (from -1..0 s to +1..3 s) or its
    restart (from -4..-1 s to +0.5 s) is not seen by the video (domain randomisation: on BAR-ATH
    these cues are seen less often than in the nominal simulation).
    """
    if aerial:
        x, y = project_ground(bx, by, bz, L, W)
    else:
        x, y = bx.astype(float), by.astype(float)
    e = _ar1(len(x), noise_sd, noise_tau, rng)
    x, y = x + e[:, 0], y + e[:, 1]
    edge = np.fmin(np.fmin(x, L - x), np.fmin(y, W - y))
    miss = (edge < -OUT_MARGIN) | blacklist_mask(x, y, L, W, cells)
    miss = _random_gaps(~dead, miss_live, gaps_live, rng, miss)
    miss = _random_gaps(dead, miss_dead, gaps_dead, rng, miss)
    T = len(x)
    for a, b in runs_of(dead):
        if rng.random() < hide_start:
            miss[max(a - int(rng.uniform(0, 1) * FPS), 0):a + int(rng.uniform(1, 3) * FPS)] = True
        if rng.random() < hide_end:
            miss[max(b - int(rng.uniform(1, 4) * FPS), a):min(b + FPS // 2, T)] = True
    x, y = np.where(miss, np.nan, x), np.where(miss, np.nan, y)
    if ghost is not None:
        gpx, gpy, frac = ghost
        for a, b in runs_of(miss & dead):
            if b - a < 5 or rng.random() >= frac:
                continue
            ref = (bx[a], by[a])
            j = int(np.nanargmin(np.hypot(gpx[a] - ref[0], gpy[a] - ref[1])))
            x[a:b], y[a:b] = gpx[a:b, j] + 0.3, gpy[a:b, j] + 0.3
    return x.astype(np.float32), y.astype(np.float32)


# -------------------------------------------------------------------------------- features
def _since_until(mask):
    """Seconds since the current True run started / until it ends (0 outside runs)."""
    _, since, until = db._run_lengths(mask)
    return since, until


def _time_to_events(ev, T, cap_s=30.0):
    """Seconds since the last event (<= t) and until the next one (>= t), capped."""
    cap = cap_s * FPS
    idx = np.flatnonzero(ev)
    t = np.arange(T)
    if len(idx) == 0:
        return np.full(T, cap_s, np.float32), np.full(T, cap_s, np.float32)
    j = np.searchsorted(idx, t, side="right") - 1
    since = np.where(j >= 0, t - idx[np.clip(j, 0, None)], cap)
    k = np.searchsorted(idx, t, side="left")
    until = np.where(k < len(idx), idx[np.clip(k, None, len(idx) - 1)] - t, cap)
    return (np.minimum(since, cap) / FPS).astype(np.float32), (np.minimum(until, cap) / FPS).astype(np.float32)


def keeper_proxy(px, py, team):
    """Deepest player of each team, own side from the two team centroids. Returns (T,2) x, y."""
    T = px.shape[0]
    kx, ky = np.full((T, 2), np.nan), np.full((T, 2), np.nan)
    with np.errstate(all="ignore"):
        cx = [np.nanmedian(px[:, team == k], axis=1) for k in (0, 1)]
        for k in (0, 1):
            m = np.flatnonzero(team == k)
            X = px[:, m].astype(float)
            left = cx[k] < cx[1 - k]                       # defends x = 0
            Xs = np.where(left[:, None], X, -X)
            Xs = np.where(np.isfinite(Xs), Xs, np.inf)
            j = np.argmin(Xs, axis=1)
            ok = np.isfinite(Xs[np.arange(T), j])
            kx[ok, k] = px[np.arange(T), m[j]][ok]
            ky[ok, k] = py[np.arange(T), m[j]][ok]
    return kx, ky


def video_features(px, py, team, vx_, vy_, L, W):
    """Boundary-oriented features of the video ball for ONE half (names prefixed v_)."""
    T = len(vx_)
    f = {}
    with np.errstate(all="ignore"):
        xs, ys, vx, vy = v1.smooth_ball(vx_.astype(float), vy_.astype(float))
        vis = np.isfinite(xs)
        spd = np.hypot(vx, vy)
        f["v_vis"] = vis.astype(np.float32)
        visS = pd.Series(vis.astype(np.float32))
        for w in (2, 6):
            f[f"v_vis_m{w}"] = visS.rolling(w * FPS, center=True, min_periods=1).mean()
        _, gs, gu = db._run_lengths(~vis)
        f["v_gap_since"], f["v_gap_until"] = gs, gu
        f["v_spd"] = spd
        spdS = pd.Series(spd)
        f["v_spd_m1"] = spdS.rolling(FPS, center=True, min_periods=3).median()
        f["v_spd_p1"] = spdS.rolling(FPS, min_periods=3).median()                      # past 1 s
        f["v_spd_f1"] = spdS[::-1].rolling(FPS, min_periods=3).median()[::-1]          # next 1 s
        f["v_spd_max_p1"] = spdS.rolling(FPS, min_periods=1).max()
        # signed distance to the lines (negative outside), margins, aerial suspicion
        ex, ey = np.fmin(xs, L - xs), np.fmin(ys, W - ys)
        edge = np.fmin(ex, ey)
        f["v_edge"], f["v_edge_x"], f["v_edge_y"] = edge, ex, ey
        jump = (spdS.rolling(FPS, center=True, min_periods=1).max() > 25).astype(np.float32)
        f["v_aerial_m1"] = jump
        for mg, nm in ((0.5, "o05"), (1.5, "o15")):
            o = pd.Series(np.where(vis, (edge < -mg).astype(np.float32), np.nan))
            o0 = o.fillna(0)
            f[f"v_{nm}"] = o
            f[f"v_{nm}_p2"] = o0.rolling(2 * FPS, min_periods=1).max()
            f[f"v_{nm}_p6"] = o0.rolling(6 * FPS, min_periods=1).max()
            f[f"v_{nm}_f2"] = o0[::-1].rolling(2 * FPS, min_periods=1).max()[::-1]
            f[f"v_{nm}_m4"] = o.rolling(4 * FPS, center=True, min_periods=1).mean()
        out_ev = np.r_[False, (np.nan_to_num(edge, nan=1) < -0.5)[1:] & (np.nan_to_num(edge, nan=1) >= -0.5)[:-1]]
        f["v_out_since"], f["v_out_until"] = _time_to_events(out_ev, T)
        # still ball
        sm = np.nan_to_num(np.asarray(f["v_spd_m1"]), nan=99)
        still = vis & (sm < STILL_V)
        s_since, s_until = _since_until(still)
        f["v_still_since"], f["v_still_until"] = s_since, s_until
        f["v_still_m4"] = pd.Series(still.astype(np.float32)).rolling(4 * FPS, center=True, min_periods=1).mean()
        f["v_still_past_max10"] = pd.Series(s_since).rolling(10 * FPS, min_periods=1).max()
        # kick from a still ball: >= 1 s still, then > KICK_V within 0.5 s
        sp0 = np.nan_to_num(spd)
        fut = pd.Series(sp0)[::-1].rolling(FPS // 2, min_periods=1).max()[::-1].values
        kick = (s_since >= 1.0) & (~np.r_[still[1:], False]) & (fut > KICK_V)
        f["v_kick_since"], f["v_kick_until"] = _time_to_events(kick, T)
        kick_long = (s_since >= 3.0) & (~np.r_[still[1:], False]) & (fut > KICK_V)
        f["v_kick3_since"], f["v_kick3_until"] = _time_to_events(kick_long, T)
        # placement spots
        gx = np.fmin(xs, L - xs)
        dy = np.abs(ys - W / 2)
        f["v_d_corner"] = np.hypot(gx, np.fmin(ys, W - ys))
        f["v_d_centre"] = np.hypot(xs - L / 2, dy)
        f["v_d_pen"] = np.hypot(gx - 11.0, dy)
        f["v_in_6m"] = ((gx > -0.5) & (gx < 6.0) & (dy < 9.5)).astype(np.float32)
        f["v_goal_dy"] = dy
        # players around the ball
        d = np.hypot(px - xs[:, None], py - ys[:, None])
        f["v_near1"] = np.nanmin(d, axis=1)
        dsrt = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        f["v_near2"] = np.where(np.isfinite(dsrt[:, 1]), dsrt[:, 1], np.nan)
        f["v_n3"] = np.where(vis, (d < 3).sum(1), np.nan)
        f["v_n10"] = np.where(vis, (d < 10).sum(1), np.nan)
        # keeper proxy
        kx, ky = keeper_proxy(px, py, team)
        dk = np.fmin(np.hypot(kx[:, 0] - xs, ky[:, 0] - ys), np.hypot(kx[:, 1] - xs, ky[:, 1] - ys))
        f["v_d_keeper"] = dk
        kg = np.fmin(np.fmin(kx[:, 0], L - kx[:, 0]), np.fmin(kx[:, 1], L - kx[:, 1]))
        f["v_keeper_goal"] = kg
        hold = (dk < 1.5) & (gx < 16.5) & (dy < 20.2)
        hs, hu = _since_until(hold)
        f["v_keep_since"], f["v_keep_until"] = hs, hu
    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    return out


def build_features(px, py, team, atd_x, atd_y, vid_x, vid_y, L, W):
    F2 = db.context_features(db.base_features(px, py, team, atd_x, atd_y, L, W))
    FV = video_features(px, py, team, vid_x, vid_y, L, W)
    return pd.concat([F2, FV], axis=1)


# ---------------------------------------------------------------------- calibration (label-free)
def video_gap_stats(source, v2_dead):
    """Gap lengths and miss rates of the BAR-ATH video track (sources 1/2) inside / outside v2 D.

    Uses the v2 dead-ball decoding as the state proxy (no ground truth). Returns
    (gaps_live, gaps_dead, miss_live, miss_dead).
    """
    vid = (source == 1) | (source == 2)
    graded = np.zeros(len(source), bool)
    for a, b in v1.HALVES:
        graded[a:b] = True
    D, Lv = v2_dead & graded, graded & ~v2_dead
    out = {}
    for name, ctx in (("live", Lv), ("dead", D)):
        g = []
        for a, b in v1.HALVES:
            for s, e in runs_of(~vid[a:b]):
                if ctx[a + s:a + e].mean() > 0.9:
                    g.append(e - s)
        out[name] = np.array(g, int)
    return out["live"], out["dead"], float(1 - vid[Lv].mean()), float(1 - vid[D].mean())


# ------------------------------------------------------------------------ training data
RAND = dict(place_s=(0.0, 8.0), hide_start=0.3, hide_end=0.3, ghost_frac=0.25, noise=(0.07, 0.2))


def sportec_half_features(hh, seed, atd_gaps, gl, gd, ml, md, cells, rand=RAND,
                          q_dead=0.965, miss_live_atd=0.165):
    """Simulated ATD + video balls for one Sportec half (domain-randomised) -> feature frame."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = synth_dead_ball(hh["bx"], hh["by"], hh["bz"], hh["dead"], hh["px"], hh["py"],
                                 hh["L"], hh["W"], rng, 0.5, 0.5, rand["place_s"])
    nsd = rng.uniform(*rand["noise"])
    vx_, vy_ = simulate_video_ball(sx, sy, sz, hh["dead"], hh["L"], hh["W"], rng, gl, gd, ml, md, cells,
                                   noise_sd=nsd, ghost=(hh["px"], hh["py"], rand["ghost_frac"]),
                                   hide_start=rand["hide_start"], hide_end=rand["hide_end"])
    ax_, ay_ = db.degrade_ball(sx, sy, hh["dead"], np.random.default_rng(seed + 7), atd_gaps, q_dead,
                               miss_live_atd)
    return build_features(hh["px"], hh["py"], hh["team"], ax_, ay_, vx_, vy_, hh["L"], hh["W"])


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)


# ------------------------------------------------------------------------------ decoding
# Operating point from Sportec LOMO (out-of-fold probabilities, randomised test simulation):
# p_hi 0.98 -> fewest false D (a false D splits a possession: costlier than a missed D); p_lo 0.6
# -> median start / end bias +-0.12 s (0.7-0.8 score higher on the spellF1 proxy but shrink every
# stoppage by 0.2-0.5 s at both ends); smoothing 1 s, merge 1 s, minimum 3 s as v2.
DECODE = dict(p_hi=0.98, p_lo=0.6, min_dur_s=3.0, merge_s=1.0, smooth_s=1.0)


def decode(p, **kw):
    prm = dict(DECODE)
    prm.update(kw)
    return db.decode(p, prm["p_hi"], prm["p_lo"], prm["min_dur_s"], prm["merge_s"], prm["smooth_s"])


# --------------------------------------------------------------------------- application
def bar_features(P, B, slot_team, fused):
    """Feature frames of the two graded BAR-ATH halves (ATD ball + video ball sources 1/2)."""
    L, W = v1.PITCH
    vid = (fused["source"] == 1) | (fused["source"] == 2)
    vx = np.where(vid, fused["video_x"], np.nan)
    vy = np.where(vid, fused["video_y"], np.nan)
    for a, b in v1.HALVES:
        yield a, b, build_features(P[a:b, :, 0] * L, P[a:b, :, 1] * W, slot_team, B[a:b, 0] * L,
                                   B[a:b, 1] * W, vx[a:b], vy[a:b], L, W)


def train_sportec(sportec_dir, atd_gaps, gl, gd, ml, md, cells, step=5, log=print):
    X, y = [], []
    for mi, mid in enumerate(SPORTEC_IDS):
        path = os.path.join(sportec_dir, f"{mid}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        for hi, hh in enumerate(load_sportec(path)):
            F = sportec_half_features(hh, 3000 + 10 * mi + hi, atd_gaps, gl, gd, ml, md, cells)
            X.append(F.values[::step].astype(np.float32))
            y.append(hh["dead"][::step])
        log(f"  features {mid}")
    model = make_model().fit(np.concatenate(X), np.concatenate(y))
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="challenge data dir (contains tactical/)")
    ap.add_argument("--sportec", required=True, help="dir with the 7 Sportec J03W*.npz (fetch_sportec.py)")
    ap.add_argument("--ball", required=True, help="balle_fusionnee.npz (video_ball.py)")
    ap.add_argument("--v2", required=True, help="v2_dead_ball.csv (state proxy for video gap stats)")
    ap.add_argument("--out", required=True, help="output npz (p_dead float32, dead bool, ...)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-set-pieces", action="store_true")
    a = ap.parse_args()
    P, B, slot_team = v1.load_tactical(a.data, a.cache)
    fused = np.load(a.ball, allow_pickle=True)
    v2_dead = pd.read_csv(a.v2)["label"].values == "D"
    gl, gd, ml, md = video_gap_stats(fused["source"], v2_dead)
    print(f"video track: miss {ml:.3f} live / {md:.3f} dead (v2 proxy); {len(gl)} / {len(gd)} gaps")
    model = train_sportec(a.sportec, db.ball_gap_lengths(B), gl, gd, ml, md, fused["blacklist_cells"])
    tags = db.set_piece_starts(a.data)
    n = v1.N_FRAMES
    p_dead = np.full(n, np.nan, np.float32)
    dead = np.zeros(n, bool)
    dead_raw = np.zeros(n, bool)
    graded = np.zeros(n, bool)
    for s, e, F in bar_features(P, B, slot_team, fused):
        p = model.predict_proba(F.values.astype(np.float32))[:, 1].astype(np.float32)
        d = decode(p)
        dead_raw[s:e] = d
        if not a.no_set_pieces:
            d = db.bridge_set_pieces(d, tags[(tags >= s) & (tags < e)] - s)
        p_dead[s:e], dead[s:e], graded[s:e] = p, d, True
    np.savez_compressed(a.out, p_dead=p_dead, dead=dead, dead_no_tags=dead_raw, graded=graded,
                        params=np.array(str(dict(DECODE=DECODE, RAND=RAND, miss_live=ml, miss_dead=md))))
    n_runs = sum(len(runs_of(dead[s:e])) for s, e in v1.HALVES)
    print(f"D {dead[graded].mean():.1%} of graded frames, {n_runs} stoppages -> {a.out}")


if __name__ == "__main__":
    main()

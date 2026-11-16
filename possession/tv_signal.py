"""Signal TV : synchronise the broadcast feed on the tactical timeline and flag, for every
tactical frame, what the TV director was showing (live play, off-pitch shot, replay).

No learning and no tuning on the match: every threshold below is fixed a priori from physical
reasoning about optical tracking (documented next to each constant). The broadcast video itself is
NOT read; only the two Metrica ATD files and the broadcast homography `valid` column are inputs.

Method
1. Both ATD files are loaded in pitch metres (same normalised frame, same team ids).
   The distance between a broadcast frame b and a tactical frame t is a truncated, directional
   Chamfer distance: for every player visible on the broadcast frame, the distance to the nearest
   tactical player of the SAME team, truncated at CAP metres, averaged. Directional because the
   TV sees only part of the pitch whereas the tactical camera sees everyone.
2. Anchors: every ANCHOR_STEP broadcast frames, a 3-frame clip (1 s apart) of a valid,
   well-populated view is matched against the WHOLE tactical match (same spacing). Anchors whose
   best mean distance is < MATCH_TOL give an offset t - b. Offsets are clustered (tolerance
   CLUSTER_TOL); each cluster is a stretch of constant offset (live TV keeps real time; replays
   replace live pictures instead of delaying them). Cluster borders are cuts in the recording.
3. Each cluster is refined on sampled valid frames (local search +-LOCAL_R frames) and a robust
   linear model offset(b) absorbs any clock drift. The border between two consecutive clusters is
   placed between the last frame confirmed by cluster k and the first confirmed by cluster k+1;
   what lies between is an ambiguous cut zone.
4. Per broadcast frame with valid homography and >= MIN_PLAYERS players: residual at the nominal
   offset (min over +-3 frames). Live if the 1-s running median (inside the valid run) is
   < LIVE_TOL. Valid-but-not-live stretches are matched BACKWARDS in time (replays): first in the
   previous REPLAY_WIN_S seconds, then over the whole past if needed; a RANSAC line t = r*b + c with
   playback rate r in [RATE_MIN, RATE_MAX] identifies a replay and the replayed tactical frames.
5. Projection on the 199,319 tactical frames (see TV_STATE codes).

Usage
    python possession/tv_signal.py --data <challenge-barcelona/data> --out tv_signal.npz \
        [--qc tv_signal_qc.json]

Output npz (one value per TACTICAL video frame 0..199318, except the `bc_*` keys which are per
BROADCAST frame 0..164487: bc_state (same codes as tv_state, 0/6 unused), bc_res (residual, m),
bc_off (nominal offset t-b, NaN in cut zones)):
    tv_state   uint8  0 unknown (frame not broadcast)       1 live pitch view, verified by ATD match
                      2 off-pitch shot (homography invalid: close-up, crowd, graphics, slow-mo c.-up)
                      3 replay identified (valid view matched to an EARLIER tactical moment)
                      4 pitch view, not live, unmatched (other-angle / old replay, bad calibration)
                      5 pitch view, too few tracked players to verify (< MIN_PLAYERS)
                      6 cut zone: TV showed something non-live OR did not cover this frame
    tv_shown   bool   tv_state == 1
    tv_offpitch bool  tv_state in (2, 4)       (the TV is not showing live play)
    tv_replay  bool   tv_state == 3
    tv_unknown bool   tv_state in (0, 6)
    tv_block_s float32  duration (s) of the maximal non-live TV block containing the frame
                        (states 2,3,4 contiguous on the broadcast timeline), 0 elsewhere
    tv_block_has_replay bool  that block contains an identified replay (state 3)
    b_frame    int32  broadcast frame shown at this tactical instant (-1 if none / cut zone)
    match_res  float32 Chamfer residual (m) of that broadcast frame at nominal offset (NaN if n/a)
    replayed   uint8  how many identified replays later showed this tactical frame
    replay_lag_s float32 delay (s) between this frame and its first replay on TV (NaN if never)
"""
import argparse
import json
import os
import resource
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

PITCH = (105.3, 68.0)
N_TACT = 199319
FPS = 25.0

# ---------------------------------------------------------------- a-priori constants
CAP = 5.0          # m. Truncation of each player distance: a spurious/duplicate detection costs at
                   # most CAP; 5 m >> optical tracking error (~0.5-1 m) of both ATD.
MATCH_TOL = 1.5    # m. Mean distance for an anchor/sample to count as a match: ~2x the combined
                   # error of two independent optical trackers.
LIVE_TOL = 2.0     # m. Per-frame live decision (1-s median): a bit above MATCH_TOL to tolerate
                   # crowded views; random configurations of a match sit at ~CAP.
MIN_PLAYERS = 4    # broadcast players needed to evaluate a frame (fewer -> configuration not unique)
ANCHOR_STEP = 5000 # broadcast frames between anchors (200 s): finer than any plausible edit
CLUSTER_TOL = 25   # frames (1 s): two anchors with offsets closer than this share a cluster
LOCAL_R = 50       # frames (2 s): local refinement window around a cluster offset
REPLAY_WIN_S = 300 # s. A replay usually shows the last few minutes; search there first
RATE_MIN, RATE_MAX = 0.15, 1.2   # replay playback rate (super slow-mo .. slightly fast)
RANSAC_TOL = 15    # frames (0.6 s) inlier tolerance for the replay line
MIN_RUN = 25       # frames (1 s): shortest not-live stretch searched as a replay
MAX_SAMPLES = 30   # samples per stretch for the backward search (stride >= 10 frames)


# ---------------------------------------------------------------- loading
def _teams(xml_path):
    root = ET.parse(xml_path).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def load_atd_xy(txt_path, xml_path):
    """Players in metres (N, S, 2) float32 with NaN, plus slot team. Ball ignored."""
    team = _teams(xml_path)
    rows = []
    with open(txt_path) as fh:
        for line in fh:
            pl = line.split(":")[1]
            rows.append(np.array(pl.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 3)[:, :2])
    P = np.stack(rows) * np.array(PITCH, np.float32)
    return P, team


def compact(P, team, k):
    """Per team: finite slots first, truncated to the team's max count; NaN -> far sentinel."""
    out = []
    for tm in (0, 1):
        X = P[:, team == tm]
        fin = np.isfinite(X[:, :, 0])
        order = np.argsort(~fin, axis=1, kind="stable")
        X = np.take_along_axis(X, order[:, :, None], axis=1)
        kk = int(fin.sum(1).max()) if k is None else k
        X = X[:, :max(kk, 1)].copy()
        m = np.isfinite(X[:, :, 0])
        X[~m] = 1e4
        out.append((X, m))
    return out


# ---------------------------------------------------------------- distances
class Matcher:
    def __init__(self, Pb, team_b, Pt, team_t):
        self.B = compact(Pb, team_b, None)
        self.T = compact(Pt, team_t, None)
        self.nb = sum(m.sum(1) for _, m in self.B)

    def one(self, b, ts, chunk=40000):
        """Distance of broadcast frame b to tactical frames ts (vector)."""
        ts = np.asarray(ts)
        out = np.empty(len(ts), np.float32)
        qs = [X[b][m[b]] for X, m in self.B]
        n = sum(len(q) for q in qs)
        if n == 0:
            out[:] = np.nan
            return out
        for i in range(0, len(ts), chunk):
            t = ts[i:i + chunk]
            tot = np.zeros(len(t), np.float32)
            for q, (T, _) in zip(qs, self.T):
                if len(q) == 0:
                    continue
                Tt = T[t]
                d2 = ((Tt[:, None, :, :] - q[None, :, None, :]) ** 2).sum(-1).min(2)
                tot += np.minimum(np.sqrt(d2), CAP).sum(1)
            out[i:i + chunk] = tot / n
        return out

    def pairs(self, bs, ts, chunk=20000):
        bs, ts = np.asarray(bs), np.asarray(ts)
        out = np.empty(len(bs), np.float32)
        for i in range(0, len(bs), chunk):
            b, t = bs[i:i + chunk], ts[i:i + chunk]
            tot = np.zeros(len(b), np.float32)
            n = np.zeros(len(b), np.float32)
            for (Xb, mb), (T, _) in zip(self.B, self.T):
                q, mq = Xb[b], mb[b]
                d2 = ((T[t][:, None, :, :] - q[:, :, None, :]) ** 2).sum(-1).min(2)
                tot += np.where(mq, np.minimum(np.sqrt(d2), CAP), 0).sum(1)
                n += mq.sum(1)
            out[i:i + chunk] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
        return out


def runs_of(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def ransac_line(x, y, rng):
    if len(x) < 3:
        return None
    best = (0, None)
    for _ in range(400):
        i, j = rng.choice(len(x), 2, replace=False)
        if x[i] == x[j]:
            continue
        sl = (y[j] - y[i]) / (x[j] - x[i])
        if not RATE_MIN <= sl <= RATE_MAX:
            continue
        ic = y[i] - sl * x[i]
        inl = np.abs(y - (sl * x + ic)) < RANSAC_TOL
        if inl.sum() > best[0]:
            best = (inl.sum(), (sl, ic))
    if best[1] is None or best[0] < 3:
        return None
    sl, ic = best[1]
    inl = np.abs(y - (sl * x + ic)) < RANSAC_TOL
    sl2, ic2 = np.polyfit(x[inl], y[inl], 1)
    if RATE_MIN <= sl2 <= RATE_MAX:
        sl, ic = sl2, ic2
    inl = np.abs(y - (sl * x + ic)) < RANSAC_TOL
    return float(sl), float(ic), inl


# ---------------------------------------------------------------- pipeline
def synchronise(M, valid, log=print):
    NB = len(valid)
    nb = M.nb
    ok = valid & (nb >= 8)
    n_t = M.T[0][0].shape[0]
    ts_all = np.arange(n_t)
    # --- 2. anchors
    anchors = []
    for a in range(0, NB, ANCHOR_STEP):
        c = [b for b in range(a, min(a + ANCHOR_STEP, NB)) if ok[b]]
        clip = []
        for b in c:
            if not clip or b - clip[-1] >= 25:
                clip.append(b)
            if len(clip) == 3:
                break
        if len(clip) < 3:
            continue
        S = np.zeros(n_t, np.float32)
        for b in clip:
            S += M.one(b, np.clip(ts_all + (b - clip[0]), 0, n_t - 1))
        S /= len(clip)
        k = int(np.argmin(S))
        anchors.append(dict(b=int(clip[0]), t=k, off=k - int(clip[0]), score=float(S[k]),
                            median=float(np.median(S))))
    good = [x for x in anchors if x["score"] < MATCH_TOL]
    log(f"anchors {len(anchors)}, matched {len(good)}")
    # --- cluster offsets in broadcast order (consecutive anchors with close offsets)
    clusters = []
    for x in sorted(good, key=lambda z: z["b"]):
        for c in clusters:
            if abs(np.median(c["offs"]) - x["off"]) <= CLUSTER_TOL:
                c["offs"].append(x["off"]); c["bs"].append(x["b"]); break
        else:
            clusters.append(dict(offs=[x["off"]], bs=[x["b"]]))
    clusters = [c for c in clusters if len(c["offs"]) >= 2]   # single anchor = likely a replay
    clusters.sort(key=lambda c: min(c["bs"]))
    # --- 3. local refinement: assign sampled valid frames to the best cluster
    samp = np.flatnonzero(valid & (nb >= 6))[::10]
    best_r = np.full(len(samp), np.inf, np.float32)
    best_k = np.full(len(samp), -1)
    best_o = np.zeros(len(samp))
    for k, c in enumerate(clusters):
        o0 = int(np.median(c["offs"]))
        for d in range(-LOCAL_R, LOCAL_R + 1):
            r = M.pairs(samp, np.clip(samp + o0 + d, 0, n_t - 1))
            upd = r < best_r
            best_r[upd], best_k[upd], best_o[upd] = r[upd], k, o0 + d
    for k, c in enumerate(clusters):
        m = (best_k == k) & (best_r < MATCH_TOL)
        x, y = samp[m], best_o[m]
        med = np.median(y)
        inl = np.abs(y - med) <= 5
        p = np.polyfit(x[inl], y[inl], 1) if inl.sum() > 10 else np.array([0.0, med])
        res = y[inl] - np.polyval(p, x[inl])
        c.update(poly=p.tolist(), b_first=int(x.min()), b_last=int(x.max()), n_samples=int(m.sum()),
                 off_mad=float(np.median(np.abs(res))), off_med=float(med),
                 frac_outside_5=float(1 - inl.mean()))
    # monotonic sanity: clusters must not interleave
    for c1, c2 in zip(clusters, clusters[1:]):
        if c1["b_last"] >= c2["b_first"]:
            raise RuntimeError("interleaved clusters: edit model violated")
    # refine each cut: every valid frame of the gap is tested under both neighbouring offsets;
    # the 1-s running median of the residual decides (same rule as the per-frame live test).
    from scipy.ndimage import median_filter
    cut_qc = []
    for c1, c2 in zip(clusters, clusters[1:]):
        gap = np.arange(c1["b_last"] + 1, c2["b_first"])
        gap = gap[valid[gap] & (nb[gap] >= MIN_PLAYERS)]
        d = dict(b_last_sampled=c1["b_last"], b_first_sampled=c2["b_first"], n_gap_eval=int(len(gap)))
        if len(gap):
            rs = []
            for c in (c1, c2):
                tn = np.round(gap + np.polyval(c["poly"], gap)).astype(int)
                r = np.full(len(gap), np.inf, np.float32)
                for dd in range(-3, 4):
                    r = np.fmin(r, M.pairs(gap, np.clip(tn + dd, 0, n_t - 1)))
                rs.append(median_filter(r, size=min(25, len(r) | 1), mode="nearest"))
            m1, m2 = gap[rs[0] < MATCH_TOL], gap[rs[1] < MATCH_TOL]
            new_last = int(m1.max()) if len(m1) else c1["b_last"]
            new_first = int(m2.min()) if len(m2) else c2["b_first"]
            if new_last < new_first:
                c1["b_last"], c2["b_first"] = new_last, new_first
                d.update(accepted=True)
            else:
                d.update(accepted=False)
        d.update(b_last=c1["b_last"], b_first=c2["b_first"])
        cut_qc.append(d)
    # ranges: cluster k owns [lo_k, hi_k) in broadcast frames; gaps between confirmations = cut zone
    for i, c in enumerate(clusters):
        c["b_lo"] = 0 if i == 0 else c["b_first"]
        c["b_hi"] = NB if i == len(clusters) - 1 else c["b_last"] + 1
    qc = dict(cuts=cut_qc, anchors=anchors, clusters=[{k: v for k, v in c.items() if k not in ("offs", "bs")}
                                         | dict(anchor_offsets=c["offs"]) for c in clusters],
              sample_best_res=dict(n=int(len(samp)),
                                   frac_lt_match_tol=float((best_r < MATCH_TOL).mean())))
    return clusters, qc


def offset_of(clusters, b):
    """Nominal offset for broadcast frames b (array); NaN in cut zones."""
    b = np.asarray(b)
    off = np.full(b.shape, np.nan)
    for c in clusters:
        m = (b >= c["b_lo"]) & (b < c["b_hi"])
        off[m] = np.polyval(c["poly"], b[m])
    return off


def nominal_t_any(clusters, b):
    """Tactical instant at which broadcast frame b was aired (nearest cluster inside cut zones)."""
    o = offset_of(clusters, np.array([b]))[0]
    if np.isnan(o):
        near = min(clusters, key=lambda c: min(abs(c["b_lo"] - b), abs(c["b_hi"] - b)))
        o = np.polyval(near["poly"], b)
    return float(b + o)


def classify_broadcast(M, valid, clusters, log=print):
    NB = len(valid)
    nb = M.nb
    n_t = M.T[0][0].shape[0]
    b_all = np.arange(NB)
    off = offset_of(clusters, b_all)
    evalm = valid & (nb >= MIN_PLAYERS) & np.isfinite(off)
    be = np.flatnonzero(evalm)
    tn = np.round(be + off[be]).astype(int)
    rr = np.full(len(be), np.inf, np.float32)
    for d in range(-3, 4):
        rr = np.fmin(rr, M.pairs(be, np.clip(tn + d, 0, n_t - 1)))
    res = np.full(NB, np.nan, np.float32)
    res[be] = rr
    # 1-s running median inside each valid run (frames with too few players are skipped)
    from scipy.ndimage import median_filter
    live = np.zeros(NB, bool)
    for a, b in runs_of(valid):
        idx = np.arange(a, b)
        idx = idx[np.isfinite(res[idx])]
        if len(idx) == 0:
            continue
        sm = median_filter(res[idx], size=min(25, len(idx) | 1), mode="nearest")
        live[idx] = sm < LIVE_TOL
    # broadcast state: 2 invalid, 1 live, 4 not-live (to be refined into 3), 5 too few players
    st = np.full(NB, 2, np.uint8)
    st[valid] = 5
    st[evalm & live] = 1
    st[evalm & ~live] = 4
    # cut zones (valid frames with no cluster) -> not-live candidates as well
    st[valid & (nb >= MIN_PLAYERS) & ~np.isfinite(off)] = 4
    return st, res, off


def find_replays(M, st, clusters, log=print):
    """Backward search on not-live valid stretches (state 4). Returns list of replay dicts."""
    NB = len(st)
    n_t = M.T[0][0].shape[0]
    rng = np.random.default_rng(0)
    nb = M.nb
    reps = []
    cand = st == 4
    for a, b in runs_of(cand):
        if b - a < MIN_RUN:
            continue
        idx = np.arange(a, b)
        idx = idx[nb[idx] >= MIN_PLAYERS]
        if len(idx) < 3:
            continue
        stride = max(10, int(np.ceil(len(idx) / MAX_SAMPLES)))
        sb = idx[::stride]
        # reference: nominal tactical time of the stretch (nearest cluster if in a cut zone)
        offb = offset_of(clusters, sb)
        if np.isnan(offb).any():
            near = min(clusters, key=lambda c: min(abs(c["b_lo"] - a), abs(c["b_hi"] - a)))
            offb = np.where(np.isnan(offb), np.polyval(near["poly"], sb), offb)
        tnom = np.round(sb + offb).astype(int)
        result = None
        for stage, lo_fn in (("recent", lambda t: max(0, t - int(REPLAY_WIN_S * FPS))),
                             ("full", lambda t: 0)):
            pts = []
            for b, t in zip(sb, tnom):
                tc = np.arange(lo_fn(t), min(n_t, t + LOCAL_R))
                s = M.one(b, tc)
                k = int(np.argmin(s))
                pts.append((b, tc[k], s[k]))
            pts = np.array(pts, float)
            okm = pts[:, 2] < MATCH_TOL
            fit = ransac_line(pts[okm, 0], pts[okm, 1], rng) if okm.sum() >= 3 else None
            if fit is not None and fit[2].sum() >= max(3, 0.3 * len(pts)):
                result = (stage, fit, pts, okm)
                break
        d = dict(b0=int(a), b1=int(b), n_samples=int(len(sb)))
        if result is None:
            d.update(found=False)
        else:
            stage, (sl, ic, inl), pts, okm = result
            xb = pts[okm, 0][inl]
            d.update(found=True, stage=stage, rate=sl, intercept=ic, n_inliers=int(inl.sum()),
                     frac_inliers=float(inl.sum() / len(pts)),
                     b_in0=int(xb.min()), b_in1=int(xb.max()),
                     src_t0=int(round(sl * xb.min() + ic)), src_t1=int(round(sl * xb.max() + ic)),
                     lag_s=float((tnom[0] - (sl * sb[0] + ic)) / FPS),
                     med_res=float(np.median(pts[okm, 2][inl])))
        reps.append(d)
        log(f"stretch {a}-{b}: {d.get('stage', 'no match')} rate={d.get('rate')} lag={d.get('lag_s')}")
    return reps


def project(st, res, clusters, reps, n_t=N_TACT):
    NB = len(st)
    st = st.copy()
    # replay frames: the inlier span of each identified replay stretch, state 4 -> 3
    for r in reps:
        if r.get("found"):
            seg = slice(r["b_in0"], r["b_in1"] + 1)
            m = st[seg] == 4
            st[seg][m] = 3
    # non-live blocks on the broadcast timeline (states 2,3,4 contiguous)
    nonlive = np.isin(st, (2, 3, 4))
    blk_s = np.zeros(NB, np.float32)
    blk_rep = np.zeros(NB, bool)
    for a, b in runs_of(nonlive):
        blk_s[a:b] = (b - a) / FPS
        blk_rep[a:b] = (st[a:b] == 3).any()
    tv_state = np.zeros(n_t, np.uint8)
    b_frame = np.full(n_t, -1, np.int32)
    match_res = np.full(n_t, np.nan, np.float32)
    tv_block_s = np.zeros(n_t, np.float32)
    tv_block_rep = np.zeros(n_t, bool)
    t_all = np.arange(n_t)
    for c in clusters:
        a1, a0 = c["poly"]                      # off(b) = a1*b + a0  ->  t = (1+a1) b + a0
        b = np.round((t_all - a0) / (1 + a1)).astype(int)
        m = (b >= c["b_lo"]) & (b < c["b_hi"]) & (b >= 0) & (b < NB)
        tv_state[m] = st[b[m]]
        b_frame[m] = b[m]
        match_res[m] = res[b[m]]
        tv_block_s[m] = blk_s[b[m]]
        tv_block_rep[m] = blk_rep[b[m]]
    # cut zones between consecutive clusters: both hypotheses
    for c1, c2 in zip(clusters, clusters[1:]):
        for c in (c1, c2):
            a1, a0 = c["poly"]
            t_lo = int(np.ceil((1 + a1) * (c1["b_last"] + 1) + a0))
            t_hi = int(np.floor((1 + a1) * (c2["b_first"] - 1) + a0))
            seg = np.arange(max(t_lo, 0), min(t_hi + 1, n_t))
            seg = seg[b_frame[seg] < 0]
            tv_state[seg] = 6
    # replayed tactical frames
    replayed = np.zeros(n_t, np.uint8)
    lag = np.full(n_t, np.nan, np.float32)
    for r in reps:
        if not r.get("found"):
            continue
        t0, t1 = sorted((r["src_t0"], r["src_t1"]))
        seg = slice(max(t0, 0), min(t1 + 1, n_t))
        replayed[seg] = np.minimum(replayed[seg].astype(int) + 1, 255)
        # first replay of each frame: replays are processed in broadcast order
        tv_frame_of_replay = nominal_t_any(clusters, r["b_in0"])
        lg = (tv_frame_of_replay - np.arange(seg.start, seg.stop)) / FPS
        cur = lag[seg]
        lag[seg] = np.where(np.isnan(cur), lg, cur)
    return dict(tv_state=tv_state, tv_shown=tv_state == 1, tv_offpitch=np.isin(tv_state, (2, 4)),
                tv_replay=tv_state == 3, tv_unknown=np.isin(tv_state, (0, 6)),
                tv_block_s=tv_block_s, tv_block_has_replay=tv_block_rep,
                b_frame=b_frame, match_res=match_res, replayed=replayed, replay_lag_s=lag), st


def run(data_dir, log=print):
    t0 = time.time()
    bd, td = os.path.join(data_dir, "broadcast"), os.path.join(data_dir, "tactical")
    Pb, team_b = load_atd_xy(os.path.join(bd, "BAR-ATH_broadcast_atd.txt"),
                             os.path.join(bd, "BAR-ATH_broadcast_atd.xml"))
    Pt, team_t = load_atd_xy(os.path.join(td, "BAR-ATH_tactical_atd.txt"),
                             os.path.join(td, "BAR-ATH_tactical_atd.xml"))
    valid = np.loadtxt(os.path.join(bd, "BAR-ATH_broadcast_homography.csv"), delimiter=",",
                       skiprows=1, usecols=11).astype(bool)
    M = Matcher(Pb, team_b, Pt, team_t)
    del Pb, Pt
    log(f"loaded in {time.time() - t0:.0f}s; broadcast {len(valid)} frames, valid {valid.mean():.3f}")
    clusters, qc = synchronise(M, valid, log)
    log(f"clusters: {[(c['b_lo'], c['b_hi'], round(c['off_med'])) for c in clusters]} "
        f"({time.time() - t0:.0f}s)")
    st, res, off = classify_broadcast(M, valid, clusters, log)
    reps = find_replays(M, st, clusters, log)
    out, st = project(st, res, clusters, reps)
    # ---- QC
    ev = np.isfinite(res)
    h, e = np.histogram(res[ev], bins=np.arange(0, CAP + 0.25, 0.25))
    qc.update(
        n_broadcast=int(len(valid)), frac_valid=float(valid.mean()),
        broadcast_state_counts={int(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))},
        residual_hist=dict(edges=e.round(2).tolist(), counts=h.tolist()),
        frac_evaluated_live=float((st[ev] == 1).mean()),
        sensitivity_live_tol={str(x): float((res[ev] < x).mean()) for x in (1.5, 2.0, 2.5)},
        replays=reps, runtime_s=time.time() - t0,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10),
        tactical_state_counts={int(k): int(v) for k, v in
                               zip(*np.unique(out["tv_state"], return_counts=True))},
        constants=dict(CAP=CAP, MATCH_TOL=MATCH_TOL, LIVE_TOL=LIVE_TOL, MIN_PLAYERS=MIN_PLAYERS,
                       ANCHOR_STEP=ANCHOR_STEP, CLUSTER_TOL=CLUSTER_TOL, LOCAL_R=LOCAL_R,
                       REPLAY_WIN_S=REPLAY_WIN_S, RATE_MIN=RATE_MIN, RATE_MAX=RATE_MAX,
                       RANSAC_TOL=RANSAC_TOL, MIN_RUN=MIN_RUN, MAX_SAMPLES=MAX_SAMPLES))
    return out, qc, dict(bc_state=st, bc_res=res, bc_off=off.astype(np.float32))


# ---------------------------------------------------------------- label-free comparison with a D track
HALVES = ((11185, 90652), (110808, 190198))


def collective_speed(data_dir):
    """Median ATD speed (m/s) of visible tactical players, per frame (descriptive statistic only)."""
    out = []
    with open(os.path.join(data_dir, "tactical", "BAR-ATH_tactical_atd.txt")) as fh:
        for line in fh:
            a = np.array(line.split(":")[1].replace(";", ",").split(","), dtype=np.float32)[2::3]
            out.append(np.nanmedian(a) if np.isfinite(a).any() else np.nan)
    return np.array(out, np.float32)


def _q(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {str(q): round(float(np.percentile(x, q)), 2) for q in qs} if len(x) else {}


def compare_with_dead(tv, dead, speed=None, qc=None):
    """Agreement between the TV signal and a dead-ball track (bool per tactical frame), WITHOUT
    any label: shares, overlaps and lags, plus disagreement categories described by statistics."""
    n = len(dead)
    graded = np.zeros(n, bool)
    for a, b in HALVES:
        graded[a:b] = True
    st = tv["tv_state"]
    D = dead & graded
    out = dict(graded_frames=int(graded.sum()), frac_D_graded=float(D[graded].mean()))
    out["by_state"] = {int(k): dict(frames=int((graded & (st == k)).sum()),
                                    p_D=round(float(D[graded & (st == k)].mean()), 3))
                       for k in range(7) if (graded & (st == k)).any()}
    Dr = runs_of(D)
    Ds, De = np.array([a for a, _ in Dr]), np.array([b for _, b in Dr])
    nonlive = np.isin(st, (2, 3, 4))
    blocks = []
    for a, b in runs_of(nonlive & graded):
        ov = np.flatnonzero((Ds < b) & (De > a))
        blocks.append(dict(a=int(a), b=int(b), dur=(b - a) / FPS, fracD=float(D[a:b].mean()),
                           n_ov=len(ov), lag_start=(a - Ds[ov[0]]) / FPS if len(ov) else np.nan,
                           lag_end=(b - De[ov[-1]]) / FPS if len(ov) else np.nan,
                           replay=bool((st[a:b] == 3).any()), frac_inv=float((st[a:b] == 2).mean()),
                           spd=float(np.nanmedian(speed[a:b])) if speed is not None else np.nan))
    dur = np.array([x["dur"] for x in blocks])
    w = dur * FPS
    fracD = np.array([x["fracD"] for x in blocks])
    nov = np.array([x["n_ov"] for x in blocks])
    out["tv_blocks"] = dict(
        n=len(blocks), frames=int(w.sum()), duration_s=_q(dur),
        frac_frames_in_D=round(float(np.average(fracD, weights=w)), 3),
        frac_blocks_overlapping_D=round(float((nov > 0).mean()), 3),
        lag_block_start_minus_D_start_s=_q([x["lag_start"] for x in blocks]),
        lag_block_end_minus_D_end_s=_q([x["lag_end"] for x in blocks]),
        by_duration={})
    for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 60), (60, 1e9)):
        m = (dur >= lo) & (dur < hi)
        if m.any():
            out["tv_blocks"]["by_duration"][f"[{lo},{hi if hi < 1e9 else 'inf'})s"] = dict(
                n=int(m.sum()), frac_overlap_D=round(float((nov[m] > 0).mean()), 3),
                frac_frames_in_D=round(float(np.average(fracD[m], weights=w[m])), 3))
    # D runs coverage
    drs = []
    for a, b in Dr:
        drs.append(dict(dur=(b - a) / FPS, cov=float(nonlive[a:b].mean()), any=bool(nonlive[a:b].any()),
                        shown=float((st[a:b] == 1).mean()), unk=float(np.isin(st[a:b], (0, 6)).mean()),
                        spd=float(np.nanmedian(speed[a:b])) if speed is not None else np.nan))
    ddur = np.array([x["dur"] for x in drs])
    dcov = np.array([x["cov"] for x in drs])
    dany = np.array([x["any"] for x in drs])
    dunk = np.array([x["unk"] for x in drs])
    out["D_runs"] = dict(n=len(drs), duration_s=_q(ddur),
                         frac_runs_with_tv_nonlive=round(float(dany.mean()), 3),
                         frac_D_frames_tv_nonlive=round(float(np.average(dcov, weights=ddur)), 3),
                         frac_D_frames_tv_unknown=round(float(np.average(dunk, weights=ddur)), 3),
                         by_duration={})
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
        m = (ddur >= lo) & (ddur < hi)
        if m.any():
            out["D_runs"]["by_duration"][f"[{lo},{hi if hi < 1e9 else 'inf'})s"] = dict(
                n=int(m.sum()), frac_with_tv_nonlive=round(float(dany[m].mean()), 3),
                frac_frames_tv_nonlive=round(float(np.average(dcov[m], weights=ddur[m])), 3))
    # disagreement categories
    spd_b = np.array([x["spd"] for x in blocks])
    rep_b = np.array([x["replay"] for x in blocks])
    lag_e = np.array([x["lag_end"] for x in blocks])
    lag_s = np.array([x["lag_start"] for x in blocks])
    cat = {}
    m = (fracD < 0.2) & (dur < 2)
    cat["A_tv_off_bref_sans_D (<2 s)"] = dict(n=int(m.sum()), frames=int(w[m].sum()),
                                              med_speed=_q(spd_b[m], (50,)))
    m = (fracD < 0.2) & (dur >= 2)
    cat["B_tv_off_long_sans_D (>=2 s, <20% D)"] = dict(
        n=int(m.sum()), frames=int(w[m].sum()), duration_s=_q(dur[m]), with_replay=int(rep_b[m].sum()),
        med_speed=_q(spd_b[m], (50,)), blocks=[(blocks[i]["a"], blocks[i]["b"]) for i in np.flatnonzero(m)])
    m = np.isfinite(lag_e) & (lag_e > 3)
    cat["C_retour_tv_tardif (bloc finit >3 s apres la fin du D)"] = dict(
        n=int(m.sum()), excess_s=_q(lag_e[m]), with_replay=int(rep_b[m].sum()))
    m = np.isfinite(lag_s) & (lag_s > 10)
    cat["D_depart_tv_tardif (bloc commence >10 s apres le debut du D)"] = dict(
        n=int(m.sum()), delay_s=_q(lag_s[m]), with_replay=int(rep_b[m].sum()))
    m = ~dany & (dunk < 0.5)
    cat["E_D_sans_tv_hors_jeu (TV reste sur le jeu)"] = dict(
        n=int(m.sum()), duration_s=_q(ddur[m]),
        med_speed=_q(np.array([x["spd"] for x in drs])[m], (50,)))
    if speed is not None:
        cat["reference_med_speed"] = dict(
            graded_D=_q(speed[D], (50,)), graded_not_D=_q(speed[graded & ~D], (50,)),
            tv_shown=_q(speed[graded & (st == 1)], (50,)), tv_nonlive=_q(speed[graded & nonlive], (50,)))
    out["disagreements"] = cat
    # replays: where is the replayed action relative to the next D start?
    if qc is not None:
        rl = []
        for r in qc.get("replays", []):
            if not r.get("found"):
                continue
            t1 = max(r["src_t0"], r["src_t1"])
            nxt = Ds[Ds >= t1 - 10 * FPS]
            inD = bool(D[min(t1, n - 1)])
            rl.append(dict(src_end=t1, src_end_in_D=inD,
                           next_D_start_minus_src_end_s=float((nxt[0] - t1) / FPS) if len(nxt) else None,
                           lag_s=r["lag_s"], rate=r["rate"]))
        out["replays_vs_D"] = rl
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="challenge-barcelona/data (tactical/ and broadcast/)")
    ap.add_argument("--out", required=True, help="output npz (per tactical frame)")
    ap.add_argument("--qc", default=None, help="optional QC json (anchors, clusters, replays, histograms)")
    ap.add_argument("--from-npz", action="store_true",
                    help="reuse an existing --out npz (and --qc json) instead of recomputing")
    ap.add_argument("--dead", default=None, help="optional frame,label CSV (D = dead ball) to compare")
    ap.add_argument("--compare-out", default=None, help="json for the label-free comparison")
    a = ap.parse_args()
    if a.from_npz:
        z = np.load(a.out)
        out = {k: z[k] for k in z.files}
        qc = json.load(open(a.qc)) if a.qc and os.path.exists(a.qc) else None
    else:
        out, qc, bstuff = run(a.data)
        np.savez_compressed(a.out, **out, **bstuff)
        if a.qc:
            with open(a.qc, "w") as fh:
                json.dump(qc, fh, indent=1, default=float)
        print("tactical tv_state counts:", qc["tactical_state_counts"])
    if a.dead:
        import csv
        with open(a.dead) as fh:
            lab = np.array([r["label"] for r in csv.DictReader(fh)])
        cmp_ = compare_with_dead(out, lab == "D", collective_speed(a.data), qc)
        with open(a.compare_out or os.path.splitext(a.out)[0] + "_vs_dead.json", "w") as fh:
            json.dump(cmp_, fh, indent=1, default=float)
        print(json.dumps({k: cmp_[k] for k in ("tv_blocks", "D_runs")}, default=float)[:2000])


if __name__ == "__main__":
    main()

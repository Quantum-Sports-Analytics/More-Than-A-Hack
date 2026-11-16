"""Dead-ball detector D v4 with stoppage type, learnt and tuned ONLY on external data (Sportec).

Data: Sportec Open DFL Dataset (Bassek, Rein, Weber & Memmert 2025, Sci Data 12:195,
doi 10.1038/s41597-025-04505-y, CC-BY 4.0), 7 matches: positions (npz from fetch_sportec.py)
plus the raw DFL event files (DFL_03_02_events_raw_*.xml, Hugging Face mirror pysport/idsse-data).

1. Events <-> tracking alignment (align_events). Shots carry CalculatedFrame / CalculatedTimestamp
   (tracking clock): they give the exact frame <-> time map of each half (t = a + N / 25). The
   manually keyed EventTime has a per-half offset (median over restarts) and ~1 s noise, so each
   restart (ThrowIn, CornerKick, GoalKick, FreeKick, KickOff, Penalty, RefereeBall) is matched one
   to one to a dead -> alive transition of BallStatus within +-4 s after offset correction.
2. Stoppage typology (stoppages): every BallStatus==0 run gets the type of the restart matched to
   its end; unmatched runs fall back on the events inside the run (RefereeBall > Foul > KickOff >
   ThrowIn) then on the geometry of the exit / restart points (dead_bounds.stoppage_types).
3. Training simulation (simulate_obs), domain-randomised over wide a-priori ranges and NOT fitted
   to any statistic of the provided match: the frozen Sportec dead ball is re-synthesised
   (dead_bounds.synth_dead_ball), lofted balls are projected to the ground from a random camera,
   a VIDEO ball (noise, gaps, out-of-pitch drop beyond 2 m and spot blacklist as coded in
   video_ball.py, ghost tracks, teleport outliers, hidden exits / restarts, <=12-frame bridges)
   and an ATD ball (own noise and gaps) are simulated, and fused like video_ball.py
   (video > bridge > ATD > none, with the source code). Players get random noise and dropouts.
4. Features (features): v2 player kinematics (dead_ball.base/context_features, players only),
   restart configurations without player identity (player on a touchline, at a corner flag,
   keeper proxy in his 6 m box with no opponent in the box, both teams in their own half with an
   empty centre circle, ring of players at ~9 m from the ball), ball geometry (signed distance to
   the lines with margins, still ball then impulse, spots) and BALL CONFIDENCE (visibility runs,
   source mix, ATD-ball visibility, jumps / teleports above 75 m/s and across gaps).
5. Models: HistGradientBoosting P(dead) per frame and a multiclass stoppage-type model on dead
   frames. Decoding: centred mean, hysteresis, short-hole merge, minimum duration; operating
   point chosen per test match on the 6 other matches (out-of-fold probabilities) with a cost that
   weighs a false D frame twice a missed one plus a fixed penalty per false D run.

Stop-type codes (stop_type, int8): 0 live, 1 touche (throw-in), 2 corner, 3 six_metres (goal kick),
4 coup_franc (free kick, incl. offside / foul), 5 engagement (kick-off after a goal), 6 penalty,
7 balle_arbitre (dropped ball), 8 inconnu.

CLI
  # 1-2: alignment + typology tables (downloads the 7 event files if missing)
  python possession/dead_v4.py events --sportec <dir J03W*.npz> --events-dir <dir> --out-dir <dir>
  # 4: leave-one-match-out validation (D v4 and D v2 replayed under the same simulation)
  python possession/dead_v4.py cv --sportec <dir> --events-dir <dir> --work <dir> --out-dir <dir>
  # apply to Barca-Athletic: writes dead_v4.npz (199,319 frames)
  python possession/dead_v4.py apply --sportec <dir> --events-dir <dir> --data <data dir> \
      --ball balle_fusionnee.npz --out dead_v4.npz [--decode-json decode.json] [--cache <dir>]
Nothing is fitted, tuned or calibrated on the provided match, its video or its ground truth.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1  # noqa: E402
import dead_ball as db  # noqa: E402
import dead_bounds as db3  # noqa: E402

FPS = 25
SPORTEC_IDS = db.SPORTEC_IDS
HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
RESTARTS = {"ThrowIn": 1, "CornerKick": 2, "GoalKick": 3, "FreeKick": 4, "KickOff": 5, "Penalty": 6,
            "RefereeBall": 7}
STOP_NAMES = {0: "vivant", 1: "touche", 2: "corner", 3: "six_metres", 4: "coup_franc", 5: "engagement",
              6: "penalty", 7: "balle_arbitre", 8: "inconnu"}
GEO = {"touche": 1, "corner": 2, "6m": 3, "faute_autre": 4, "but": 5, "coup_envoi_autre": 5, "penalty": 6,
       "sortie_but_autre": 8}
MATCH_WIN = 100            # frames (+-4 s) for restart <-> BallStatus matching
VIDEO_OUT_MARGIN = 2.0     # video_ball.OUT_MARGIN
VIDEO_MAX_BRIDGE = 12      # video_ball.MAX_GAP
JUMP_V = 75.0              # m/s: frame-to-frame ball speed above this is a teleport
STILL_V, KICK_V = db3.STILL_V, db3.KICK_V


def runs_of(mask):
    e = np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


# =============================================================================== 1. events
def _com(mid):
    return "000001" if mid in ("J03WMX", "J03WN1") else "000002"


def fetch_events(events_dir):
    os.makedirs(events_dir, exist_ok=True)
    for mid in SPORTEC_IDS:
        f = f"DFL_03_02_events_raw_DFL-COM-{_com(mid)}_DFL-MAT-{mid}.xml"
        p = os.path.join(events_dir, f)
        if not os.path.exists(p):
            r = subprocess.run(["curl", "-sL", "--fail", "-o", p + ".part", HF + f])
            if r.returncode:
                raise RuntimeError(f"download failed {f}")
            os.replace(p + ".part", p)


def _ts(s):
    return datetime.fromisoformat(s).timestamp() if s else np.nan


def parse_events(path):
    rows = []
    for ev in ET.parse(path).getroot().iter("Event"):
        ch = list(ev)
        if not ch:
            continue
        c, a = ch[0], ev.attrib
        rows.append(dict(eid=a["EventId"], type=c.tag, t=_ts(a["EventTime"]), tdec=_ts(c.get("DecisionTimestamp")),
                         x=float(a.get("X-Position", "nan")), y=float(a.get("Y-Position", "nan")),
                         cframe=float(a.get("CalculatedFrame", "nan")), ctime=_ts(a.get("CalculatedTimestamp"))))
    return pd.DataFrame(rows)


def align_events(mid, npz, events_dir):
    """Events of one match with frame indices per half (relative to hN_n0).

    Returns (events DataFrame with columns half, f, fdec, f_corr, fdec_corr, up, res),
    per-half summary dict.
    """
    z = np.load(npz) if isinstance(npz, str) else npz
    ev = parse_events(os.path.join(events_dir, f"DFL_03_02_events_raw_DFL-COM-{_com(mid)}_DFL-MAT-{mid}.xml"))
    ev["half"], ev["f"], ev["fdec"] = "", np.nan, np.nan
    summ = {}
    for h, lo, hi in (("h1", 0, 100000), ("h2", 100000, 10 ** 9)):
        an = ev[ev.cframe.notna() & (ev.cframe >= lo) & (ev.cframe < hi)]
        a0 = float(np.median(an.ctime - an.cframe / FPS))
        resid = np.abs(an.ctime - an.cframe / FPS - a0).max() * FPS
        n0, T = int(z[f"{h}_n0"]), len(z[f"{h}_BS"])
        fr = np.round((ev.t.values - a0) * FPS) - n0
        ok = (fr >= -FPS * 60) & (fr < T + FPS * 60) & (ev.half.values == "")
        ev.loc[ok, "half"] = h
        ev.loc[ok, "f"] = fr[ok]
        ev.loc[ok, "fdec"] = (np.round((ev.tdec.values - a0) * FPS) - n0)[ok]
        summ[h] = dict(anchors=len(an), anchor_resid_frames=float(resid))
    ev["f_corr"], ev["fdec_corr"], ev["up"], ev["res"] = np.nan, np.nan, -1, np.nan
    for h in ("h1", "h2"):
        bs = z[f"{h}_BS"]
        up = np.flatnonzero((bs[1:] == 1) & (bs[:-1] != 1)) + 1
        sel = (ev.half == h) & ev.type.isin(list(RESTARTS)) & ~((ev.type == "KickOff") & (ev.f < 150))
        r = ev[sel]
        d = np.array([fe - up[np.argmin(np.abs(up - fe))] for fe in r.f]) if len(up) else np.array([])
        off = int(np.median(d[np.abs(d) < 125])) if (np.abs(d) < 125).any() else 0
        hm = ev.half == h
        ev.loc[hm, "f_corr"] = ev.loc[hm, "f"] - off
        ev.loc[hm, "fdec_corr"] = ev.loc[hm, "fdec"] - off
        pairs = sorted((abs(fe - u), i, j) for i, fe in zip(r.index, ev.loc[r.index, "f_corr"])
                       for j, u in enumerate(up) if abs(fe - u) <= MATCH_WIN)
        ui, uj = set(), set()
        for dd, i, j in pairs:
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            ev.at[i, "up"] = up[j]
            ev.at[i, "res"] = ev.at[i, "f_corr"] - up[j]
        s = ev[sel]
        summ[h].update(offset_frames=off, restarts=int(sel.sum()), matched=int((s.up >= 0).sum()),
                       abs_res_med_s=float(np.nanmedian(np.abs(s.res)) / FPS),
                       abs_res_p90_s=float(np.nanpercentile(np.abs(s.res), 90) / FPS))
    return ev, summ


def stoppages(mid, npz, ev):
    """One row per BallStatus==0 run of the trimmed halves with type and boundary measurements."""
    z = np.load(npz) if isinstance(npz, str) else npz
    L, W = float(z["L"]), float(z["W"])
    out = []
    for h in ("h1", "h2"):
        bs = z[f"{h}_BS"]
        al = np.flatnonzero(bs == 1)
        A0, B0 = al[0], al[-1] + 1
        bx, by = z[f"{h}_BX"][A0:B0] + L / 2, z[f"{h}_BY"][A0:B0] + W / 2
        dead = bs[A0:B0] == 0
        geo = db3.stoppage_types(dict(L=L, W=W, bx=bx, by=by, dead=dead))
        eh = ev[ev.half == h]
        by_up = {int(u) - A0: i for u, i in zip(eh.up, eh.index) if u >= 0}
        spd = np.r_[0, np.hypot(np.diff(bx), np.diff(by)) * FPS]
        for k, (a, b) in enumerate(runs_of(dead)):
            i = by_up.get(b)
            inside = set(eh[(eh.f_corr - A0 >= a - 50) & (eh.f_corr - A0 < b)].type)
            if i is not None:
                typ, src = RESTARTS[ev.at[i, "type"]], "reprise"
            elif "RefereeBall" in inside:
                typ, src = 7, "evenement_interne"
            elif "Foul" in inside or "Offside" in inside:
                typ, src = 4, "evenement_interne"
            elif "KickOff" in inside:
                typ, src = 5, "evenement_interne"
            elif "ThrowIn" in inside:
                typ, src = 1, "evenement_interne"
            else:
                typ, src = GEO[geo[k]], "geometrie"
            w = np.arange(max(a - 50, 0), min(a + 6, len(bx)))
            edge = np.fmin(np.fmin(bx[w], L - bx[w]), np.fmin(by[w], W - by[w]))
            o = np.flatnonzero(edge < 0)
            kk = np.flatnonzero(spd[b + 1:min(b + 75, len(spd))] > 3.0)
            cause = ",".join(sorted(inside - {"Delete", "Play", "OtherBallAction", "TacklingGame", "BallClaiming"}))
            out.append(dict(mid=mid, half=h, a=int(a), b=int(b), dur_s=(b - a) / FPS, type=typ,
                            type_name=STOP_NAMES[typ], type_source=src, cause=cause,
                            dec_vs_start_s=(ev.at[i, "fdec_corr"] - A0 - a) / FPS if i is not None else np.nan,
                            dec_vs_end_s=(ev.at[i, "fdec_corr"] - A0 - b) / FPS if i is not None else np.nan,
                            event_vs_end_s=(ev.at[i, "f_corr"] - A0 - b) / FPS if i is not None else np.nan,
                            exit_vs_start_s=(w[o[0]] - a) / FPS if len(o) else np.nan,
                            kick_vs_end_s=(kk[0] + 1) / FPS if len(kk) else np.nan))
    return pd.DataFrame(out)


def sportec_tables(sportec_dir, events_dir):
    fetch_events(events_dir)
    S, A = [], []
    for mid in SPORTEC_IDS:
        npz = os.path.join(sportec_dir, f"{mid}.npz")
        ev, summ = align_events(mid, npz, events_dir)
        S.append(stoppages(mid, npz, ev))
        for h, d in summ.items():
            A.append(dict(match=mid, half=h, **d))
    return pd.concat(S, ignore_index=True), pd.DataFrame(A)


# =========================================================================== 2. simulation
def _ar1(T, d, sd, tau_s, rng):
    a = np.exp(-1.0 / max(tau_s * FPS, 1.0))
    z = rng.normal(size=(T, d)) * sd * np.sqrt(1 - a * a)
    e = np.empty((T, d))
    e[0] = rng.normal(size=d) * sd
    for i in range(1, T):
        e[i] = a * e[i - 1] + z[i]
    return e


def _add_gaps(cand, target, med_frames, sigma, rng, miss):
    """Add lognormal-length gaps starting on cand frames until the missed share of cand reaches target."""
    miss = miss.copy()
    idx = np.flatnonzero(cand)
    if len(idx) == 0 or target <= 0:
        return miss
    need = int(target * len(idx)) - int((miss & cand).sum())
    tries = 0
    while need > 0 and tries < 100000:
        tries += 1
        s = int(rng.choice(idx))
        n = max(1, int(np.exp(rng.normal(np.log(med_frames), sigma))))
        seg = cand[s:s + n] & ~miss[s:s + n]
        miss[s:s + n] |= cand[s:s + n]
        need -= int(seg.sum())
    return miss


def draw_params(rng, stress=False):
    """Wide a-priori ranges (none of them estimated on the provided match)."""
    u = rng.uniform
    p = dict(cam_h=u(15, 45), cam_back=u(15, 60), cam_dx=u(-15, 15), cam_side=rng.choice([-1, 1]),
             project_atd=rng.random() < 0.5,
             v_sd=u(0.05, 1.0), v_tau=u(0.3, 4.0), a_sd=u(0.3, 3.0), a_tau=u(0.3, 3.0),
             v_miss_live=u(0.02, 0.45), a_miss_live=u(0.02, 0.45),
             v_gap_live_s=u(0.2, 3.0), v_gap_dead_s=u(1.0, 15.0), a_gap_live_s=u(0.5, 10.0), a_gap_dead_s=u(2.0, 30.0),
             gap_sigma=u(0.5, 1.3), spot_r=u(0.4, 1.0), a_out_margin=u(0.0, 5.0), a_out_drop=u(0.5, 1.0),
             ghost_frac=u(0.0, 0.4), tele_rate=u(0.0, 0.01), hide_start=u(0.0, 0.6), hide_end=u(0.0, 0.6),
             p_pass=u(0.2, 0.8), p_dribble=u(0.2, 0.8), place_max=u(3.0, 10.0),
             pl_sd=u(0.0, 0.4), pl_drop=u(0.0, 0.15))
    for c in ("v", "a"):
        live = p[f"{c}_miss_live"]
        if rng.random() < 0.8:          # missing more often in stoppages (ball out of play / out of frame)
            p[f"{c}_miss_dead"] = live + rng.random() * (0.95 - live)
        else:                           # uninformative case
            p[f"{c}_miss_dead"] = float(np.clip(live * u(0.8, 1.2), 0, 0.95))
    if stress:                          # pessimistic corner of the ranges
        p.update(v_sd=u(0.5, 1.5), v_miss_live=u(0.3, 0.6), a_miss_live=u(0.3, 0.6), ghost_frac=u(0.3, 0.6),
                 tele_rate=u(0.005, 0.02), hide_start=u(0.4, 0.8), hide_end=u(0.4, 0.8), pl_sd=u(0.3, 0.6),
                 pl_drop=u(0.1, 0.25))
        p["v_miss_dead"] = p["v_miss_live"] + rng.random() * (0.95 - p["v_miss_live"]) * 0.5
        p["a_miss_dead"] = p["a_miss_live"] + rng.random() * (0.95 - p["a_miss_live"]) * 0.5
    return p


def _project(bx, by, bz, L, W, p):
    cx, cy = L / 2 + p["cam_dx"], (W + p["cam_back"]) if p["cam_side"] > 0 else -p["cam_back"]
    h = p["cam_h"]
    z = np.clip(np.nan_to_num(bz), 0, h - 2)
    k = h / (h - z)
    return cx + (bx - cx) * k, cy + (by - cy) * k


def simulate_obs(hf, rng, p):
    """Simulated (players, video, ATD, fused ball + source) for one Sportec half dict."""
    L, W, dead = hf["L"], hf["W"], hf["dead"]
    px, py = hf["px"].astype(np.float64), hf["py"].astype(np.float64)
    T, S = px.shape
    sx, sy, sz = db3.synth_dead_ball(hf["bx"], hf["by"], hf["bz"], dead, px, py, L, W, rng,
                                     p["p_pass"], p["p_dribble"], (0.0, p["place_max"]))
    # ---- players: AR(1) noise (common per player) + dropout segments
    if p["pl_sd"] > 0:
        e = _ar1(T, 2 * S, p["pl_sd"], 2.0, rng)
        px, py = px + e[:, :S], py + e[:, S:]
    if p["pl_drop"] > 0:
        for j in range(S):
            m = _add_gaps(np.ones(T, bool), p["pl_drop"], 5 * FPS, 1.0, rng, np.zeros(T, bool))
            px[m, j] = np.nan
            py[m, j] = np.nan
    # ---- video ball
    gx, gy = _project(sx, sy, sz, L, W, p)
    e = _ar1(T, 2, p["v_sd"], p["v_tau"], rng)
    vx, vy = gx + e[:, 0], gy + e[:, 1]
    edge = np.fmin(np.fmin(vx, L - vx), np.fmin(vy, W - vy))
    spots = ((L / 2, W / 2), (11.0, W / 2), (L - 11.0, W / 2))
    on_spot = np.zeros(T, bool)
    for a, b in spots:
        on_spot |= np.hypot(vx - a, vy - b) < p["spot_r"]
    miss = (edge < -VIDEO_OUT_MARGIN) | on_spot | ~np.isfinite(vx)
    gl, gd, sg = p["v_gap_live_s"] * FPS, p["v_gap_dead_s"] * FPS, p["gap_sigma"]
    miss = _add_gaps(~dead, p["v_miss_live"], gl, sg, rng, miss)
    miss = _add_gaps(dead, p["v_miss_dead"], gd, sg, rng, miss)
    for a, b in runs_of(dead):
        if rng.random() < p["hide_start"]:
            miss[max(a - int(rng.uniform(0, 1) * FPS), 0):a + int(rng.uniform(1, 3) * FPS)] = True
        if rng.random() < p["hide_end"]:
            miss[max(b - int(rng.uniform(1, 4) * FPS), a):min(b + FPS // 2, T)] = True
    vx, vy = np.where(miss, np.nan, vx), np.where(miss, np.nan, vy)
    for a, b in runs_of(miss):                          # ghost track on a player during a gap
        if b - a < 10 or rng.random() >= p["ghost_frac"]:
            continue
        s = a + int(rng.uniform(0, 0.5) * (b - a))
        n = int(rng.uniform(0.2, 1.0) * (b - s))
        if n < 3:
            continue
        ref = (sx[s], sy[s])
        dd = np.hypot(px[s] - ref[0], py[s] - ref[1])
        if not np.isfinite(dd).any():
            continue
        j = int(np.nanargmin(dd))
        vx[s:s + n], vy[s:s + n] = px[s:s + n, j] + 0.3, py[s:s + n, j] + 0.3
    n_tel = rng.binomial(T, p["tele_rate"])             # teleport outliers (false detections)
    for s in rng.integers(0, T, n_tel):
        n = int(rng.integers(1, 9))
        vx[s:s + n], vy[s:s + n] = rng.uniform(0, L), rng.uniform(0, W)
    vis = np.isfinite(vx)
    src = np.where(vis, 1, 0).astype(np.int8)
    for a, b in runs_of(~vis):                          # <= 12-frame bridges inside a track
        if a > 0 and b < T and (b - a) <= VIDEO_MAX_BRIDGE:
            w = (np.arange(1, b - a + 1) / (b - a + 1))
            vx[a:b] = vx[a - 1] + w * (vx[b] - vx[a - 1])
            vy[a:b] = vy[a - 1] + w * (vy[b] - vy[a - 1])
            src[a:b] = 2
    # ---- ATD ball
    if p["project_atd"]:
        ax, ay = gx.copy(), gy.copy()
    else:
        ax, ay = sx.copy(), sy.copy()
    e = _ar1(T, 2, p["a_sd"], p["a_tau"], rng)
    ax, ay = ax + e[:, 0], ay + e[:, 1]
    aedge = np.fmin(np.fmin(ax, L - ax), np.fmin(ay, W - ay))
    amiss = (aedge < -p["a_out_margin"]) & (rng.random(T) < p["a_out_drop"])
    amiss = _add_gaps(~dead, p["a_miss_live"], p["a_gap_live_s"] * FPS, sg, rng, amiss)
    amiss = _add_gaps(dead, p["a_miss_dead"], p["a_gap_dead_s"] * FPS, sg, rng, amiss)
    ax, ay = np.where(amiss, np.nan, ax), np.where(amiss, np.nan, ay)
    # ---- fusion as video_ball.py: video (detected or bridged) > ATD > none
    use_a = (src == 0) & np.isfinite(ax)
    fx, fy = np.where(use_a, ax, vx), np.where(use_a, ay, vy)
    src = np.where(use_a, 3, src).astype(np.int8)
    return dict(px=px.astype(np.float32), py=py.astype(np.float32), team=hf["team"], fx=fx, fy=fy, src=src,
                ax=ax, ay=ay, L=L, W=W)


# ============================================================================== 3. features
def _since_until_s(mask, cap_s=60.0):
    _, s, u = db._run_lengths(np.asarray(mask, bool))
    return np.minimum(s, cap_s), np.minimum(u, cap_s)


def _roll(x, w_s, how="mean", side="c"):
    s = pd.Series(np.asarray(x, np.float64))
    w = max(1, int(w_s * FPS))
    if side == "c":
        r = s.rolling(w, center=True, min_periods=1)
    elif side == "p":
        r = s.rolling(w, min_periods=1)
    else:
        r = s[::-1].rolling(w, min_periods=1)
    v = getattr(r, how)()
    return (v[::-1] if side == "f" else v).values.astype(np.float32)


def _smooth_by_source(x, y, src):
    """Savitzky-Golay inside runs of constant source (a source switch never makes an impulse)."""
    xs, ys = np.full(len(x), np.nan), np.full(len(x), np.nan)
    vx, vy = xs.copy(), ys.copy()
    for m in ((src == 1) | (src == 2), src == 3):          # video track (detected + bridged) / ATD
        bx = np.where(m, x, np.nan).astype(float)
        by = np.where(m, y, np.nan).astype(float)
        a, b, c, d = v1.smooth_ball(bx, by)
        xs[m], ys[m], vx[m], vy[m] = a[m], b[m], c[m], d[m]
    return xs, ys, vx, vy


def config_features(px, py, team, L, W):
    """Restart configurations from players only (team membership, no player identity)."""
    f = {}
    with np.errstate(all="ignore"):
        dt = np.fmin(py, W - py)                                        # distance to a touchline
        f["c_touch_min"] = np.nanmin(dt, axis=1)
        f["c_touch_n1"] = (dt < 1.0).sum(1).astype(np.float32)
        dc = np.full(px.shape, np.inf)
        for cx in (0.0, L):
            for cy in (0.0, W):
                dc = np.fmin(dc, np.hypot(px - cx, py - cy))
        dc = np.where(np.isfinite(px), dc, np.nan)
        f["c_corner_min"] = np.nanmin(dc, axis=1)
        f["c_corner_n3"] = (dc < 3.0).sum(1).astype(np.float32)
        kx, ky = db3.keeper_proxy(px, py, team)
        gk_box, opp_in, opp_in_pen = [], [], []
        cxs = [np.nanmedian(px[:, team == k], axis=1) for k in (0, 1)]
        for k in (0, 1):
            left = cxs[k] < cxs[1 - k]
            gx = np.where(left, kx[:, k], L - kx[:, k])
            gk_box.append(((gx < 6.0) & (np.abs(ky[:, k] - W / 2) < 9.2)).astype(np.float32))
            opx = px[:, team != k]
            ogx = np.where(left[:, None], opx, L - opx)
            inbox = (ogx < 16.5) & (np.abs(py[:, team != k] - W / 2) < 20.2)
            opp_in.append(inbox.sum(1).astype(np.float32))
            opp_in_pen.append(np.where(gk_box[-1] > 0, inbox.sum(1), 99).astype(np.float32))
        f["c_gk_in6"] = np.fmax(*gk_box)
        f["c_gk6_opp_in_box"] = np.fmin(*opp_in_pen)                    # 0 -> goal-kick shape
        f["c_box_opp_max"] = np.fmax(*opp_in)
        # kick-off: each team in its own half, empty centre circle
        own = []
        for k in (0, 1):
            left = cxs[k] < cxs[1 - k]
            xk = px[:, team == k]
            ok = np.where(left[:, None], xk <= L / 2 + 0.5, xk >= L / 2 - 0.5)
            n = np.isfinite(xk).sum(1)
            own.append(np.where(n > 0, (ok & np.isfinite(xk)).sum(1) / np.maximum(n, 1), np.nan))
        f["c_own_half_min"] = np.fmin(*own)
        dcen = np.hypot(px - L / 2, py - W / 2)
        f["c_centre_n9"] = (dcen < 9.15).sum(1).astype(np.float32)
        f["c_centre_min"] = np.nanmin(dcen, axis=1)
        # penalty: players near a spot and box emptiness
        dpen = np.fmin(np.hypot(px - 11.0, py - W / 2), np.hypot(px - (L - 11.0), py - W / 2))
        f["c_pen_min"] = np.nanmin(dpen, axis=1)
    out = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})
    for c in list(out.columns):
        out[c + "_m1"] = _roll(out[c], 1.0)
        out[c + "_p3"] = _roll(out[c], 3.0, side="p")
        out[c + "_f3"] = _roll(out[c], 3.0, side="f")
    return out


def ball_v4_features(px, py, team, fx, fy, src, ax, L, W):
    """Fused-ball geometry, still/impulse and confidence features (prefix b_)."""
    T = len(fx)
    f = {}
    with np.errstate(all="ignore"):
        xs, ys, vx, vy = _smooth_by_source(fx, fy, src)
        vis = np.isfinite(xs)
        spd = np.hypot(vx, vy)
        # ---- confidence
        f["b_vis"] = vis.astype(np.float32)
        for w in (1, 4, 10):
            f[f"b_vis_m{w}"] = _roll(vis, w)
        f["b_gap_len"], f["b_gap_since"], f["b_gap_until"] = [np.minimum(v, 60) for v in db._run_lengths(~vis)]
        for code, nm in ((1, "vid"), (2, "bri"), (3, "atd")):
            m = (src == code).astype(np.float32)
            f[f"b_src_{nm}_m2"] = _roll(m, 2)
            f[f"b_src_{nm}_m6"] = _roll(m, 6)
        avis = np.isfinite(ax).astype(np.float32)
        for w in (2, 6, 20):
            f[f"b_atd_vis_m{w}"] = _roll(avis, w)
        f["b_atd_gap_since"], f["b_atd_gap_until"] = _since_until_s(avis == 0)
        raw = np.hypot(np.diff(fx), np.diff(fy)) * FPS
        jump = np.r_[False, np.nan_to_num(raw, nan=0) > JUMP_V]
        # teleport across a gap: displacement / gap duration
        gj = np.zeros(T, bool)
        last = -1
        for i in np.flatnonzero(np.isfinite(fx)):
            if last >= 0 and i - last > 1 and np.hypot(fx[i] - fx[last], fy[i] - fy[last]) / ((i - last) / FPS) > 40:
                gj[i] = True
            last = i
        f["b_jump_m1"] = _roll(jump, 1, "sum")
        f["b_jump_m5"] = _roll(jump, 5, "sum")
        f["b_gjump_m5"] = _roll(gj, 5, "sum")
        f["b_jump_since"], f["b_jump_until"] = db3._time_to_events(jump | gj, T)
        f["b_src_switch_m4"] = _roll(np.r_[False, src[1:] != src[:-1]], 4, "sum")
        # ---- kinematics / geometry
        f["b_spd"] = spd
        f["b_spd_m1"] = _roll(spd, 1, "median")
        f["b_spd_p1"] = _roll(spd, 1, "median", "p")
        f["b_spd_f1"] = _roll(spd, 1, "median", "f")
        f["b_spd_max_p1"] = _roll(np.nan_to_num(spd), 1, "max", "p")
        ex, ey = np.fmin(xs, L - xs), np.fmin(ys, W - ys)
        edge = np.fmin(ex, ey)
        f["b_edge"], f["b_edge_x"], f["b_edge_y"] = edge, ex, ey
        f["b_edge_m2"] = _roll(edge, 2, "median")
        for mg, nm in ((0.5, "o05"), (1.5, "o15"), (3.0, "o30")):
            o = np.where(vis, (edge < -mg).astype(np.float32), np.nan)
            o0 = np.nan_to_num(o)
            f[f"b_{nm}"] = o
            f[f"b_{nm}_p2"] = _roll(o0, 2, "max", "p")
            f[f"b_{nm}_p6"] = _roll(o0, 6, "max", "p")
            f[f"b_{nm}_f2"] = _roll(o0, 2, "max", "f")
        e0 = np.nan_to_num(edge, nan=1)
        out_ev = np.r_[False, (e0[1:] < -0.5) & (e0[:-1] >= -0.5)]
        f["b_out_since"], f["b_out_until"] = db3._time_to_events(out_ev, T)
        sm = np.nan_to_num(f["b_spd_m1"], nan=99)
        still = vis & (sm < STILL_V)
        f["b_still_since"], f["b_still_until"] = _since_until_s(still)
        f["b_still_m4"] = _roll(still, 4)
        s_since = f["b_still_since"]
        fut = _roll(np.nan_to_num(spd), 0.5, "max", "f")
        kick = (s_since >= 1.0) & ~np.r_[still[1:], False] & (fut > KICK_V)
        f["b_kick_since"], f["b_kick_until"] = db3._time_to_events(kick, T)
        kick3 = (s_since >= 3.0) & ~np.r_[still[1:], False] & (fut > KICK_V)
        f["b_kick3_since"], f["b_kick3_until"] = db3._time_to_events(kick3, T)
        gx, dy = np.fmin(xs, L - xs), np.abs(ys - W / 2)
        f["b_d_corner"] = np.hypot(gx, np.fmin(ys, W - ys))
        f["b_d_centre"] = np.hypot(xs - L / 2, dy)
        f["b_d_pen"] = np.hypot(gx - 11.0, dy)
        f["b_in6"] = ((gx > -0.5) & (gx < 6.0) & (dy < 9.5)).astype(np.float32)
        f["b_goal_dy"] = dy
        d = np.hypot(px - xs[:, None], py - ys[:, None])
        f["b_near1"] = np.nanmin(d, axis=1)
        ds = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        f["b_near2"] = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan)
        f["b_n3"] = np.where(vis, (d < 3).sum(1), np.nan)
        f["b_n10"] = np.where(vis, (d < 10).sum(1), np.nan)
        ring = (d > 8.0) & (d < 10.5)
        f["b_ring9"] = np.where(vis, ring.sum(1), np.nan)
        f["b_ring9_m2"] = _roll(np.nan_to_num(f["b_ring9"]), 2)
        f["b_in9"] = np.where(vis, ((d < 8.0) & (d > 1.5)).sum(1), np.nan)
        kx, ky = db3.keeper_proxy(px, py, team)
        dk = np.fmin(np.hypot(kx[:, 0] - xs, ky[:, 0] - ys), np.hypot(kx[:, 1] - xs, ky[:, 1] - ys))
        f["b_d_keeper"] = dk
        hold = (dk < 1.5) & (gx < 16.5) & (dy < 20.2)
        f["b_keep_since"], f["b_keep_until"] = _since_until_s(hold)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def features(obs):
    P = db.context_features(db.base_features(obs["px"], obs["py"], obs["team"], None, None, obs["L"], obs["W"]),
                            players_only=True)
    C = config_features(obs["px"], obs["py"], obs["team"], obs["L"], obs["W"])
    B = ball_v4_features(obs["px"], obs["py"], obs["team"], obs["fx"], obs["fy"], obs["src"], obs["ax"],
                         obs["L"], obs["W"])
    return pd.concat([P, C, B], axis=1)


def features_v2(obs):
    """D v2 feature set (ATD-like ball only), for the replay."""
    return db.context_features(db.base_features(obs["px"], obs["py"], obs["team"], obs["ax"], obs["ay"],
                                                obs["L"], obs["W"]))


# =============================================================================== 4. models
def make_model(multiclass=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def load_halves(sportec_dir, stops):
    """Sportec halves (trimmed) with bz and per-frame stop type."""
    out = {}
    for mid in SPORTEC_IDS:
        hs = []
        for k, hh in enumerate(db3.load_sportec(os.path.join(sportec_dir, f"{mid}.npz"))):
            h = ("h1", "h2")[k]
            st = np.zeros(len(hh["dead"]), np.int8)
            for r in stops[(stops.mid == mid) & (stops.half == h)].itertuples():
                st[r.a:r.b] = r.type
            hh["stop"] = st
            hs.append(hh)
        out[mid] = hs
    return out


DECODE_GRID = [dict(p_hi=ph, p_lo=pl, min_dur_s=md, smooth_s=sm, merge_s=1.0)
               for ph in (0.8, 0.9, 0.95, 0.98) for pl in (0.4, 0.5, 0.6, 0.7) for md in (2.0, 3.0, 4.0)
               for sm in (0.5, 1.0) if pl < ph]
V2_DECODE = dict(p_hi=0.9, p_lo=0.6, min_dur_s=3.0, smooth_s=1.0, merge_s=1.0)
FP_RUN_PENALTY_S = 2.0      # a false D run also costs 2 s (it splits a possession)


def decode(p, prm):
    return db.decode(p, prm["p_hi"], prm["p_lo"], prm["min_dur_s"], prm["merge_s"], prm["smooth_s"])


def decode_cost(true, pred):
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    nfr = sum(1 for a, b in runs_of(pred) if true[a:b].mean() < 0.5)
    return 2 * fp + fn + FP_RUN_PENALTY_S * FPS * nfr


def evaluate(true, pred, stop_type):
    """Frame + stoppage metrics for one half. stop_type: per-frame true type."""
    tp = int((pred & true).sum())
    r = dict(tp=tp, fp=int((pred & ~true).sum()), fn=int((~pred & true).sum()))
    pr = runs_of(pred)
    rows = []
    for a, b in runs_of(true):
        ov = [(min(b, d) - max(a, c), c, d) for c, d in pr if c < b and d > a]
        if ov:
            _, c, d = max(ov)
            rows.append(dict(type=int(stop_type[a]), dur=(b - a) / FPS, found=True, ds=(c - a) / FPS, de=(d - b) / FPS))
        else:
            rows.append(dict(type=int(stop_type[a]), dur=(b - a) / FPS, found=False, ds=np.nan, de=np.nan))
    r["false_runs"] = sum(1 for c, d in pr if true[c:d].mean() < 0.5)
    r["pred_runs"] = len(pr)
    return r, rows


# ========================================================================= 5. simulation cache
N_TRAIN_SIMS = 2
TRAIN_STEP = 5


def _seed(mid, h, kind, k):
    return (SPORTEC_IDS.index(mid) * 1000 + (0 if h == "h1" else 500) + {"train": 0, "test": 100, "stress": 200}[kind]
            + k) * 7919 + 13


def build_cache(H, work, log=print, threads_note=""):
    """Simulated features for every Sportec half: N_TRAIN_SIMS training draws (subsampled), one nominal
    test draw and one pessimistic (stress) test draw (full resolution). float16 npy files in work/."""
    os.makedirs(work, exist_ok=True)
    cols = None
    for mid in SPORTEC_IDS:
        for k_h, hf in enumerate(H[mid]):
            h = ("h1", "h2")[k_h]
            for kind, n in (("train", N_TRAIN_SIMS), ("test", 1), ("stress", 1)):
                for k in range(n):
                    stem = os.path.join(work, f"{mid}_{h}_{kind}{k}")
                    if os.path.exists(stem + "_y.npy"):
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng(_seed(mid, h, kind, k))
                    p = draw_params(rng, stress=(kind == "stress"))
                    obs = simulate_obs(hf, rng, p)
                    F4, F2 = features(obs), features_v2(obs)
                    T = len(F4)
                    idx = np.arange(int(rng.integers(0, TRAIN_STEP)), T, TRAIN_STEP) if kind == "train" else np.arange(T)
                    np.save(stem + "_X4.npy", F4.values[idx].astype(np.float16))
                    np.save(stem + "_X2.npy", F2.values[idx].astype(np.float16))
                    np.save(stem + "_y.npy", np.stack([hf["dead"][idx].astype(np.int8), hf["stop"][idx]], 1))
                    cols = (list(F4.columns), list(F2.columns))
                    with open(os.path.join(work, "params.jsonl"), "a") as fh:
                        fh.write(json.dumps(dict(mid=mid, half=h, kind=kind, k=k, **{a: (float(b) if not isinstance(b, (bool, np.bool_)) else bool(b)) for a, b in p.items()})) + "\n")
                    log(f"{mid} {h} {kind}{k}: {T} frames, {time.time() - t0:.0f}s")
    if cols is not None:
        with open(os.path.join(work, "columns.json"), "w") as fh:
            json.dump(dict(v4=cols[0], v2=cols[1]), fh)
    with open(os.path.join(work, "columns.json")) as fh:
        return json.load(fh)


def _load(work, mids, kind, which, n=1):
    X, Y = [], []
    for mid in mids:
        for h in ("h1", "h2"):
            for k in range(n):
                stem = os.path.join(work, f"{mid}_{h}_{kind}{k}")
                X.append(np.load(stem + f"_X{which}.npy").astype(np.float32))
                Y.append(np.load(stem + "_y.npy"))
    return X, Y


def train_models(work, mids, which):
    X, Y = _load(work, mids, "train", which, N_TRAIN_SIMS)
    X, Y = np.concatenate(X), np.concatenate(Y)
    m_dead = make_model().fit(X, Y[:, 0])
    m_type = None
    if which == "4":
        d = Y[:, 0] == 1
        m_type = make_model().fit(X[d], Y[d, 1])
    return m_dead, m_type


def run_types(dead, P_type, classes):
    """Per decoded D run: argmax of the mean type probability -> per-frame stop_type."""
    st = np.zeros(len(dead), np.int8)
    for a, b in runs_of(dead):
        st[a:b] = classes[int(np.argmax(P_type[a:b].mean(0)))]
    return st


def choose_decoder(probs, grid=DECODE_GRID):
    """probs: list of (p, true_dead) halves -> grid point with the lowest total cost."""
    best = None
    for prm in grid:
        c = sum(decode_cost(t, decode(p, prm)) for p, t in probs)
        if best is None or c < best[0]:
            best = (c, prm)
    return best[1]


def summarise(rows_frames, rows_stops):
    tp = sum(r["tp"] for r in rows_frames)
    fp = sum(r["fp"] for r in rows_frames)
    fn = sum(r["fn"] for r in rows_frames)
    st = pd.DataFrame(rows_stops)
    f = st[st.found]
    out = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
               stoppages=len(st), found=int(st.found.sum()), found_rate=float(st.found.mean()),
               false_runs=sum(r["false_runs"] for r in rows_frames),
               start_abs_med_s=float(f.ds.abs().median()), start_abs_p90_s=float(f.ds.abs().quantile(.9)),
               end_abs_med_s=float(f.de.abs().median()), end_abs_p90_s=float(f.de.abs().quantile(.9)),
               start_signed_med_s=float(f.ds.median()), end_signed_med_s=float(f.de.median()))
    return out


def per_type_table(rows_stops, label):
    st = pd.DataFrame(rows_stops)
    g = []
    for t, d in st.groupby("type"):
        f = d[d.found]
        g.append(dict(modele=label, type=STOP_NAMES[t], n=len(d), trouves=float(d.found.mean()),
                      debut_med=float(f.ds.abs().median()) if len(f) else np.nan,
                      debut_p90=float(f.ds.abs().quantile(.9)) if len(f) else np.nan,
                      debut_signe_med=float(f.ds.median()) if len(f) else np.nan,
                      fin_med=float(f.de.abs().median()) if len(f) else np.nan,
                      fin_p90=float(f.de.abs().quantile(.9)) if len(f) else np.nan,
                      fin_signe_med=float(f.de.median()) if len(f) else np.nan,
                      type_juste=float(d.type_ok[d.found].astype(float).mean()) if "type_ok" in d and len(f) else np.nan))
    return pd.DataFrame(g)


# Official scorer (possession_metrics.py) from the organisers' repo, cloned next to this one by default.
EVAL_DIR = os.environ.get("MTAH_EVAL_DIR", os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "more-than-a-hack-2027", "evaluation")))


def event_f1(true, pred, iou=0.5):
    """Stoppage-level F1: a predicted D run and a true one match when their IoU >= iou (one to one)."""
    tr, pr = runs_of(true), runs_of(pred)
    pairs = []
    for i, (a, b) in enumerate(tr):
        for j, (c, d) in enumerate(pr):
            if c < b and d > a:
                inter = min(b, d) - max(a, c)
                pairs.append((inter / (max(b, d) - min(a, c)), i, j))
    pairs.sort(reverse=True)
    ui, uj, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < iou:
            break
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        tp += 1
    return tp, len(pr) - tp, len(tr) - tp


def spell_proxy(true, pred, poss, pm):
    """spellF1 of the official scorer with ORACLE Sportec teams (BallPossession 1/2 -> H/A): only the
    dead-ball separators differ between truth and prediction."""
    team = np.where(poss == 1, 0, 1).astype(np.int8)
    gt = np.where(true, 2, team).astype(np.int8)
    pr = np.where(pred, 2, team).astype(np.int8)
    return pm.spell_scores(gt, pr, [(0, len(gt))])["mean_f1"]


def _oof_fold(work, mid, which):
    path = os.path.join(work, f"oof{which}_{mid}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        out = {}
        for kind in ("test", "stress"):
            for k in (0, 1):
                key = f"{kind}_{k}"
                r = dict(p=z[key + "_p"], dead=z[key + "_dead"], stop=z[key + "_stop"])
                if which == "4":
                    r["ptype"], r["classes"] = z[key + "_ptype"], z["classes"]
                out[(kind, k)] = r
        return out
    rest = [m for m in SPORTEC_IDS if m != mid]
    m_dead, m_type = train_models(work, rest, which)
    out, sav = {}, {}
    for kind in ("test", "stress"):
        X, Y = _load(work, [mid], kind, which)
        for k, (x, y) in enumerate(zip(X, Y)):
            r = dict(p=m_dead.predict_proba(x)[:, 1].astype(np.float32), dead=y[:, 0] == 1, stop=y[:, 1])
            if m_type is not None:
                r["ptype"] = m_type.predict_proba(x).astype(np.float16)
                r["classes"] = m_type.classes_
            out[(kind, k)] = r
            for a, b in r.items():
                if a != "classes":
                    sav[f"{kind}_{k}_{a}"] = b
    if m_type is not None:
        sav["classes"] = m_type.classes_
    np.savez_compressed(path + ".tmp.npz", **sav)
    os.replace(path + ".tmp.npz", path)
    return out


def cross_validate(sportec_dir, events_dir, work, out_dir, log=print):
    sys.path.insert(0, EVAL_DIR)
    import possession_metrics as pm  # official scorer code (no ground truth is read)
    os.makedirs(out_dir, exist_ok=True)
    stops, align = sportec_tables(sportec_dir, events_dir)
    stops.to_csv(os.path.join(out_dir, "sportec_arrets.csv"), index=False)
    align.to_csv(os.path.join(out_dir, "sportec_alignement.csv"), index=False)
    H = load_halves(sportec_dir, stops)
    build_cache(H, work, log)
    oof = {}
    for mid in SPORTEC_IDS:
        t0 = time.time()
        for which in ("4", "2"):
            for key, r in _oof_fold(work, mid, which).items():
                oof[(which, key[0], mid, key[1])] = r
        log(f"fold {mid}: {time.time() - t0:.0f}s")
    res, per_type, per_match = {}, [], []
    configs = (("4", "test", "cv"), ("4", "stress", "cv"), ("2", "test", "v2"), ("2", "stress", "v2"),
               ("2", "test", "cv"), ("2", "stress", "cv"))
    for which, kind, dec_mode in configs:
        label = f"D{'v4' if which == '4' else 'v2'}_{kind}_{'decodeur_v2' if dec_mode == 'v2' else 'decodeur_cv'}"
        fr, sr, ev_c, sp = [], [], [0, 0, 0], []
        for mid in SPORTEC_IDS:
            if dec_mode == "v2":
                prm = V2_DECODE
            else:
                pool = [(oof[(which, "test", m, k)]["p"], oof[(which, "test", m, k)]["dead"])
                        for m in SPORTEC_IDS if m != mid for k in (0, 1)]
                prm = choose_decoder(pool)
            fm, sm, em, spm = [], [], [0, 0, 0], []
            for k in (0, 1):
                r = oof[(which, kind, mid, k)]
                pred = decode(r["p"], prm)
                f1, s1 = evaluate(r["dead"], pred, r["stop"])
                if which == "4":
                    st = run_types(pred, r["ptype"].astype(np.float32), r["classes"])
                    for row, (a, b) in zip(s1, runs_of(r["dead"])):
                        if row["found"]:
                            vals = st[a:b][pred[a:b]]
                            row["type_ok"] = bool(len(vals) and np.bincount(vals).argmax() == row["type"])
                e = event_f1(r["dead"], pred)
                em = [u + v for u, v in zip(em, e)]
                spm.append(spell_proxy(r["dead"], pred, H[mid][k]["poss"], pm))
                fm.append(f1)
                sm += s1
            summ = summarise(fm, sm)
            summ.update(event_f1_iou05=2 * em[0] / max(2 * em[0] + em[1] + em[2], 1), spell_proxy=float(np.mean(spm)))
            per_match.append(dict(modele=label, match=mid, **summ, decodeur=json.dumps(prm)))
            fr += fm
            sr += sm
            ev_c = [u + v for u, v in zip(ev_c, em)]
            sp += spm
        out = summarise(fr, sr)
        out["frame_f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9)
        out["event_f1_iou05"] = 2 * ev_c[0] / max(2 * ev_c[0] + ev_c[1] + ev_c[2], 1)
        out["event_tp_fp_fn"] = ev_c
        out["spell_proxy_mean"] = float(np.mean(sp))
        out["false_runs_per_match"] = out["false_runs"] / len(SPORTEC_IDS)
        if which == "4":
            st_df = pd.DataFrame(sr)
            out["type_accuracy"] = float(st_df.type_ok[st_df.found].astype(float).mean())
        res[label] = out
        per_type.append(per_type_table(sr, label))
        log(f"{label}: " + json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in out.items()}))
    pd.DataFrame(per_match).to_csv(os.path.join(out_dir, "cv_par_match.csv"), index=False)
    pd.concat(per_type).to_csv(os.path.join(out_dir, "cv_par_type.csv"), index=False)
    # type confusion (v4, nominal test)
    conf = []
    for mid in SPORTEC_IDS:
        prm = json.loads([r for r in per_match if r["modele"] == "Dv4_test_decodeur_cv" and r["match"] == mid][0]["decodeur"])
        for k in (0, 1):
            r = oof[("4", "test", mid, k)]
            pred = decode(r["p"], prm)
            st = run_types(pred, r["ptype"].astype(np.float32), r["classes"])
            for a, b in runs_of(r["dead"]):
                vals = st[a:b][pred[a:b]]
                if len(vals):
                    conf.append((STOP_NAMES[int(r["stop"][a])], STOP_NAMES[int(np.bincount(vals).argmax())]))
    pd.crosstab(pd.Series([c[0] for c in conf], name="vrai"), pd.Series([c[1] for c in conf], name="predit")).to_csv(
        os.path.join(out_dir, "cv_confusion_type.csv"))
    pool = [(oof[("4", "test", m, k)]["p"], oof[("4", "test", m, k)]["dead"]) for m in SPORTEC_IDS for k in (0, 1)]
    final = choose_decoder(pool)
    with open(os.path.join(out_dir, "decode_v4.json"), "w") as fh:
        json.dump(final, fh)
    with open(os.path.join(out_dir, "cv_resume.json"), "w") as fh:
        json.dump(dict(resultats=res, decodeur_final=final, critere="2*faux D + D manques + 2 s par faux arret (frames)"),
                  fh, indent=1)
    return res, final


# ============================================================================ 6. BAR-ATH apply
def apply_match(data_dir, ball_npz, work, sportec_dir, events_dir, decode_prm, cache_dir=None, log=print):
    cols = build_cache(load_halves(sportec_dir, sportec_tables(sportec_dir, events_dir)[0]), work, log)
    m_dead, m_type = train_models(work, list(SPORTEC_IDS), "4")
    P, B, slot_team = v1.load_tactical(data_dir, cache_dir)
    fb = np.load(ball_npz)
    Lp, Wp = v1.PITCH
    N = v1.N_FRAMES
    p_dead = np.full(N, np.nan, np.float32)
    dead = np.zeros(N, bool)
    stop_type = np.zeros(N, np.int8)
    p_type = np.zeros((N, 8), np.float16)
    graded = np.zeros(N, bool)
    runs_tab = []
    for a, b in v1.HALVES:
        graded[a:b] = True
        obs = dict(px=P[a:b, :, 0] * Lp, py=P[a:b, :, 1] * Wp, team=slot_team, fx=fb["x"][a:b].astype(float),
                   fy=fb["y"][a:b].astype(float), src=fb["source"][a:b].astype(np.int8),
                   ax=B[a:b, 0].astype(float) * Lp, ay=B[a:b, 1].astype(float) * Wp, L=Lp, W=Wp)
        F = features(obs)[cols["v4"]]
        p = m_dead.predict_proba(F.values.astype(np.float32))[:, 1]
        d = decode(p, decode_prm)
        pt = m_type.predict_proba(F.values.astype(np.float32))
        st = run_types(d, pt, m_type.classes_)
        p_dead[a:b], dead[a:b], stop_type[a:b] = p, d, st
        for j, c in enumerate(m_type.classes_):
            p_type[a:b, int(c) - 1] = pt[:, j]
        for s, e in runs_of(d):
            runs_tab.append((a + s, a + e, int(st[s]), float(p[s:e].mean())))
        log(f"half {a}-{b}: D {d.mean():.1%}, {len(runs_of(d))} stoppages")
    return dict(p_dead=p_dead, dead=dead, stop_type=stop_type, graded=graded, p_type=p_type,
                runs=np.array(runs_tab, np.float64).reshape(-1, 4))



def label_free_stats(out, ball_npz=None, v2_csv=None):
    """Statistics of the BAR-ATH output that need no label (and decide nothing)."""
    g, d = out["graded"], out["dead"]
    runs = out["runs"]
    dur = (runs[:, 1] - runs[:, 0]) / FPS if len(runs) else np.zeros(0)
    s = dict(frames_notees=int(g.sum()), part_D_notees=float(d[g].mean()), n_arrets=int(len(runs)),
             duree_med_s=float(np.median(dur)), duree_p10_s=float(np.percentile(dur, 10)),
             duree_p90_s=float(np.percentile(dur, 90)), arrets_courts_lt5s=int((dur < 5).sum()),
             arrets_par_mi_temps=[int(((runs[:, 0] >= a) & (runs[:, 0] < b)).sum()) for a, b in v1.HALVES])
    pt = {}
    for c in np.unique(runs[:, 2]).astype(int) if len(runs) else []:
        m = runs[:, 2] == c
        pt[STOP_NAMES[c]] = dict(n=int(m.sum()), duree_med_s=float(np.median(dur[m])), part_frames=float(dur[m].sum() * FPS / g.sum()))
    s["par_type"] = pt
    if ball_npz is not None:
        fb = np.load(ball_npz)
        src = fb["source"]
        s["balle_visible_dans_D"] = float((src[d & g] > 0).mean())
        s["balle_visible_hors_D"] = float((src[~d & g] > 0).mean())
        xs, ys = fb["x"].astype(float), fb["y"].astype(float)
        Lp, Wp = v1.PITCH
        edge = np.fmin(np.fmin(xs, Lp - xs), np.fmin(ys, Wp - ys))
        out_m = np.nan_to_num(edge, nan=1) < -0.5
        spd = np.r_[0, np.hypot(np.diff(xs), np.diff(ys)) * FPS]
        n_out = n_kick = 0
        for a, b, _, _ in runs:
            a, b = int(a), int(b)
            n_out += bool(out_m[max(a - 50, 0):a + 50].any())
            w = np.nan_to_num(spd[max(b - 50, 0):b + 50], nan=0)
            n_kick += bool((w > KICK_V).any())
        s["arrets_avec_sortie_balle_pm2s"] = float(n_out / max(len(runs), 1))
        s["arrets_avec_balle_rapide_fin_pm2s"] = float(n_kick / max(len(runs), 1))
    if v2_csv is not None:
        lab = pd.read_csv(v2_csv)["label"].astype(str).values
        d2 = lab == "D"
        s["v2_part_D_notees"] = float(d2[g].mean())
        s["accord_frames_v2"] = float((d2[g] == d[g]).mean())
        s["v4_D_et_v2_vivant_part"] = float((d & ~d2 & g).sum() / g.sum())
        s["v4_vivant_et_v2_D_part"] = float((~d & d2 & g).sum() / g.sum())
        r2 = []
        for a, b in v1.HALVES:
            r2 += [(a + u, a + v) for u, v in runs_of(d2[a:b])]
        s["v2_n_arrets"] = len(r2)
        s["v4_arrets_chevauchant_v2"] = float(np.mean([d2[int(a):int(b)].any() for a, b, _, _ in runs])) if len(runs) else np.nan
        s["v2_arrets_chevauchant_v4"] = float(np.mean([d[a:b].any() for a, b in r2])) if r2 else np.nan
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("events", "cv", "apply"):
        s = sub.add_parser(name)
        s.add_argument("--sportec", required=True, help="directory with the 7 J03W*.npz (fetch_sportec.py)")
        s.add_argument("--events-dir", required=True, help="directory for the DFL event xml (downloaded if missing)")
        if name == "events":
            s.add_argument("--out-dir", required=True)
        if name in ("cv", "apply"):
            s.add_argument("--work", required=True, help="cache of simulated Sportec features")
        if name == "cv":
            s.add_argument("--out-dir", required=True)
        if name == "apply":
            s.add_argument("--data", required=True)
            s.add_argument("--ball", required=True, help="balle_fusionnee.npz (x, y, source)")
            s.add_argument("--out", required=True)
            s.add_argument("--decode-json", default=None, help="decoder from `cv` (decode_v4.json)")
            s.add_argument("--cache", default=None)
            s.add_argument("--v2-csv", default=None, help="optional D v2 CSV for the label-free agreement stats")
            s.add_argument("--stats-out", default=None, help="optional JSON of label-free statistics")
    a = ap.parse_args()
    if a.cmd == "events":
        os.makedirs(a.out_dir, exist_ok=True)
        S, A = sportec_tables(a.sportec, a.events_dir)
        S.to_csv(os.path.join(a.out_dir, "sportec_arrets.csv"), index=False)
        A.to_csv(os.path.join(a.out_dir, "sportec_alignement.csv"), index=False)
        print(A.to_string())
    elif a.cmd == "cv":
        res, final = cross_validate(a.sportec, a.events_dir, a.work, a.out_dir, log=lambda s: print(s, flush=True))
        print(json.dumps(res, indent=1), "\nfinal decoder:", final)
    else:
        prm = json.load(open(a.decode_json)) if a.decode_json else dict(DECODE_DEFAULT)
        out = apply_match(a.data, a.ball, a.work, a.sportec, a.events_dir, prm, a.cache, log=lambda s: print(s, flush=True))
        np.savez_compressed(a.out, **out, stop_type_codes=json.dumps(STOP_NAMES), decode=json.dumps(prm))
        print("wrote", a.out)
        if a.stats_out:
            st = label_free_stats(out, a.ball, a.v2_csv)
            with open(a.stats_out, "w") as fh:
                json.dump(st, fh, indent=1, ensure_ascii=False)
            print(json.dumps(st, indent=1, ensure_ascii=False))


DECODE_DEFAULT = dict(p_hi=0.95, p_lo=0.6, min_dur_s=3.0, smooth_s=1.0, merge_s=1.0)

if __name__ == "__main__":
    main()

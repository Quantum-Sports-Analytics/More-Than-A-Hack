"""Dead-ball detector D v5: multi-source (Sportec + SkillCorner + Metrica) training and validation.

External data only (nothing is fitted, tuned or chosen on the provided Barca-Athletic match):
* Sportec Open DFL (Bassek et al. 2025, Sci Data 12:195, CC-BY 4.0), 7 matches, 25 fps, BallStatus labels,
  ball SIMULATED by dead_v4.simulate_obs (frozen dead ball re-synthesised, randomised degradation).
* SkillCorner Open Data (MIT), 19 A-League matches (1959846 excluded: faulty labels), broadcast tracking
  10 fps resampled to 25 fps, REAL detected ball (used as the "video" source), labels derived from
  phases_of_play (weak boundaries, h_label_conf). Frames with < 10 positioned players (untracked TV
  shots) are excluded from training and from the main frame metrics; no "number of tracked players"
  feature is used.
* Metrica Sports sample data (3 matches, 25 fps, no explicit licence: internal use, source cited). Its
  semi-manual ball is removed during stoppages (label leak), so it is NEVER used raw: it is frozen like
  Sportec during stoppages and re-simulated (BZ = 0), or dropped entirely ("sans balle" mode).

Everything is computed per frame at 25 fps and STORED / LEARNT / DECODED at 5 Hz (one frame in five).
Features = the 217 D v4 features + long-range context (Lang et al. 2022: aggregated collective
statistics shifted by +-30..80 s, 30/60 s means) + restart-trigger timings (Vidal-Codina et al. 2022:
time since / until a player on the touchline, at a corner flag, keeper in his 6 m box with an empty box,
kick-off shape, still ball).

CLI
  python possession/dead_v5.py cache  --work <dir> --ext <dir with SC_*.npz, metrica_*.npz> --sportec <dir> \
        --stops <sportec_arrets.csv> [--worker i --nworkers n]
  python possession/dead_v5.py eval   --work <dir> --out-dir <dir>         (zero-shot v4/v2, v5 LOMO, transfer)
  python possession/dead_v5.py apply  --work <dir> --data <data> --ball balle_fusionnee.npz --model <pkl> \
        --decode-json decode_v5.json --out dead_v5.npz [--v2-csv ...] [--stats-out ...]
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1  # noqa: E402
import dead_ball as db  # noqa: E402
import dead_bounds as db3  # noqa: E402
import dead_v4 as dv  # noqa: E402

FPS = 25
STEP = 5                      # 25 fps -> 5 Hz
HZ = FPS // STEP
SC_EXCLUDE = ("1959846",)
MIN_PLAYERS = 10
STOP_NAMES = dv.STOP_NAMES
runs_of = dv.runs_of


# ================================================================================ loading
def _trim(bs):
    al = np.flatnonzero(bs == 1)
    return al[0], al[-1] + 1


def load_ext(path, kind):
    """SkillCorner / Metrica npz (Sportec layout + extras) -> list of half dicts."""
    z = np.load(path)
    L, W = float(z["L"]), float(z["W"])
    out = []
    for h in ("h1", "h2"):
        bs = z[f"{h}_BS"]
        a, b = _trim(bs)
        px = (z[f"{h}_PX"][a:b] + L / 2).astype(np.float32)
        py = (z[f"{h}_PY"][a:b] + W / 2).astype(np.float32)
        npl = np.isfinite(px).sum(1)
        d = dict(px=px, py=py, team=z[f"{h}_team"], bx=(z[f"{h}_BX"][a:b] + L / 2).astype(np.float32),
                 by=(z[f"{h}_BY"][a:b] + W / 2).astype(np.float32), bs=bs[a:b].copy(), dead=bs[a:b] == 0,
                 stop=z[f"{h}_stop"][a:b].astype(np.int8), poss=z[f"{h}_BP"][a:b].astype(np.int8),
                 conf=z[f"{h}_label_conf"][a:b].astype(np.uint8), L=L, W=W,
                 valid=(npl >= MIN_PLAYERS) & (bs[a:b] >= 0), half=h)
        bz = z[f"{h}_BZ"][a:b].astype(np.float32)
        d["bz"] = np.where(np.isfinite(bz), bz, 0).astype(np.float32)
        if kind == "metrica":
            d["bx"], d["by"] = _freeze_like_sportec(d["bx"], d["by"], d["dead"])
        out.append(d)
    return out


def _freeze_like_sportec(bx, by, dead):
    """Metrica's ball is removed / hand-placed during stoppages: freeze it at the last live position
    (Sportec convention) and make sure a restart spot exists, so that synth_dead_ball re-synthesises
    the dead-ball trajectory exactly as for Sportec (no information from the provided ball)."""
    bx, by = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    T = len(bx)
    # fill live NaNs by linear interpolation (the simulation then re-degrades the ball)
    ok = np.isfinite(bx)
    if ok.sum() > 2:
        idx = np.arange(T)
        bx = np.interp(idx, idx[ok], bx[ok])
        by = np.interp(idx, idx[ok], by[ok])
    for a, b in runs_of(dead):
        p0 = (bx[a - 1], by[a - 1]) if a > 0 else (bx[a], by[a])
        bx[a:b], by[a:b] = p0
    return bx.astype(np.float32), by.astype(np.float32)


def load_sportec_halves(sportec_dir, stops, mids=None):
    out = {}
    for mid in (mids or dv.SPORTEC_IDS):
        hs = []
        for k, hh in enumerate(db3.load_sportec(os.path.join(sportec_dir, f"{mid}.npz"))):
            h = ("h1", "h2")[k]
            st = np.zeros(len(hh["dead"]), np.int8)
            for r in stops[(stops.mid == mid) & (stops.half == h)].itertuples():
                st[r.a:r.b] = r.type
            hh.update(stop=st, valid=np.ones(len(st), bool), conf=np.full(len(st), 2, np.uint8),
                      bs=hh["dead"].astype(np.int8) ^ 1, half=h)
            hs.append(hh)
        out[mid] = hs
    return out


# ============================================================================ observations
def obs_real_ball(hf):
    """SkillCorner: detected ball = video source; <= 12-frame bridges; ATD channel = same detections."""
    fx, fy = hf["bx"].astype(float).copy(), hf["by"].astype(float).copy()
    T = len(fx)
    vis = np.isfinite(fx)
    src = np.where(vis, 1, 0).astype(np.int8)
    for a, b in runs_of(~vis):
        if a > 0 and b < T and (b - a) <= dv.VIDEO_MAX_BRIDGE:
            w = np.arange(1, b - a + 1) / (b - a + 1)
            fx[a:b] = fx[a - 1] + w * (fx[b] - fx[a - 1])
            fy[a:b] = fy[a - 1] + w * (fy[b] - fy[a - 1])
            src[a:b] = 2
    return dict(px=hf["px"], py=hf["py"], team=hf["team"], fx=fx, fy=fy, src=src,
                ax=hf["bx"].astype(float), ay=hf["by"].astype(float), L=hf["L"], W=hf["W"])


def obs_no_ball(hf):
    T = len(hf["px"])
    nan = np.full(T, np.nan)
    return dict(px=hf["px"], py=hf["py"], team=hf["team"], fx=nan, fy=nan.copy(), src=np.zeros(T, np.int8),
                ax=nan.copy(), ay=nan.copy(), L=hf["L"], W=hf["W"])


def mask_ball_windows(obs, rng, frac=0.5, win_s=60.0):
    """Training augmentation (C2): the ball (video and ATD) is fully hidden in random windows."""
    T = len(obs["fx"])
    w = int(win_s * FPS)
    for s in range(0, T, w):
        if rng.random() < frac:
            for k in ("fx", "fy", "ax", "ay"):
                obs[k] = obs[k].copy()
                obs[k][s:s + w] = np.nan
            obs["src"] = obs["src"].copy()
            obs["src"][s:s + w] = 0
    return obs


# ================================================================================ features
LONG_CORE = ("spd_med", "frac_slow", "disp2_med", "cspd_max", "spread_max", "ext_x", "b_vis")


def extra_features(F):
    out = {}
    for c in LONG_CORE:
        ref = pd.Series(F[c].values.astype(np.float64)).rolling(2 * FPS, center=True, min_periods=1).median()
        for sh in (30, 45, 60, 80):
            out[f"L_{c}_p{sh}"] = ref.shift(sh * FPS).values
            out[f"L_{c}_f{sh}"] = ref.shift(-sh * FPS).values
        for w in (30, 60):
            out[f"L_{c}_m{w}"] = ref.rolling(w * FPS, center=True, min_periods=1).mean().values
        out[f"L_{c}_dev30"] = (ref - ref.rolling(30 * FPS, center=True, min_periods=1).mean()).values
    T = len(F)
    with np.errstate(invalid="ignore"):
        trig = {
            "touch": F["c_touch_min_m1"].values < 0.5,
            "corner": F["c_corner_min_m1"].values < 1.5,
            "gk6": (F["c_gk_in6_m1"].values > 0.5) & (F["c_gk6_opp_in_box_m1"].values < 0.5),
            "kickoff": (F["c_own_half_min_m1"].values > 0.95) & (F["c_centre_min_m1"].values < 2.0),
            "still": F["b_still_m4"].values > 0.5,
        }
    for k, m in trig.items():
        m = np.nan_to_num(m.astype(float)) > 0
        rise = np.r_[m[:1], m[1:] & ~m[:-1]]
        fall = np.r_[~m[1:] & m[:-1], False]
        out[f"T_{k}_on_since"], out[f"T_{k}_on_until"] = db3._time_to_events(rise, T, 60.0)
        out[f"T_{k}_off_since"], out[f"T_{k}_off_until"] = db3._time_to_events(fall, T, 60.0)
        out[f"T_{k}_m10"] = dv._roll(m, 10)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in out.items()})


def features_v5(obs):
    F = dv.features(obs)
    return pd.concat([F, extra_features(F)], axis=1)


# =================================================================================== cache
def cache_specs(ext_dir):
    sp = []
    for mid in dv.SPORTEC_IDS:
        for h in ("h1", "h2"):
            for var in ("train0", "train1", "test0", "stress0"):
                sp.append(("sportec", mid, h, var))
    for f in sorted(glob.glob(os.path.join(ext_dir, "metrica_*.npz"))):
        mid = os.path.basename(f)[:-4]
        for h in ("h1", "h2"):
            for var in ("train0", "train1", "test0", "stress0", "noball"):
                sp.append(("metrica", mid, h, var))
    for f in sorted(glob.glob(os.path.join(ext_dir, "SC_*.npz"))):
        mid = os.path.basename(f)[:-4]
        if mid[3:] in SC_EXCLUDE:
            continue
        for h in ("h1", "h2"):
            for var in ("real", "mask", "noball"):
                sp.append(("skillcorner", mid, h, var))
    return sp


def _seed(src, mid, h, var):
    if src == "sportec":                                   # same draws as dead_v4's cache
        kind, k = var[:-1], int(var[-1])
        return dv._seed(mid, h, kind, k)
    import zlib
    return zlib.crc32(f"{src}|{mid}|{h}|{var}".encode())


def build_one(spec, halves, work):
    src, mid, h, var = spec
    stem = os.path.join(work, f"{src}__{mid}__{h}__{var}")
    if os.path.exists(stem + "_y.npy"):
        return None
    hf = halves[h]
    rng = np.random.default_rng(_seed(src, mid, h, var))
    if src in ("sportec", "metrica") and var != "noball":
        p = dv.draw_params(rng, stress=var.startswith("stress"))
        obs = dv.simulate_obs(hf, rng, p)
        if var == "train1":
            obs = mask_ball_windows(obs, rng)
    elif var == "noball":
        obs = obs_no_ball(hf)
    else:
        obs = obs_real_ball(hf)
        if var == "mask":
            obs = mask_ball_windows(obs, rng)
    F5 = features_v5(obs)
    F2 = dv.features_v2(obs)
    idx = np.arange(0, len(F5), STEP)
    np.save(stem + "_X5.npy", F5.values[idx].astype(np.float16))
    np.save(stem + "_X2.npy", F2.values[idx].astype(np.float16))
    y = np.stack([hf["dead"][idx].astype(np.int8), hf["stop"][idx], hf["valid"][idx].astype(np.int8),
                  hf["conf"][idx].astype(np.int8), hf["poss"][idx].astype(np.int8), hf["bs"][idx].astype(np.int8)], 1)
    np.save(stem + "_y.npy", y)
    return list(F5.columns), list(F2.columns)


def run_cache(work, ext_dir, sportec_dir, stops_csv, worker=0, nworkers=1, log=print):
    os.makedirs(work, exist_ok=True)
    specs = cache_specs(ext_dir)[worker::nworkers]
    stops = pd.read_csv(stops_csv)
    loaded = {}
    for spec in specs:
        src, mid, h, var = spec
        key = (src, mid)
        if key not in loaded:
            loaded.clear()
            if src == "sportec":
                hs = load_sportec_halves(sportec_dir, stops, [mid])[mid]
            else:
                hs = load_ext(os.path.join(ext_dir, mid + ".npz"), src)
            loaded[key] = {("h1", "h2")[i]: x for i, x in enumerate(hs)}
        t0 = time.time()
        cols = build_one(spec, loaded[key], work)
        if cols is not None:
            with open(os.path.join(work, "columns_v5.json"), "w") as fh:
                json.dump(dict(v5=cols[0], v2=cols[1]), fh)
            log(f"{spec}: {time.time() - t0:.0f}s")


# ================================================================================ decoders
def fill_invalid(p, valid):
    """Probabilities on invalid frames (no positioned players) are unknown: linear interpolation."""
    p = np.asarray(p, np.float64).copy()
    if valid.all() or not valid.any():
        return p
    idx = np.arange(len(p))
    p[~valid] = np.interp(idx[~valid], idx[valid], p[valid])
    return p


def hyst(p, prm, hz=HZ):
    """Hysteresis decoder (dead_ball.decode semantics) with durations in seconds at rate hz."""
    ps = pd.Series(p).rolling(max(1, int(round(prm["smooth_s"] * hz))), center=True, min_periods=1).mean().values
    out = np.zeros(len(p), bool)
    for a, b in runs_of(ps > prm["p_lo"]):
        if ps[a:b].max() > prm["p_hi"]:
            out[a:b] = True
    for a, b in runs_of(~out):
        if a > 0 and b < len(out) and (b - a) < prm["merge_s"] * hz:
            out[a:b] = True
    for a, b in runs_of(out):
        if (b - a) < prm["min_dur_s"] * hz:
            out[a:b] = False
    return out


def hsmm(p, prm, hz=HZ):
    """Two-state semi-Markov Viterbi: D runs >= dmin_s, LIVE runs >= lmin_s, switch cost lam.

    Emissions: w*log p + b (dead), w*log(1-p) (live). Recurrence on 'in state with run length >= min':
      D(t) = max(D(t-1) + eD[t], Lv(t-nd) - lam + sum eD[t-nd+1..t]),  same for Lv.
    """
    eps = 1e-4
    pc = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    eD = prm["w"] * np.log(pc) + prm["b"]
    eL = prm["w"] * np.log1p(-pc)
    nd, nl = max(1, int(round(prm["dmin_s"] * hz))), max(1, int(round(prm["lmin_s"] * hz)))
    T = len(pc)
    cD, cL = np.r_[0, np.cumsum(eD)], np.r_[0, np.cumsum(eL)]
    D = np.full(T, -np.inf)
    Lv = np.full(T, -np.inf)
    chD = np.zeros(T, np.int8)
    chL = np.zeros(T, np.int8)
    lam = prm["lam"]
    for t in range(T):
        # stay
        sd = D[t - 1] + eD[t] if t > 0 else -np.inf
        sl = Lv[t - 1] + eL[t] if t > 0 else -np.inf
        # enter (run [t-nd+1, t]) from the other state (or from the start)
        s0 = t - nd + 1
        if s0 == 0:
            ed = cD[t + 1]
        elif s0 > 0:
            ed = Lv[s0 - 1] - lam + cD[t + 1] - cD[s0]
        else:
            ed = -np.inf
        s1 = t - nl + 1
        if s1 == 0:
            el = cL[t + 1]
        elif s1 > 0:
            el = D[s1 - 1] - lam + cL[t + 1] - cL[s1]
        else:
            el = -np.inf
        if ed > sd:
            D[t], chD[t] = ed, 1
        else:
            D[t] = sd
        if el > sl:
            Lv[t], chL[t] = el, 1
        else:
            Lv[t] = sl
    out = np.zeros(T, bool)
    t = T - 1
    state = 1 if D[t] >= Lv[t] else 0
    while t >= 0:
        if state == 1:
            if chD[t]:
                s0 = t - nd + 1
                out[s0:t + 1] = True
                t, state = s0 - 1, 0
            else:
                out[t] = True
                t -= 1
        else:
            if chL[t]:
                t, state = t - nl, 1
            else:
                t -= 1
    return out


HYST_GRID = [dict(kind="hyst", p_hi=ph, p_lo=pl, min_dur_s=md, smooth_s=sm, merge_s=1.0)
             for ph in (0.8, 0.9, 0.95) for pl in (0.4, 0.5, 0.6) for md in (2.0, 3.0, 4.0) for sm in (0.6, 1.0, 2.0)]
HSMM_GRID = [dict(kind="hsmm", w=w, lam=lam, b=b, dmin_s=dm, lmin_s=2.0)
             for w in (0.2, 0.5, 1.0) for lam in (1.0, 4.0, 12.0) for b in (-0.2, 0.0, 0.2) for dm in (3.0, 4.0)]


def decode5(p, prm, hz=HZ):
    return hsmm(p, prm, hz) if prm["kind"] == "hsmm" else hyst(p, prm, hz)


# ================================================================================ metrics
def _cost(true, pred, ok):
    fp = int((pred & ~true & ok).sum())
    fn = int((~pred & true & ok).sum())
    nfr = sum(1 for a, b in runs_of(pred) if (true[a:b] | ~ok[a:b]).mean() < 0.5)
    return 2 * fp + fn + 2.0 * HZ * nfr


def choose(units, grid):
    """units: list of dicts with p (filled), dead, ok -> grid point with the lowest pooled cost."""
    best = None
    for prm in grid:
        c = sum(_cost(u["dead"], decode5(u["p"], prm), u["ok"]) for u in units)
        if best is None or c < best[0]:
            best = (c, prm)
    return best[1]


def half_metrics(u, pred, pm=None, ptype=None, classes=None):
    """u: dead, ok (bs >= 0), valid (players), stop, poss; arrays at 5 Hz."""
    true, ok, valid = u["dead"], u["ok"], u["valid"] & u["ok"]
    r = {}
    for nm, m in (("v", valid), ("all", ok)):
        r[f"tp_{nm}"] = int((pred & true & m).sum())
        r[f"fp_{nm}"] = int((pred & ~true & m).sum())
        r[f"fn_{nm}"] = int((~pred & true & m).sum())
    tr = runs_of(true & ok)
    pr = runs_of(pred)
    r["false_runs"] = sum(1 for a, b in pr if (true[a:b] | ~ok[a:b]).mean() < 0.5)
    r["pred_runs"] = len(pr)
    tp, fp, fn = dv.event_f1(true & ok, pred)
    r["ev_tp"], r["ev_fp"], r["ev_fn"] = tp, fp, fn
    st = None
    if ptype is not None:
        st = dv.run_types(pred, ptype, classes)
    rows = []
    w1 = HZ
    for a, b in tr:
        ov = [(min(b, d) - max(a, c), c, d) for c, d in pr if c < b and d > a]
        row = dict(type=int(u["stop"][a]), found=bool(ov), ds=np.nan, de=np.nan, type_ok=np.nan,
                   start_ok=bool(valid[max(a - w1, 0):a + w1].all()), end_ok=bool(valid[max(b - w1, 0):b + w1].all()))
        if ov:
            _, c, d = max(ov)
            row["ds"], row["de"] = (c - a) / HZ, (d - b) / HZ
            if st is not None:
                vals = st[a:b][pred[a:b]]
                row["type_ok"] = float(len(vals) > 0 and np.bincount(vals).argmax() == row["type"])
        rows.append(row)
    if pm is not None and u.get("poss") is not None:
        poss = pd.Series(np.where(u["poss"] > 0, u["poss"], np.nan)).ffill().bfill().fillna(1).values
        r["spell"] = dv.spell_proxy(true & ok, pred, poss.astype(np.int8), pm)
    return r, rows


def summarise5(frs, rows, n_matches):
    g = lambda k: sum(f[k] for f in frs)  # noqa: E731
    st = pd.DataFrame(rows)
    out = {}
    for nm in ("v", "all"):
        tp, fp, fn = g(f"tp_{nm}"), g(f"fp_{nm}"), g(f"fn_{nm}")
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"P_{nm}"], out[f"R_{nm}"], out[f"F1_{nm}"] = P, R, 2 * P * R / max(P + R, 1e-9)
    tp, fp, fn = g("ev_tp"), g("ev_fp"), g("ev_fn")
    out["F1_arret_iou05"] = 2 * tp / max(2 * tp + fp + fn, 1)
    out["faux_arrets_par_match"] = g("false_runs") / n_matches
    out["arrets"], out["trouves"] = len(st), float(st.found.mean())
    fs = st[st.found & st.start_ok]
    fe = st[st.found & st.end_ok]
    out["debut_p50"], out["debut_p90"] = float(fs.ds.abs().median()), float(fs.ds.abs().quantile(.9))
    out["fin_p50"], out["fin_p90"] = float(fe.de.abs().median()), float(fe.de.abs().quantile(.9))
    out["debut_1s"], out["fin_1s"] = float((fs.ds.abs() <= 1).mean()), float((fe.de.abs() <= 1).mean())
    if st.type_ok.notna().any():
        out["type_juste"] = float(st.type_ok[st.found].mean())
    sp = [f["spell"] for f in frs if "spell" in f]
    if sp:
        out["spell_proxy"] = float(np.mean(sp))
    return out


def per_type5(rows, label):
    st = pd.DataFrame(rows)
    res = []
    for t, d in st.groupby("type"):
        fs, fe = d[d.found & d.start_ok], d[d.found & d.end_ok]
        res.append(dict(config=label, type=STOP_NAMES.get(int(t), str(t)), n=len(d), trouves=float(d.found.mean()),
                        debut_p50=float(fs.ds.abs().median()) if len(fs) else np.nan,
                        debut_p90=float(fs.ds.abs().quantile(.9)) if len(fs) else np.nan,
                        fin_p50=float(fe.de.abs().median()) if len(fe) else np.nan,
                        fin_p90=float(fe.de.abs().quantile(.9)) if len(fe) else np.nan,
                        type_juste=float(d.type_ok[d.found].mean()) if d.type_ok.notna().any() else np.nan))
    return pd.DataFrame(res)


# ================================================================================ data sets
def units(work, src, var, mids=None):
    out = []
    for f in sorted(glob.glob(os.path.join(work, f"{src}__*__{var}_y.npy"))):
        _, mid, h, _ = os.path.basename(f)[:-6].split("__")
        if mids is not None and mid not in mids:
            continue
        out.append((mid, h, f[:-6]))
    return out


def load_y(stem):
    y = np.load(stem + "_y.npy")
    return dict(dead=y[:, 0] == 1, stop=y[:, 1], valid=y[:, 2] == 1, conf=y[:, 3], poss=y[:, 4], ok=y[:, 5] >= 0)


SRC_W = {"sportec": 1.0, "metrica": 1.0, "skillcorner": 0.5}
TRAIN_VARS = {"sportec": ("train0", "train1"), "metrica": ("train0", "train1"), "skillcorner": ("real", "mask")}
TEST_VARS = {"sportec": ("test0", "stress0"), "metrica": ("test0", "stress0", "noball"), "skillcorner": ("real", "noball")}


def matches(work, src):
    return sorted({m for m, _, _ in units(work, src, TRAIN_VARS[src][0])})


SUB = {"train0": 3, "real": 3, "train1": 6, "mask": 6}


def train_set(work, include, sub=None, which="5"):
    """include: {src: list of match ids}. Rows: valid, label known (bs >= 0); weights per source and
    label confidence (conf 0 -> 0, conf 1 -> half)."""
    X, y, w, ys = [], [], [], []
    for src, mids in include.items():
        for var in TRAIN_VARS[src]:
            for mid, h, stem in units(work, src, var, set(mids)):
                Y = load_y(stem)
                keep = Y["valid"] & Y["ok"]
                keep[np.arange(len(keep)) % (sub or SUB[var]) != 0] = False
                x = np.load(stem + f"_X{which}.npy")[keep].astype(np.float32)
                cw = np.where(Y["conf"][keep] >= 2, 1.0, np.where(Y["conf"][keep] == 1, 0.5, 0.0))
                X.append(x)
                y.append(Y["dead"][keep])
                ys.append(Y["stop"][keep])
                w.append(SRC_W[src] * cw)
    return np.concatenate(X), np.concatenate(y), np.concatenate(w), np.concatenate(ys)


def make_hgb(n_iter=250):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0)


def fit_v5(work, include):
    X, y, w, ys = train_set(work, include)
    m = make_hgb().fit(X, y, sample_weight=w)
    d = y & (ys >= 1) & (ys <= 7) & (w > 0)
    idx = np.flatnonzero(d)[::3]
    mt = make_hgb(150).fit(X[idx], ys[idx], sample_weight=w[idx])
    return m, mt


def predict_units(work, m, mt, src, var, mids, which="5", ncol=None):
    out = {}
    for mid, h, stem in units(work, src, var, set(mids)):
        x = np.load(stem + f"_X{which}.npy").astype(np.float32)
        if ncol is not None:
            x = x[:, :ncol]
        r = dict(p=m.predict_proba(x)[:, 1].astype(np.float32))
        if mt is not None:
            r["pt"] = mt.predict_proba(x).astype(np.float16)
        out[(mid, h)] = r
    return out


# ============================================================================ evaluation driver
def _pk(path, fn):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    r = fn()
    pickle.dump(r, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)
    return r


def stage_zero_shot(work, models_pkl, oof_dir, log=print):
    """D v4 (final Sportec model) and D v2 replayed (final) on SkillCorner / Metrica; Sportec = their OOF."""
    M = pickle.load(open(models_pkl, "rb"))
    ncol = len(json.load(open(os.path.join(oof_dir, "columns.json")))["v4"])
    out = {}
    for src in ("skillcorner", "metrica"):
        for var in TEST_VARS[src]:
            mids = matches(work, src)
            for k, v in predict_units(work, M["v4_dead"], M["v4_type"], src, var, mids, "5", ncol).items():
                out[("v4", src, var) + k] = v
            for k, v in predict_units(work, M["v2_dead"], None, src, var, mids, "2").items():
                out[("v2", src, var) + k] = v
            log(f"zero-shot {src} {var}")
    for which, fam in (("4", "v4"), ("2", "v2")):
        for mid in dv.SPORTEC_IDS:
            z = np.load(os.path.join(oof_dir, f"oof{which}_{mid}.npz"))
            for kind in ("test", "stress"):
                for k, h in enumerate(("h1", "h2")):
                    r = dict(p=z[f"{kind}_{k}_p"][::STEP].astype(np.float32))
                    if which == "4":
                        pt = z[f"{kind}_{k}_ptype"][::STEP].astype(np.float32)
                        cl = z["classes"]
                        full = np.zeros((len(pt), 8), np.float16)       # columns = codes 1..8
                        for j, c in enumerate(cl):
                            full[:, int(c) - 1] = pt[:, j]
                        r["pt"] = full
                    out[(fam, "sportec", kind + "0", mid, h)] = r
    return out


def _std_types(m_type, pt):
    full = np.zeros((len(pt), 8), np.float16)
    for j, c in enumerate(m_type.classes_):
        full[:, int(c) - 1] = pt[:, j]
    return full


def _predict_fold(work, m, mt, tests, fam, out):
    for src, mids in tests.items():
        for var in TEST_VARS[src]:
            for k, v in predict_units(work, m, mt, src, var, mids).items():
                v["pt"] = _std_types(mt, v["pt"].astype(np.float32))
                out[(fam, src, var) + k] = v


def stage_lomo(work, log=print, n_sc_folds=5):
    allm = {s: matches(work, s) for s in ("sportec", "metrica", "skillcorner")}
    folds = [("sportec", [m]) for m in allm["sportec"]] + [("metrica", [m]) for m in allm["metrica"]]
    sc = allm["skillcorner"]
    folds += [("skillcorner", sc[i::n_sc_folds]) for i in range(n_sc_folds)]
    out = {}
    for src, held in folds:
        path = os.path.join(work, f"pred_lomo_{src}_{held[0]}.pkl")

        def run():
            t0 = time.time()
            inc = {s: [m for m in ms if not (s == src and m in held)] for s, ms in allm.items()}
            m, mt = fit_v5(work, inc)
            o = {}
            _predict_fold(work, m, mt, {src: held}, "v5", o)
            log(f"lomo {src} {held}: {time.time() - t0:.0f}s")
            return o
        out.update(_pk(path, run))
    return out


def stage_transfer(work, log=print):
    allm = {s: matches(work, s) for s in ("sportec", "metrica", "skillcorner")}
    out = {}
    for held in ("skillcorner", "metrica", "sportec"):
        path = os.path.join(work, f"pred_transfer_sans_{held}.pkl")

        def run():
            t0 = time.time()
            inc = {s: ms for s, ms in allm.items() if s != held}
            m, mt = fit_v5(work, inc)
            o = {}
            _predict_fold(work, m, mt, {held: allm[held]}, "v5tr", o)
            log(f"transfer sans {held}: {time.time() - t0:.0f}s")
            if held == "skillcorner":
                pickle.dump(dict(dead=m, type=mt), open(os.path.join(work, "model_v5_sans_skillcorner.pkl"), "wb"))
            return o
        out.update(_pk(path, run))
    return out


def make_units(work, preds, fam, src, var):
    us = []
    for mid in matches(work, src):
        for h in ("h1", "h2"):
            key = (fam, src, var, mid, h)
            if key not in preds:
                continue
            stem = os.path.join(work, f"{src}__{mid}__{h}__{var}")
            Y = load_y(stem)
            pr = preds[key]
            n = min(len(pr["p"]), len(Y["dead"]))
            u = {k: v[:n] for k, v in Y.items()}
            u["p"] = fill_invalid(pr["p"][:n], u["valid"] & u["ok"])
            u["pt"] = pr.get("pt")
            u["mid"] = mid
            us.append(u)
    return us


NOMINAL = {"sportec": "test0", "metrica": "test0", "skillcorner": "real"}


def evaluate_family(work, preds, fam, pm, decoders, fixed=None, log=print):
    """decoders: 'hyst' / 'hsmm'. Decoder for source X chosen on the family's nominal predictions of the
    OTHER sources (never on X). fixed: optional fixed hysteresis parameters (v4 / v2 as deployed)."""
    rows_sum, rows_type, rows_match, chosen = [], [], [], {}
    srcs = [s for s in ("sportec", "skillcorner", "metrica") if any(k[0] == fam and k[1] == s for k in preds)]
    for kind in decoders:
        for src in srcs:
            if kind == "fixed":
                prm = fixed
            else:
                pool = []
                for o in srcs:
                    if o != src:
                        pool += make_units(work, preds, fam, o, NOMINAL[o])
                if not pool:
                    continue
                prm = choose(pool, HYST_GRID if kind == "hyst" else HSMM_GRID)
            chosen[(fam, kind, src)] = prm
            for var in TEST_VARS[src]:
                us = make_units(work, preds, fam, src, var)
                if not us:
                    continue
                frs, rows = [], []
                for u in us:
                    pred = decode5(u["p"], prm)
                    pt = u["pt"].astype(np.float32) if u["pt"] is not None else None
                    f, r = half_metrics(u, pred, pm, pt, np.arange(1, 9) if pt is not None else None)
                    frs.append(f)
                    rows += r
                    rows_match.append(dict(fam=fam, decodeur=kind, src=src, var=var, match=u["mid"], **f))
                nm = len({u["mid"] for u in us})
                sm = summarise5(frs, rows, nm)
                rows_sum.append(dict(fam=fam, decodeur=kind, src=src, var=var, n_matchs=nm, **sm, params=json.dumps(prm)))
                rows_type.append(per_type5(rows, f"{fam}|{kind}|{src}|{var}"))
                log(f"{fam} {kind} {src} {var}: F1arret {sm['F1_arret_iou05']:.3f} P {sm['P_v']:.3f} R {sm['R_v']:.3f} faux/m {sm['faux_arrets_par_match']:.2f}")
    return rows_sum, rows_type, rows_match, chosen


# ============================================================================== BAR-ATH apply
def bar_features(data_dir, ball_npz, cache_dir=None):
    """Per graded half: v5 feature frame (25 fps) of the tactical ATD + fused ball (inputs only)."""
    P, B, slot_team = v1.load_tactical(data_dir, cache_dir)
    fb = np.load(ball_npz)
    Lp, Wp = v1.PITCH
    out = []
    for a, b in v1.HALVES:
        obs = dict(px=P[a:b, :, 0] * Lp, py=P[a:b, :, 1] * Wp, team=slot_team, fx=fb["x"][a:b].astype(float),
                   fy=fb["y"][a:b].astype(float), src=fb["source"][a:b].astype(np.int8),
                   ax=B[a:b, 0].astype(float) * Lp, ay=B[a:b, 1].astype(float) * Wp, L=Lp, W=Wp)
        out.append((a, b, features_v5(obs)))
    return out


def apply_recipe(halves_feats, recipe, tcn_p=None):
    """recipe: {"components": [{"kind": "hgb", "pkl": path, "ncol": 217|319, "weight": w}, {"kind": "tcn", "weight": w}],
    "decode": {...}}; tcn_p: {half_start: (p5, pt5)} from dead_v5_tcn.py predict-bar. Decoding at 5 Hz as in validation."""
    N = v1.N_FRAMES
    res = dict(p_dead=np.full(N, np.nan, np.float32), dead=np.zeros(N, bool), stop_type=np.zeros(N, np.int8),
               graded=np.zeros(N, bool), p_type=np.zeros((N, 8), np.float16))
    runs_tab = []
    for a, b, F in halves_feats:
        X = F.values.astype(np.float32)
        idx5 = np.arange(0, len(X), STEP)
        p5, pt5, wsum = np.zeros(len(idx5)), np.zeros((len(idx5), 8)), 0.0
        for c in recipe["components"]:
            if c["kind"] == "hgb":
                M = pickle.load(open(c["pkl"], "rb"))
                x5 = X[idx5][:, :c["ncol"]]
                p5 += c["weight"] * M["dead"].predict_proba(x5)[:, 1]
                pt5 += c["weight"] * _std_types(M["type"], M["type"].predict_proba(x5)).astype(np.float64)
            else:
                tp, tt = tcn_p[a]
                p5 += c["weight"] * tp[:len(idx5)]
                pt5 += c["weight"] * tt[:len(idx5)].astype(np.float64)
            wsum += c["weight"]
        p5, pt5 = p5 / wsum, pt5 / wsum
        d5 = decode5(p5, recipe["decode"])
        st5 = dv.run_types(d5, pt5, np.arange(1, 9))
        T = b - a
        up = np.minimum(np.arange(T) // STEP, len(idx5) - 1)
        res["graded"][a:b] = True
        res["p_dead"][a:b] = np.interp(np.arange(T), idx5, p5).astype(np.float32)
        res["dead"][a:b] = d5[up]
        res["stop_type"][a:b] = st5[up]
        res["p_type"][a:b] = pt5[up].astype(np.float16)
        for s0, e0 in runs_of(res["dead"][a:b]):
            runs_tab.append((a + s0, a + e0, int(res["stop_type"][a + s0]), float(res["p_dead"][a + s0:a + e0].mean())))
    res["runs"] = np.array(runs_tab, np.float64).reshape(-1, 4)
    return res


def main_apply(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ball", required=True)
    ap.add_argument("--recipe", required=True, help="JSON recipe (components + decode), see decode_v5.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--bar-feat", default=None, help="cache of the BAR-ATH 5 Hz features (written if missing)")
    ap.add_argument("--feat-out", default=None, help="write the 5 Hz BAR-ATH v4 features for dead_v5_tcn.py")
    ap.add_argument("--tcn-pred", default=None, help="npz from dead_v5_tcn.py predict-bar")
    ap.add_argument("--v2-csv", default=None)
    ap.add_argument("--stats-out", default=None)
    a = ap.parse_args(argv)
    recipe = json.load(open(a.recipe))
    if a.bar_feat and os.path.exists(a.bar_feat):
        z = np.load(a.bar_feat)
        cols = json.loads(str(z["columns"]))
        hf = [(int(s0), int(e0), pd.DataFrame(np.repeat(z[f"X_{s0}"].astype(np.float32), STEP, 0)[:e0 - s0], columns=cols))
              for s0, e0 in v1.HALVES]
    else:
        hf = bar_features(a.data, a.ball, a.cache)
        if a.bar_feat:
            np.savez_compressed(a.bar_feat, columns=json.dumps(list(hf[0][2].columns)),
                                **{f"X_{s0}": F.values[::STEP].astype(np.float16) for s0, e0, F in hf})
    if a.feat_out:
        np.savez_compressed(a.feat_out, **{f"X_{s}": F.values[::STEP, :217].astype(np.float16) for s, e, F in hf},
                            **{f"valid_{s}": np.ones(len(F.values[::STEP]), bool) for s, e, F in hf})
    tcn_p = None
    if a.tcn_pred:
        z = np.load(a.tcn_pred)
        tcn_p = {s: (z[f"p_{s}"], z[f"pt_{s}"]) for s, e, F in hf}
    if any(c["kind"] == "tcn" for c in recipe["components"]) and tcn_p is None:
        print("TCN predictions required: run dead_v5_tcn.py predict-bar on --feat-out first")
        return
    out = apply_recipe(hf, recipe, tcn_p)
    np.savez_compressed(a.out, **out, stop_type_codes=json.dumps(STOP_NAMES), decode=json.dumps(recipe))
    print("wrote", a.out)
    if a.stats_out:
        st = dv.label_free_stats(out, a.ball, a.v2_csv)
        json.dump(st, open(a.stats_out, "w"), indent=1, ensure_ascii=False)
        print(json.dumps(st, indent=1, ensure_ascii=False))


def main_cache(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--work", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--sportec", required=True)
    ap.add_argument("--stops", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    a = ap.parse_args(argv)
    run_cache(a.work, a.ext, a.sportec, a.stops, a.worker, a.nworkers, log=lambda m: print(m, flush=True))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cache":
    main_cache()
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "apply":
    main_apply()

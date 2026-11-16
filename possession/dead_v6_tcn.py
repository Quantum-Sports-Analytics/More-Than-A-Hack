"""D v6: full MS-TCN training (GPU) for the dead-ball detector, external data only.

Recipe, on Sportec (CC-BY 4.0), SkillCorner Open Data (MIT,
1959846 excluded) and Metrica sample (internal use, source cited; raw ball never used):
* inputs per 5 Hz step: the 217 D v4 features (robust-standardised per fold) + validity flag + a DeepSets
  encoder per team over player tokens (x, y, vx, vy, speed, signed distance to the nearest line, mask;
  up to 14 present players per team; phi = MLP 7-32-32, mean + max pooling -> 128 dims);
* trunk: 3 stages x 9 dilated residual layers (dilations 1..256, 64 channels, dropout 0.3, non-causal);
  heads: P(D), D start, D end (Gaussian targets sigma 1 s, BCE pos_weight 20), stop type (8 classes);
  loss = BCE(D) + 0.15 TMSE + boundaries + 0.5 type, per-step weights = source weight x label confidence;
* augmentations (pre-computed variants, features recomputed): ball fully hidden in 50 % of the crops
  (no-ball variant), ball hidden in 60-s windows, players masked outside a REAL SkillCorner camera footprint
  (own footprint for SkillCorner; footprints re-centred on the ball for Sportec / Metrica) plus 0-3 random
  visible players, x / y flips, team swap, time stretch +-10 %;
* training: up to 40 epochs, cosine schedule, early stopping (patience 6) on an internal validation match
  per training source, 3 seeds averaged; checkpoints per fold and seed in the bucket (Spot preemption);
* decoding (local): hysteresis or HSMM (dead_v5), then each D boundary is moved to the maximum of the
  start / end head within +-2 s when that maximum exceeds thr_b (chosen on the other sources).

Commands
  VM (Cloud Batch, GPU):  python dead_v6_tcn.py vm --root <bucket dir> [--smoke]
  local evaluation:       python dead_v6_tcn.py eval --results <dir> --zero-shot pred_zero_shot.pkl --out-dir <dir>
  local BAR-ATH tokens:   python dead_v6_tcn.py bar-tokens --data <data dir> --out bar_tok5.npz
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import tarfile
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FPS, STEP, HZ = 25, 5, 5
K_TOK = 14
NCOL = 217
SC_EXCLUDE = ("1959846",)
VARS = {"sportec": ("train0", "train1", "aug0", "aug1", "noball", "test0", "stress0"),
        "metrica": ("train0", "train1", "aug0", "aug1", "noball", "test0", "stress0"),
        "skillcorner": ("real", "mask", "aug0", "aug1", "noball")}
BALL_TRAIN = {"sportec": ("train0", "train1", "aug0", "aug1"), "metrica": ("train0", "train1", "aug0", "aug1"),
              "skillcorner": ("real", "mask", "aug0", "aug1")}
NOBALL = "noball"
TEST_VARS = {"sportec": ("test0", "stress0", "noball"), "metrica": ("test0", "stress0", "noball"),
             "skillcorner": ("real", "noball")}
NOMINAL = {"sportec": "test0", "metrica": "test0", "skillcorner": "real"}
SRC_W = {"sportec": 1.0, "metrica": 1.0, "skillcorner": 0.5}
N_SC_FOLDS = 5


def log(root, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if root:
        with open(os.path.join(root, "log.txt"), "a") as fh:
            fh.write(line + "\n")


# ================================================================================ tokens
def player_tokens(px, py, team, L, W, step=STEP, K=K_TOK):
    """(T5, 2, K, 7) float16: up to K present players per team per 5 Hz step."""
    px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
    T = len(px)
    vx = np.full(px.shape, np.nan)
    vy = vx.copy()
    vx[2:-2] = (px[4:] - px[:-4]) * FPS / 4
    vy[2:-2] = (py[4:] - py[:-4]) * FPS / 4
    idx = np.arange(0, T, step)
    out = np.zeros((len(idx), 2, K, 7), np.float16)
    for k in (0, 1):
        cols = np.flatnonzero(np.asarray(team) == k)
        if len(cols) == 0:
            continue
        X, Y = px[idx][:, cols], py[idx][:, cols]
        VX, VY = vx[idx][:, cols], vy[idx][:, cols]
        pres = np.isfinite(X) & np.isfinite(Y)
        order = np.argsort(~pres, axis=1, kind="stable")[:, :K]
        g = lambda A: np.take_along_axis(A, order, 1)  # noqa: E731
        Xs, Ys, VXs, VYs, Ps = g(X), g(Y), g(VX), g(VY), g(pres)
        spd = np.hypot(VXs, VYs)
        edge = np.fmin(np.fmin(Xs, L - Xs), np.fmin(Ys, W - Ys))
        feats = np.stack([Xs / L - 0.5, Ys / W - 0.5, VXs / 10, VYs / 10, spd / 10, edge / 10, Ps.astype(float)], -1)
        feats = np.where(Ps[..., None], np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        feats = np.clip(feats, -5, 5)
        n = feats.shape[1]
        out[:, k, :n] = feats.astype(np.float16)
    return out


# ============================================================================ augmentation
def _in_quad(px, py, quad):
    """px, py (T, S); quad (T, 4, 2) ordered convex; NaN quad -> True (no masking)."""
    inside = np.ones(px.shape, bool)
    ok = np.isfinite(quad).all((1, 2))
    if not ok.any():
        return inside
    q = quad[ok]
    X, Y = px[ok], py[ok]
    s = None
    sgn = []
    for i in range(4):
        a, b = q[:, i], q[:, (i + 1) % 4]
        cr = (b[:, 0, None] - a[:, 0, None]) * (Y - a[:, 1, None]) - (b[:, 1, None] - a[:, 1, None]) * (X - a[:, 0, None])
        sgn.append(cr)
    sgn = np.stack(sgn, 0)
    s = (sgn >= 0).all(0) | (sgn <= 0).all(0)
    inside[ok] = s | ~np.isfinite(X)
    return inside


def build_cam_pool(sc_files, n_max=20000):
    """Camera footprints relative to the detected ball (metres), from SkillCorner."""
    offs = []
    for f in sc_files:
        z = np.load(f)
        for h in ("h1", "h2"):
            cam = z[f"{h}_cam"]
            bx, by = z[f"{h}_BX"], z[f"{h}_BY"]
            ok = np.isfinite(cam).all(1) & np.isfinite(bx)
            for i in np.flatnonzero(ok)[::25]:
                q = cam[i].reshape(4, 2) - np.array([bx[i], by[i]])
                offs.append(q)
    offs = np.array(offs, np.float32)
    if len(offs) > n_max:
        offs = offs[np.random.default_rng(0).choice(len(offs), n_max, replace=False)]
    return offs


def augment_half(hf, rng, cam_pool=None, own_cam=None):
    """Time stretch +-10 %, x / y flips, team swap, camera-footprint player masking, 0-3 extra masked players."""
    L, W = hf["L"], hf["W"]
    T = len(hf["px"])
    r = rng.uniform(0.9, 1.1)
    t = np.arange(0, T - 1, r)
    i0 = np.floor(t).astype(int)
    fr = (t - i0)[:, None]
    lin = lambda A: A[i0] * (1 - fr) + A[i0 + 1] * fr if A.ndim == 2 else A[i0] * (1 - fr[:, 0]) + A[i0 + 1] * fr[:, 0]  # noqa: E731
    out = dict(hf)
    for k in ("px", "py"):
        out[k] = lin(hf[k].astype(np.float64)).astype(np.float32)
    for k in ("bx", "by", "bz"):
        out[k] = lin(hf[k].astype(np.float64)).astype(np.float32)
    for k in ("dead", "stop", "valid", "conf", "poss", "bs"):
        out[k] = hf[k][i0]
    cam = own_cam[i0] if own_cam is not None else None
    if rng.random() < 0.5:
        out["px"] = (L - out["px"]).astype(np.float32)
        out["bx"] = (L - out["bx"]).astype(np.float32)
        if cam is not None:
            cam = cam.copy()
            cam[:, :, 0] = L - cam[:, :, 0]
    if rng.random() < 0.5:
        out["py"] = (W - out["py"]).astype(np.float32)
        out["by"] = (W - out["by"]).astype(np.float32)
        if cam is not None:
            cam = cam.copy()
            cam[:, :, 1] = W - cam[:, :, 1]
    if rng.random() < 0.5:
        out["team"] = (1 - np.asarray(hf["team"])).astype(np.int8)
        out["poss"] = np.where(out["poss"] == 1, 2, np.where(out["poss"] == 2, 1, out["poss"])).astype(np.int8)
    Tn = len(out["px"])
    if cam is None and cam_pool is not None and len(cam_pool):
        # footprint re-centred on the (true) ball, a new random footprint every 10 s
        cam = np.full((Tn, 4, 2), np.nan, np.float32)
        bx = np.nan_to_num(out["bx"], nan=L / 2)
        by = np.nan_to_num(out["by"], nan=W / 2)
        for s in range(0, Tn, 10 * FPS):
            q = cam_pool[rng.integers(len(cam_pool))]
            e = min(Tn, s + 10 * FPS)
            cam[s:e] = q[None] + np.stack([bx[s:e], by[s:e]], 1)[:, None, :]
    px = out["px"]
    hide = np.zeros(px.shape, bool)
    if cam is not None and rng.random() < 0.8:
        hide |= ~_in_quad(px, out["py"], cam)
    for s in range(0, Tn, 10 * FPS):              # 0-3 extra visible players hidden per 10 s
        n = int(rng.integers(0, 4))
        if n:
            e = min(Tn, s + 10 * FPS)
            vis = np.flatnonzero((np.isfinite(px[s:e]) & ~hide[s:e]).any(0))
            if len(vis):
                j = rng.choice(vis, min(n, len(vis)), replace=False)
                hide[s:e, j] = True
    out["hide"] = hide                            # applied to the observations after the ball simulation
    return out


# =================================================================================== cache
def _crc(*a):
    return zlib.crc32("|".join(map(str, a)).encode())


def cache_specs(inputs):
    sp = []
    for mid in ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"):
        for h in ("h1", "h2"):
            for var in VARS["sportec"]:
                sp.append(("sportec", mid, h, var))
    for f in sorted(glob.glob(os.path.join(inputs, "metrica_*.npz"))):
        for h in ("h1", "h2"):
            for var in VARS["metrica"]:
                sp.append(("metrica", os.path.basename(f)[:-4], h, var))
    for f in sorted(glob.glob(os.path.join(inputs, "SC_*.npz"))):
        mid = os.path.basename(f)[:-4]
        if mid[3:] in SC_EXCLUDE:
            continue
        for h in ("h1", "h2"):
            for var in VARS["skillcorner"]:
                sp.append(("skillcorner", mid, h, var))
    return sp


def _load_halves(src, mid, inputs):
    import pandas as pd
    import dead_v5 as d5
    if src == "sportec":
        stops = pd.read_csv(os.path.join(inputs, "sportec_arrets.csv"))
        hs = d5.load_sportec_halves(inputs, stops, [mid])[mid]
    else:
        hs = d5.load_ext(os.path.join(inputs, mid + ".npz"), src)
        if src == "skillcorner":
            z = np.load(os.path.join(inputs, mid + ".npz"))
            for k, h in enumerate(("h1", "h2")):
                bs = z[f"{h}_BS"]
                al = np.flatnonzero(bs == 1)
                a, b = al[0], al[-1] + 1
                cam = z[f"{h}_cam"][a:b].reshape(-1, 4, 2).astype(np.float32)
                cam[:, :, 0] += hs[k]["L"] / 2
                cam[:, :, 1] += hs[k]["W"] / 2
                hs[k]["cam"] = cam
    return {("h1", "h2")[i]: x for i, x in enumerate(hs)}


def build_variant(spec, hf, cam_pool, cache):
    import dead_v4 as dv
    import dead_v5 as d5
    src, mid, h, var = spec
    stem = os.path.join(cache, f"{src}__{mid}__{h}__{var}")
    if os.path.exists(stem + "_y.npy"):
        return False
    if var.startswith("aug"):
        rng = np.random.default_rng(_crc(src, mid, h, var, "v6"))
        hf2 = augment_half(hf, rng, cam_pool=None if src == "skillcorner" else cam_pool,
                           own_cam=hf.get("cam") if src == "skillcorner" else None)
        if src == "skillcorner":
            obs = d5.obs_real_ball(hf2)
            obs["px"], obs["py"] = hf2["px"], hf2["py"]
        else:
            obs = dv.simulate_obs(hf2, rng, dv.draw_params(rng))
        if rng.random() < 0.5:
            obs = d5.mask_ball_windows(obs, rng)
        hide = hf2["hide"]
        obs["px"] = np.where(hide, np.nan, obs["px"]).astype(np.float32)
        obs["py"] = np.where(hide, np.nan, obs["py"]).astype(np.float32)
        lab = hf2
    else:
        # identical to dead_v5.build_one for the shared variants (same seeds -> same test sets)
        rng = None if var == "noball" else np.random.default_rng(d5._seed(src, mid, h, var))
        if src in ("sportec", "metrica") and var != "noball":
            p = dv.draw_params(rng, stress=var.startswith("stress"))
            obs = dv.simulate_obs(hf, rng, p)
            if var == "train1":
                obs = d5.mask_ball_windows(obs, rng)
        elif var == "noball":
            obs = d5.obs_no_ball(hf)
        else:
            obs = d5.obs_real_ball(hf)
            if var == "mask":
                obs = d5.mask_ball_windows(obs, rng)
        lab = hf
    F = dv.features(obs)
    idx = np.arange(0, len(F), STEP)
    np.save(stem + "_X.npy", F.values[idx].astype(np.float16))
    np.save(stem + "_P.npy", player_tokens(obs["px"], obs["py"], obs["team"], obs["L"], obs["W"]))
    y = np.stack([lab["dead"][idx].astype(np.int8), lab["stop"][idx], lab["valid"][idx].astype(np.int8),
                  lab["conf"][idx].astype(np.int8), lab["poss"][idx].astype(np.int8), lab["bs"][idx].astype(np.int8)], 1)
    np.save(stem + "_y.npy", y)
    return True


def _cache_worker(args):
    specs, inputs, cache, cam_pool_path, root, bcache = args
    import warnings
    warnings.filterwarnings("ignore")
    cam_pool = np.load(cam_pool_path)
    cur, halves = None, None
    specs = [sp for sp in specs if not os.path.exists(os.path.join(cache, "__".join(sp) + "_y.npy"))]
    for spec in specs:
        src, mid = spec[0], spec[1]
        if (src, mid) != cur:
            halves = _load_halves(src, mid, inputs)
            cur = (src, mid)
        t0 = time.time()
        if build_variant(spec, halves[spec[2]], cam_pool, cache):
            stem = "__".join(spec)
            if bcache:                                  # persist at once (Spot preemption)
                for suf in ("_X.npy", "_P.npy", "_y.npy"):
                    shutil.copy(os.path.join(cache, stem + suf), os.path.join(bcache, stem + suf + ".part"))
                for suf in ("_X.npy", "_P.npy", "_y.npy"):
                    os.replace(os.path.join(bcache, stem + suf + ".part"), os.path.join(bcache, stem + suf))
            log(None, f"cache {spec} {time.time() - t0:.0f}s")
    return len(specs)


def run_cache(root, inputs, cache, specs, nproc=7, bcache=None):
    import multiprocessing as mp
    os.makedirs(cache, exist_ok=True)
    if bcache:
        os.makedirs(bcache, exist_ok=True)
        n = 0
        for f in os.listdir(bcache):                    # restore what a previous (preempted) VM built
            if f.endswith(".npy") and not os.path.exists(os.path.join(cache, f)):
                shutil.copy(os.path.join(bcache, f), os.path.join(cache, f))
                n += 1
        log(root, f"cache: {n} files restored from the bucket")
    pool_path = os.path.join(cache, "cam_pool.npy")
    if not os.path.exists(pool_path):
        scf = [f for f in sorted(glob.glob(os.path.join(inputs, "SC_*.npz"))) if os.path.basename(f)[3:-4] not in SC_EXCLUDE]
        np.save(pool_path, build_cam_pool(scf))
    groups = {}
    for s in specs:
        groups.setdefault((s[0], s[1]), []).append(s)
    jobs = [(g, inputs, cache, pool_path, root, bcache) for g in groups.values()]
    with mp.get_context("fork").Pool(nproc) as p:
        for _ in p.imap_unordered(_cache_worker, jobs):
            pass


# ============================================================================== data on GPU
def units(cache, src, var):
    out = []
    for f in sorted(glob.glob(os.path.join(cache, f"{src}__*__{var}_y.npy"))):
        _, mid, h, _ = os.path.basename(f)[:-6].split("__")
        out.append((mid, h, f[:-6]))
    return out


def matches(cache, src):
    return sorted({m for m, _, _ in units(cache, src, VARS[src][0])})


def fold_def(cache, fold):
    allm = {s: matches(cache, s) for s in ("sportec", "metrica", "skillcorner")}
    parts = fold.split(":")
    if parts[0] == "transfer":
        held = {parts[1]: allm[parts[1]]}
        inc = {s: m for s, m in allm.items() if s != parts[1]}
    elif parts[0] == "lomo":
        src = parts[1]
        hm = allm[src][int(parts[2])::N_SC_FOLDS] if src == "skillcorner" else [parts[2]]
        held = {src: hm}
        inc = {s: [m for m in ms if not (s == src and m in hm)] for s, ms in allm.items()}
    else:
        held, inc = {}, allm
    # internal validation: last match of each training source (deterministic)
    val = {s: ms[-1:] for s, ms in inc.items() if len(ms) > 1}
    tr = {s: [m for m in ms if m not in val.get(s, [])] for s, ms in inc.items()}
    return tr, val, held


def load_unit(stem):
    X = np.load(stem + "_X.npy")[:, :NCOL]
    P = np.load(stem + "_P.npy")
    y = np.load(stem + "_y.npy")
    return X, P, y


class TrainData:
    """All training variants on the device; crops drawn per base half (ball variant or no-ball variant)."""

    def __init__(self, cache, inc, device):
        import torch
        self.dev = device
        self.bases = []
        for src, mids in inc.items():
            for mid in mids:
                for h in ("h1", "h2"):
                    vs = {}
                    for var in BALL_TRAIN[src] + (NOBALL,):
                        stem = os.path.join(cache, f"{src}__{mid}__{h}__{var}")
                        if os.path.exists(stem + "_y.npy"):
                            vs[var] = stem
                    if vs:
                        self.bases.append((src, vs))
        sample = []
        for src, vs in self.bases:
            for v in vs.values():
                sample.append(np.load(v + "_X.npy")[::13, :NCOL].astype(np.float32))
        S = np.concatenate(sample)
        self.med = np.nan_to_num(np.nanmedian(S, 0)).astype(np.float32)
        q = np.nanpercentile(S, [25, 75], 0)
        self.iqr = np.where(np.nan_to_num(q[1] - q[0]) > 1e-6, q[1] - q[0], 1.0).astype(np.float32)
        self.T = []
        for src, vs in self.bases:
            d = {}
            for var, stem in vs.items():
                X, P, y = load_unit(stem)
                d[var] = self._to_dev(X, P, y, src)
            self.T.append((src, d))
        self.lens = np.array([min(v["n"] for v in d.values()) for _, d in self.T], float)

    def norm(self, X, valid):
        import torch
        Z = (torch.as_tensor(X, dtype=torch.float32, device=self.dev) - torch.as_tensor(self.med, device=self.dev)) / \
            torch.as_tensor(self.iqr, device=self.dev)
        Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10, 10)
        v = torch.as_tensor(valid, dtype=torch.float32, device=self.dev)[:, None]
        return torch.cat([Z, v], 1).half()

    def _to_dev(self, X, P, y, src):
        import torch
        dead, stop, valid, conf, ok = y[:, 0] == 1, y[:, 1], y[:, 2] == 1, y[:, 3], y[:, 5] >= 0
        w = SRC_W[src] * np.where(conf >= 2, 1.0, np.where(conf == 1, 0.5, 0.0)) * (valid & ok)
        st, en = gauss_targets(dead & ok)
        lab = np.stack([dead.astype(np.float32), w.astype(np.float32), st, en, np.clip(stop.astype(np.float32) - 1, -1, 7)])
        return dict(Z=self.norm(X, valid), P=torch.as_tensor(P, device=self.dev),
                    lab=torch.as_tensor(lab, device=self.dev), n=len(y))

    def batch(self, rng, B=16, crop=480):
        import torch
        zs, ps, ls = [], [], []
        for _ in range(B):
            i = rng.choice(len(self.T), p=self.lens / self.lens.sum())
            src, d = self.T[i]
            if NOBALL in d and rng.random() < 0.5:
                u = d[NOBALL]
            else:
                ks = [k for k in d if k != NOBALL]
                u = d[ks[rng.integers(len(ks))]]
            s = int(rng.integers(0, max(1, u["n"] - crop)))
            zs.append(u["Z"][s:s + crop])
            ps.append(u["P"][s:s + crop])
            ls.append(u["lab"][:, s:s + crop])
        Lm = min(z.shape[0] for z in zs)
        return (torch.stack([z[:Lm] for z in zs]).float(), torch.stack([p[:Lm] for p in ps]).float(),
                torch.stack([l[:, :Lm] for l in ls]))


def gauss_targets(dead):
    T = len(dead)
    st, en = np.zeros(T, np.float32), np.zeros(T, np.float32)
    e = np.diff(np.r_[0, dead.astype(np.int8), 0])
    t = np.arange(T)
    for a in np.flatnonzero(e == 1):
        lo, hi = max(a - 15, 0), min(a + 16, T)
        st[lo:hi] = np.maximum(st[lo:hi], np.exp(-0.5 * ((t[lo:hi] - a) / HZ) ** 2))
    for b in np.flatnonzero(e == -1):
        lo, hi = max(b - 15, 0), min(b + 16, T)
        en[lo:hi] = np.maximum(en[lo:hi], np.exp(-0.5 * ((t[lo:hi] - b) / HZ) ** 2))
    return st, en


# =================================================================================== model
def build_model(cin=NCOL + 1, ch=64, stages=3, layers=9, drop=0.3):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.norm = nn.LayerNorm(128)

        def forward(self, P):                       # P (B, T, 2, K, 7)
            m = P[..., 6:7]
            h = self.phi(P) * m
            cnt = m.sum(3).clamp(min=1)
            mean = h.sum(3) / cnt
            mx = (h - (1 - m) * 1e4).max(3).values * (m.sum(3) > 0.5).float()
            return self.norm(torch.cat([mean, mx], -1).flatten(2))          # (B, T, 128)

    class Stage(nn.Module):
        def __init__(self, c_in):
            super().__init__()
            self.inp = nn.Conv1d(c_in, ch, 1)
            self.n0 = nn.GroupNorm(1, ch)
            self.norms = nn.ModuleList([nn.GroupNorm(1, ch) for _ in range(layers)])
            self.convs = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
            self.mix = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(layers)])
            self.drop = nn.Dropout(drop)
            self.nout = nn.GroupNorm(1, ch)
            self.out = nn.Conv1d(ch, 11, 1)

        def forward(self, x):
            h = self.n0(self.inp(x))
            for nrm, c, m in zip(self.norms, self.convs, self.mix):
                h = h + self.drop(m(Fnn.relu(c(nrm(h)))))
            return self.out(self.nout(h))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ds = DeepSets()
            self.stages = nn.ModuleList([Stage(cin + 128)] + [Stage(11) for _ in range(stages - 1)])

        def forward(self, Z, P):                    # Z (B, T, cin)
            x = torch.cat([Z, self.ds(P)], -1).transpose(1, 2)
            outs = [self.stages[0](x)]
            for s in self.stages[1:]:
                o = outs[-1]
                outs.append(s(torch.cat([torch.sigmoid(o[:, :3]), torch.softmax(o[:, 3:], 1)], 1)))
            return outs

    return Net()


def loss_fn(outs, lab):
    import torch
    import torch.nn.functional as Fnn
    dead, w, st, en, stop = lab[:, 0], lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4].long()
    pw = torch.tensor(20.0, device=lab.device)
    tot = 0.0
    for o in outs:
        ld = (Fnn.binary_cross_entropy_with_logits(o[:, 0], dead, reduction="none") * w).sum() / w.sum().clamp(min=1)
        lp = Fnn.logsigmoid(o[:, 0:1])
        tm = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=16).mean()
        wb = (w > 0).float()
        lb = ((Fnn.binary_cross_entropy_with_logits(o[:, 1], st, pos_weight=pw, reduction="none")
               + Fnn.binary_cross_entropy_with_logits(o[:, 2], en, pos_weight=pw, reduction="none")) * wb).sum() / wb.sum().clamp(min=1) / 20
        m = (dead > 0.5) & (stop >= 0) & (w > 0)
        lt = Fnn.cross_entropy(o[:, 3:].permute(0, 2, 1)[m], stop[m]) if m.any() else torch.zeros((), device=lab.device)
        tot = tot + ld + 0.15 * tm + lb + 0.5 * lt
    return tot


def predict_seq(model, Z, P):
    """Z (T, cin) half tensor on device, P (T, 2, K, 7) -> dict of numpy arrays."""
    import torch
    model.eval()
    with torch.no_grad():
        o = model(Z[None].float(), P[None].float())[-1][0]
    return dict(p=torch.sigmoid(o[0]).cpu().numpy().astype(np.float32), ps=torch.sigmoid(o[1]).cpu().numpy().astype(np.float16),
                pe=torch.sigmoid(o[2]).cpu().numpy().astype(np.float16), pt=torch.softmax(o[3:], 0).T.cpu().numpy().astype(np.float16))


def val_loss(model, data, vunits):
    import torch
    import torch.nn.functional as Fnn
    model.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for u in vunits:
            o = model(u["Z"][None].float(), u["P"][None].float())[-1][0]
            w = u["lab"][1]
            tot += float((Fnn.binary_cross_entropy_with_logits(o[0], u["lab"][0], reduction="none") * w).sum())
            n += float(w.sum())
    return tot / max(n, 1)


def train_seed(data, vunits, seed, epochs, iters, device, root, tag, ckpt_dir, patience=6):
    import torch
    path = os.path.join(ckpt_dir, f"{tag}_s{seed}.pt")
    model = build_model().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state"])
        log(root, f"{tag} seed {seed}: checkpoint reused")
        return model
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * iters)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        for _ in range(iters):
            Z, P, lab = data.batch(rng)
            loss = loss_fn(model(Z, P), lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach())
        vl = val_loss(model, data, vunits)
        log(root, f"{tag} s{seed} ep{ep} train {tl / iters:.3f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, best_val=best, med=data.med, iqr=data.iqr), path + ".tmp")
    os.replace(path + ".tmp", path)
    return model


def run_fold(root, cache, fold, seeds, epochs, iters, device, bar=None):
    import torch
    res_path = os.path.join(root, "results", f"pred_{fold.replace(':', '_')}.pkl")
    if os.path.exists(res_path):
        log(root, f"{fold}: already done")
        return
    tr, val, held = fold_def(cache, fold)
    t0 = time.time()
    data = TrainData(cache, tr, device)
    vunits = []
    for src, mids in val.items():
        for mid in mids:
            for h in ("h1", "h2"):
                stem = os.path.join(cache, f"{src}__{mid}__{h}__{NOMINAL[src]}")
                if os.path.exists(stem + "_y.npy"):
                    X, P, y = load_unit(stem)
                    vunits.append(data._to_dev(X, P, y, src))
    log(root, f"{fold}: {len(data.T)} train halves, {len(vunits)} val halves, load {time.time() - t0:.0f}s")
    models = [train_seed(data, vunits, s, epochs, iters, device, root, fold.replace(":", "_"), os.path.join(root, "ckpt"))
              for s in range(seeds)]
    out = {}
    for src, mids in held.items():
        for var in TEST_VARS[src]:
            for mid, h, stem in units(cache, src, var):
                if mid not in mids:
                    continue
                X, P, y = load_unit(stem)
                Z = data.norm(X, y[:, 2] == 1)
                Pt = torch.as_tensor(P, device=device)
                rs = [predict_seq(m, Z, Pt) for m in models]
                out[("v6", src, var, mid, h)] = {k: np.mean([r[k].astype(np.float32) for r in rs], 0).astype(rs[0][k].dtype) for k in rs[0]}
    if bar is not None:
        Xb, Pb = bar
        res = {}
        for s in Xb:
            Z = data.norm(Xb[s], np.ones(len(Xb[s]), bool))
            rs = [predict_seq(m, Z, torch.as_tensor(Pb[s], device=device)) for m in models]
            for k in rs[0]:
                res[f"{k}_{s}"] = np.mean([r[k].astype(np.float32) for r in rs], 0)
        np.savez_compressed(os.path.join(root, "results", "bar_v6_pred.npz"), **res)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    pickle.dump(out, open(res_path + ".tmp", "wb"))
    os.replace(res_path + ".tmp", res_path)
    log(root, f"{fold}: done in {time.time() - t0:.0f}s")
    del data, vunits
    torch.cuda.empty_cache()


FOLDS = (["transfer:skillcorner", "transfer:metrica", "transfer:sportec"]
         + [f"lomo:sportec:{m}" for m in ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")]
         + ["all"]
         + [f"lomo:metrica:metrica_g{i}" for i in (1, 2, 3)]
         + [f"lomo:skillcorner:{k}" for k in range(N_SC_FOLDS)])


def status(root, **kw):
    p = os.path.join(root, "status.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(st, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def main_vm(a):
    root = a.root
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    os.makedirs(os.path.join(root, "ckpt"), exist_ok=True)
    inputs = a.inputs or os.path.join(root, "inputs")
    cache = a.cache or "/tmp/cache_v6"
    tarp = os.path.join(root, "cache_v6.tar")
    log(root, f"start smoke={a.smoke}")
    status(root, stage="cache")
    specs = cache_specs(inputs)
    if a.smoke:
        keep = {("sportec", "J03WMX"), ("metrica", "metrica_g1"), ("skillcorner", "SC_1874553"), ("skillcorner", "SC_1886347")}
        specs = [s for s in specs if (s[0], s[1]) in keep]
    t0 = time.time()
    run_cache(root, inputs, cache, specs, nproc=a.nproc, bcache=os.path.join(root, "cache"))
    log(root, f"cache ready ({len(specs)} variants) {time.time() - t0:.0f}s")
    if not os.path.exists(os.path.join(root, "results", "labels_y.tar")):
        # labels for the local evaluation
        with tarfile.open(os.path.join(root, "results", "labels_y.tar"), "w") as t:
            for f in glob.glob(os.path.join(cache, "*_y.npy")):
                t.add(f, arcname=os.path.basename(f))
    import torch                                     # imported only after the forked cache workers
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(root, f"device={dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")
    status(root, device=dev)
    bar = None
    bx = os.path.join(inputs, "bar_X5.npz")
    bt = os.path.join(inputs, "bar_tok5.npz")
    if os.path.exists(bx) and os.path.exists(bt):
        zx, zt = np.load(bx), np.load(bt)
        bar = ({k[2:]: zx[k][:, :NCOL] for k in zx.files if k.startswith("X_")}, {k[2:]: zt[k] for k in zt.files if k.startswith("P_")})
    folds = ["transfer:skillcorner"] if a.smoke else FOLDS
    for i, fold in enumerate(folds):
        status(root, stage="train", fold=fold, fold_index=i, n_folds=len(folds))
        run_fold(root, cache, fold, 1 if a.smoke else a.seeds, 1 if a.smoke else a.epochs, 20 if a.smoke else a.iters,
                 dev, bar=bar if (fold == "all" or a.smoke) else None)
    status(root, stage="done")
    log(root, "all done")


# ============================================================================ local helpers
def bar_tokens(data_dir, out):
    import v1
    P, B, slot_team = v1.load_tactical(data_dir, None)
    Lp, Wp = v1.PITCH
    res = {}
    for a, b in v1.HALVES:
        res[f"P_{a}"] = player_tokens(P[a:b, :, 0] * Lp, P[a:b, :, 1] * Wp, slot_team, Lp, Wp)
    np.savez_compressed(out, **res)


def refine(pred, ps, pe, thr, win=2 * HZ):
    """Move each decoded boundary to the start / end head maximum within +-win steps if above thr."""
    out = pred.copy()
    if thr is None:
        return out
    import dead_v5 as d5
    T = len(out)
    for a, b in d5.runs_of(pred):
        lo, hi = max(0, a - win), min(T, a + win + 1)
        j = lo + int(np.argmax(ps[lo:hi]))
        na = j if ps[j] > thr else a
        lo, hi = max(0, b - win), min(T, b + win + 1)
        j = lo + int(np.argmax(pe[lo:hi]))
        nb = j if pe[j] > thr else b
        if nb - na >= 2 * HZ:
            out[a:b] = False
            out[na:nb] = True
    return out


# ============================================================================ local evaluation
THR_GRID = (None, 0.2, 0.3, 0.45, 0.6)


def make_units6(ywork, preds, fam, src, var):
    """Units with labels (from the VM label tar) + p / ps / pe / pt; p filled on invalid frames."""
    import dead_v5 as d5
    us = []
    for key, pr in preds.items():
        if key[0] != fam or key[1] != src or key[2] != var:
            continue
        mid, h = key[3], key[4]
        f = os.path.join(ywork, f"{src}__{mid}__{h}__{var}_y.npy")
        if not os.path.exists(f):
            continue
        y = np.load(f)
        Y = dict(dead=y[:, 0] == 1, stop=y[:, 1], valid=y[:, 2] == 1, conf=y[:, 3], poss=y[:, 4], ok=y[:, 5] >= 0)
        n = min(len(pr["p"]), len(y))
        u = {k: v[:n] for k, v in Y.items()}
        u["p"] = d5.fill_invalid(pr["p"][:n], u["valid"] & u["ok"])
        u["ps"] = pr["ps"][:n].astype(np.float32) if pr.get("ps") is not None else None
        u["pe"] = pr["pe"][:n].astype(np.float32) if pr.get("pe") is not None else None
        u["pt"] = pr.get("pt")
        u["mid"] = mid
        us.append(u)
    return us


def short_path(m, u, sp, hz=HZ):
    """Second acceptance path, for short stoppages (quick throw-ins and free kicks) that the hysteresis rejects
    because P(dead) has no time to reach p_hi. A run of p >= sp["p_mid"] that the main decoder left live is accepted
    when it lasts sp["min_s"]..sp["max_s"] seconds AND the start head and the end head both peak >= sp["b_thr"]
    within 2 s of its edges. Returns (mask with the accepted runs added, mask of the added runs only).
    Off unless the recipe has a "short" entry; its values must be chosen on external matches (dead_v6_short_tune.py)."""
    import dead_v5 as d5
    out, added = m.copy(), np.zeros(len(m), bool)
    if not sp or u.get("ps") is None or u.get("pe") is None:
        return out, added
    p, ps, pe, w = u["p"], u["ps"], u["pe"], 2 * hz
    for a, b in d5.runs_of(np.asarray(p) >= sp["p_mid"]):
        if m[a:b].any() or not (sp["min_s"] * hz <= b - a <= sp["max_s"] * hz):
            continue
        if ps[max(a - w, 0):a + w + 1].max() >= sp["b_thr"] and pe[max(b - w, 0):b + w + 1].max() >= sp["b_thr"]:
            out[a:b] = True
            added[a:b] = True
    return out, added


def decode6_parts(u, prm):
    """(decoded mask, mask of the runs that came from the short path)."""
    import dead_v5 as d5
    m = d5.decode5(u["p"], prm)
    if prm.get("thr_b") is not None and u.get("ps") is not None:
        m = refine(m, u["ps"], u["pe"], prm["thr_b"])
    return short_path(m, u, prm.get("short"))


def decode6(u, prm):
    return decode6_parts(u, prm)[0]


def choose6(pool, kind):
    """Decoder on the pool (other sources), then the boundary threshold given that decoder."""
    import dead_v5 as d5
    prm = dict(d5.choose(pool, d5.HYST_GRID if kind == "hyst" else d5.HSMM_GRID))
    if all(u.get("ps") is not None for u in pool):
        best = None
        for thr in THR_GRID:
            q = dict(prm, thr_b=thr)
            c = sum(d5._cost(u["dead"], decode6(u, q), u["ok"]) for u in pool)
            if best is None or c < best[0]:
                best = (c, thr)
        prm["thr_b"] = best[1]
    return prm


def evaluate6(ywork, preds, fam, pm, kinds=("hyst", "hsmm"), fixed=None, log_=print):
    import pandas as pd
    import dead_v5 as d5
    rows, rtype, chosen = [], [], {}
    srcs = [s for s in ("sportec", "skillcorner", "metrica") if any(k[0] == fam and k[1] == s for k in preds)]
    for kind in kinds:
        for src in srcs:
            if kind == "fixed":
                prm = fixed
            else:
                pool = []
                for o in srcs:
                    if o != src:
                        pool += make_units6(ywork, preds, fam, o, NOMINAL[o])
                if not pool:
                    continue
                prm = choose6(pool, kind)
            chosen[(fam, kind, src)] = prm
            for var in TEST_VARS[src]:
                us = make_units6(ywork, preds, fam, src, var)
                if not us:
                    continue
                frs, rr = [], []
                for u in us:
                    pred = decode6(u, prm)
                    pt = u["pt"].astype(np.float32) if u["pt"] is not None else None
                    f, r = d5.half_metrics(u, pred, pm, pt, np.arange(1, 9) if pt is not None else None)
                    frs.append(f)
                    rr += r
                nm = len({u["mid"] for u in us})
                sm = d5.summarise5(frs, rr, nm)
                rows.append(dict(fam=fam, decodeur=kind, src=src, var=var, n_matchs=nm, **sm, params=json.dumps(prm)))
                rtype.append(d5.per_type5(rr, f"{fam}|{kind}|{src}|{var}"))
                log_(f"{fam} {kind} {src} {var}: F1arret {sm['F1_arret_iou05']:.3f} faux/m {sm['faux_arrets_par_match']:.2f} spell {sm.get('spell_proxy', np.nan):.3f}")
    return pd.DataFrame(rows), (pd.concat(rtype) if rtype else pd.DataFrame()), chosen


def ensemble6(A, fa, B, fb, name, wa=0.5):
    """p / pt averaged; boundary heads (ps / pe) taken from whichever member has them."""
    out = {}
    for k, v in A.items():
        if k[0] != fa:
            continue
        kb = (fb,) + k[1:]
        if kb not in B:
            continue
        w = B[kb]
        n = min(len(v["p"]), len(w["p"]))
        r = dict(p=(wa * v["p"][:n] + (1 - wa) * w["p"][:n]).astype(np.float32))
        for h in ("ps", "pe"):
            src_ = v if v.get(h) is not None else w
            r[h] = src_[h][:n] if src_.get(h) is not None else None
        if v.get("pt") is not None and w.get("pt") is not None:
            r["pt"] = (wa * v["pt"][:n].astype(np.float32) + (1 - wa) * w["pt"][:n].astype(np.float32)).astype(np.float16)
        else:
            r["pt"] = v.get("pt") if v.get("pt") is not None else w.get("pt")
        out[(name,) + k[1:]] = r
    return out


OUT_TYPES = {1: "touch", 2: "goal", 3: "goal"}          # throw-in -> touchlines; corner, goal kick -> goal lines


def ball_veto(d5_, added, st5, a, bx, by, veto):
    """Drops a SHORT-PATH run typed throw-in / corner / goal kick when the cleaned ball (ball_clean.py) was seen
    around its start and never came within veto["margin_m"] of the line it should have crossed. Runs of the main
    decoder are never touched (their ball often leaves in the air or unseen), nor are free kicks. Unknown ball: kept."""
    import ball_clean as bc
    import dead_v5 as d5
    out, n = d5_.copy(), 0
    for s, e in d5.runs_of(added):
        line = OUT_TYPES.get(int(st5[s]))
        if line is None:
            continue
        f0 = a + s * STEP
        m = bc.line_margin(bx, by, f0 - int(veto["pre_s"] * 25), f0 + int(veto["post_s"] * 25), line)
        if np.isfinite(m) and m > veto["margin_m"]:
            out[s:e] = False
            n += 1
    return out, n


def apply_bar(bar_x5, v6_pred, v4_pkl, recipe, ball=None, aerial=None):
    """BAR-ATH output with the dead_v4.npz interface. recipe: {"w_v4": w, "decode": {...}} (w_v4 = 0 -> v6 only,
    1 -> v4 only). Decoding at 5 Hz, boundary refinement with the v6 start / end heads, nearest upsampling.
    Optional, both off in decode_v6.json: recipe["decode"]["short"] (short_path) and recipe["veto"] (ball_veto,
    needs ball = balle_fusionnee.npz and aerial = aerial.npz, already loaded)."""
    import pickle as pk
    import v1
    import dead_v4 as dv
    import dead_v5 as d5
    zx = np.load(bar_x5)
    zp = np.load(v6_pred)
    M = pk.load(open(v4_pkl, "rb"))
    N = v1.N_FRAMES
    res = dict(p_dead=np.full(N, np.nan, np.float32), dead=np.zeros(N, bool), stop_type=np.zeros(N, np.int8),
               graded=np.zeros(N, bool), p_type=np.zeros((N, 8), np.float16))
    runs_tab = []
    w = float(recipe["w_v4"])
    for a, b in v1.HALVES:
        X = zx[f"X_{a}"].astype(np.float32)
        p4 = M["dead"].predict_proba(X[:, :NCOL])[:, 1]
        pt4 = d5._std_types(M["type"], M["type"].predict_proba(X[:, :NCOL])).astype(np.float64)
        p6, pt6 = zp[f"p_{a}"].astype(np.float64), zp[f"pt_{a}"].astype(np.float64)
        n = min(len(p4), len(p6))
        p5 = w * p4[:n] + (1 - w) * p6[:n]
        pt5 = w * pt4[:n] + (1 - w) * pt6[:n]
        u = dict(p=p5, ps=zp[f"ps_{a}"][:n].astype(np.float64), pe=zp[f"pe_{a}"][:n].astype(np.float64))
        d5_, added = decode6_parts(u, recipe["decode"])
        st5 = dv.run_types(d5_, pt5, np.arange(1, 9))
        res.setdefault("short_added", 0)
        res.setdefault("short_vetoed", 0)
        res["short_added"] += len(d5.runs_of(added))
        if recipe.get("veto") and added.any():
            if ball is None:
                raise ValueError("recipe['veto'] needs the fused ball (and aerial.npz for the flight correction)")
            import ball_clean as bc
            if "_clean" not in res:
                res["_clean"] = bc.clean_ball(ball, aerial, recipe["veto"])
            d5_, n_veto = ball_veto(d5_, added, st5, a, res["_clean"][0], res["_clean"][1], recipe["veto"])
            res["short_vetoed"] += n_veto
            st5 = dv.run_types(d5_, pt5, np.arange(1, 9))
        T = b - a
        idx5 = np.arange(n) * STEP
        up = np.minimum(np.arange(T) // STEP, n - 1)
        res["graded"][a:b] = True
        res["p_dead"][a:b] = np.interp(np.arange(T), idx5, p5).astype(np.float32)
        res["dead"][a:b] = d5_[up]
        res["stop_type"][a:b] = st5[up]
        res["p_type"][a:b] = pt5[up].astype(np.float16)
        for s0, e0 in d5.runs_of(res["dead"][a:b]):
            runs_tab.append((a + s0, a + e0, int(res["stop_type"][a + s0]), float(res["p_dead"][a + s0:a + e0].mean())))
    res["runs"] = np.array(runs_tab, np.float64).reshape(-1, 4)
    res.pop("_clean", None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vm", "bar-tokens"])
    ap.add_argument("--root")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--nproc", type=int, default=7)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--data")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "vm":
        main_vm(a)
    else:
        bar_tokens(a.data, a.out)


if __name__ == "__main__":
    main()

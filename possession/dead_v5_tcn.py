"""MS-TCN-lite for D v5 (CPU, torch), trained on the dead_v5 5 Hz feature cache (external data only).

Input per 5 Hz step: the 217 D v4 features (robust-standardised on the training rows, NaN -> 0) plus a
validity flag. Trunk: 1x1 projection to 64 channels, then 2 stages x 8 dilated residual layers
(dilations 1..128, kernel 3, dropout 0.3, non-causal: receptive field ~ +-100 s); the second stage
refines the first stage's outputs (MS-TCN, Farha & Gall 2019). Heads: P(D), D start, D end (Gaussian
targets sigma = 1 s, BCE pos_weight 20), stop type (8 classes, CE on dead steps). Loss = BCE(D) +
0.15 TMSE (tau 4) + boundaries + 0.5 type; per-step weights = source weight x label confidence
(SkillCorner 0.5; confidence 0 -> 0) x validity. Crops of 96 s drawn on the fly.

  python possession/dead_v5_tcn.py --work <dead_v5 cache> --fold transfer:skillcorner
  folds: transfer:<src> | lomo:sportec:<mid> | lomo:metrica:<mid> | lomo:skillcorner:<k>  | all (final model)
Writes <work>/pred_tcn_<fold>.pkl with keys ("tcn", src, var, mid, h) -> {p, pt} (same as dead_v5).
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

HZ = 5
SRC_W = {"sportec": 1.0, "metrica": 1.0, "skillcorner": 0.5}
TRAIN_VARS = {"sportec": ("train0", "train1"), "metrica": ("train0", "train1"), "skillcorner": ("real", "mask")}
TEST_VARS = {"sportec": ("test0", "stress0"), "metrica": ("test0", "stress0", "noball"), "skillcorner": ("real", "noball")}
N_SC_FOLDS = 5
CROP = 480


def units(work, src, var):
    out = []
    for f in sorted(glob.glob(os.path.join(work, f"{src}__*__{var}_y.npy"))):
        _, mid, h, _ = os.path.basename(f)[:-6].split("__")
        out.append((mid, h, f[:-6]))
    return out


def matches(work, src):
    return sorted({m for m, _, _ in units(work, src, TRAIN_VARS[src][0])})


class Stage(nn.Module):
    def __init__(self, cin, ch=64, layers=8, drop=0.3):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.convs = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
        self.mix = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(layers)])
        self.drop = nn.Dropout(drop)
        self.out = nn.Conv1d(ch, 11, 1)             # dead, start, end, 8 types

    def forward(self, x):
        h = self.inp(x)
        for c, m in zip(self.convs, self.mix):
            h = h + self.drop(m(Fnn.relu(c(h))))
        return self.out(h)


class TCN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = Stage(cin)
        self.s2 = Stage(11)

    def forward(self, x):
        o1 = self.s1(x)
        o2 = self.s2(torch.cat([torch.sigmoid(o1[:, :3]), torch.softmax(o1[:, 3:], 1)], 1))
        return [o1, o2]


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


def load_half(stem, ncol, mmap=False):
    X = np.load(stem + "_X5.npy", mmap_mode="r" if mmap else None)[:, :ncol]
    y = np.load(stem + "_y.npy")
    return X, y


def build_train(work, include, ncol):
    data = []
    for src, mids in include.items():
        for var in TRAIN_VARS[src]:
            for mid, h, stem in units(work, src, var):
                if mid not in mids:
                    continue
                X, y = load_half(stem, ncol, mmap=True)
                dead, stop, valid, conf, ok = y[:, 0] == 1, y[:, 1], y[:, 2] == 1, y[:, 3], y[:, 5] >= 0
                w = SRC_W[src] * np.where(conf >= 2, 1.0, np.where(conf == 1, 0.5, 0.0)) * (valid & ok)
                st, en = gauss_targets(dead & ok)
                data.append(dict(X=X, dead=dead.astype(np.float32), w=w.astype(np.float32), st=st, en=en,
                                 stop=np.clip(stop.astype(np.int64) - 1, -1, 7), valid=valid.astype(np.float32)))
    return data


def norm_stats(data):
    sample = np.concatenate([np.asarray(d["X"][::11]).astype(np.float32) for d in data])
    med = np.nanmedian(sample, 0)
    q = np.nanpercentile(sample, [25, 75], 0)
    iqr = np.where((q[1] - q[0]) > 1e-6, q[1] - q[0], 1.0)
    return np.nan_to_num(med).astype(np.float32), np.nan_to_num(iqr, nan=1.0).astype(np.float32)


def to_input(X, valid, med, iqr):
    Z = (np.asarray(X).astype(np.float32) - med) / iqr
    Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
    return np.concatenate([Z, valid[:, None].astype(np.float32)], 1).T     # (C, T)


def train(data, med, iqr, epochs=12, seed=0, log=print, iters=120):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = data[0]["X"].shape[1] + 1
    model = TCN(cin)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    lens = np.array([len(d["dead"]) for d in data])
    n_crops = min(int(lens.sum() / CROP), 16 * iters)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (n_crops // 16 + 1))
    pw = torch.tensor(20.0)
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        model.train()
        for it in range(n_crops // 16):
            xs, ys = [], []
            for _ in range(16):
                i = rng.choice(len(data), p=lens / lens.sum())
                d = data[i]
                s = int(rng.integers(0, max(1, len(d["dead"]) - CROP)))
                sl = slice(s, s + CROP)
                xs.append(to_input(d["X"][sl], d["valid"][sl], med, iqr))
                ys.append(np.stack([d["dead"][sl], d["w"][sl], d["st"][sl], d["en"][sl], d["stop"][sl].astype(np.float32)]))
            L = min(x.shape[1] for x in xs)
            x = torch.from_numpy(np.stack([a[:, :L] for a in xs]))
            y = torch.from_numpy(np.stack([a[:, :L] for a in ys]))
            dead, w, st, en, stop = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4].long()
            loss = 0.0
            for o in model(x):
                ld = (Fnn.binary_cross_entropy_with_logits(o[:, 0], dead, reduction="none") * w).sum() / w.sum().clamp(min=1)
                lp = Fnn.logsigmoid(o[:, 0:1])
                tm = torch.clamp((lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2, max=16).mean()
                wb = (w > 0).float()
                lb = ((Fnn.binary_cross_entropy_with_logits(o[:, 1], st, pos_weight=pw, reduction="none")
                       + Fnn.binary_cross_entropy_with_logits(o[:, 2], en, pos_weight=pw, reduction="none")) * wb).sum() / wb.sum().clamp(min=1) / 20
                m = (dead > 0.5) & (stop >= 0) & (w > 0)
                lt = Fnn.cross_entropy(o[:, 3:].permute(0, 2, 1)[m], stop[m]) if m.any() else torch.tensor(0.0)
                loss = loss + ld + 0.15 * tm + lb + 0.5 * lt
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        log(f"  epoch {ep}: loss {tot / max(1, n_crops // 16):.3f} ({time.time() - t0:.0f}s)")
    return model


def predict(model, X, valid, med, iqr):
    model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(to_input(X, valid, med, iqr)[None]))[-1][0]
    return dict(p=torch.sigmoid(o[0]).numpy().astype(np.float32), ps=torch.sigmoid(o[1]).numpy().astype(np.float16),
                pe=torch.sigmoid(o[2]).numpy().astype(np.float16), pt=torch.softmax(o[3:], 0).numpy().T.astype(np.float16))


def run_fold(work, fold, epochs, log=print, iters=120):
    ncol = 217
    allm = {s: matches(work, s) for s in ("sportec", "metrica", "skillcorner")}
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
    data = build_train(work, inc, ncol)
    med, iqr = norm_stats(data)
    log(f"fold {fold}: {len(data)} train halves, {sum(len(d['dead']) for d in data)} steps")
    model = train(data, med, iqr, epochs, log=log, iters=iters)
    del data
    out = {}
    for src, mids in held.items():
        for var in TEST_VARS[src]:
            for mid, h, stem in units(work, src, var):
                if mid not in mids:
                    continue
                X, y = load_half(stem, ncol)
                out[("tcn", src, var, mid, h)] = predict(model, X, y[:, 2] == 1, med, iqr)
    if not held:
        torch.save(dict(state=model.state_dict(), med=med, iqr=iqr, ncol=ncol), os.path.join(work, "tcn_final.pt"))
    return out


def predict_bar(work, bar_in, out):
    """Final TCN (fold 'all') on the BAR-ATH 5 Hz v4 features (inputs only) -> p_<start>, pt_<start>."""
    ck = torch.load(os.path.join(work, "tcn_final.pt"), weights_only=False)
    z = np.load(bar_in)
    X0 = z[z.files[0]]
    model = TCN(X0.shape[1] + 1)
    model.load_state_dict(ck["state"])
    res = {}
    for k in z.files:
        X = z[k]
        r = predict(model, X, np.ones(len(X), bool), ck["med"], ck["iqr"])
        s = k.split("_", 1)[1]
        res[f"p_{s}"], res[f"pt_{s}"] = r["p"], r["pt"]
    np.savez_compressed(out, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-bar", default=None, help="bar_tcn_in.npz -> writes --bar-out")
    ap.add_argument("--bar-out", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--fold", default=[], nargs="*")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--iters", type=int, default=120, help="iterations (batches of 16 crops) per epoch")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.predict_bar:
        predict_bar(a.work, a.predict_bar, a.bar_out)
        return
    for fold in a.fold:
        path = os.path.join(a.work, f"pred_tcn_{fold.replace(':', '_')}.pkl")
        if os.path.exists(path):
            continue
        t0 = time.time()
        out = run_fold(a.work, fold, a.epochs, log=lambda m: print(m, flush=True), iters=a.iters)
        pickle.dump(out, open(path + ".tmp", "wb"))
        os.replace(path + ".tmp", path)
        print(f"fold {fold} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

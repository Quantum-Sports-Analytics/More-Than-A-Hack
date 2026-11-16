"""Stage 1 (dead ball) final assembly: dead-ball detector (D v4, or D v2 as fallback) + TV signal.

The TV signal (possession/tv_signal.py) is NOT tunable on external data (Sportec has no broadcast),
so it is only allowed to CONFIRM or EXTEND a dead ball found by the detector, never to create one.
All constants below are fixed a priori (broadcast-production reasoning) or come from the detector's
own decoder chosen on Sportec (p_lo); none is chosen on the provided match.

Rules
  R1 merge  : two detector D runs separated by a live gap <= MERGE_GAP_S, in which >= MERGE_COVER
              of the frames are TV non-live (off-pitch / replay / unmatched pitch view), are merged
              (a director does not cut away from a few seconds of live play between two stoppages).
  R2 extend : frames contiguous to a detector D run, with p_dead >= p_lo (the detector's own
              hysteresis low threshold) AND TV non-live, are added to the run (at most EXT_MAX_S
              per side). Needs a per-frame p_dead (D v4); skipped for the D v2 CSV.
  Never     : a frame with no contact with a detector D run is never set to D by the TV.

Outputs (per TACTICAL video frame 0..N-1, N = 199 319)
  stage1.npz : dead bool, p_dead float32 (NaN if unknown), stop_type int8 (codes of dead_v4.py),
               d_source int8 (0 live, 1 detector, 2 TV merge R1, 3 TV extension R2), tv_state uint8,
               graded bool, runs int32 (n,4: start, end_exclusive, stop_type, d_source_max)
  stage1.csv : frame,dead,p_dead,stop_type,stop_name,d_source,tv_state,graded
  stage1_runs.csv : one line per dead-ball run.

Usage
  python possession/stage1.py --dead dead_v4.npz --decode-json out/decode_v4.json --tv tv_signal.npz --out stage1
  python possession/stage1.py --dead v2_dead_ball.csv --tv tv_signal.npz --out stage1_v2   # fallback
"""
import argparse
import json

import numpy as np
import pandas as pd

FPS = 25
N_FRAMES = 199319
GRADED = [(11185, 90652), (110808, 190198)]
STOP_NAMES = ["vivant", "touche", "corner", "six_metres", "coup_franc", "engagement", "penalty",
              "balle_arbitre", "inconnu"]
TV_NONLIVE = (2, 3, 4)          # off-pitch, identified replay, pitch view not matching live play
MERGE_GAP_S = 4.0
MERGE_COVER = 0.8
EXT_MAX_S = 5.0


def graded_mask(n=N_FRAMES):
    g = np.zeros(n, bool)
    for a, b in GRADED:
        g[a:b] = True
    return g


def runs_of(m):
    m = np.asarray(m, bool).astype(np.int8)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def load_dead(path, decode_json=None):
    if path.endswith(".npz"):
        z = np.load(path)
        dead = z["dead"].astype(bool)
        p = z["p_dead"].astype(np.float32) if "p_dead" in z.files else np.full(len(dead), np.nan, np.float32)
        st = z["stop_type"].astype(np.int8) if "stop_type" in z.files else np.where(dead, 8, 0).astype(np.int8)
        p_lo = None
        if decode_json:
            dj = json.load(open(decode_json))
            p_lo = float(dj["p_lo"] if "p_lo" in dj else dj["decode"]["p_lo"])   # dead_v4 / dead_v5 recipes
        return dead, p, st, p_lo
    d = pd.read_csv(path)
    lab = d["label"].to_numpy() if "label" in d else d.iloc[:, 1].to_numpy()
    dead = lab == "D"
    return dead, np.full(len(dead), np.nan, np.float32), np.where(dead, 8, 0).astype(np.int8), None


def combine(dead, p, st, tv_state, p_lo=None):
    n = len(dead)
    nonlive = np.isin(tv_state, TV_NONLIVE)
    src = np.where(dead, 1, 0).astype(np.int8)
    out = dead.copy()
    st = st.copy()
    rs = runs_of(out)
    # R1 merge
    for (a0, b0), (a1, b1) in zip(rs[:-1], rs[1:]):
        gap = a1 - b0
        if 0 < gap <= MERGE_GAP_S * FPS and nonlive[b0:a1].mean() >= MERGE_COVER:
            out[b0:a1] = True
            src[b0:a1] = 2
            st[b0:a1] = st[a1]            # the stoppage type is the one of the restart that ends it
    # R2 extension (needs p_dead and the detector's p_lo)
    if p_lo is not None and np.isfinite(p).any():
        ok = nonlive & (p >= p_lo)
        emax = int(EXT_MAX_S * FPS)
        for a, b in runs_of(out):
            i = a - 1
            while i >= 0 and a - i <= emax and ok[i] and not out[i]:
                i -= 1
            if i + 1 < a:
                out[i + 1:a] = True; src[i + 1:a] = 3; st[i + 1:a] = st[a]
            j = b
            while j < n and j - b < emax and ok[j] and not out[j]:
                j += 1
            if j > b:
                out[b:j] = True; src[b:j] = 3; st[b:j] = st[b - 1]
    st[~out] = 0
    st[out & (st == 0)] = 8
    return out, st, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dead", required=True, help="dead_v4.npz or a per-frame CSV with label D")
    ap.add_argument("--decode-json", default=None)
    ap.add_argument("--tv", required=True, help="tv_signal.npz")
    ap.add_argument("--out", default="stage1")
    a = ap.parse_args()
    dead, p, st, p_lo = load_dead(a.dead, a.decode_json)
    tv = np.load(a.tv)["tv_state"]
    assert len(dead) == len(tv) == N_FRAMES, (len(dead), len(tv))
    g = graded_mask()
    out, st2, src = combine(dead & g, p, st, tv, p_lo)
    out &= g
    runs = []
    for s, e in runs_of(out):
        runs.append((s, e, int(np.bincount(st2[s:e]).argmax()), int(src[s:e].max())))
    runs = np.array(runs, np.int32).reshape(-1, 4)
    np.savez_compressed(a.out + ".npz", dead=out, p_dead=p, stop_type=st2, d_source=src, tv_state=tv,
                        graded=g, runs=runs, stop_type_codes=json.dumps(dict(enumerate(STOP_NAMES))))
    pd.DataFrame(dict(frame=np.arange(N_FRAMES), dead=out.astype(int), p_dead=np.round(p, 4), stop_type=st2,
                      stop_name=np.array(STOP_NAMES)[st2], d_source=src, tv_state=tv, graded=g.astype(int))
                 ).to_csv(a.out + ".csv", index=False)
    rd = pd.DataFrame(runs, columns=["debut", "fin_exclue", "stop_type", "d_source_max"])
    rd["duree_s"] = (rd.fin_exclue - rd.debut) / FPS
    rd["stop_name"] = np.array(STOP_NAMES)[rd.stop_type]
    rd.to_csv(a.out + "_runs.csv", index=False)
    summ = dict(n_runs=len(runs), frac_D_graded=float(out[g].mean()), frac_D_detector=float((dead & g)[g].mean()),
                frames_R1=int((src == 2).sum()), frames_R2=int((src == 3).sum()), p_lo=p_lo,
                runs_by_type=rd.stop_name.value_counts().to_dict(),
                median_dur_by_type=rd.groupby("stop_name").duree_s.median().round(1).to_dict())
    json.dump(summ, open(a.out + "_resume.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()

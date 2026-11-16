"""Chooses the short-stoppage path of D v6 (dead_v6_tcn.short_path) on EXTERNAL matches only.

The main decoder of decode_v6.json is kept as it is; only the extra acceptance path for short stoppages
(quick throw-ins and free kicks) is searched. Nothing here reads the provided match.

Inputs, both written by the GPU training run of dead_v6_tcn.py (`vm` command), in <root>/results/:
  pred_*.pkl     out-of-fold predictions, keys ("v6", source, variant, match, half) -> p, ps, pe, pt at 5 Hz
                 (transfer:* folds = a whole source held out; lomo:* folds = one match held out)
  labels_y.tar   per-half labels (<source>__<match>__<half>__<variant>_y.npy), extracted to --labels

Protocol (the one dead_v6_tcn.choose6 uses for the decoder): for each source, the setting is chosen on the OTHER
sources (lowest pooled cost of dead_v5._cost: wrong frames + a penalty per false stoppage) and scored on that
source. "off" is always in the grid, so a source where the short path only adds false stoppages keeps it off.
The final setting is the one with the lowest cost over all sources.

  python possession/dead_v6_short_tune.py --results <root>/results --labels <dir with *_y.npy> \
      --recipe outputs/etage1/decode_v6.json --out-dir <dir>
  -> short_tune.csv (per source: before / after), decode_v6_short.json (recipe with the chosen "short" entry,
     or identical to the input recipe if "off" wins)

The ball-in-pitch veto (recipe["veto"]) is NOT tuned here: the predictions carry no ball track. Its constants are
physical orders of magnitude fixed beforehand (see ball_clean.py), like the TV rules of stage1.py.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dead_v5 as d5  # noqa: E402
import dead_v6_tcn as d6  # noqa: E402

GRID = [None] + [dict(p_mid=pm_, b_thr=bt, min_s=mn, max_s=mx)
                 for pm_ in (0.4, 0.5, 0.6, 0.7) for bt in (0.7, 0.8, 0.9, 0.95) for mn in (1.5, 2.0, 3.0) for mx in (10.0, 20.0)]


def load_preds(results):
    """All out-of-fold predictions; a whole-source hold-out (transfer) wins over a one-match hold-out (lomo)."""
    preds = {}
    files = sorted(glob.glob(os.path.join(results, "pred_*.pkl")), key=lambda f: "transfer" not in os.path.basename(f))
    for f in files:
        if os.path.basename(f) == "pred_all.pkl":
            continue                                   # the final model saw every match: not out of fold
        for k, v in pickle.load(open(f, "rb")).items():
            preds.setdefault(k, v)
    return preds


def score(units, prm, metrics_module=None):
    frs, rows = [], []
    for u in units:
        pred = d6.decode6(u, prm)
        f, r = d5.half_metrics(u, pred, metrics_module)
        frs.append(f)
        rows += r
    return d5.summarise5(frs, rows, len({u["mid"] for u in units}))


def cost(units, prm):
    return sum(d5._cost(u["dead"], d6.decode6(u, prm), u["ok"]) for u in units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scorer", default=None, help="folder of possession_metrics.py, for the spellF1 proxy (optional)")
    a = ap.parse_args()
    metrics_module = None
    if a.scorer:
        sys.path.insert(0, a.scorer)
        import possession_metrics as metrics_module
    recipe = json.load(open(a.recipe))
    base = {k: v for k, v in recipe["decode"].items() if k != "short"}
    preds = load_preds(a.results)
    srcs = [s for s in ("sportec", "skillcorner", "metrica") if any(k[0] == "v6" and k[1] == s for k in preds)]
    if not srcs:
        raise SystemExit("no ('v6', ...) predictions in " + a.results)
    nominal = {s: d6.make_units6(a.labels, preds, "v6", s, d6.NOMINAL[s]) for s in srcs}
    print({s: len(u) for s, u in nominal.items()}, "halves per source")
    costs = {s: [cost(nominal[s], dict(base, short=g)) for g in GRID] for s in srcs}
    rows = []
    for s in srcs:
        others = [o for o in srcs if o != s]
        best = int(np.argmin(np.sum([costs[o] for o in others], 0))) if others else 0
        for var in d6.TEST_VARS[s]:
            us = d6.make_units6(a.labels, preds, "v6", s, var)
            if not us:
                continue
            for name, g in (("base", None), ("short", GRID[best])):
                sm = score(us, dict(base, short=g), metrics_module)
                rows.append(dict(source=s, variant=var, decoder=name, short=json.dumps(g), n_matches=len({u["mid"] for u in us}),
                                 F1_stoppage=sm["F1_arret_iou05"], false_per_match=sm["faux_arrets_par_match"],
                                 found=sm["trouves"], F1_frames=sm["F1_all"], spell_proxy=sm.get("spell_proxy", np.nan)))
    final = GRID[int(np.argmin(np.sum([costs[s] for s in srcs], 0)))]
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "short_tune.csv"), index=False)
    print(df.round(3).to_string(index=False))
    out = dict(recipe, decode=dict(base, **({"short": final} if final else {})))
    json.dump(out, open(os.path.join(a.out_dir, "decode_v6_short.json"), "w"), indent=1, ensure_ascii=False)
    print("chosen on all sources:", final or "off (the short path does not pay on external matches)")


if __name__ == "__main__":
    main()

"""D v6 on BAR-ATH from the committed weights, without retraining (CPU, about 1 minute).

Same computation as the `bar` branch of dead_v6_tcn.run_fold (which only runs at the end of the GPU training):
  1. 5 Hz features of the provided match (dead_v5.bar_features: 217 D v4 features + context)   -> bar_X5.npz
  2. player tokens (dead_v6_tcn.bar_tokens)                                                     -> bar_tok5.npz
  3. MS-TCN, mean of the 3 seeds, inputs normalised with the med / iqr stored in each checkpoint -> bar_v6_pred.npz
  4. decoding with the recipe (dead_v6_tcn.apply_bar)                                            -> dead_v6.npz
On CPU the network probabilities differ from the GPU run by < 1e-3; the decoded stoppages are identical.
Usage: python possession/dead_v6_infer.py --data data --ball outputs/intermediaires/balle_fusionnee.npz \
           --models models/etage1 --recipe outputs/etage1/decode_v6.json --out-dir <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dead_v5 as d5  # noqa: E402
import dead_v6_tcn as d6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ball", required=True)
    ap.add_argument("--models", default="models/etage1")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aerial", default=None, help="aerial.npz, only read when the recipe has a 'veto' entry")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    bx, bt = os.path.join(a.out_dir, "bar_X5.npz"), os.path.join(a.out_dir, "bar_tok5.npz")
    bp = os.path.join(a.out_dir, "bar_v6_pred.npz")

    hf = d5.bar_features(a.data, a.ball, None)
    np.savez_compressed(bx, columns=json.dumps(list(hf[0][2].columns)),
                        **{f"X_{s0}": F.values[::d5.STEP].astype(np.float16) for s0, e0, F in hf})
    d6.bar_tokens(a.data, bt)
    zx, zt = np.load(bx), np.load(bt)
    keys = [k[2:] for k in zx.files if k.startswith("X_")]
    per = {s: [] for s in keys}
    for seed in range(3):
        ck = torch.load(os.path.join(a.models, f"tcn_all_s{seed}.pt"), map_location="cpu", weights_only=False)
        model = d6.build_model()
        model.load_state_dict(ck["state"])
        model.float()
        for s in keys:
            X = zx["X_" + s][:, :d6.NCOL]
            Z = (torch.as_tensor(X, dtype=torch.float32) - torch.as_tensor(ck["med"])) / torch.as_tensor(ck["iqr"])
            Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10, 10)
            Z = torch.cat([Z, torch.ones(len(X), 1)], 1).half()          # same rounding as TrainData.norm
            per[s].append(d6.predict_seq(model, Z, torch.as_tensor(zt["P_" + s])))
    res = {f"{k}_{s}": np.mean([r[k].astype(np.float32) for r in per[s]], 0) for s in keys for k in per[s][0]}
    np.savez_compressed(bp, **res)

    recipe = json.load(open(a.recipe))
    ball = np.load(a.ball) if recipe.get("veto") else None
    aerial = np.load(a.aerial, allow_pickle=True) if recipe.get("veto") and a.aerial else None
    out = d6.apply_bar(bx, bp, os.path.join(a.models, "model_v4_final.pkl"), recipe, ball, aerial)
    added, vetoed = out.pop("short_added"), out.pop("short_vetoed")
    if recipe["decode"].get("short"):
        print(f"short path: {added} stoppages added, {vetoed} of them dropped by the ball-in-pitch veto")
    np.savez_compressed(os.path.join(a.out_dir, "dead_v6.npz"), **out, stop_type_codes=json.dumps(d5.STOP_NAMES),
                        decode=json.dumps(recipe, ensure_ascii=False))
    print("stoppages:", len(out["runs"]), "| dead share of graded frames:", round(float(out["dead"][out["graded"]].mean()), 4))


if __name__ == "__main__":
    main()

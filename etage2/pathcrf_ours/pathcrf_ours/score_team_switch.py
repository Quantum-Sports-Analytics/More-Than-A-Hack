"""Score the team-switch penalty sweep (spellF1 option 1) on the out-of-fold SkillCorner files.

For every folder outputs/oof_ts/lam_<λ>/ (written by run_skillcorner_oof.py --team-switch λ on the 20 matches with
the 2-fold B.B models), computes with the challenge's own metric code (evaluation/possession_metrics via
pathcrf_test/score_cv), pooled over the 20 matches, at the 0.2 s step level (truth and prediction on the same grid):
  path      the CRF path's sender team (what the penalty changes), gaps forward-filled inside live stretches;
  viterbi   today's recipe: p_home + two-state Viterbi, cost chosen per fold on the other 18 matches (COSTS grid).
Reports spellF1, macroF1-like frame accuracy on live steps, predicted / true possessions (team spells on live steps).

    .venv/bin/python -m pathcrf_ours.score_team_switch [--root outputs/oof_ts] -> outputs/oof_ts/score_team_switch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import HACK_ROOT, OUTPUTS

sys.path.insert(0, str(HACK_ROOT / "evaluation"))
sys.path.insert(0, str(HACK_ROOT / "pipeline"))
sys.path.insert(0, str(HACK_ROOT / "pathcrf_test"))
from common_pl import folds  # noqa: E402
from score_cv import COSTS, IOUS, spell_counts, spell_from_counts, viterbi_batch  # noqa: E402
import possession_metrics as pm  # noqa: E402

H, A, D = pm.H, pm.A, pm.D                      # the challenge codes: 0, 1, 2 (X = 3)
CODES = np.array([H, A], dtype=np.int8)


def codes(team: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Per step: H / A on live steps from team 0 / 1 (NaN -> forward-filled, else H), D when dead."""
    t = pd.Series(team).ffill().bfill().fillna(0).to_numpy()
    return np.where(alive, CODES[(t > 0.5).astype(int)], D).astype(np.int8)


def possessions(c: np.ndarray) -> int:
    live = c != D
    ch = np.flatnonzero((c[1:] != c[:-1]) & live[1:] & live[:-1])
    starts = np.flatnonzero(live[1:] & ~live[:-1])
    return int(len(ch) + len(starts) + (1 if live[0] else 0))


def score_dir(d: Path, F) -> dict:
    tabs = {int(f.stem.split("_")[-1]): pd.read_parquet(f) for f in sorted(d.glob("oof_bb_*.parquet"))}
    mids = sorted(tabs)
    cost = np.array(COSTS, float)
    counts_path = np.zeros((len(IOUS), 3))
    counts_vit = np.zeros((len(mids), len(COSTS), len(IOUS), 3))
    acc = {"path": [0, 0], "vit": np.zeros((len(COSTS), 2))}
    poss = {"true": 0, "path": 0, "vit": np.zeros(len(COSTS))}
    per_match = {}
    for j, m in enumerate(mids):
        t = tabs[m]
        alive = t["alive_true"].to_numpy(dtype=bool)
        gt = codes(t["team_true"].to_numpy(), alive)
        pth = codes(t["team_path"].to_numpy(), alive)
        c = spell_counts(gt, pth)
        counts_path += c
        per_match[m] = {"path_spell": round(spell_from_counts(c), 4)}
        live = alive & (gt != D)
        acc["path"][0] += int((pth[live] == gt[live]).sum()); acc["path"][1] += int(live.sum())
        poss["true"] += possessions(gt); poss["path"] += possessions(pth)
        ph = pd.Series(t["p_home"].to_numpy()).ffill().bfill().fillna(0.5).to_numpy()
        team = np.zeros((len(t), len(COSTS)), int)
        d_ = np.diff(np.r_[0, alive.astype(int), 0])
        for s, e in zip(np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)):
            team[s:e] = viterbi_batch(np.repeat(ph[s:e, None], len(COSTS), axis=1), cost)   # 0 = home
        for k in range(len(COSTS)):
            pv = np.where(alive, CODES[team[:, k]], D).astype(np.int8)
            counts_vit[j, k] = spell_counts(gt, pv)
            acc["vit"][k] += [int((pv[live] == gt[live]).sum()), int(live.sum())]
            poss["vit"][k] += possessions(pv)
    # per-fold cost choice for the Viterbi baseline (honest), pooled
    oof = np.zeros((len(IOUS), 3)); chosen = {}
    acc_v = [0, 0]; poss_v = 0
    for f in F:
        train = [i for i, m in enumerate(mids) if m not in f]
        pooled = counts_vit[train].sum(axis=0)
        k = int(np.argmax([spell_from_counts(pooled[c]) for c in range(len(COSTS))]))
        for m in f:
            if m in mids:
                i = mids.index(m); chosen[m] = COSTS[k]; oof += counts_vit[i, k]
    best_k = int(np.argmax([spell_from_counts(counts_vit.sum(axis=0)[c]) for c in range(len(COSTS))]))
    return {"matches": len(mids),
            "path": {"spell_f1": round(spell_from_counts(counts_path), 4), "frame_acc": round(acc["path"][0] / acc["path"][1], 4),
                     "possessions": poss["path"]},
            "viterbi_fold_cost": {"spell_f1": round(spell_from_counts(oof), 4), "costs": sorted(set(chosen.values()))},
            "viterbi_best_cost": {"cost": COSTS[best_k], "spell_f1": round(spell_from_counts(counts_vit.sum(axis=0)[best_k]), 4),
                                  "frame_acc": round(acc["vit"][best_k][0] / acc["vit"][best_k][1], 4), "possessions": int(poss["vit"][best_k])},
            "possessions_true": poss["true"], "per_match_path_spell": per_match}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OUTPUTS / "oof_ts"))
    a = ap.parse_args()
    root = Path(a.root)
    F = folds()
    res = {}
    for d in sorted(root.glob("lam_*"), key=lambda p: float(p.name[4:])):
        if len(list(d.glob("oof_bb_*.parquet"))) == 0:
            continue
        lam = float(d.name[4:])
        res[d.name] = score_dir(d, F)
        r = res[d.name]
        print(f"lambda {lam:5.1f} | path: spell {r['path']['spell_f1']:.4f} acc {r['path']['frame_acc']:.4f} poss {r['path']['possessions']}"
              f" | viterbi fold-cost spell {r['viterbi_fold_cost']['spell_f1']:.4f} (best cost {r['viterbi_best_cost']['cost']}: "
              f"{r['viterbi_best_cost']['spell_f1']:.4f}, poss {r['viterbi_best_cost']['possessions']}) | true poss {r['possessions_true']} ({r['matches']} matches)", flush=True)
    # honest choice of lambda: per fold, the lambda with the best pooled path spellF1 on the other 18 matches
    lams = sorted(res, key=lambda k: float(k[4:]))
    if len(lams) > 1 and all(res[k]["matches"] == res[lams[0]]["matches"] for k in lams):
        import pandas as pd
        per = {k: {m: pd.read_parquet(f) for m, f in ((int(f.stem.split("_")[-1]), f) for f in sorted((root / k).glob("oof_bb_*.parquet")))} for k in lams}
        counts = {k: {} for k in lams}
        for k in lams:
            for m, t in per[k].items():
                alive = t["alive_true"].to_numpy(dtype=bool)
                counts[k][m] = spell_counts(codes(t["team_true"].to_numpy(), alive), codes(t["team_path"].to_numpy(), alive))
        mids = sorted(per[lams[0]])
        oof = np.zeros((len(IOUS), 3)); chosen = {}
        for f in F:
            train = [m for m in mids if m not in f]
            best = max(lams, key=lambda k: spell_from_counts(sum(counts[k][m] for m in train)))
            for m in f:
                if m in mids:
                    chosen[m] = float(best[4:]); oof += counts[best][m]
        res["fold_choice"] = {"spell_f1": round(spell_from_counts(oof), 4), "lambda_per_match": chosen}
        print(f"fold-chosen lambda: spell {res['fold_choice']['spell_f1']:.4f}, lambdas {sorted(set(chosen.values()))}")
    json.dump(res, open(root / "score_team_switch.json", "w"), indent=1)


if __name__ == "__main__":
    main()

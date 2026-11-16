"""Full-match spellF1 (challenge scorer) of pathcrf_ours checkpoints on a few SkillCorner matches (plan B metric of
the 3 phaseb validation matches), from run_skillcorner.py outputs.

    .venv/bin/python -m pathcrf_ours.run_skillcorner --weights W --tag T [--team-switch λ] --matches 2010085 2011166 2013725
    .venv/bin/python -m pathcrf_ours.score_val_matches --tags T1 T2 ... [--matches ...]
Per tag: p_home + two-state Viterbi at each cost of the grid, and the decoded path's own team (team_pcrf; what
--team-switch changes). Frames: SkillCorner live frames with a known team, replays out (pathcrf_test/score_skillcorner).
Writes pathcrf_test/outputs/score_val_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .paths import HACK_ROOT

sys.path.insert(0, str(HACK_ROOT / "evaluation"))
sys.path.insert(0, str(HACK_ROOT / "pathcrf_test"))
from score_skillcorner import codes_from_team, decode, scores  # noqa: E402

PIPE = HACK_ROOT / "pipeline" / "cache"
OUT = HACK_ROOT / "pathcrf_test" / "outputs"
VAL = [2010085, 2011166, 2013725]
COSTS = (3, 5, 7, 10, 15, 20, 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--matches", nargs="*", type=int, default=VAL)
    a = ap.parse_args()
    for tag in a.tags:
        parts = []
        for m in a.matches:
            t = pd.read_parquet(PIPE / f"sc_{m}.parquet", columns=["match", "frame", "replay", "dead", "team", "blue_team"])
            pc = pd.read_parquet(OUT / f"sc_{m}_{tag}.parquet")
            parts.append(t.merge(pc.drop(columns="match"), on="frame", how="left"))
        df = pd.concat(parts, ignore_index=True)
        usable = ~df["replay"].to_numpy().astype(bool)
        dead = df["dead"].to_numpy().astype(bool)
        team_true = df["team"].to_numpy()
        live = usable & ~dead & (team_true >= 0)
        live_dec = usable & ~dead
        gt = codes_from_team(df, np.where(live, team_true, -1), usable)
        ph = df["p_home_pcrf"].ffill().bfill().fillna(0.5).to_numpy()
        res = {"tag": tag, "matches": a.matches, "live_frames": int(live.sum()),
               "covered_share": round(float(df.loc[live, "covered"].fillna(False).mean()), 4), "viterbi": {}}
        for c in COSTS:
            team = decode(df, ph, c, live_dec)
            res["viterbi"][str(c)] = {k: round(v, 4) if isinstance(v, float) else v for k, v in scores(df, gt, codes_from_team(df, team, usable)).items() if not isinstance(v, dict)}
        tp = pd.Series(df["team_pcrf"].to_numpy()).ffill().bfill().fillna(0).to_numpy().astype(int)
        res["path"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in scores(df, gt, codes_from_team(df, np.where(live_dec, tp, -1), usable)).items() if not isinstance(v, dict)}
        best = max(res["viterbi"], key=lambda c: res["viterbi"][c]["spell_f1"])
        print(f"{tag:22s} path spell {res['path']['spell_f1']:.4f} macro {res['path']['macro_f1']:.4f} poss {res['path']['possessions_pred']}/{res['path']['possessions_true']} | "
              f"viterbi best cost {best}: {res['viterbi'][best]['spell_f1']:.4f}; cost 7 {res['viterbi']['7']['spell_f1']:.4f} cost 10 {res['viterbi']['10']['spell_f1']:.4f} | live {res['live_frames']}", flush=True)
        json.dump(res, open(OUT / f"score_val_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

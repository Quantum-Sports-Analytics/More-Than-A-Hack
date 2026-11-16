"""A pathcrf_ours checkpoint on the 20 SkillCorner matches, written in the format of pathcrf_test/run_skillcorner.py
(the paper-weights runner) so the stacking heads (pathcrf_test/stack_v1*.py) can be fitted on them.

    .venv/bin/python -m pathcrf_ours.run_skillcorner --weights outputs/runs/replica_fold6/model/best_acc_ep16.pt --tag phaseA
    -> pathcrf_test/outputs/sc_<match>_<tag>.parquet (match, frame, p_home_pcrf, team_pcrf, covered)

Input: the processed SkillCorner tables of Phase B (cache/processed/sc_<match>.parquet, 25 fps interpolated from
10 fps, episodes = our dead-ball rules on the possession events). Back to 10 fps: frame = round(0.4 * frame_id).
Only valid as stacking input for a model that never trained on SkillCorner (Phase A: IDSSE only). Phase B / B.B
trained on 17 of these matches, so their SkillCorner predictions would leak into a head fitted on them.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .data.skillcorner import SPLITS
from .infer import infer_match, load_net, pick_device
from .paths import HACK_ROOT

OUT = HACK_ROOT / "pathcrf_test" / "outputs"
CACHE = Path(__file__).resolve().parents[1] / "cache" / "processed"


def all_matches() -> list:
    return sorted({m for v in SPLITS.values() for m in v}) if isinstance(SPLITS, dict) else sorted(SPLITS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="one checkpoint, or several = emission ensemble (board item 28)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--team-switch", nargs="+", type=float, default=[0.0], help="spellF1 option 1: team-switch penalty inside the CRF decode (0 = paper); several values = one output per value")
    ap.add_argument("--trans-scale", nargs="+", type=float, default=[1.0], help="multiplier on the learned transition scores (1 = paper); several values = one output per value")
    ap.add_argument("--tta", action="store_true", help="average the 4 pitch mirrors of every checkpoint")
    ap.add_argument("--stride-div", type=int, default=2, help="inference window stride = window / stride_div (2 = paper)")
    ap.add_argument("--window-seconds", type=float, default=None, help="inference window length (default: the data config, 10 s)")
    ap.add_argument("--force-last", action="store_true", help="force each episode's final state from the next episode's first set piece (official correct_episode_lasts)")
    a = ap.parse_args()
    cfg = DataConfig()
    if a.window_seconds:
        cfg = replace(cfg, window_seconds=a.window_seconds)
    device = pick_device(a.device)
    nets = [load_net(Path(w), device) for w in a.weights]
    single = len(a.team_switch) == 1 and len(a.trans_scale) == 1
    variants = [(a.tag if single else f"{a.tag}_l{lam:g}_a{alpha:g}", lam, alpha) for lam in a.team_switch for alpha in a.trans_scale]
    mids = a.matches or all_matches()
    log = {"weights": a.weights, "tta": a.tta, "stride_div": a.stride_div, "window_seconds": cfg.window_seconds,
           "variants": [v[0] for v in variants], "matches": {}}
    for mid in mids:
        t0 = time.time()
        proc = pd.read_parquet(CACHE / f"sc_{mid}.parquet")
        edges_v, stats = infer_match(nets, proc, cfg, device, windowed=True, force_last=a.force_last, evaluate=False,
                                     tta=a.tta, stride_div=a.stride_div, variants=variants)
        proc = proc.set_index("frame_id")
        frames = np.arange(int(np.rint(proc.index.min() * 0.4)), int(np.rint(proc.index.max() * 0.4)) + 1)
        for name, edges in edges_v.items():
            edges = edges.set_index("frame_id") if "frame_id" in edges.columns else edges
            e = edges.join(proc[["episode_id"]])
            e = e[e["episode_id"] > 0].copy()
            team = e["edge_src"].astype(str).str.split("_").str[0]
            e["team"] = team.where(team.isin(["home", "away"])).groupby(e["episode_id"]).ffill()
            e["frame"] = np.rint(e.index.to_numpy() * 0.4).astype(int)
            agg = e.groupby("frame").agg(p_home_pcrf=("p_home", "mean"),
                                         team=("team", lambda s: s.mode().iloc[0] if s.notna().any() else None))
            out = pd.DataFrame({"match": int(mid), "frame": frames})
            out["p_home_pcrf"] = agg["p_home_pcrf"].reindex(frames).to_numpy()
            out["team_pcrf"] = agg["team"].reindex(frames).map({"home": 0.0, "away": 1.0}).to_numpy()
            out["covered"] = out["p_home_pcrf"].notna()
            out.to_parquet(OUT / f"sc_{mid}_{name}.parquet", index=False)
        log["matches"][mid] = {"seconds": round(time.time() - t0), "covered_rows": int(out["covered"].sum()),
                               "episodes_skipped_nan": stats.get("episodes_skipped_nan", 0)}
        print(mid, log["matches"][mid], flush=True)
    json.dump(log, open(OUT / f"run_skillcorner_log_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

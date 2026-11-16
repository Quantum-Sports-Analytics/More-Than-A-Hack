"""Out-of-fold per-step PathCRF probabilities on the 20 SkillCorner matches, for the fusion model
(two-fold training, 2026-11-13).

Each match is scored by the 2-fold model that did NOT see it: matches of half 1 by the model trained on half 2
(`--weights-h2`) and the other way round. Input = the damaged draw `cache/damaged/sc_<match>_v<k>.parquet` (players
damaged like Metrica) with the ball damaged on the fly like in training / validation (`data/ball.damage_ball`, one draw
per 10 s chunk of each episode, seeded by match and chunk), the condition closest to Barça–Athletic.

Output `<out>/oof_bb_<match>.parquet`, one row per 0.2 s step: match, frame (25 Hz SkillCorner frame_id of the step,
whole-match numbering), period, p_home (soft sender mass of the home team, same computation as run_metrica.py /
infer_match), p_self (softmax mass of the player self-loop states), team_true (0 home, 1 away, NaN when dead),
alive_true. Inside an episode the steps are the ones the model scored (episode start + 0.2 s k); between episodes a
0.2 s grid is continued so that every period is covered (p_home NaN there).
With --air also writes `<out>/air_<match>.parquet` (frame 25 Hz, p_air) from airball/outputs/oof_air.parquet.

    .venv/bin/python -m pathcrf_ours.run_skillcorner_oof --weights-h1 outputs/runs/kf1_bb/model/last.pt \\
        --weights-h2 outputs/runs/kf2_bb/model/last.pt --out outputs/oof_bb --air
    # smoke with one model for every match: --weights-h1 X --weights-h2 X --matches 1886347
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import labels as L
from .data.ball import BallDamageConfig, damage_ball
from .infer import infer_match, load_net, pick_device
from .paths import CACHE, HACK_ROOT, OUTPUTS

HALF1 = [1874553, 1899585, 1927964, 1959846, 1996435, 2006229, 2007448, 2010085, 2013725, 2016236]
HALF2 = [1886347, 1925299, 1953632, 1986691, 1996436, 2006363, 2007721, 2011166, 2015213, 2017461]
STEP = 5                                    # 25 Hz frames per 0.2 s step


def damage_match_ball(t: pd.DataFrame, cfg: DataConfig, seed: int) -> pd.DataFrame:
    """Ball damaged per 10 s chunk of every episode, on the step frames only (the only ones the model reads)."""
    t = t.copy()
    every, W = cfg.sample_every, cfg.window_steps
    bcfg = BallDamageConfig()
    t["ball_missing"] = t["ball_x"].isna()
    for k, ep in enumerate(sorted(e for e in t["episode_id"].unique() if e > 0)):
        ix = np.flatnonzero(t["episode_id"].to_numpy() == ep)
        down = ix[L.downsample_index(len(ix), every)]
        ball = t[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)[down]
        for c0 in range(0, len(down), W):
            rng = np.random.default_rng([seed, int(ep), c0])
            dmg = damage_ball(ball[c0:c0 + W].copy(), cfg.target_fps, rng, bcfg)
            rows = down[c0:c0 + W]
            t.iloc[rows, t.columns.get_loc("ball_x")] = dmg[:, 0]
            t.iloc[rows, t.columns.get_loc("ball_y")] = dmg[:, 1]
            t.iloc[rows, t.columns.get_loc("ball_missing")] = np.isnan(dmg[:, 0])
    return t


def step_rows(t: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per 0.2 s step: the model's own steps inside episodes, a 5-frame grid elsewhere."""
    e = edges.join(t.set_index("frame_id")[["period_id", "episode_id", "ball_state", "ball_owning_team_id"]])
    tp = e["edge_src"].astype(str).str.split("_").str[0]
    e["team_path"] = tp.where(tp.isin(["home", "away"])).map({"home": 0.0, "away": 1.0})
    e["team_path"] = e["team_path"].groupby(e["episode_id"]).ffill()
    out = []
    for p, ep_ in e.groupby("period_id"):
        frames = ep_.index.to_numpy()
        steps = set(frames[ep_["step"].to_numpy(dtype=bool)])
        f, last = int(frames.min()), int(frames.max())
        chosen = []
        step_arr = np.array(sorted(steps), dtype=np.int64)
        while f <= last:
            nxt = step_arr[step_arr >= f]
            if len(nxt) and nxt[0] - f < STEP:        # a model step within reach: snap to it
                f = int(nxt[0])
            chosen.append(f)
            f += STEP
        out.append(ep_.loc[chosen])
    r = pd.concat(out)
    team = r["ball_owning_team_id"].map({"home": 0.0, "away": 1.0})
    alive = r["ball_state"].eq("alive").to_numpy()
    return pd.DataFrame({"frame": r.index.to_numpy(np.int64), "period": r["period_id"].to_numpy(np.int64),
                         "p_home": r["p_home"].astype(float).to_numpy(), "p_self": r["p_self"].astype(float).to_numpy(),
                         "team_true": np.where(alive, team.to_numpy(), np.nan), "alive_true": alive,
                         "team_path": r["team_path"].astype(float).to_numpy(),
                         "model_step": r["step"].to_numpy(dtype=bool)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-h1", required=True, nargs="+", help="model(s) trained on half 1 (score the half-2 matches); several = emission ensemble (board item 34)")
    ap.add_argument("--weights-h2", required=True, nargs="+", help="model(s) trained on half 2 (score the half-1 matches)")
    ap.add_argument("--out", default=str(OUTPUTS / "oof_bb"))
    ap.add_argument("--matches", nargs="*", type=int, default=None)
    ap.add_argument("--variant", type=int, default=1, help="damaged draw to read (cache/damaged/sc_<m>_v<k>.parquet)")
    ap.add_argument("--no-ball-damage", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--air", action="store_true", help="also write air_<match>.parquet from airball/outputs/oof_air.parquet")
    ap.add_argument("--team-switch", type=float, default=0.0, help="option 1: team-switch penalty inside the CRF decode")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride-div", type=int, default=2, help="inference window stride = window / stride_div (2 = paper); board item 28")
    ap.add_argument("--tta", action="store_true", help="average the 4 pitch mirrors")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig()
    device = pick_device(a.device)
    def _load(paths):
        ns = [load_net(Path(w), device) for w in paths]
        return ns[0] if len(ns) == 1 else ns
    nets = {"h1": _load(a.weights_h1), "h2": _load(a.weights_h2)}
    mids = a.matches or sorted(HALF1 + HALF2)
    air = pd.read_parquet(HACK_ROOT / "airball" / "outputs" / "oof_air.parquet", columns=["match", "frame", "p_air"]) if a.air else None
    log = {"weights": {"h1": a.weights_h1, "h2": a.weights_h2}, "variant": a.variant, "ball_damage": not a.no_ball_damage, "team_switch": a.team_switch,
           "seed": a.seed, "stride_div": a.stride_div, "tta": a.tta, "matches": {}}
    for m in mids:
        t0 = time.time()
        which = "h2" if m in HALF1 else "h1"                 # the model that did not see m
        t = pd.read_parquet(CACHE / "damaged" / f"sc_{m}_v{a.variant}.parquet")
        if not a.no_ball_damage:
            t = damage_match_ball(t, cfg, a.seed * 1000003 + m)
        edges, stats = infer_match(nets[which], t, cfg, device, windowed=True, force_last=False, evaluate=False, team_switch=a.team_switch,
                                   tta=a.tta, stride_div=a.stride_div)
        edges = edges.set_index("frame_id") if "frame_id" in edges.columns else edges
        r = step_rows(t, edges)
        r.insert(0, "match", m)
        r.to_parquet(out / f"oof_bb_{m}.parquet", index=False)
        live = r["alive_true"].to_numpy()
        info = {"model": which, "seconds": round(time.time() - t0), "steps": int(len(r)), "live_steps": int(live.sum()),
                "p_home_on_live": round(float(r.loc[live, "p_home"].notna().mean()), 4),
                "model_steps": int(r["model_step"].sum()), "episodes_skipped_nan": stats.get("episodes_skipped_nan", 0),
                "ball_missing_share": round(float(t["ball_missing"].mean()), 3) if "ball_missing" in t else None}
        ok = live & r["p_home"].notna().to_numpy()
        if ok.any():
            info["team_acc_argmax"] = round(float(((r.loc[ok, "p_home"] < 0.5).astype(int) == r.loc[ok, "team_true"]).mean()), 4)
        if air is not None:
            am = air[air["match"] == m]
            pd.DataFrame({"frame": np.rint(am["frame"].to_numpy() * 2.5).astype(np.int64), "p_air": am["p_air"].to_numpy()}) \
                .drop_duplicates("frame").to_parquet(out / f"air_{m}.parquet", index=False)
        log["matches"][str(m)] = info
        print(m, info, flush=True)
    json.dump(log, open(out / "run_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Run a pathcrf_ours checkpoint on Barça–Athletic, through the slot adapter.

    python -m pathcrf_ours.run_metrica --weights outputs/runs/phasebb/model/best_acc_ep10.pt --tag <tag> \
        --dead-csv <stage-1 frame,label file> --dead-tag v6 --team-switch 3 --stride-div 4

Input: pathcrf_test/outputs/metrica_virtual<_suffix>.parquet, the adapter's 10 fps table (period, clock_s,
video_frame, blue_team, ball_x/y, ball_missing, home_1..11_x/y, away_1..11_x/y, *_filled). The keeper rule
(step 0) has already been applied by the adapter; `_filled` = no real slot fed the track = present 0 for the model.
Live stretches: the frames that are neither D nor X in the stage-1 file given with --dead-csv.
Output: pathcrf_test/outputs/metrica_pcrf_<tag>.parquet (row, period, video_frame, blue_team, alive_<tag>,
p_home_pcrf_<tag>, team_pcrf_<tag>), read by pathcrf_test/make_submission.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import DataConfig
from .data.features import add_kinematics
from .data.skillcorner import upsample_period
from .data.sportec import label_episodes
from .infer import infer_match, load_net, pick_device
from .paths import HACK_ROOT

OUT = HACK_ROOT / "pathcrf_test" / "outputs"
NAMES = [f"{t}_{k}" for t in ("home", "away") for k in range(1, 12)]
# The team's fused ball (challenge-barcelona/possession/video_ball.py): video detection > bridged video > Metrica's
# tracking ball > nothing, with the two inputs kept side by side. Its x/y is the merged ball every chain here uses.
BALL_NPZ = Path.home() / "Documents" / "challenge-barcelona" / "outputs" / "intermediaires" / "balle_fusionnee.npz"
PITCH_NPZ = 105.3                                   # the npz pitch length (v1.PITCH); y is up there, down here


def ball_variant(v: pd.DataFrame, variant: str, conf_min: float = 0.0, drop_bridged: bool = False):
    """10 fps adapter rows -> (ball_x, ball_y in the adapter's frame, missing) read from balle_fusionnee.npz.
    fused = the team's fused ball; video = the video detections only (bridged included); atd = Metrica's ball only.
    conf_min hides video detections below that confidence; drop_bridged hides the bridged (interpolated) video rows.
    Hidden rows keep the adapter's gap-filled position so the table stays finite; `missing` tells the model."""
    z = np.load(BALL_NPZ, allow_pickle=False)
    vf = np.clip(np.round(v["video_frame"].to_numpy()).astype(int), 0, len(z["x"]) - 1)
    src = z["source"][vf]
    if variant == "fused":
        x, y, miss = z["x"][vf], z["y"][vf], src == 0
    elif variant == "video":
        x, y, miss = z["video_x"][vf], z["video_y"][vf], ~np.isin(src, (1, 2))
    elif variant == "atd":
        x, y = z["atd_x"][vf], z["atd_y"][vf]
        miss = np.isnan(x)
    else:
        raise ValueError(variant)
    if conf_min > 0:
        miss = miss | ((src == 1) & ~(np.nan_to_num(z["video_conf"][vf], nan=1.0) >= conf_min))
    if drop_bridged:
        miss = miss | (src == 2)
    miss = miss | np.isnan(x) | np.isnan(y)
    ax, ay = v["ball_x"].to_numpy(float), v["ball_y"].to_numpy(float)
    bx = np.where(miss, ax, x * 105.0 / PITCH_NPZ)
    by = np.where(miss, ay, 68.0 - y)
    return bx, by, miss


def adapter_to_processed(v: pd.DataFrame, alive: np.ndarray, cfg: DataConfig, ball_missing: Optional[np.ndarray] = None) -> pd.DataFrame:
    """10 fps adapter rows -> 25 fps processed-like table with ball_state, `<player>_present` and `ball_missing`
    (rows where the chain's ball was hidden and imputed, so the ball model must not trust it)."""
    parts = []
    for p in (1, 2):
        m = (v["period"] == p).to_numpy()
        if not m.any():
            continue
        rows = np.flatnonzero(m)
        frames10 = v["row"].to_numpy()[rows].astype(float)              # consecutive 10 fps rows
        cols10 = {}
        for n in NAMES:
            cols10[f"{n}_x"] = v[f"{n}_x"].to_numpy(float)[rows]
            cols10[f"{n}_y"] = v[f"{n}_y"].to_numpy(float)[rows]
        cols10["ball_x"] = v["ball_x"].to_numpy(float)[rows]
        cols10["ball_y"] = v["ball_y"].to_numpy(float)[rows]
        labels10 = {"alive": alive[rows], "src_row": rows,
                    "ball_missing": (ball_missing[rows] if ball_missing is not None else np.zeros(len(rows), bool))}
        for n in NAMES:
            labels10[f"{n}_present"] = ~v[f"{n}_filled"].to_numpy(bool)[rows]
        up = upsample_period(frames10, cols10, labels10, cfg.source_fps)
        up["period_id"] = p
        parts.append(up)
    df = pd.concat(parts, ignore_index=True)
    df["ball_state"] = np.where(df.pop("alive"), "alive", "dead")
    df["ball_owning_team_id"] = None
    df = label_episodes(df)
    df = add_kinematics(df, cfg, objects=NAMES)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, nargs="+", help="one checkpoint, or several for an emission ensemble (board item 34)")
    ap.add_argument("--tag", required=True, help="suffix of the output: metrica_pcrf_<tag>.parquet")
    ap.add_argument("--virtual", default="", help="adapter suffix, e.g. shape -> metrica_virtual_shape.parquet")
    ap.add_argument("--dead-csv", default=None, help="25 fps frame,label file whose D / X frames define the live stretches")
    ap.add_argument("--dead-tag", default="v6")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--team-switch", type=float, default=0.0, help="spellF1 option 1: team-switch penalty inside the CRF decode (0 = paper)")
    ap.add_argument("--stride-div", type=int, default=2, help="inference window stride = window / stride_div (2 = paper); board item 28")
    ap.add_argument("--tta", action="store_true", help="average the 4 pitch mirrors")
    ap.add_argument("--ball", default="adapter", choices=["adapter", "fused", "video", "atd"], help="adapter = the chain's merged ball (default); the others come from balle_fusionnee.npz")
    ap.add_argument("--conf-min", type=float, default=0.0, help="hide video detections with confidence below this (with --ball fused/video)")
    ap.add_argument("--drop-bridged", action="store_true", help="hide the bridged (interpolated) video rows (with --ball fused/video)")
    a = ap.parse_args()

    t0 = time.time()
    cfg = DataConfig()
    device = pick_device(a.device)
    nets = [load_net(Path(w), device) for w in a.weights]
    net = nets[0] if len(nets) == 1 else nets
    vsfx = f"_{a.virtual}" if a.virtual else ""
    v = pd.read_parquet(OUT / f"metrica_virtual{vsfx}.parquet")
    if not a.dead_csv:
        raise SystemExit("--dead-csv is required: the live stretches come from the stage-1 file")
    lab = pd.read_csv(a.dead_csv)["label"].to_numpy().astype(str)
    vf = np.clip(np.round(v["video_frame"].to_numpy()).astype(int), 0, len(lab) - 1)
    alive = {a.dead_tag: ~np.isin(lab[vf], ["D", "X"])}
    ball_missing = v["ball_missing"].to_numpy(dtype=bool)   # written by metrica_adapter.py (ball not observed after step 0)
    if a.ball != "adapter":
        v = v.copy()
        v["ball_x"], v["ball_y"], ball_missing = ball_variant(v, a.ball, a.conf_min, a.drop_bridged)

    out = v[["row", "period", "video_frame", "blue_team"]].copy()
    log = {"weights": a.weights, "presence": bool(nets[0].cfg.presence), "ball": a.ball,
           "conf_min": a.conf_min, "drop_bridged": a.drop_bridged, "ball_missing_share": round(float(ball_missing.mean()), 4)}
    for tag, al in alive.items():
        t1 = time.time()
        proc = adapter_to_processed(v, al, cfg, ball_missing=ball_missing)
        edges, stats = infer_match(net, proc, cfg, device, windowed=True, force_last=False, evaluate=False, team_switch=a.team_switch,
                                   tta=a.tta, stride_div=a.stride_div)
        edges = edges.set_index("frame_id")
        proc = proc.set_index("frame_id")
        # back to 10 fps rows: the 25 fps frame nearest to each row (the row's own time)
        e = edges.join(proc[["src_row", "episode_id"]])
        e = e[e["episode_id"] > 0]
        team = e["edge_src"].astype(str).str.split("_").str[0]
        team = team.where(team.isin(["home", "away"]))
        e["team"] = team.groupby(e["episode_id"]).ffill()
        e["is_out"] = e["edge_src"].astype(str).str.startswith("out_").astype(float)   # the ball is outside (holder = an outside node)
        per_row = e.groupby("src_row").agg(p_home=("p_home", "mean"), out=("is_out", "mean"),
                                           team=("team", lambda s: s.mode().iloc[0] if s.notna().any() else None))
        out[f"alive_{tag}"] = al
        out[f"p_home_pcrf_{tag}"] = per_row["p_home"].reindex(np.arange(len(v))).to_numpy()
        out[f"team_pcrf_{tag}"] = per_row["team"].reindex(np.arange(len(v))).map({"home": 0, "away": 1}).to_numpy()
        out[f"out_pcrf_{tag}"] = per_row["out"].reindex(np.arange(len(v))).to_numpy()
        cov = out.loc[al, f"team_pcrf_{tag}"].notna().mean()
        log[tag] = {"seconds": round(time.time() - t1), "live_rows": int(al.sum()), "covered_share_of_live": round(float(cov), 4),
                    "episodes": int(proc["episode_id"].nunique() - 1), "episodes_skipped_nan": stats.get("episodes_skipped_nan", 0)}
        print(tag, log[tag], flush=True)
    out.to_parquet(OUT / f"metrica_pcrf_{a.tag}.parquet", index=False)
    log["seconds"] = round(time.time() - t0)
    json.dump(log, open(OUT / f"run_metrica_log_{a.tag}.json", "w"), indent=1)
    print(f"wrote {OUT / f'metrica_pcrf_{a.tag}.parquet'}; now run: airball/.venv/bin/python pathcrf_test/score_metrica.py --tag {a.tag}")


if __name__ == "__main__":
    main()

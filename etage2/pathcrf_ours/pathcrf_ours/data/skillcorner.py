"""SkillCorner open data -> the same "processed" match table as the Sportec pipeline (Phase B).

SkillCorner gives 10 fps tracking (22 players on every frame that is not a TV replay, 0 players on
replays), a ball with x, y, z, and `player_possession` events (who holds the ball, frame_start..frame_end,
and the game interruption that follows). We turn that into the PathCRF training table:

    frame_id (25 fps, global), period_id, timestamp, ball_state (alive / dead), ball_owning_team_id,
    player_id (per-frame label: holder name, NaN in flight, out_<side> on the frame the ball goes out),
    home_<id>_x/y, away_<id>_x/y, ball_x/y/z, then kinematics, episode_id, phase_id (pipeline helpers).

Rules (same as pipeline/labels.py, the main pipeline's labels, so both read SkillCorner the same way):
- holder = the possessing player on [frame_start, frame_end];
- a gap between two possessions with no interruption is "in flight / loose" (NaN label): the step labels
  then give sender = last holder, receiver = next holder, as in the Sportec labels;
- an interruption (throw-in, corner, goal kick, goal, free kick...) makes the ball dead from the end of the
  possession to the restart kick (last 5 s before the next possession: the frame of the biggest 3-D ball
  acceleration if above 10 m/s^2, else the next possession's start);
- for throw-ins, corners, goal kicks and goals the dead stretch starts when the ball crosses a line
  (searched from 1 s before the possession ends to 5 s after); that frame gets the label out_<side>
  (side by the pitch diagonals, as in touches.classify_out_sides; a goal -> out_left / out_right);
- each half is dead from its first frame to the kick-off kick;
- TV replays (frames with no players) are dead.

Coordinates: metres, rescaled from the match's pitch (104-106 x 68) to 105 x 68, origin bottom-left, and
rotated by 180 degrees in the halves where the home team attacks right-to-left, so that the home team
attacks left-to-right in both halves (kloppy's STATIC_HOME_AWAY convention used for Sportec).

10 fps -> 25 fps: linear interpolation between consecutive 10 fps frames (never across a missing frame);
labels and ball state are taken from the nearest 10 fps frame.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import PITCH_X, PITCH_Y, DataConfig
from .features import add_kinematics, processed_columns
from .sportec import label_episodes, label_phases

SC_FPS = 10
RESTART_WINDOW, RESTART_ACCEL = 50, 10.0          # frames (10 fps), m/s^2
LINE_TYPES = ("throw_in", "corner", "goal_kick", "goal")
LOOSE_MAX_S = 30.0


def skillcorner_dir() -> Path:
    env = os.environ.get("SKILLCORNER_DIR")
    if env:
        return Path(env)
    cands = [
        Path(__file__).resolve().parents[3] / "skillcorner",
        Path.home() / "Documents" / "Quantum Football Event Detection.nosync" / "skillcorner-opendata",
    ]
    for c in cands:
        if (c / "data" / "matches").exists():
            return c
    return cands[0]


def match_dir(mid: str | int) -> Path:
    return skillcorner_dir() / "data" / "matches" / str(mid)


def all_match_ids() -> List[str]:
    d = skillcorner_dir() / "data" / "matches"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.isdigit())


SPLITS = {
    "train": ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691", "1996435",
              "1996436", "2006229", "2006363", "2007448", "2007721"],
    "validation": ["2010085", "2011166", "2013725"],
    "test": ["2015213", "2016236", "2017461"],
}


# ------------------------------------------------------------------------------------------ reading
def read_meta(mid: str) -> dict:
    meta = json.loads((match_dir(mid) / f"{mid}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: ("home" if p["team_id"] == home_id else "away") for p in meta["players"]}
    gk = {p["id"] for p in meta["players"] if (p.get("player_role") or {}).get("acronym") == "GK"}
    sides = meta.get("home_team_side") or ["left_to_right", "left_to_right"]
    return {"L": float(meta["pitch_length"]), "W": float(meta["pitch_width"]), "team_of": team_of, "gk": gk,
            "home_side": {i + 1: s for i, s in enumerate(sides)}, "periods": meta.get("match_periods", [])}


def read_tracking(mid: str) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """(ball table with one row per 10 fps frame; X, Y per player id as arrays aligned with the rows)."""
    rows, pl = [], []
    with open(match_dir(mid) / f"{mid}_tracking_extrapolated.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            b = d["ball_data"]
            rows.append((d["frame"], d["period"], b["x"], b["y"], b["z"], b["is_detected"], len(d["player_data"]) > 0))
            for q in d["player_data"]:
                pl.append((i, q["player_id"], q["x"], q["y"]))
    ball = pd.DataFrame(rows, columns=["frame", "period", "x", "y", "z", "is_detected", "has_players"])
    ball["period"] = ball["period"].fillna(0).astype(int)
    for c in ("x", "y", "z"):
        ball[c] = ball[c].astype(float)
    ball["seen"] = ball["is_detected"].eq(True)
    pl = pd.DataFrame(pl, columns=["row", "pid", "x", "y"])
    n = len(ball)
    X, Y = {}, {}
    for pid, g in pl.groupby("pid"):
        X[pid] = np.full(n, np.nan)
        Y[pid] = np.full(n, np.nan)
        X[pid][g["row"].to_numpy()] = g["x"].to_numpy(float)
        Y[pid][g["row"].to_numpy()] = g["y"].to_numpy(float)
    return ball, X, Y


def read_possessions(mid: str) -> pd.DataFrame:
    ev = pd.read_csv(match_dir(mid) / f"{mid}_dynamic_events.csv", low_memory=False)
    pos = ev[ev["event_type"] == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    return pos[["frame_start", "frame_end", "player_id", "game_interruption_after", "period"]]


# ------------------------------------------------------------------------------------------ geometry
def to_pitch(x: np.ndarray, y: np.ndarray, L: float, W: float, flip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-origin metres on an L x W pitch -> 0..105 x 0..68, rotated by 180 degrees where `flip`."""
    xs = (x + L / 2) * PITCH_X / L
    ys = (y + W / 2) * PITCH_Y / W
    xs = np.where(flip, PITCH_X - xs, xs)
    ys = np.where(flip, PITCH_Y - ys, ys)
    return xs, ys


def out_side(x: float, y: float) -> Optional[str]:
    """Which outside node a ball at (x, y) (0..105 x 0..68 frame) left by, using the pitch diagonals."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    r = PITCH_Y / PITCH_X
    d1, d2 = y - r * x, y + r * x - PITCH_Y
    if d1 > 0 and d2 < 0:
        return "out_left"
    if d1 < 0 and d2 > 0:
        return "out_right"
    if d1 < 0 and d2 < 0:
        return "out_bottom"
    if d1 > 0 and d2 > 0:
        return "out_top"
    return None


def ball_accel_3d(ball: pd.DataFrame) -> np.ndarray:
    """Frame-to-frame 3-D ball acceleration at 10 fps, never across halves (pipeline/labels.py rule)."""
    a = np.full(len(ball), np.nan)
    for p in ball["period"].unique():
        m = (ball["period"] == p).to_numpy()
        xyz = ball.loc[m, ["x", "y", "z"]].to_numpy()
        v = np.diff(xyz, axis=0) * SC_FPS
        acc = np.linalg.norm(np.diff(v, axis=0) * SC_FPS, axis=1)
        idx = np.flatnonzero(m)
        a[idx[2:]] = acc
    return np.nan_to_num(a, nan=-np.inf)


# ------------------------------------------------------------------------------------------ labels
def frame_labels_10(ball: pd.DataFrame, pos: pd.DataFrame, meta: dict, names: Dict[int, str]) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
    """Per 10 fps frame: (player label, dead mask, owning team code 0 home / 1 away / -1).

    `ball` must already be in the 0..105 x 0..68 frame (for the out side)."""
    n = len(ball)
    period = ball["period"].to_numpy()
    bx, by = ball["x"].to_numpy(), ball["y"].to_numpy()
    with np.errstate(invalid="ignore"):
        outside = (bx < 0) | (bx > PITCH_X) | (by < 0) | (by > PITCH_Y)
    accel = ball_accel_3d(ball)
    label = np.full(n, None, dtype=object)
    dead = np.zeros(n, bool)
    team = np.full(n, -1, int)

    fs = pos["frame_start"].to_numpy(int)
    fe = pos["frame_end"].to_numpy(int)
    gi = pos["game_interruption_after"].to_numpy(object)
    pname = [names.get(p) for p in pos["player_id"]]
    pteam = np.array([0 if (nm or "").startswith("home") else (1 if nm else -1) for nm in pname])

    def restart_kick(s: int, e: int) -> int:
        lo = max(s, e - RESTART_WINDOW)
        w = accel[lo:e]
        if len(w) and w.max() > RESTART_ACCEL:
            return lo + int(np.argmax(w))
        return e

    for k in range(len(pos)):
        if pname[k] is None:
            continue
        a, b = max(fs[k], 0), min(fe[k], n - 1)
        label[a : b + 1] = pname[k]
        team[a : b + 1] = pteam[k]
        if k + 1 >= len(pos) or period[min(fe[k], n - 1)] != period[min(fs[k + 1], n - 1)]:
            continue
        if isinstance(gi[k], str):
            s0, e0 = fe[k] + 1, fs[k + 1]
            if not (0 < e0 - s0 < 3000):
                continue
            s = s0
            if gi[k].startswith(LINE_TYPES):
                lo, hi = max(0, fe[k] - SC_FPS), min(e0, s0 + 5 * SC_FPS)
                hit = np.flatnonzero(outside[lo:hi])
                if len(hit):
                    f_out = lo + int(hit[0])
                    if gi[k].startswith("goal"):
                        side = "out_left" if bx[f_out] < PITCH_X / 2 else "out_right"
                    else:
                        side = out_side(bx[f_out], by[f_out])
                    if side is not None:
                        label[f_out] = side
                        if f_out <= fe[k]:              # the ball was out before the possession "ended"
                            label[f_out + 1 : fe[k] + 1] = None
                        s = f_out + 1
            e = restart_kick(s, e0)
            dead[s:e] = True
            team[e:e0] = pteam[k + 1]
        elif 0 < fs[k + 1] - fe[k] <= LOOSE_MAX_S * SC_FPS:
            team[fe[k] + 1 : fs[k + 1]] = pteam[k]       # in flight / loose: the earlier team keeps it
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if not len(idx):
            continue
        first = np.flatnonzero((fs >= idx[0]) & (fs <= idx[-1]))
        if len(first):
            k = first[0]
            e = restart_kick(idx[0], fs[k])
            dead[idx[0] : e] = True
            team[e : fs[k]] = pteam[k]
    dead |= period == 0
    dead |= ~ball["has_players"].to_numpy()
    team[dead] = -1
    return pd.Series(label, dtype=object), dead, team


# ------------------------------------------------------------------------------------------ 10 -> 25 fps
def upsample_period(frames10: np.ndarray, cols10: Dict[str, np.ndarray], labels10: Dict[str, np.ndarray], fps_out: float) -> pd.DataFrame:
    """Linear interpolation of `cols10` onto a fps_out grid; `labels10` copied from the nearest 10 fps frame."""
    t10 = frames10 / SC_FPS
    k0, k1 = int(np.ceil(t10[0] * fps_out)), int(np.floor(t10[-1] * fps_out))
    k = np.arange(k0, k1 + 1)
    pos = k / fps_out * SC_FPS - frames10[0]          # position on the 10 fps row axis
    i = np.clip(np.floor(pos).astype(int), 0, len(frames10) - 1)
    w = pos - i
    j = np.clip(i + 1, 0, len(frames10) - 1)
    exact = w < 1e-9
    out = {"frame_id": k, "timestamp": np.round(k / fps_out, 2)}
    for name, v in cols10.items():
        a, b = v[i], v[j]
        val = np.where(exact, a, a * (1 - w) + b * w)
        out[name] = val
    near = np.clip(np.rint(pos).astype(int), 0, len(frames10) - 1)
    for name, v in labels10.items():
        out[name] = v[near]
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ main
def process_skillcorner(mid: str, cfg: DataConfig, cache_dir: Optional[Path] = None, force: bool = False) -> pd.DataFrame:
    mid = str(mid)
    key = f"sc_{mid}"
    out = Path(cache_dir) / "processed" / f"{key}.parquet" if cache_dir else None
    if out is not None and out.exists() and not force:
        return pd.read_parquet(out)

    meta = read_meta(mid)
    ball, X, Y = read_tracking(mid)
    pos = read_possessions(mid)
    names = {pid: f"{meta['team_of'][pid]}_{pid}" for pid in X if pid in meta["team_of"]}

    period = ball["period"].to_numpy()
    flip = np.array([meta["home_side"].get(int(p), "left_to_right") == "right_to_left" for p in period])
    ball = ball.copy()
    ball["x"], ball["y"] = to_pitch(ball["x"].to_numpy(), ball["y"].to_numpy(), meta["L"], meta["W"], flip)
    ball.loc[~ball["seen"], ["x", "y", "z"]] = np.nan
    label10, dead10, team10 = frame_labels_10(ball, pos, meta, names)

    cols10: Dict[str, np.ndarray] = {"ball_x": ball["x"].to_numpy(), "ball_y": ball["y"].to_numpy(), "ball_z": ball["z"].to_numpy()}
    for pid, nm in names.items():
        xs, ys = to_pitch(X[pid], Y[pid], meta["L"], meta["W"], flip)
        cols10[f"{nm}_x"], cols10[f"{nm}_y"] = xs, ys
    labels10 = {"player_id": label10.to_numpy(dtype=object), "dead": dead10, "team": team10}

    parts = []
    frames = ball["frame"].to_numpy()
    for p in (1, 2):
        m = period == p
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        assert np.all(np.diff(frames[idx]) == 1), f"{mid}: non-contiguous frames in period {p}"
        up = upsample_period(frames[idx], {k: v[idx] for k, v in cols10.items()}, {k: v[idx] for k, v in labels10.items()}, cfg.source_fps)
        up["period_id"] = p
        parts.append(up)
    df = pd.concat(parts, ignore_index=True)
    df["ball_state"] = np.where(df.pop("dead"), "dead", "alive")
    df["ball_owning_team_id"] = df.pop("team").map({0: "home", 1: "away", -1: None})
    df["player_id"] = df["player_id"].astype(object)

    df = label_episodes(df)
    df = cut_episodes_after_out(df)
    df = add_kinematics(df, cfg)
    df, _ = label_phases(df)
    df = df[processed_columns(df)].copy()
    df.attrs["match"] = key
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    return df


def cut_episodes_after_out(df: pd.DataFrame) -> pd.DataFrame:
    """SHORTCUT 4 for SkillCorner: an episode ends on its first out_<side> frame; whatever follows before the
    next stoppage becomes dead (a restart within a few frames of the out would otherwise keep the episode alive and
    put an outside -> player edge in the labels, which the CRF forbids). Episodes are then re-numbered."""
    df = df.copy()
    lab = df["player_id"].astype(str)
    is_out = lab.str.startswith(("out_", "goal_")).to_numpy()
    ep = df["episode_id"].to_numpy()
    state = df["ball_state"].to_numpy().copy()
    for e in np.unique(ep[ep > 0]):
        idx = np.flatnonzero(ep == e)
        hits = np.flatnonzero(is_out[idx])
        if len(hits) and hits[0] < len(idx) - 1:
            state[idx[hits[0] + 1 :]] = "dead"
    df["ball_state"] = state
    df = df.drop(columns=["episode_id"])
    return label_episodes(df)


def is_skillcorner(match_id: str) -> bool:
    return str(match_id).startswith("sc_")

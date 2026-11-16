"""Ball-touch refinement: turn synchronised events plus the ball trajectory into one touch list per episode.

1. Every event stays at its own (ELASTIC-synchronised) frame; its position is the ball's at that frame.
2. The ball path of the episode is simplified with RDP (Ramer-Douglas-Peucker keeps only the corners
   of a curve, epsilon 0.5 m); corners with a direction change under 15 degrees are dropped.
3. Events and corners are aligned with Needleman-Wunsch (a sequence alignment): a pair scores
   100 * (1 - d / 5 m), clipped to [-100, 100], impossible beyond 5 m; gaps cost -10; several events may
   share one corner ("repeat", -20) when the pair scores at least 50.
4. Corners with no event become extra touches by the nearest player within 5 m, unless they fall
   within 5 frames of a neighbouring event; when the neighbouring events are less than 50 frames apart
   and by the same team, only that team's players are candidates.
5. After the last event, the first touch with the ball outside [1, 104] x [1, 67] becomes "out", or
   "goal" when the episode's last shot was successful; everything later is cut.
6. Same-frame event duplicates are reduced to one (prefer the previous holder, then the latest event);
   an un-typed corner touch whose player equals both neighbours is dropped; player touches take the
   player's position.
7. Out/goal sides: the pitch diagonals decide left/right/bottom/top; goals are left if x < 10, right
   if x > 95.

Per-frame label: a frame carries player P only when the touches before and after it are both by P
(ffill == bfill); frames between touches of different players (ball in flight) are NaN; before the
first touch and after the last touch the nearest touch's player is used; after an out, the out node.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdp import rdp

from ..config import PITCH_X, PITCH_Y, DataConfig

TOUCH_COLS = ["frame_id", "period_id", "episode_id", "timestamp", "player_id", "event_type", "start_x", "start_y"]


# -------------------------------------------------------------------------------------------- RDP
def simplify_ball_path(frame_ids: np.ndarray, xy: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Returns [n, 3] rows (frame_id, x, y) of the kept corners, first and last point included."""
    pts = np.column_stack([frame_ids.astype(float) * cfg.rdp_frame_scale, xy])
    kept = np.array(rdp(pts, epsilon=cfg.rdp_epsilon))
    if len(kept) < 3:
        kept[:, 0] = np.round(kept[:, 0] / cfg.rdp_frame_scale)
        return kept
    d = np.diff(kept[:, 1:], axis=0)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    cos = np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)
    turn = np.where(np.arccos(cos) >= np.deg2rad(cfg.rdp_min_angle))[0] + 1
    idx = np.concatenate([[0], turn, [len(kept) - 1]])
    kept = kept[idx]
    kept[:, 0] = np.round(kept[:, 0] / cfg.rdp_frame_scale)
    return kept


# ------------------------------------------------------------------------------- Needleman-Wunsch
def align_events_to_corners(
    event_xy: np.ndarray, corner_xy: np.ndarray, max_dist: float, gap: float = -10.0, repeat_pen: float = -20.0, repeat_thr: float = 50.0
) -> List[Tuple[Optional[int], Optional[int], str]]:
    """Alignment path as (event_pos, corner_pos, move) with move in {match, repeat, event_only, corner_only}."""
    n_e, n_c = len(event_xy), len(corner_xy)
    if n_e == 0 and n_c == 0:
        return []
    if n_e == 0:
        return [(None, j, "corner_only") for j in range(n_c)]
    if n_c == 0:
        return [(i, None, "event_only") for i in range(n_e)]
    dist = np.linalg.norm(event_xy[:, None, :] - corner_xy[None, :, :], axis=2)
    score = np.clip(100.0 * (1.0 - dist / max_dist), -100.0, 100.0)
    score[dist >= max_dist] = -1e9
    dp = np.zeros((n_e + 1, n_c + 1))
    tr = np.zeros((n_e + 1, n_c + 1), dtype=np.int8)  # 0 match, 1 event gap, 2 corner gap, 3 repeat
    for i in range(1, n_e + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
        tr[i, 0] = 1
    for j in range(1, n_c + 1):
        dp[0, j] = dp[0, j - 1] + gap
        tr[0, j] = 2
    for i in range(1, n_e + 1):
        for j in range(1, n_c + 1):
            s = score[i - 1, j - 1]
            diag = dp[i - 1, j - 1] + s
            up = dp[i - 1, j] + gap
            left = dp[i, j - 1] + gap
            rep = dp[i - 1, j] + s + repeat_pen if s >= repeat_thr else -np.inf
            if diag >= up and diag >= left and diag >= rep:
                dp[i, j], tr[i, j] = diag, 0
            elif rep >= up and rep >= left:
                dp[i, j], tr[i, j] = rep, 3
            elif up >= left:
                dp[i, j], tr[i, j] = up, 1
            else:
                dp[i, j], tr[i, j] = left, 2
    path = []
    i, j = n_e, n_c
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tr[i, j] in (0, 3):
            path.append((i - 1, j - 1, "match" if tr[i, j] == 0 else "repeat"))
            if tr[i, j] == 0:
                i, j = i - 1, j - 1
            else:
                i -= 1
        elif i > 0 and (j == 0 or tr[i, j] == 1):
            path.append((i - 1, None, "event_only"))
            i -= 1
        else:
            path.append((None, j - 1, "corner_only"))
            j -= 1
    path.reverse()
    return path


# --------------------------------------------------------------------------------------- episode
def episode_touches(
    ep_tracking: pd.DataFrame, ep_events: pd.DataFrame, home: List[str], away: List[str], cfg: DataConfig
) -> Optional[pd.DataFrame]:
    """ep_tracking indexed by frame_id (one episode); ep_events the synced events of that episode."""
    if len(ep_tracking) < 25 or ep_events.empty:
        return None
    ep_events = ep_events.sort_values("frame_id", kind="stable")
    ball = ep_tracking[["ball_x", "ball_y"]]
    corners = simplify_ball_path(ball.index.to_numpy(), ball.to_numpy(), cfg)
    ev_xy = np.column_stack([ep_events["frame_id"].map(ball["ball_x"]), ep_events["frame_id"].map(ball["ball_y"])])
    path = align_events_to_corners(ev_xy, corners[:, 1:], cfg.touch_max_dist)

    rows = []
    for e_pos, c_pos, move in path:
        if e_pos is not None:
            ev = ep_events.iloc[e_pos]
            rows.append({"frame_id": int(ev["frame_id"]), "event_idx": int(ev.name), "player_id": ev["player_id"], "event_type": ev["spadl_type"], "from_event": True})
        else:
            rows.append({"frame_id": int(corners[c_pos, 0]), "event_idx": np.nan, "player_id": None, "event_type": None, "from_event": False})
    touches = pd.DataFrame(rows)
    touches["start_x"] = touches["frame_id"].map(ball["ball_x"])
    touches["start_y"] = touches["frame_id"].map(ball["ball_y"])

    # corners without an event -> nearest player within max_dist
    teams = {"home": home, "away": away}
    ev_mask = touches["from_event"].to_numpy()
    for i in np.where(~ev_mask)[0]:
        f = int(touches.at[i, "frame_id"])
        cands = home + away
        prev_ev = np.where(ev_mask[:i])[0]
        next_ev = np.where(ev_mask[i + 1 :])[0]
        if len(prev_ev) and len(next_ev):
            pi, ni = prev_ev[-1], i + 1 + next_ev[0]
            pf, nf = int(touches.at[pi, "frame_id"]), int(touches.at[ni, "frame_id"])
            if f < pf + 5 or f > nf - 5:
                continue
            pt, nt = str(touches.at[pi, "player_id"])[:4], str(touches.at[ni, "player_id"])[:4]
            if nf - pf < 50 and pt == nt and pt in teams:
                cands = teams[pt]
        row = ep_tracking.loc[f]
        px = row[[f"{p}_x" for p in cands]].dropna().astype(float)
        py = row[[f"{p}_y" for p in cands]].dropna().astype(float)
        if px.empty:
            continue
        d = np.sqrt((px.to_numpy() - touches.at[i, "start_x"]) ** 2 + (py.to_numpy() - touches.at[i, "start_y"]) ** 2)
        k = int(d.argmin())
        if d[k] >= cfg.touch_max_dist:
            continue
        touches.at[i, "player_id"] = px.index[k][:-2]

    # ball leaving the pitch after the last event -> out / goal, cut the rest
    if ev_mask.any():
        last_ev_frame = touches.loc[ev_mask, "frame_id"].max()
        inside = touches["start_x"].between(1, PITCH_X - 1) & touches["start_y"].between(1, PITCH_Y - 1)
        out_pts = touches[(touches["frame_id"] > last_ev_frame) & ~inside]
        if not out_pts.empty:
            oi = out_pts.index[0]
            shots = ep_events[ep_events["spadl_type"] == "shot"]
            scored = bool(shots["success"].iloc[-1]) if not shots.empty else False
            touches.at[oi, "player_id"] = "goal" if scored else "out"
            touches = touches[touches["frame_id"] <= touches.at[oi, "frame_id"]].copy()

    touches = touches.dropna(subset=["player_id"]).sort_values(["frame_id", "event_idx"], ignore_index=True)

    # same-frame duplicates among event touches
    ev_rows = touches[touches["from_event"]]
    drop, prev_pid = [], None
    for _, grp in ev_rows.groupby("frame_id", sort=False):
        grp = grp.sort_values("event_idx")
        if len(grp) == 1:
            prev_pid = grp.iloc[-1]["player_id"]
            continue
        if prev_pid is not None and (grp["player_id"] == prev_pid).any():
            keep = grp[grp["player_id"] == prev_pid]["event_idx"].astype(float).idxmax()
        else:
            keep = grp["event_idx"].astype(float).idxmax()
        drop += [k for k in grp.index if k != keep]
        prev_pid = grp.loc[keep, "player_id"]
    if drop:
        touches = touches.drop(index=drop)

    untyped = touches["event_type"].isna()
    same_prev = touches["player_id"].eq(touches["player_id"].shift(1))
    same_next = touches["player_id"].eq(touches["player_id"].shift(-1))
    touches = touches.loc[~(untyped & same_prev & same_next)].copy()

    is_player = touches["player_id"].astype(str).str.startswith(("home_", "away_"))
    for i in touches.index[is_player]:
        p, f = touches.at[i, "player_id"], int(touches.at[i, "frame_id"])
        touches.at[i, "start_x"] = round(float(ep_tracking.at[f, f"{p}_x"]), 2)
        touches.at[i, "start_y"] = round(float(ep_tracking.at[f, f"{p}_y"]), 2)

    touches["period_id"] = int(ep_events["period_id"].iloc[0])
    touches["timestamp"] = touches["frame_id"].map(ep_tracking["timestamp"])
    return touches.reset_index(drop=True)


def classify_out_sides(touches: pd.DataFrame) -> pd.DataFrame:
    touches = touches.copy()
    out = touches["player_id"] == "out"
    goal = touches["player_id"] == "goal"
    touches.loc[out, "event_type"] = "out"
    touches.loc[goal, "event_type"] = "goal"
    r = PITCH_Y / PITCH_X
    d1 = touches["start_y"] - r * touches["start_x"]
    d2 = touches["start_y"] + r * touches["start_x"] - PITCH_Y
    touches.loc[out & (d1 > 0) & (d2 < 0), "player_id"] = "out_left"
    touches.loc[out & (d1 < 0) & (d2 > 0), "player_id"] = "out_right"
    touches.loc[out & (d1 < 0) & (d2 < 0), "player_id"] = "out_bottom"
    touches.loc[out & (d1 > 0) & (d2 > 0), "player_id"] = "out_top"
    touches.loc[goal & (touches["start_x"] < 10), "player_id"] = "goal_left"
    touches.loc[goal & (touches["start_x"] > PITCH_X - 10), "player_id"] = "goal_right"
    unresolved = touches["player_id"].isin(["out", "goal"])
    touches.loc[unresolved, "player_id"] = np.nan
    return touches


def sanitize_after_out(touches: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """SHORTCUT 4: cut every episode at its first out/goal touch. Returns (touches, affected episode ids)."""
    keep, affected = [], []
    for ep, grp in touches.groupby("episode_id", sort=False):
        is_out = grp["player_id"].astype(str).str.startswith(("out_", "goal_")).to_numpy()
        if is_out.any():
            first = int(np.argmax(is_out))
            if first < len(grp) - 1:
                affected.append(int(ep))
            grp = grp.iloc[: first + 1]
        keep.append(grp)
    return pd.concat(keep, ignore_index=True), affected


def match_touches(tracking: pd.DataFrame, events: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, List[int]]:
    """tracking: full match with frame_id and episode_id columns; events: synced events with episode_id."""
    from .sportec import players_of

    home, away = players_of(tracking)
    t = tracking.set_index("frame_id")
    parts = []
    for ep in tracking["episode_id"].unique():
        if ep == 0:
            continue
        ep_t = t[t["episode_id"] == ep]
        ep_e = events[events["episode_id"] == ep]
        tt = episode_touches(ep_t, ep_e, home, away, cfg)
        if tt is not None and not tt.empty:
            tt["episode_id"] = int(ep)
            parts.append(tt)
    touches = pd.concat(parts, ignore_index=True)
    touches = classify_out_sides(touches).dropna(subset=["player_id"])
    affected: List[int] = []
    if cfg.sanitize_after_out:
        touches, affected = sanitize_after_out(touches)
    return touches[TOUCH_COLS].reset_index(drop=True), affected


# ------------------------------------------------------------------------------- per-frame label
def frame_labels(tracking: pd.DataFrame, touches: pd.DataFrame) -> pd.Series:
    """Per-frame `player_id` label aligned with `tracking` (which has frame_id and episode_id)."""
    lab = tracking[["frame_id", "episode_id"]].merge(touches[["frame_id", "player_id"]], how="left", on="frame_id")
    assert len(lab) == len(tracking), "duplicate touch frames"
    out = pd.Series(np.nan, index=tracking.index, dtype=object)
    lab.index = tracking.index
    for ep, grp in lab.groupby("episode_id"):
        if ep == 0:
            continue
        pid = grp["player_id"]
        valid = pid.dropna()
        if valid.empty:
            continue
        f, b = pid.ffill(), pid.bfill()
        val = pd.Series(np.where(f == b, f, np.nan), index=grp.index, dtype=object)
        val.loc[: valid.index[0]] = valid.iloc[0]
        val.loc[valid.index[-1] :] = valid.iloc[-1]
        out.loc[grp.index] = val
    return out

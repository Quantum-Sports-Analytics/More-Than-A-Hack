"""Event-level evaluation, as in the paper.

1. Decoded edges -> events: self-loop runs shorter than `min_dur` frames (0.4 s) are overwritten by the
   following edge; then every change of edge is an event: a new player self-loop = control, a new
   player -> X edge = kick, a new outside self-loop = out.
2. Both true touches and predicted events are simplified to kick / control / out: out stays out; if the
   next event of the episode is by the same player -> control, by another player -> kick; the last
   event of an episode is typed by its original event type (outgoing types -> kick, incoming -> control).
   Fouls are dropped from the truth.
3. Within each episode, true and predicted lists are aligned by Needleman-Wunsch (pair score = time,
   distance, same player, same type; gaps -10). A pair counts as matched when player, type agree and
   |dt| <= 1 s. Precision / recall / F1 per type and in total.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import OUTSIDE_XY

PASS_LIKE_OPEN = ["pass", "cross", "shot", "clearance", "shot_block", "keeper_punch", "keeper_save", "keeper_sweeper"]
SET_PIECE = ["throw_in", "goalkick", "corner_short", "corner_crossed", "freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
OUTGOING = PASS_LIKE_OPEN + SET_PIECE + ["tackle", "bad_touch", "dispossessed", "kick"]
INCOMING = ["control", "interception", "ball_recovery", "keeper_claim", "keeper_pick_up"]
SIMPLE_TYPES = ["kick", "control", "out"]


def is_player(node) -> bool:
    return isinstance(node, str) and node.startswith(("home_", "away_"))


def node_xy(tracking: pd.DataFrame, frame: int, node: str) -> Tuple[float, float]:
    if is_player(node):
        return float(tracking.at[frame, f"{node}_x"]), float(tracking.at[frame, f"{node}_y"])
    return OUTSIDE_XY.get(node, (np.nan, np.nan))


def overwrite_short_self_loops(tuples: pd.Series, min_dur: int) -> pd.Series:
    if min_dur == 0 or tuples.empty:
        return tuples
    out = tuples.copy()
    run_id = tuples.ne(tuples.shift()).cumsum()
    pos = {f: i for i, f in enumerate(tuples.index)}
    for _, grp in tuples.groupby(run_id):
        edge = grp.iloc[0]
        if edge[0] != edge[1] or len(grp) >= min_dur:
            continue
        last = pos[grp.index[-1]]
        if last + 1 >= len(tuples):
            continue
        out.loc[grp.index] = [tuples.iloc[last + 1]] * len(grp)
    return out


def events_from_edges(tracking: pd.DataFrame, edges: pd.DataFrame, min_dur: int = 10) -> pd.DataFrame:
    t = tracking.set_index("frame_id").sort_index() if "frame_id" in tracking.columns else tracking
    e = edges.set_index("frame_id").sort_index() if "frame_id" in edges.columns else edges
    rows: List[dict] = []
    for ep in sorted(t["episode_id"].unique()):
        if ep == 0:
            continue
        et = t[t["episode_id"] == ep]
        ee = e.loc[et.index].dropna(subset=["edge_src", "edge_dst"])
        if ee.empty:
            continue
        tuples = pd.Series(list(zip(ee["edge_src"], ee["edge_dst"])), index=ee.index)
        tuples = overwrite_short_self_loops(tuples, min_dur)
        prev = None
        ep_rows = []
        for f, cur in tuples.items():
            if prev is not None and cur == prev:
                prev = cur
                continue
            if cur[0] == cur[1]:
                if prev is None:
                    prev = cur
                    continue
                kind = "control" if is_player(cur[0]) else "out"
            elif is_player(cur[0]):
                kind = "kick"
            else:
                prev = cur
                continue
            x, y = node_xy(t, f, cur[0])
            ep_rows.append({"frame_id": int(f), "period_id": int(et.at[f, "period_id"]), "episode_id": int(ep), "seconds": float(et.at[f, "timestamp"]), "player_id": cur[0], "receiver_id": cur[1], "event_type": kind, "start_x": x, "start_y": y})
            prev = cur
        rows += ep_rows
    return pd.DataFrame(rows)


def simplify_events(events: pd.DataFrame, drop_fouls: bool) -> pd.DataFrame:
    ev = events.copy()
    if drop_fouls:
        ev = ev[ev["event_type"] != "foul"].copy()
    if "seconds" not in ev.columns:
        ev["seconds"] = ev["timestamp"].astype(float)
    ev = ev.sort_values("frame_id", ignore_index=True, kind="stable")
    nxt = ev.groupby("episode_id")["player_id"].shift(-1)
    simple = pd.Series(index=ev.index, dtype=object)
    is_out = ev["event_type"] == "out"
    simple[is_out] = "out"
    non_out = ~is_out
    same = ev["player_id"] == nxt
    simple[non_out & same] = "control"
    simple[non_out & ~same & nxt.notna()] = "kick"
    last = non_out & nxt.isna()
    simple[last & ev["event_type"].isna()] = "kick"
    simple[last & ev["event_type"].isin(OUTGOING)] = "kick"
    simple[last & ev["event_type"].isin(INCOMING)] = "control"
    ev["event_type"] = simple
    return ev[ev["event_type"].isin(SIMPLE_TYPES)].copy()


def _pair_score(a: pd.Series, b: pd.Series) -> float:
    if a["frame_id"] == b["frame_id"]:
        ts = 25.0
    else:
        td = round(abs(a["seconds"] - b["seconds"]), 2)
        ts = 0.0 if td >= 3.0 else 25.0 * (1.0 - td / 3.0)
    d = float(np.hypot(a["start_x"] - b["start_x"], a["start_y"] - b["start_y"]))
    ds = 25.0 * (1.0 - min(d, 25.0) / 25.0)
    return ts + ds + (25.0 if a["player_id"] == b["player_id"] else 0.0) + (25.0 if a["event_type"] == b["event_type"] else 0.0)


def needleman_wunsch(true_ev: pd.DataFrame, pred_ev: pd.DataFrame, gap: float = -10.0) -> List[Tuple[Optional[int], Optional[int]]]:
    n, m = len(true_ev), len(pred_ev)
    if n == 0 or m == 0:
        return [(i, None) for i in range(n)] + [(None, j) for j in range(m)]
    dp = np.zeros((n + 1, m + 1))
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[1:, 0] = gap * np.arange(1, n + 1)
    dp[0, 1:] = gap * np.arange(1, m + 1)
    ptr[1:, 0], ptr[0, 1:] = 1, 2
    T = [true_ev.iloc[i] for i in range(n)]
    P = [pred_ev.iloc[j] for j in range(m)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1, j - 1] + _pair_score(T[i - 1], P[j - 1])
            dele = dp[i - 1, j] + gap
            ins = dp[i, j - 1] + gap
            if match >= dele and match >= ins:
                dp[i, j], ptr[i, j] = match, 0
            elif dele >= ins:
                dp[i, j], ptr[i, j] = dele, 1
            else:
                dp[i, j], ptr[i, j] = ins, 2
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i, j] == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or ptr[i, j] == 1):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


def event_scores(true_touches: pd.DataFrame, pred_events: pd.DataFrame, tol_s: float = 1.0) -> pd.DataFrame:
    """Precision / recall / F1 per simplified type and total. `true_touches` = the touches parquet."""
    tr = true_touches.copy()
    if "seconds" not in tr.columns:
        tr["seconds"] = tr["timestamp"].astype(float)
    tr = simplify_events(tr, drop_fouls=True)
    pr = simplify_events(pred_events, drop_fouls=False)
    stats = {t: {"pred": 0, "true": 0, "matched": 0} for t in SIMPLE_TYPES}
    for t in SIMPLE_TYPES:
        stats[t]["true"] = int((tr["event_type"] == t).sum())
        stats[t]["pred"] = int((pr["event_type"] == t).sum())
    for ep in np.sort(pd.Index(tr["episode_id"]).union(pr["episode_id"]).unique()):
        a, b = tr[tr["episode_id"] == ep], pr[pr["episode_id"] == ep]
        for i, j in needleman_wunsch(a, b):
            if i is None or j is None:
                continue
            x, y = a.iloc[i], b.iloc[j]
            if x["player_id"] == y["player_id"] and x["event_type"] == y["event_type"] and abs(x["seconds"] - y["seconds"]) <= tol_s:
                stats[x["event_type"]]["matched"] += 1
    df = pd.DataFrame(stats).T
    df.loc["total"] = df.sum()
    df["precision"] = np.where(df["pred"] > 0, df["matched"] / df["pred"], 0.0)
    df["recall"] = np.where(df["true"] > 0, df["matched"] / df["true"], 0.0)
    pr_sum = df["precision"] + df["recall"]
    df["f1"] = np.where(pr_sum > 0, 2 * df["precision"] * df["recall"] / pr_sum, 0.0)
    return df

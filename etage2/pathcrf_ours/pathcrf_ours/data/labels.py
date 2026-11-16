"""From per-frame player labels to per-step (sender, receiver) node indices.

Downsampling: step i covers the source rows (down_idx[i-1], down_idx[i]]; its label is the last
non-NaN player label inside that bucket. Then
    src = ffill then bfill   (who last had the ball)
    dst = bfill then ffill   (who gets it next)
so a self-loop (src == dst) means "holds", src != dst means "in flight from src to dst".
goal_left / goal_right map to the out_left / out_right nodes.
Fix-up: "A->A at t, B->B at t+1" is impossible for the CRF, so step t becomes A->B.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import OUTSIDE_NODES


def node_order(left_players: List[str], right_players: List[str]) -> List[str]:
    return list(left_players) + list(right_players) + list(OUTSIDE_NODES)


def node_index(order: List[str]) -> Dict[str, int]:
    idx = {name: i for i, name in enumerate(order)}
    idx["goal_left"] = idx["out_left"]
    idx["goal_right"] = idx["out_right"]
    return idx


def downsample_index(n_frames: int, every: int) -> np.ndarray:
    down = np.arange(0, n_frames, every, dtype=np.int64)
    if len(down) == 0 or down[-1] != n_frames - 1:
        down = np.append(down, n_frames - 1)
    return down


def step_labels(player_id: pd.Series, down_idx: np.ndarray, index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """player_id: per-frame labels of one episode (positional). Returns (src, dst) as float arrays (NaN if unmapped)."""
    vals = player_id.reset_index(drop=True)
    per_step = []
    prev = -1
    for cur in down_idx:
        bucket = vals.iloc[prev + 1 : cur + 1].dropna()
        per_step.append(bucket.iloc[-1] if not bucket.empty else np.nan)
        prev = int(cur)
    s = pd.Series(per_step, dtype=object)
    with pd.option_context("future.no_silent_downcasting", True):
        src = s.ffill().bfill().map(index).to_numpy(dtype=float)
        dst = s.bfill().ffill().map(index).to_numpy(dtype=float)
    return fix_self_loop_jumps(src, dst)


def fix_self_loop_jumps(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(src) > 1:
        loop_now = src[:-1] == dst[:-1]
        loop_next = src[1:] == dst[1:]
        change = src[:-1] != src[1:]
        bad = loop_now & loop_next & change
        if bad.any():
            dst = dst.copy()
            dst[:-1] = np.where(bad, src[1:], dst[:-1])
    return src, dst

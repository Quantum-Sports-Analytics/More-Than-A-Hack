"""Processed match -> fixed-length training windows.

A processed match (see pipeline.py) is a 25 Hz table with phase_id, episode_id, player_id (per-frame
label) and x, y, vx, vy, speed, accel per object. For every phase (constant set of players) the nodes
are ordered left-keeper team, right team, outside nodes; for every in-play episode of at least
`window_seconds` the frames are sampled every `sample_every` (25 -> 5 fps), labelled per step, and cut
into windows of `window_steps` with stride `window_stride` (the last window is aligned to the end).

Node features (8): team flag (-1 left, +1 right, 0 outside), goalkeeper flag, x, y, vx, vy, speed, accel.
Teams are padded to `team_size` with zero rows.

Presence (Phase B): every sample also carries `present [T, N]` (1 = a real observation of that node at
that step, 0 = a padded row or a frame filled by the damage/adapter). It comes from the `<player>_present`
columns when the table has them (damaged tables), else from the padding only. The training loop appends it
as a 9th feature when the model has `presence=True`.

Variants (Phase B): a match may be given as a list of tables (the clean one and K damaged draws). All of
them are indexed; `resample(rng)` picks one variant per match for the coming epoch, so every epoch sees a
differently broken copy of every match while the epoch length stays that of one copy.

Flip augmentation (each axis independently with p = 0.5):
    "legacy": the official behaviour. flip_x mirrors x and vx of every node, swaps the two team blocks,
              remaps player labels by +-team_size and swaps out_left/out_right labels; flip_y mirrors y
              and vy and swaps out_bottom/out_top labels. The outside-node rows are NOT reordered and the
              team flag is NOT re-signed, so after a flip the out labels point at the wrong outside row.
    "fixed":  same, plus the outside rows are swapped to match the labels and the team flag is re-signed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import FEATURE_NAMES, OUTSIDE_NODES, OUTSIDE_XY, PITCH_X, PITCH_Y, DataConfig
from . import labels as L
from .ball import N_BALL_FEATURES, BallDamageConfig, ball_node_features, damage_ball
from .sportec import detect_keepers, label_phases, team_of

PHYS = ("x", "y", "vx", "vy", "speed", "accel")
I_TEAM, I_GK, I_X, I_Y, I_VX, I_VY = 0, 1, 2, 3, 4, 5


def add_outside_columns(tracking: pd.DataFrame) -> pd.DataFrame:
    tracking = tracking.copy()
    for name, (x, y) in OUTSIDE_XY.items():
        tracking[f"{name}_x"], tracking[f"{name}_y"] = x, y
        for k in ("vx", "vy", "speed", "accel"):
            tracking[f"{name}_{k}"] = 0.0
        tracking[f"{name}_present"] = True
    return tracking


MatchTables = Union[pd.DataFrame, Sequence[pd.DataFrame]]


class WindowDataset(Dataset):
    def __init__(self, matches: Dict[str, MatchTables], cfg: DataConfig, team_size: int = 11, flip: Optional[str] = None, seed: int = 0,
                 ball_damage: Optional[bool] = None, deterministic: bool = False):
        """ball_damage: damage the ball of every window (None = cfg.ball_damage); deterministic: the damage of window i is
        drawn from a generator seeded with (seed, i), so a validation set is the same at every epoch."""
        self.cfg = cfg
        self.team_size = team_size
        self.flip = cfg.flip if flip is None else flip
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.ball_damage = cfg.ball_damage if ball_damage is None else ball_damage
        self.deterministic = deterministic
        self.ball_cfg = BallDamageConfig()
        self.n_nodes = 2 * team_size + len(OUTSIDE_NODES)
        self.samples: List[dict] = []
        self.groups: Dict[str, Dict[int, List[int]]] = {}      # match -> variant -> sample indices
        for match_id, tables in matches.items():
            tables = [tables] if isinstance(tables, pd.DataFrame) else list(tables)
            self.groups[match_id] = {}
            for v, tracking in enumerate(tables):
                start = len(self.samples)
                self._add_match(match_id, tracking, v)
                self.groups[match_id][v] = list(range(start, len(self.samples)))
        self.active: List[int] = [i for m in self.groups for i in self.groups[m][0]]

    # -------------------------------------------------------------------------------- construction
    def _add_match(self, match_id: str, tracking: pd.DataFrame, variant: int = 0) -> None:
        cfg, ts = self.cfg, self.team_size
        every, W = cfg.sample_every, cfg.window_steps
        tracking, phases = label_phases(tracking)
        tracking = add_outside_columns(tracking)
        has_present = any(c.endswith("_present") and not c.startswith("out_") for c in tracking.columns)
        for phase, prow in phases.iterrows():
            active = list(prow["active_players"])
            if not active:
                continue
            pt = tracking[tracking["phase_id"] == phase]
            left_gk, right_gk = detect_keepers(pt)
            left_team, right_team = team_of(left_gk), team_of(right_gk)
            left = [p for p in active if p.startswith(left_team)]
            right = [p for p in active if p.startswith(right_team)]
            order = L.node_order(left, right)
            index = L.node_index(order)
            cols = [f"{p}_{f}" for p in order for f in PHYS]
            pcols = [f"{p}_present" for p in order]
            flags = np.zeros((len(order), 2), dtype=np.float32)
            for i, p in enumerate(order):
                flags[i, 0] = -1.0 if p.startswith(left_team) else (1.0 if p.startswith(right_team) else 0.0)
                flags[i, 1] = 1.0 if p in (left_gk, right_gk) else 0.0
            for ep in pt["episode_id"].unique():
                if ep == 0:
                    continue
                et = tracking[tracking["episode_id"] == ep]
                if len(et) < cfg.window_seconds * cfg.source_fps:
                    continue
                pids = et["player_id"].dropna()
                pids = pids[~pids.astype(str).str.startswith(("out_", "goal_"))]
                if pids.empty or any(et[f"{p}_x"].isna().any() for p in pids.unique()):
                    continue
                if et[cols].isna().any().any():
                    continue
                feats = et[cols].to_numpy(dtype=np.float32).reshape(len(et), len(order), len(PHYS))
                feats = np.concatenate([np.broadcast_to(flags[None], (len(et), len(order), 2)), feats], axis=-1)
                if has_present:
                    pres = et[pcols].to_numpy(dtype=bool)
                else:
                    pres = np.ones((len(et), len(order)), dtype=bool)
                down = L.downsample_index(len(et), every)
                src, dst = L.step_labels(et["player_id"], down, index)
                if np.isnan(src).any() or np.isnan(dst).any():
                    continue
                n_pl = len(order) - len(OUTSIDE_NODES)
                if ((src >= n_pl) & (src != dst)).any():      # outside -> another node is not a CRF state (out -> out is)
                    continue
                fd, pd_ = feats[down], pres[down]
                lab = np.stack([src, dst], axis=1).astype(np.int64)
                if len(fd) < W:
                    continue
                bxy = et[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)[down] if cfg.ball else None
                starts = list(range(0, len(fd) - W + 1, cfg.window_stride))
                starts = starts[:-1] + [len(fd) - W]
                x_pad, lab_pad, pres_pad = self._pad(fd, lab, pd_, order, left_team, right_team)
                for s in starts:
                    self.samples.append(
                        {
                            "x": x_pad[s : s + W],
                            "y": lab_pad[s : s + W],
                            "present": pres_pad[s : s + W],
                            "ball": bxy[s : s + W] if bxy is not None else None,
                            "meta": {"match": match_id, "variant": variant, "phase": int(phase), "episode": int(ep),
                                     "start_frame": int(et["frame_id"].iloc[down[s]]), "order": order},
                        }
                    )

    def _pad(self, feats: np.ndarray, lab: np.ndarray, pres: np.ndarray, order: List[str], left_team: str, right_team: str):
        ts, N = self.team_size, self.n_nodes
        T, n, F = feats.shape
        left = [i for i, p in enumerate(order) if p.startswith(left_team)][:ts]
        right = [i for i, p in enumerate(order) if p.startswith(right_team)][:ts]
        outside = [i for i, p in enumerate(order) if p in OUTSIDE_NODES]
        x = np.zeros((T, N, F), dtype=np.float32)
        present = np.zeros((T, N), dtype=np.float32)
        slots = {}
        for k, i in enumerate(left):
            x[:, k] = feats[:, i]
            present[:, k] = pres[:, i]
            slots[i] = k
        for k, i in enumerate(right):
            x[:, ts + k] = feats[:, i]
            present[:, ts + k] = pres[:, i]
            slots[i] = ts + k
        for k, i in enumerate(outside):
            x[:, 2 * ts + k] = feats[:, i]
            present[:, 2 * ts + k] = 1.0
            slots[i] = 2 * ts + k
        remap = np.vectorize(lambda i: slots[int(i)])
        return x, remap(lab).astype(np.int64), present

    # ------------------------------------------------------------------------------------- variants
    def resample(self, rng: Optional[np.random.Generator] = None, variant: Optional[int] = None) -> None:
        """Pick one variant per match (random, or a fixed index clipped to what the match has)."""
        rng = rng or self.rng
        active = []
        for m, vs in self.groups.items():
            keys = sorted(vs)
            v = keys[min(variant, len(keys) - 1)] if variant is not None else keys[int(rng.integers(len(keys)))]
            active += vs[v]
        self.active = active

    def n_variants(self) -> int:
        return max(len(v) for v in self.groups.values()) if self.groups else 0

    # ------------------------------------------------------------------------------------- access
    def __len__(self) -> int:
        return len(self.active)

    def __getitem__(self, i: int):
        s = self.samples[self.active[i]]
        x, y, present = s["x"].copy(), s["y"].copy(), s["present"].copy()
        ball = s["ball"].copy() if s["ball"] is not None else None
        rng = np.random.default_rng([self.seed, i]) if self.deterministic else self.rng
        if self.flip != "none":
            x, y, present, ball = self.random_flip(x, y, present, self.flip, rng, self.team_size, ball)
        if ball is not None:
            if self.ball_damage:
                ball = damage_ball(ball, self.cfg.target_fps, rng, self.ball_cfg)
            ballf = ball_node_features(ball, x[:, :, [I_X, I_Y]], self.cfg.target_fps)
        else:
            ballf = np.zeros((x.shape[0], x.shape[1], 0), dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(present), torch.from_numpy(ballf)

    @staticmethod
    def random_flip(x: np.ndarray, y: np.ndarray, present: np.ndarray, mode: str, rng: np.random.Generator, ts: int, ball: Optional[np.ndarray] = None):
        flip_x, flip_y = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
        return WindowDataset.apply_flip(x, y, present, flip_x, flip_y, mode, ts, ball)

    @staticmethod
    def apply_flip(x: np.ndarray, y: np.ndarray, present: np.ndarray, flip_x: bool, flip_y: bool, mode: str, ts: int, ball: Optional[np.ndarray] = None):
        """present: [N] or [T, N]; the team blocks are swapped along its last axis. ball [T, 2] is mirrored with the players."""
        o_l, o_r, o_b, o_t = 2 * ts, 2 * ts + 1, 2 * ts + 2, 2 * ts + 3
        if flip_x:
            x[:, :, I_X] = PITCH_X - x[:, :, I_X]
            x[:, :, I_VX] = -x[:, :, I_VX]
            if ball is not None:
                ball[:, 0] = PITCH_X - ball[:, 0]
            x = np.concatenate([x[:, ts : 2 * ts], x[:, :ts], x[:, 2 * ts :]], axis=1)
            present = np.concatenate([present[..., ts : 2 * ts], present[..., :ts], present[..., 2 * ts :]], axis=-1)
            lab = y.copy()
            y = np.where(lab < ts, lab + ts, np.where(lab < 2 * ts, lab - ts, lab))
            y = np.where(lab == o_l, o_r, np.where(lab == o_r, o_l, y))
            if mode == "fixed":
                x[:, :, I_TEAM] = -x[:, :, I_TEAM]
                x[:, [o_l, o_r]] = x[:, [o_r, o_l]]
        if flip_y:
            x[:, :, I_Y] = PITCH_Y - x[:, :, I_Y]
            x[:, :, I_VY] = -x[:, :, I_VY]
            if ball is not None:
                ball[:, 1] = PITCH_Y - ball[:, 1]
            lab = y.copy()
            y = np.where(lab == o_b, o_t, np.where(lab == o_t, o_b, lab))
            if mode == "fixed":
                x[:, [o_b, o_t]] = x[:, [o_t, o_b]]
        return x, y, present, ball


def collate(batch):
    xs, ys, ps, bs = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ps), torch.stack(bs)

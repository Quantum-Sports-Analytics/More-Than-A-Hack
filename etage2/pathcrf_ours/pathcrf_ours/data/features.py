"""Per-object kinematics: velocity, speed and acceleration from x, y at the source frame rate.

For every player (and the ball), per period, over that object's non-NaN frames:
- players only: frames whose frame-to-frame speed exceeds `speed_outlier` (12 m/s) are set to NaN and
  refilled by linear interpolation (then bfill/ffill);
- vx, vy = Savitzky-Golay smoothing (window 15, order 2) of the frame difference x fps;
- ax, ay = Savitzky-Golay (window 9, order 2) of the velocity difference x fps;
- speed = |v|, accel = |a| (magnitudes, never signed);
- the first frame copies the second frame's velocity; accel is 0 on the first and last frame.
Nothing is normalised: metres, m/s and m/s^2 go straight into the model.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..config import DataConfig
from .sportec import players_of

KINEMATICS = ("vx", "vy", "speed", "accel")


def add_kinematics(tracking: pd.DataFrame, cfg: DataConfig, objects: List[str] | None = None) -> pd.DataFrame:
    fps = cfg.source_fps
    tracking = tracking.copy()
    if objects is None:
        home, away = players_of(tracking)
        objects = home + away + ["ball"]
    new_cols = [f"{p}_{k}" for p in objects for k in KINEMATICS]
    tracking = tracking.drop(columns=[c for c in new_cols if c in tracking.columns])
    tracking = pd.concat([tracking, pd.DataFrame(np.nan, index=tracking.index, columns=new_cols)], axis=1)
    n_removed = 0
    for p in objects:
        for period in tracking["period_id"].unique():
            pm = tracking["period_id"] == period
            x = tracking.loc[pm, f"{p}_x"].dropna()
            y = tracking.loc[pm, f"{p}_y"].dropna()
            if x.empty:
                continue
            if p != "ball":
                inst = np.sqrt(x.diff() ** 2 + y.diff() ** 2) * fps
                bad = x.index[inst > cfg.speed_outlier]
                if len(bad):
                    n_removed += len(bad)
                    idx = x.index
                    tracking.loc[bad, [f"{p}_x", f"{p}_y"]] = np.nan
                    x = tracking.loc[idx, f"{p}_x"].interpolate(method="linear").bfill().ffill()
                    y = tracking.loc[idx, f"{p}_y"].interpolate(method="linear").bfill().ffill()
                    tracking.loc[idx, f"{p}_x"] = x
                    tracking.loc[idx, f"{p}_y"] = y
            if len(x) < 3:
                continue
            w_v, o_v = cfg.savgol_vel
            w_a, o_a = cfg.savgol_acc
            vx = savgol_filter(np.diff(x.to_numpy()) * fps, window_length=w_v, polyorder=o_v)
            vy = savgol_filter(np.diff(y.to_numpy()) * fps, window_length=w_v, polyorder=o_v)
            ax = savgol_filter(np.diff(vx) * fps, window_length=w_a, polyorder=o_a)
            ay = savgol_filter(np.diff(vy) * fps, window_length=w_a, polyorder=o_a)
            tracking.loc[x.index[1:], f"{p}_vx"] = vx
            tracking.loc[x.index[1:], f"{p}_vy"] = vy
            tracking.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            tracking.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)
            for k in ("vx", "vy", "speed"):
                tracking.at[x.index[0], f"{p}_{k}"] = tracking.at[x.index[1], f"{p}_{k}"]
            tracking.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0.0
    tracking.attrs["outliers_removed"] = n_removed
    return tracking


def processed_columns(tracking: pd.DataFrame) -> List[str]:
    """The column set of a processed match: state columns + 6 physical features per object."""
    home, away = players_of(tracking)
    state = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id", "ball_state", "ball_owning_team_id"]
    if "player_id" in tracking.columns:
        state.append("player_id")
    feats = [f"{p}_{f}" for p in home + away + ["ball"] for f in ("x", "y", *KINEMATICS)]
    if "ball_z" in tracking.columns:
        feats.append("ball_z")
    return state + feats

"""Ball input for PathCRF (Phase B.B, 2026-11-12): per-node relative features and Metrica-like ball damage.

The ball never becomes a node. Every node (22 players + 4 outside anchors) gets `N_BALL_FEATURES` extra input
channels per step, computed from the (possibly damaged) ball position at 5 fps:
    dx = ball_x - node_x, dy = ball_y - node_y, dist = |ball - node|, ball speed (m/s, from the step difference),
    missing (1 when the ball is not observed at that step; the other four channels are then 0).
Metres, no normalisation, like the rest of the PathCRF inputs. No height: Metrica has none.

Damage (drawn fresh per window during training, seeded per window for validation), after the measurements on
Barça–Athletic: long gaps (the chain's merged ball is missing on
11 % of rows, the raw tracking ball on 41 %), a slow offset that grows with ball speed (0.5-1.3 m, biased across
the pitch), small jitter, glitches (2.9 per minute, median 0.7 s) that put the ball somewhere else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BALL_FEATURES = 5


@dataclass
class BallDamageConfig:
    p_window_missing: float = 0.15      # the whole 10 s window has no ball
    p_partial_gap: float = 0.15         # one gap of gap_min..gap_max seconds inside the window
    gap_min_s: float = 1.0
    gap_max_s: float = 8.0
    offset_x_sd: float = 0.7            # slow (per-window) offset, metres
    offset_y_sd: float = 1.2
    offset_y_bias: float = -0.6
    offset_speed_gain: float = 0.04     # extra offset sd per m/s of ball speed (≈ 1.1 m sd at 10 m/s, 1.7 m at 25 m/s)
    jitter_sd: float = 0.1
    p_glitch: float = 0.15              # one glitch stretch in the window
    glitch_min_s: float = 0.4
    glitch_max_s: float = 2.0
    glitch_jump_sd: float = 8.0         # metres, where the glitch sends the ball


def damage_ball(ball: np.ndarray, fps: float, rng: np.random.Generator, cfg: BallDamageConfig) -> np.ndarray:
    """ball [T, 2] (NaN = missing) -> damaged copy [T, 2]."""
    T = len(ball)
    out = ball.astype(np.float32).copy()
    if rng.random() < cfg.p_window_missing:
        out[:] = np.nan
        return out
    if rng.random() < cfg.p_partial_gap:
        L = int(round(rng.uniform(cfg.gap_min_s, cfg.gap_max_s) * fps))
        s = int(rng.integers(0, max(T - L, 1)))
        out[s : s + L] = np.nan
    seen = ~np.isnan(out[:, 0])
    if seen.sum() >= 2:
        v = np.zeros(T, dtype=np.float32)
        d = np.linalg.norm(np.diff(out, axis=0), axis=1) * fps
        v[1:] = np.nan_to_num(d, nan=0.0)
        sd_x = cfg.offset_x_sd + cfg.offset_speed_gain * v
        sd_y = cfg.offset_y_sd + cfg.offset_speed_gain * v
        ox = rng.normal(0.0, 1.0) * sd_x + rng.normal(0.0, cfg.jitter_sd, T)
        oy = cfg.offset_y_bias + rng.normal(0.0, 1.0) * sd_y + rng.normal(0.0, cfg.jitter_sd, T)
        out[:, 0] += ox
        out[:, 1] += oy
    if rng.random() < cfg.p_glitch:
        L = int(round(rng.uniform(cfg.glitch_min_s, cfg.glitch_max_s) * fps))
        s = int(rng.integers(0, max(T - L, 1)))
        jump = rng.normal(0.0, cfg.glitch_jump_sd, 2).astype(np.float32)
        out[s : s + L] += jump
    return out


def ball_node_features(ball: np.ndarray, node_xy: np.ndarray, fps: float) -> np.ndarray:
    """ball [T, 2] (NaN = missing), node_xy [T, N, 2] -> [T, N, N_BALL_FEATURES] float32."""
    T, N, _ = node_xy.shape
    f = np.zeros((T, N, N_BALL_FEATURES), dtype=np.float32)
    missing = np.isnan(ball[:, 0]) | np.isnan(ball[:, 1])
    b = np.where(missing[:, None], 0.0, ball).astype(np.float32)
    rel = b[:, None, :] - node_xy
    f[:, :, 0] = rel[:, :, 0]
    f[:, :, 1] = rel[:, :, 1]
    f[:, :, 2] = np.linalg.norm(rel, axis=-1)
    speed = np.zeros(T, dtype=np.float32)
    if T > 1:
        d = np.linalg.norm(np.diff(b, axis=0), axis=1) * fps
        both = ~missing[1:] & ~missing[:-1]
        speed[1:] = np.where(both, d, 0.0)
    f[:, :, 3] = speed[:, None]
    f[missing] = 0.0
    f[:, :, 4] = missing[:, None].astype(np.float32)
    return f

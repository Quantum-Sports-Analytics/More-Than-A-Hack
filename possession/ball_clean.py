"""Cleaned ball track for one question only: did the ball leave the pitch?

The fused ball (video_ball.py) is a ground projection through the tactical camera. Three artefacts make it say
"out" when the ball stayed in play (seen on the provided match while studying the errors of the dead-ball detector;
nothing here is fitted on it, the constants are physical orders of magnitude):
  1. a ball in the air projects beyond its true ground position (a 3 m high cross near the goal lands on the
     goal line): inside a validated flight, the fitted ground position of aerial.py replaces the projection;
  2. a ball that is probably in the air (p_air) without a validated flight has no usable ground position: unknown;
  3. a detection that stays at the same spot outside the pitch for seconds is a fixed object, not the ball: removed.
A missing ball is "unknown", never "out".
"""
import numpy as np

LENGTH, WIDTH = 105.3, 68.0
DEFAULTS = dict(p_air=0.5, static_s=2.0, static_m=0.3, fps=25)


def clean_ball(ball, aerial=None, prm=None):
    """ball: balle_fusionnee.npz (x, y); aerial: aerial.npz or None. Returns x, y (metres, NaN = unknown)."""
    prm = dict(DEFAULTS, **(prm or {}))
    x, y = np.asarray(ball["x"], float).copy(), np.asarray(ball["y"], float).copy()
    if aerial is not None:
        xg, yg = np.asarray(aerial["x_ground"], float), np.asarray(aerial["y_ground"], float)
        fitted = (np.asarray(aerial["flight_id"]) >= 0) & np.isfinite(xg)
        x[fitted], y[fitted] = xg[fitted], yg[fitted]
        in_air = (np.nan_to_num(np.asarray(aerial["p_air"], float)) > prm["p_air"]) & ~fitted
        x[in_air], y[in_air] = np.nan, np.nan
    outside = (x < 0) | (x > LENGTH) | (y < 0) | (y > WIDTH)
    n_min, tol, n, i = int(round(prm["static_s"] * prm["fps"])), prm["static_m"], len(x), 0
    static = np.zeros(n, bool)
    while i < n:
        if not np.isfinite(x[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(x[j + 1]) and abs(x[j + 1] - x[i]) < tol and abs(y[j + 1] - y[i]) < tol:
            j += 1
        if j - i + 1 >= n_min and outside[i]:
            static[i:j + 1] = True
        i = j + 1
    x[static], y[static] = np.nan, np.nan
    return x, y


def line_margin(x, y, lo, hi, line):
    """Smallest distance (m) of the ball to the touchlines (line='touch') or to the goal lines (line='goal') over
    frames [lo, hi); negative = beyond the line; NaN = ball never known there."""
    xs, ys = x[max(lo, 0):hi], y[max(lo, 0):hi]
    d = np.minimum(ys, WIDTH - ys) if line == "touch" else np.minimum(xs, LENGTH - xs)
    return float(np.nanmin(d)) if np.isfinite(d).any() else float("nan")

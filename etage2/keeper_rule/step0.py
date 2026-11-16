"""Step 0 of every prediction chain: find each team's camp, then fix the keepers.

Runs on the tracking BEFORE any model (ball in the air, dead ball, possession):
  1. Camps: in each half, the team whose outfield players' average position along the long axis (x) is further
     left defends the left goal (keeper_rule/test_density/sides.py, method "centroid"; keeper slots are left out,
     so a mislabelled keeper cannot change it). Right on 2/2 Barça–Athletic halves and 40/40 SkillCorner halves.
  2. Keeper rule v2 with those camps and the continuity fallback (keeper_rule/test_density/keeper_v2c.py): a
     keeper slot never stays in the other team's half; who carries the team's keeper is decided by speed, else by
     the smoothest track. On Barça–Athletic it gives the same result as pipeline/keeper_slots.fix_halves_v2.

Use:
    import sys; sys.path.insert(0, "keeper_rule"); import step0
    actions = step0.apply_halves(halves, team, is_gk, fps)       # metrica_io.load() halves, changed in place
    X, Y, actions, side0 = step0.apply_half(X, Y, team, is_gk, fps)   # one half (any source, e.g. SkillCorner)
side0 = x sign of team 0's own goal (-1 = left goal).
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "test_density", _HERE.parent / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import keeper_v2c as _KC  # noqa: E402
import sides as _S  # noqa: E402


def camp(X, team, is_gk):
    """x sign of team 0's own goal in this half (-1 = left goal, +1 = right goal), and the share of frames agreeing."""
    r = _S.half_sides(X, np.asarray(team), np.asarray(is_gk, bool))["centroid"]
    return r["side0"], r["share"] if r["side0"] == -1 else 1 - r["share"]


def apply_half(X, Y, team, is_gk, fps=10):
    team, is_gk = np.asarray(team), np.asarray(is_gk, bool)
    side0, _ = camp(X, team, is_gk)
    Xf, Yf, acts = _KC.fix_half(X, Y, team, is_gk, side0, fps, "continuity")
    return Xf, Yf, acts, side0


def apply_halves(halves, team, is_gk, fps=10):
    """Same interface as pipeline/keeper_slots.fix_halves(_v2): halves changed in place, actions returned."""
    out = []
    for h in halves:
        h["X"], h["Y"], acts, side0 = apply_half(h["X"], h["Y"], team, is_gk, fps)
        h["side0"] = side0
        for a in acts:
            a["period"] = h["period"]
            a["video_start"], a["video_end"] = float(h["video_pos"][a["start"]]), float(h["video_pos"][a["end"] - 1])
        out += acts
    return out

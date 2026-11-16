"""Keeper-slot clean-up on Metrica tracking, before any model.

Problem: Metrica sometimes shows two "keepers" in one penalty box. One is a real person who moves; the other is a
near-still dot (noise, often exactly frozen). The moving one can sit in the other team's keeper slot (a keeper filed
under the wrong team).

Rule, per half (the box of team t = the box on the side where t's keeper slot stands most of the half):
  1. Conflict stretch: both keeper slots inside the same box (|x| >= 52.65 - 16.5 on that side, |y| <= 20.16) for
     at least 1 s; gaps under 1 s are joined.
  2. Movement of each slot over the stretch: path length / time (m/s).
  3. Decision once per stretch: the faster slot is the box's keeper if it is at least RATIO times faster and at
     least MIN_DIFF m/s faster. Otherwise the stretch is unsure and nothing changes.
       - faster slot = the box owner's slot: the other slot (still dot) is set to missing;
       - faster slot = the other team's slot: its positions are moved into the box owner's keeper slot, and the
         other team's keeper slot is set to missing.
  4. Lone intruder: the other team's keeper slot is in the box for at least 1 s while the owner's keeper slot is
     missing: its positions are moved into the owner's slot.
Measured on BAR-ATH at 25 fps: 19 conflict stretches (600 s), 15 decided, 4 unsure.
"""
import numpy as np

GOAL_X, BOX_X, BOX_Y = 52.65, 16.5, 20.16
MIN_S = 1.0          # shortest conflict stretch, and longest gap joined
RATIO = 3.0
MIN_DIFF = 0.3       # m/s


def _runs(mask, fps):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    st = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    merged = []
    for a, b in st:
        if merged and a - merged[-1][1] <= MIN_S * fps:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= MIN_S * fps]


def _speed(x, y, fps):
    step = np.hypot(np.diff(x), np.diff(y))
    return float(np.nanmean(step) * fps) if np.isfinite(step).any() else np.nan


def fix_half(X, Y, team, is_gk, fps=10):
    """X, Y: (n, slots) positions in metres for one half. Returns fixed copies and a list of actions."""
    X, Y = X.copy(), Y.copy()
    gk = {t: int(np.flatnonzero((team == t) & is_gk)[0]) for t in (0, 1)}
    with np.errstate(all="ignore"):
        side = {t: np.sign(np.nanmedian(X[:, gk[t]])) for t in (0, 1)}
    actions = []
    for t in (0, 1):                       # box owned by team t
        o, q = gk[t], gk[1 - t]            # owner's slot, other team's slot
        inbox = {j: (side[t] * X[:, j] >= GOAL_X - BOX_X) & (np.abs(Y[:, j]) <= BOX_Y) for j in (o, q)}
        for a, b in _runs(inbox[o] & inbox[q], fps):
            vo, vq = _speed(X[a:b, o], Y[a:b, o], fps), _speed(X[a:b, q], Y[a:b, q], fps)
            fast, slow = max(vo, vq), min(vo, vq)
            act = dict(team_box=t, start=a, end=b, v_owner=round(vo, 2), v_other=round(vq, 2))
            if not (fast >= RATIO * max(slow, 0.01) and fast - slow >= MIN_DIFF):
                actions.append({**act, "action": "unsure"})
                continue
            if vq > vo:
                X[a:b, o], Y[a:b, o] = X[a:b, q], Y[a:b, q]
                actions.append({**act, "action": "other slot moves: moved into owner's slot"})
            else:
                actions.append({**act, "action": "owner's slot moves: still dot dropped"})
            X[a:b, q], Y[a:b, q] = np.nan, np.nan
        lone = inbox[q] & np.isnan(X[:, o])
        for a, b in _runs(lone, fps):
            X[a:b, o], Y[a:b, o] = X[a:b, q], Y[a:b, q]
            X[a:b, q], Y[a:b, q] = np.nan, np.nan
            actions.append(dict(team_box=t, start=a, end=b, action="lone intruder: moved into owner's slot"))
    return X, Y, actions


def fix_halves(halves, team, is_gk, fps=10):
    """Applies fix_half to every half of metrica_io.load() in place; returns the actions with video frames."""
    out = []
    for h in halves:
        h["X"], h["Y"], acts = fix_half(h["X"], h["Y"], team, is_gk, fps)
        for a in acts:
            a["period"] = h["period"]
            a["video_start"], a["video_end"] = float(h["video_pos"][a["start"]]), float(h["video_pos"][a["end"] - 1])
        out += acts
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Version 2 (2026-11-03, user): "a keeper slot never stays in the other team's half".
#
# Why: at 1H 05:46 the real Barça keeper is tracked in Athletic's keeper slot (46) inside Barça's box, while Barça's
# slot (24) is a ghost standing still 19 m from the goal line, just outside the box. Version 1 needs both slots inside
# the box (or the owner's slot missing), so it did nothing for 46 s.
#
# Sides (user): blue = home = team 0, red = away = team 1. The home keeper defends the left goal (x < 0) in the 1st
# half and the right goal in the 2nd.
#
# Rule, for the half of team t (the owner), o = t's keeper slot, q = the other team's keeper slot:
#   1. Trigger: q is in t's half (stretches of at least 1 s, gaps under 1 s joined). A keeper never plays in the
#      opponent's half, so q is not the other team's keeper there.
#   2. On every frame of the stretch, who carries t's keeper (speeds over a centred 3 s window, path length / time):
#        a. q frozen (<= FROZEN m/s: a tracking ghost; a real keeper always shifts a little) -> o;
#        b. o missing, outside t's half, or frozen -> q;
#        c. one of the two clearly faster (RATIO times and MIN_DIFF m/s) -> that one;
#        d. otherwise the one nearer t's goal (the keeper is the deepest).
#      The choice is smoothed over SMOOTH_V2 s (majority) so it does not flicker.
#   3. q carries the keeper: q's positions go into o, q is set to missing ("move"). o carries it: q is set to missing
#      ("drop", q is a ghost).
# Known risk: the other team's real keeper going up for a late set piece would be moved too.
WIN_V2 = 3.0          # s, speed window
FROZEN = 0.02         # m/s: a slot this still is a ghost
SMOOTH_V2 = 6.0       # s, majority window on the choice


def _speed_series(x, y, fps):
    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    w = int(round(WIN_V2 * fps))
    import pandas as pd
    return pd.Series(step).rolling(w, center=True, min_periods=max(1, w // 2)).mean().to_numpy() * fps


def fix_half_v2(X, Y, team, is_gk, fps=10, period=None):
    """Version 2 on one half. period 1 / 2 gives the sides from the fixed rule (home on the left in the 1st half);
    without it, from where each keeper slot stands most of the half. Returns fixed copies and the actions
    (runs of 'move' / 'drop' with start, end (excluded), team_box and mean speeds)."""
    import pandas as pd
    X, Y = X.copy(), Y.copy()
    gk = {t: int(np.flatnonzero((team == t) & is_gk)[0]) for t in (0, 1)}
    if period in (1, 2):
        side = {0: -1.0 if period == 1 else 1.0, 1: 1.0 if period == 1 else -1.0}
    else:
        with np.errstate(all="ignore"):
            side = {t: np.sign(np.nanmedian(X[:, gk[t]])) for t in (0, 1)}
    X0, Y0 = X.copy(), Y.copy()                     # decisions are made on the raw tracking
    actions = []
    for t in (0, 1):
        o, q = gk[t], gk[1 - t]
        gx = side[t] * GOAL_X
        q_in = ~np.isnan(X0[:, q]) & (side[t] * X0[:, q] > 0)
        o_in = ~np.isnan(X0[:, o]) & (side[t] * X0[:, o] > 0)
        vo, vq = _speed_series(X0[:, o], Y0[:, o], fps), _speed_series(X0[:, q], Y0[:, q], fps)
        do, dq = np.hypot(X0[:, o] - gx, Y0[:, o]), np.hypot(X0[:, q] - gx, Y0[:, q])
        with np.errstate(invalid="ignore"):
            o_fast = (vo >= RATIO * np.maximum(vq, 0.01)) & (vo - vq >= MIN_DIFF)
            q_fast = (vq >= RATIO * np.maximum(vo, 0.01)) & (vq - vo >= MIN_DIFF)
            q_frozen, o_frozen = vq <= FROZEN, vo <= FROZEN
            keeper_o = o_in & ~o_frozen & (q_frozen | o_fast | (~q_fast & (do < dq)))
            keeper_o |= o_in & q_frozen & o_frozen      # both frozen: drop the intruder, keep the owner's slot
        for a, b in _runs(q_in, fps):
            k = pd.Series(keeper_o[a:b].astype(float)).rolling(int(round(SMOOTH_V2 * fps)), center=True,
                                                               min_periods=1).mean().to_numpy() > 0.5
            d = np.diff(np.r_[0, k.astype(int), 0])
            cuts = sorted(set([0, len(k)] + list(np.flatnonzero(d != 0))))
            for s, e in zip(cuts[:-1], cuts[1:]):
                if e <= s:
                    continue
                fa, fb = a + s, a + e
                drop = bool(k[s])
                if not drop:
                    X[fa:fb, o], Y[fa:fb, o] = X0[fa:fb, q], Y0[fa:fb, q]
                X[fa:fb, q], Y[fa:fb, q] = np.nan, np.nan
                actions.append(dict(team_box=t, start=int(fa), end=int(fb), action="drop" if drop else "move",
                                    v_owner=round(float(np.nanmean(vo[fa:fb])), 2) if np.isfinite(vo[fa:fb]).any() else None,
                                    v_other=round(float(np.nanmean(vq[fa:fb])), 2) if np.isfinite(vq[fa:fb]).any() else None))
    actions.sort(key=lambda d: d["start"])
    return X, Y, actions


def fix_halves_v2(halves, team, is_gk, fps=10):
    """fix_half_v2 on every half of metrica_io.load() in place; returns the actions with video frames."""
    out = []
    for h in halves:
        h["X"], h["Y"], acts = fix_half_v2(h["X"], h["Y"], team, is_gk, fps, period=h["period"])
        for a in acts:
            a["period"] = h["period"]
            a["video_start"], a["video_end"] = float(h["video_pos"][a["start"]]), float(h["video_pos"][a["end"] - 1])
        out += acts
    return out

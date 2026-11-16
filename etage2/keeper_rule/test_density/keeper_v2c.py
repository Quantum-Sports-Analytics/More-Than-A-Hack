"""Keeper rule v2 with detected sides and a track-continuity fallback ("v2c"), 2026-11-04.

Same rule as pipeline/keeper_slots.fix_half_v2 (copied here, the original is not changed), with two differences:
  - sides come from sides.py (outfield centroid), not from the hard-coded rule;
  - fallback (step d, used only when the speed tests cannot decide) can be
      "goal"        the original: the slot nearer the owner's goal carries the keeper;
      "continuity"  the smoothest track: over each stretch, the keeper's track may follow either slot (and switch
                    between them); the route kept is the one with the smallest sum of squared steps, from the owner
                    slot's last position before the stretch to its first position after it. A real keeper moves in
                    many small steps (cheap); taking the ghost means one big jump somewhere (expensive).
                    An end is used only if the owner slot moved like a real keeper there (>= TRUST_V m/s over
                    TRUST_S s); with no trusted end, the "goal" fallback is used for that stretch.
fallback="goal" with the hard-coded sides reproduces fix_half_v2 exactly.
"""
import numpy as np
import pandas as pd

import keeper_slots as KS  # pipeline (constants and helpers only)

ANCHOR_S = 2.0        # s: how far before / after a stretch to look for the owner slot's position
TRUST_S = 3.0         # s: the owner slot is trusted at that end only if it moved like a real keeper over this window
TRUST_V = 0.3         # m/s: ... at least this fast on average (Metrica, 1H 05:45: slot 24 was already a ghost
                      #       crawling at 0.1 m/s before the stretch; anchoring on it kept the ghost)


def _anchor(x, y, idx):
    for i in idx:
        if np.isfinite(x[i]):
            return np.array([x[i], y[i]])
    return None


def _trusted(x, y, lo, hi, fps):
    lo, hi = max(lo, 0), min(hi, len(x))
    if hi - lo < 2:
        return False
    step = np.hypot(np.diff(x[lo:hi]), np.diff(y[lo:hi]))
    return bool(np.isfinite(step).sum() >= (hi - lo) // 2 and np.nanmean(step) * fps >= TRUST_V)


def continuity_path(Xo, Yo, Xq, Yq, a, b, fps):
    """Per frame of [a, b): True when the smoothest keeper track is on the owner's slot o (NaN where neither slot is
    present). Dynamic programming over two states (o, q); cost = sum of squared steps, plus the jumps from the
    anchor before a and to the anchor after b."""
    n = len(Xo)
    k, kt = int(round(ANCHOR_S * fps)), int(round(TRUST_S * fps))
    A = _anchor(Xo, Yo, range(a - 1, max(-1, a - 1 - k), -1)) if _trusted(Xo, Yo, a - kt, a, fps) else None
    B = _anchor(Xo, Yo, range(b, min(n, b + k))) if _trusted(Xo, Yo, b, b + kt, fps) else None
    if A is None and B is None:
        return None                                   # no trusted end: the caller uses the "nearer goal" fallback
    P = np.stack([np.column_stack([Xo[a:b], Yo[a:b]]), np.column_stack([Xq[a:b], Yq[a:b]])], 1)   # (m, 2, 2)
    ok = np.isfinite(P[:, :, 0])
    valid = np.flatnonzero(ok.any(1))
    out = np.full(b - a, np.nan)
    if not len(valid):
        return out
    INF = 1e18
    c = np.where(ok[valid[0]], 0.0, INF)
    if A is not None:
        c = c + np.where(ok[valid[0]], np.nan_to_num(((P[valid[0]] - A) ** 2).sum(1), nan=0), 0)
    back = np.zeros((len(valid), 2), np.int8)
    for i in range(1, len(valid)):
        p0, p1 = P[valid[i - 1]], P[valid[i]]
        d = ((p1[None, :, :] - p0[:, None, :]) ** 2).sum(2)          # from state s (rows) to state s' (cols)
        d = np.where(np.isfinite(d), d, INF)
        tot = c[:, None] + d
        back[i] = tot.argmin(0)
        c = np.where(ok[valid[i]], tot.min(0), INF)
    if B is not None:
        c = c + np.where(ok[valid[-1]], np.nan_to_num(((P[valid[-1]] - B) ** 2).sum(1), nan=0), 0)
    s = int(np.argmin(c))
    path = np.zeros(len(valid), np.int8)
    for i in range(len(valid) - 1, -1, -1):
        path[i] = s
        s = back[i, s]
    out[valid] = (path == 0).astype(float)
    return pd.Series(out).ffill().bfill().to_numpy()


def fix_half(X, Y, team, is_gk, side0, fps=10, fallback="continuity"):
    """side0: x sign of team 0's own goal (from sides.py). Returns fixed copies and the actions."""
    X, Y = X.copy(), Y.copy()
    gk = {t: int(np.flatnonzero((team == t) & is_gk)[0]) for t in (0, 1)}
    side = {0: float(side0), 1: -float(side0)}
    X0, Y0 = X.copy(), Y.copy()
    actions = []
    for t in (0, 1):
        o, q = gk[t], gk[1 - t]
        gx = side[t] * KS.GOAL_X
        q_in = ~np.isnan(X0[:, q]) & (side[t] * X0[:, q] > 0)
        o_in = ~np.isnan(X0[:, o]) & (side[t] * X0[:, o] > 0)
        vo, vq = KS._speed_series(X0[:, o], Y0[:, o], fps), KS._speed_series(X0[:, q], Y0[:, q], fps)
        do, dq = np.hypot(X0[:, o] - gx, Y0[:, o]), np.hypot(X0[:, q] - gx, Y0[:, q])
        with np.errstate(invalid="ignore"):
            o_fast = (vo >= KS.RATIO * np.maximum(vq, 0.01)) & (vo - vq >= KS.MIN_DIFF)
            q_fast = (vq >= KS.RATIO * np.maximum(vo, 0.01)) & (vq - vo >= KS.MIN_DIFF)
            q_frozen, o_frozen = vq <= KS.FROZEN, vo <= KS.FROZEN
        for a, b in KS._runs(q_in, fps):
            cont = continuity_path(X0[:, o], Y0[:, o], X0[:, q], Y0[:, q], a, b, fps) if fallback == "continuity" else None
            if cont is not None:
                tie = np.nan_to_num(cont, nan=0).astype(bool)
            else:
                with np.errstate(invalid="ignore"):
                    tie = do[a:b] < dq[a:b]
            sl = slice(a, b)
            with np.errstate(invalid="ignore"):
                keeper_o = o_in[sl] & ~o_frozen[sl] & (q_frozen[sl] | o_fast[sl] | (~q_fast[sl] & tie))
                keeper_o |= o_in[sl] & q_frozen[sl] & o_frozen[sl]
            k = pd.Series(keeper_o.astype(float)).rolling(int(round(KS.SMOOTH_V2 * fps)), center=True,
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


def fix_halves(halves, team, is_gk, fps=10, fallback="continuity"):
    """On metrica_io.load() halves, in place (same interface as keeper_slots.fix_halves); sides detected per half."""
    import sides as S
    out = []
    for h in halves:
        side0 = S.half_sides(h["X"], team, is_gk)["centroid"]["side0"]
        h["X"], h["Y"], acts = fix_half(h["X"], h["Y"], team, is_gk, side0, fps, fallback)
        for a in acts:
            a["period"] = h["period"]
            a["video_start"], a["video_end"] = float(h["video_pos"][a["start"]]), float(h["video_pos"][a["end"] - 1])
        out += acts
    return out

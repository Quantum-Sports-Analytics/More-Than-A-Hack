"""Which goal does each team defend? Side detection from where the players stand along the pitch's long axis.

Every method gives, for one half, side0 = x sign of team 0's own goal (-1 = left goal, +1 = right goal; team 1 is
the other one), plus a vote share (how clearly the frames agree: 1 or 0 = unanimous, 0.5 = a coin toss). Keeper
slots are the weak point (a keeper filed under the other team), so the main methods use outfield slots only.

Methods (per frame a vote "team 0 defends the left goal", then the majority over the half or a window):
  centroid          team 0's mean x of outfield slots < team 1's mean x
  deepest_outfield  team of the outfield player nearest each goal line (two half-votes per frame: left end, right end)
  deepest_any       the same with every slot, keeper slots included (usually the keepers themselves)
  density_mass      share of team 0's outfield player-frames with x < 0 minus team 1's
  keeper_median     median x of each team's keeper slot over the half (what keeper_slots.py falls back to); one
                    decision per half, no per-frame vote
"""
import numpy as np

METHODS = ("centroid", "deepest_outfield", "deepest_any", "density_mass", "keeper_median")
MIN_OUTFIELD = 5          # a frame votes only with at least this many outfield slots of each team present


def frame_votes(X, team, is_gk):
    """Per-frame votes in [0, 1] (1 = team 0 on the left, NaN = no vote) for the methods that vote per frame."""
    present = ~np.isnan(X)
    out_t = [(team == t) & ~is_gk for t in (0, 1)]
    cnt = [present[:, m].sum(1) for m in out_t]
    ok = (cnt[0] >= MIN_OUTFIELD) & (cnt[1] >= MIN_OUTFIELD)
    with np.errstate(all="ignore"):
        c0, c1 = np.nanmean(X[:, out_t[0]], 1), np.nanmean(X[:, out_t[1]], 1)
        v = {"centroid": np.where(ok, (c0 < c1).astype(float), np.nan)}
        for name, cols in (("deepest_outfield", ~is_gk), ("deepest_any", np.ones(len(team), bool))):
            Xc, tc = X[:, cols], team[cols]
            lo = np.argmin(np.where(np.isnan(Xc), np.inf, Xc), 1)
            hi = np.argmax(np.where(np.isnan(Xc), -np.inf, Xc), 1)
            vote = ((tc[lo] == 0).astype(float) + (tc[hi] == 1).astype(float)) / 2
            v[name] = np.where(ok, vote, np.nan)
        l0 = np.nansum(X[:, out_t[0]] < 0, 1) / cnt[0]
        l1 = np.nansum(X[:, out_t[1]] < 0, 1) / cnt[1]
        v["density_mass"] = np.where(ok, 0.5 + (l0 - l1) / 2, np.nan)
    return v


def half_sides(X, team, is_gk, votes=None):
    """{method: {"side0": -1 | +1, "share": mean vote for "team 0 on the left"}} for one half."""
    votes = votes if votes is not None else frame_votes(X, team, is_gk)
    res = {}
    for k, v in votes.items():
        s = float(np.nanmean(v))
        res[k] = {"side0": -1 if s > 0.5 else 1, "share": round(s, 4)}
    gk0, gk1 = (int(np.flatnonzero((team == t) & is_gk)[0]) for t in (0, 1))
    with np.errstate(all="ignore"):
        m0, m1 = float(np.nanmedian(X[:, gk0])), float(np.nanmedian(X[:, gk1]))
    res["keeper_median"] = {"side0": -1 if m0 < m1 else 1, "share": float(m0 < m1), "medians": [round(m0, 1), round(m1, 1)]}
    return res


def window_accuracy(votes, truth_side0, fps, minutes=(1, 2, 5, 10)):
    """{method: {"<w>min": [windows giving the true side, windows with a vote]}}, non-overlapping windows."""
    want = truth_side0 == -1
    out = {}
    for k, v in votes.items():
        out[k] = {}
        for w in minutes:
            L = int(w * 60 * fps)
            nwin = len(v) // L
            if nwin == 0:
                continue
            with np.errstate(all="ignore"):
                m = np.nanmean(v[:nwin * L].reshape(nwin, L), 1)
            m = m[np.isfinite(m)]
            out[k][f"{w}min"] = [int(((m > 0.5) == want).sum()), int(len(m))]
    return out


def first_correct_minute(votes, truth_side0, fps, step_s=10):
    """Cumulative majority from the start of the half: the earliest time (s) after which it stays right."""
    want = truth_side0 == -1
    out = {}
    for k, v in votes.items():
        ok = np.isfinite(v)
        cs, cn = np.cumsum(np.where(ok, v, 0)), np.cumsum(ok)
        idx = np.arange(step_s * fps - 1, len(v), step_s * fps)
        with np.errstate(all="ignore"):
            maj = cs[idx] / cn[idx]
        right = (maj > 0.5) == want
        right &= np.isfinite(maj)
        wrong = np.flatnonzero(~right)
        out[k] = 0.0 if not len(wrong) else (float((wrong[-1] + 1) * step_s) if wrong[-1] + 1 < len(idx) else None)
    return out


def density(X, lo=-56, hi=56):
    """Per-slot share of frames in each 1 m bin along x (rows of NaN for slots never present)."""
    bins = np.arange(lo, hi + 1)
    H = np.full((X.shape[1], len(bins) - 1), np.nan)
    for j in range(X.shape[1]):
        x = X[:, j]
        x = x[np.isfinite(x)]
        if len(x):
            H[j] = np.histogram(np.clip(x, lo, hi - 1e-6), bins)[0] / len(x)
    return H, bins


def period_for(side0):
    """keeper_slots.fix_half_v2 takes sides only through `period` (1: team 0 defends the left goal, 2: the right
    one). The detected side is passed that way, so the pipeline's code is used unchanged."""
    return 1 if side0 == -1 else 2

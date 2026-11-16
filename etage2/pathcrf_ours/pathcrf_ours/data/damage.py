"""Damage a clean processed match so that its player tracks look like Barça–Athletic's (Phase B).

What Metrica's slot data does, measured on Barça–Athletic (pathcrf_test/outputs/slot_stats.json and the
adapter report pathcrf_test/outputs/metrica_adapter.json, both after the keeper rule and the slot->track
adapter with straight-line filling):
- a player track has holes: ~525 holes per team per match, median 4.7 s, p90 27.7 s, 8 % of player-frames
  filled (keepers 8-18 %); players vanish far from the ball (median 26 m from the holder);
- when a slot comes back the adapter may hand it to the wrong player (nearest free track), so from then
  on two team-mates' identities are exchanged;
- keeper errors: a ghost dot in a keeper slot, or the real keeper in the other team's keeper slot (the
  keeper rule, keeper_rule/step0.py, fixes ~97 % of those; the rest reach the model);
- holes are filled with straight lines (nearest value at the edges), then the kinematics are recomputed
  on the filled track, so a filled player glides at constant speed with zero acceleration.

This module injects exactly those faults, in that order, on the 25 fps processed table:
    holes -> keeper errors -> keeper rule (step 0) -> identity swaps at hole ends -> straight-line fill
    -> `<player>_present` flag (False on filled frames) -> kinematics recomputed.
Labels (player_id) follow the body: after an identity swap the label carries the track's name (the team never changes). The ball columns are
not touched either (the ball is damaged separately, see data/ball.py).

`dial` scales the amount of damage (0 = clean, 1 = Metrica-like, 2 = twice as bad).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import PITCH_X, PITCH_Y, DataConfig
from .features import KINEMATICS, add_kinematics
from .sportec import detect_keepers, players_of


@dataclass
class DamageConfig:
    dial: float = 1.0
    hole_rate_per_min: float = 0.53     # holes per outfield player per minute on the pitch (525 / 11 / 90)
    gk_hole_factor: float = 1.5         # keepers vanish more often (filled 8-18 % vs 8 %)
    hole_median_s: float = 4.7
    hole_sigma: float = 1.385           # log-normal sigma: p90 = 27.7 s
    hole_min_s: float = 0.3
    hole_max_s: float = 60.0
    far_from_ball_bias: bool = True     # holes start preferably far from the ball holder (median 26 m)
    swap_prob: float = 0.25             # at the end of a hole: identity exchanged with a team-mate nearby
    swap_max_dist: float = 12.0
    keeper_error_prob: float = 0.5      # per team per half: one Metrica-like keeper error (move or ghost)
    keeper_error_median_s: float = 20.0
    keeper_error_sigma: float = 0.6
    keeper_creep_speed: float = 0.05    # m/s of a ghost dot
    keeper_rule: bool = True            # run keeper_rule/step0 after the damage, as on Metrica
    seed: int = 0


# ------------------------------------------------------------------------------------------ helpers
def _lognormal_s(rng: np.random.Generator, median: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(np.exp(np.log(median) + sigma * rng.standard_normal()), lo, hi))


def _creep(x0: float, y0: float, n: int, speed: float, fps: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """A nearly frozen dot: slow drift in a random direction plus a little shake."""
    ang = rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / fps
    x = x0 + speed * t * np.cos(ang) + rng.normal(0, 0.02, n)
    y = y0 + speed * t * np.sin(ang) + rng.normal(0, 0.02, n)
    return x, y


def _step0():
    root = Path(__file__).resolve().parents[3]
    kr = root / "keeper_rule"
    if str(kr) not in sys.path:
        sys.path.insert(0, str(kr))
    import step0  # noqa: E402

    return step0


# ------------------------------------------------------------------------------------------ main
def damage_match(proc: pd.DataFrame, dcfg: DamageConfig, data_cfg: DataConfig, rng: Optional[np.random.Generator] = None) -> Tuple[pd.DataFrame, Dict]:
    """Returns (damaged processed table with `<player>_present` columns, a dict of damage statistics)."""
    rng = rng or np.random.default_rng(dcfg.seed)
    fps = data_cfg.source_fps
    home, away = players_of(proc)
    players = home + away
    P = len(players)
    team = np.array([0 if p.startswith("home") else 1 for p in players])
    out = proc.copy()
    out = out.sort_values(["period_id", "frame_id"], ignore_index=True)
    X = out[[f"{p}_x" for p in players]].to_numpy(dtype=float)
    Y = out[[f"{p}_y" for p in players]].to_numpy(dtype=float)
    present = np.isfinite(X)                    # truth: on the pitch and not damaged
    on_pitch = present.copy()
    Xd, Yd = X.copy(), Y.copy()
    stats = {"holes": 0, "hole_frames": 0, "swaps": 0, "keeper_errors": 0, "step0_actions": 0, "dropped_frames": 0}
    dial = dcfg.dial
    if dial <= 0:
        out = _attach_present(out, players, present)
        return out, stats

    labels = out["player_id"].astype(object).to_numpy()
    name_to_col = {p: i for i, p in enumerate(players)}
    holder = np.array([name_to_col.get(l, -1) if isinstance(l, str) else -1 for l in labels])

    for period in out["period_id"].unique():
        rows = np.flatnonzero((out["period_id"] == period).to_numpy())
        a0, a1 = rows[0], rows[-1] + 1
        n = a1 - a0
        Xp, Yp = Xd[a0:a1], Yd[a0:a1]
        onp = on_pitch[a0:a1]
        hol = holder[a0:a1]
        hx = np.where(hol >= 0, X[a0:a1][np.arange(n), np.clip(hol, 0, P - 1)], np.nan)
        hy = np.where(hol >= 0, Y[a0:a1][np.arange(n), np.clip(hol, 0, P - 1)], np.nan)

        # keepers of this half (by mean distance from the centre line, per team)
        gk_idx = _keepers(Xp, team)
        is_gk = np.zeros(P, bool)
        is_gk[list(gk_idx.values())] = True

        # 1. holes ------------------------------------------------------------------------------
        hole_ends: List[Tuple[int, int]] = []
        for j in range(P):
            onj = np.flatnonzero(onp[:, j])
            if len(onj) < fps:
                continue
            first, last = onj[0], onj[-1]
            minutes = len(onj) / fps / 60
            rate = dcfg.hole_rate_per_min * dial * (dcfg.gk_hole_factor if is_gk[j] else 1.0)
            k = rng.poisson(rate * minutes)
            for _ in range(k):
                L = int(round(_lognormal_s(rng, dcfg.hole_median_s, dcfg.hole_sigma, dcfg.hole_min_s, dcfg.hole_max_s) * fps))
                if last - first - L <= 0:
                    continue
                s = None
                for _try in range(8):
                    c = int(rng.integers(first, last - L))
                    if not dcfg.far_from_ball_bias:
                        s = c
                        break
                    d = np.hypot(Xp[c, j] - hx[c], Yp[c, j] - hy[c])
                    pacc = 0.5 if not np.isfinite(d) else float(np.clip(d / 30.0, 0.1, 1.0))
                    if rng.random() < pacc:
                        s = c
                        break
                if s is None:
                    s = c
                e = min(s + L, last + 1)
                Xp[s:e, j] = np.nan
                Yp[s:e, j] = np.nan
                stats["holes"] += 1
                stats["hole_frames"] += int(e - s)
                if e <= last:
                    hole_ends.append((e, j))

        # 2. keeper errors (Metrica-like), per team ---------------------------------------------
        for t in (0, 1):
            if rng.random() >= dcfg.keeper_error_prob * min(dial, 1.0) * (dial if dial > 1 else 1.0):
                continue
            o, q = gk_idx[t], gk_idx[1 - t]
            L = int(round(_lognormal_s(rng, dcfg.keeper_error_median_s, dcfg.keeper_error_sigma, 3.0, 120.0) * fps))
            both = np.flatnonzero(np.isfinite(Xp[:, o]) & np.isfinite(Xp[:, q]))
            if len(both) <= L + 1:
                continue
            s = int(rng.choice(both[: len(both) - L]))
            e = s + L
            kind = "move" if rng.random() < 0.5 else "ghost"
            if kind == "move":                       # the real keeper shows up in the other team's keeper slot
                Xp[s:e, q], Yp[s:e, q] = X[a0 + s : a0 + e, o], Y[a0 + s : a0 + e, o]
                Xp[s:e, o], Yp[s:e, o] = _creep(X[a0 + s, o], Y[a0 + s, o], e - s, dcfg.keeper_creep_speed, fps, rng)
            else:                                    # a ghost dot 10 m in front of t's goal in the intruder slot
                gx = 10.0 if t == 0 else PITCH_X - 10.0
                Xp[s:e, q], Yp[s:e, q] = _creep(gx, PITCH_Y / 2 + 3.0, e - s, dcfg.keeper_creep_speed, fps, rng)
            stats["keeper_errors"] += 1

        # 3. keeper rule (step 0) on centred coordinates -----------------------------------------
        if dcfg.keeper_rule:
            st = _step0()
            Xc, Yc = Xp - PITCH_X / 2, Yp - PITCH_Y / 2
            Xf, Yf, acts, _side0 = st.apply_half(Xc, Yc, team, is_gk, fps=fps)
            Xp[:], Yp[:] = Xf + PITCH_X / 2, Yf + PITCH_Y / 2
            stats["step0_actions"] += len(acts)

        # 4. identity swaps at hole ends -----------------------------------------------------------
        for e, j in hole_ends:
            if rng.random() >= dcfg.swap_prob * min(dial, 1.5):
                continue
            if not np.isfinite(Xp[e, j]):
                continue
            cands = [k for k in range(P) if k != j and team[k] == team[j] and not is_gk[k] and not is_gk[j]
                     and np.isfinite(Xp[e, k]) and np.array_equal(onp[e:, k], onp[e:, j])]  # same time left on the pitch
            if not cands:
                continue
            d = np.array([np.hypot(Xp[e, k] - Xp[e, j], Yp[e, k] - Yp[e, j]) for k in cands])
            near = [k for k, dk in zip(cands, d) if dk <= dcfg.swap_max_dist]
            if not near:
                continue
            k = int(rng.choice(near))
            Xp[e:, [j, k]] = Xp[e:, [k, j]]
            Yp[e:, [j, k]] = Yp[e:, [k, j]]
            onp[e:, [j, k]] = onp[e:, [k, j]]
            # the label follows the body: from here on the track named j shows k's body and vice versa, so a ball
            # held by that body is labelled with the track's name (what the adapter does on Metrica; the team is
            # unchanged, and the player-level target stays consistent with what the model sees)
            seg = hol[e:]
            jj, kk = seg == j, seg == k
            seg[jj], seg[kk] = k, j
            stats["swaps"] += 1

        # 5. straight-line fill inside each player's span; present = still a real observation -------
        for j in range(P):
            onj = np.flatnonzero(onp[:, j])
            if not len(onj):
                continue
            first, last = onj[0], onj[-1]
            seg = slice(first, last + 1)
            real = np.isfinite(Xp[seg, j])
            present[a0 + first : a0 + last + 1, j] = real
            present[a0 : a0 + first, j] = False
            present[a0 + last + 1 : a1, j] = False
            if not real.all():
                sx = pd.Series(Xp[seg, j]).interpolate(limit_area="inside").ffill().bfill()
                sy = pd.Series(Yp[seg, j]).interpolate(limit_area="inside").ffill().bfill()
                Xp[seg, j], Yp[seg, j] = sx.to_numpy(), sy.to_numpy()
            Xp[:first, j] = np.nan
            Xp[last + 1 :, j] = np.nan
            Yp[:first, j] = np.nan
            Yp[last + 1 :, j] = np.nan
        stats["dropped_frames"] += int((~present[a0:a1] & on_pitch[a0:a1]).sum())

    out["player_id"] = np.array([players[h] if h >= 0 else labels[i] for i, h in enumerate(holder)], dtype=object)
    for i, p in enumerate(players):
        out[f"{p}_x"] = np.clip(Xd[:, i], -2.0, PITCH_X + 2.0)
        out[f"{p}_y"] = np.clip(Yd[:, i], -2.0, PITCH_Y + 2.0)
    out = add_kinematics(out, data_cfg, objects=players)
    out = _attach_present(out, players, present)
    stats["filled_share"] = float(stats["dropped_frames"] / max(on_pitch.sum(), 1))
    return out, stats


def _keepers(Xp: np.ndarray, team: np.ndarray) -> Dict[int, int]:
    gk = {}
    with np.errstate(all="ignore"):
        mx = np.nanmean(Xp, axis=0)
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        far = np.abs(np.nan_to_num(mx[cols], nan=PITCH_X / 2) - PITCH_X / 2)
        gk[t] = int(cols[np.argmax(far)])
    return gk


def _attach_present(out: pd.DataFrame, players: List[str], present: np.ndarray) -> pd.DataFrame:
    cols = {f"{p}_present": present[:, i] for i, p in enumerate(players)}
    out = out.drop(columns=[c for c in cols if c in out.columns])
    return pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)


def present_columns(tracking: pd.DataFrame) -> List[str]:
    return [c for c in tracking.columns if c.endswith("_present")]

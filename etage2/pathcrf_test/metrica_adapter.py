"""Metrica adapter for PathCRF: turn Barça–Athletic's 46 track slots into 11 steady "virtual players" per team.

STATUS 2026-11-06: judged inaccurate by hand on viz/adapter_viewer.html; NOT usable for detection as it is. The slot ->
player rules must be redesigned from viz/slot_gaps_viewer.html before anything downstream is trusted. Step 0
(camps + keeper rule) is fine and always applies.

Why: PathCRF's code needs every player of a phase present on every frame of a 10 s window, with the same identity.
Metrica's tactical tracking has 46 slots (24 Barça, 22 Athletic). A slot appears, vanishes and can come back as a
different person; only 23% of 10 s windows have 11 complete slots per team.

What it does, per half and per team, at 10 fps (after step 0: camps + keeper rule v2, as every chain):
  1. the keeper slot is the keeper track;
  2. the outfield slots feed 10 outfield tracks, frame by frame:
     - a slot that fed a track on the previous frame keeps feeding it while it is present (identity kept);
     - slots that have no track yet are matched to the free tracks (those whose slot vanished, or never used) by
       the Hungarian method (the best one-to-one pairing) on the distance to each track's last known position;
     - when more than 10 outfield slots are present (duplicates), the extra slots wait until a track is free;
  3. holes in a track (no slot for a while) are filled with a straight line between the last and next known
     positions, or the nearest known position at the start / end of a half.
Coordinates move to PathCRF's pitch: 0..105 x 0..68 m, origin in a corner, y up (the model saw random flips).

Run: airball/.venv/bin/python pathcrf_test/metrica_adapter.py [--fill line|shift|robust] [--gk-gain G] [--short F]
  --fill line (default): straight lines, outputs as below;
  --fill shift / robust: team-shape filling (see fill_team), outputs get the suffix _shape (or --suffix).
  -> pathcrf_test/outputs/metrica_virtual.parquet (one row per 10 fps row of the chain's table: row, period,
     video_frame, clock_s, home_1..home_11_{x,y}, away_1..away_11_{x,y}, filled flags, blue_team)
  -> pathcrf_test/outputs/metrica_adapter.json (how much was filled)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "airball"))
sys.path.insert(0, str(ROOT / "keeper_rule"))
import metrica_io as M  # noqa: E402
import step0  # noqa: E402

OUT = ROOT / "pathcrf_test" / "outputs"
N_OUT = 10
PX, PY = 105.0, 68.0


def link_tracks(X, Y, slots):
    """X, Y: (n, 46). slots: outfield slot indices of one team. Returns pos (n, N_OUT, 2) with NaN where no slot
    feeds the track, src (n, N_OUT) slot index or -1."""
    n = len(X)
    pres = ~np.isnan(X[:, slots])
    src = np.full((n, N_OUT), -1)
    last = np.full((N_OUT, 2), np.nan)
    cur = np.full(N_OUT, -1)
    for t in range(n):
        here = set(np.asarray(slots)[pres[t]])
        for k in range(N_OUT):
            if cur[k] >= 0 and cur[k] not in here:
                cur[k] = -1
        free = [k for k in range(N_OUT) if cur[k] < 0]
        cand = [s for s in here if s not in set(cur)]
        if free and cand:
            cost = np.zeros((len(cand), len(free)))
            for i, s in enumerate(cand):
                for j, k in enumerate(free):
                    cost[i, j] = 1e3 if np.isnan(last[k, 0]) else np.hypot(X[t, s] - last[k, 0], Y[t, s] - last[k, 1])
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                cur[free[j]] = cand[i]
        for k in range(N_OUT):
            if cur[k] >= 0:
                src[t, k] = cur[k]
                last[k] = X[t, cur[k]], Y[t, cur[k]]
    pos = np.full((n, N_OUT, 2), np.nan)
    ok = src >= 0
    rows = np.nonzero(ok)
    pos[rows[0], rows[1], 0] = X[rows[0], src[ok]]
    pos[rows[0], rows[1], 1] = Y[rows[0], src[ok]]
    return pos, src


def fill_track(v):
    """Straight line inside a track's holes, nearest value at the edges. Returns filled array and gap lengths."""
    s = pd.Series(v)
    nan = s.isna().to_numpy()
    out = s.interpolate(limit_area="inside").ffill().bfill().to_numpy()
    d = np.diff(np.r_[0, nan.astype(int), 0])
    gaps = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return out, gaps


def holes_of(nan):
    """Start (inclusive) and end (exclusive) of each run of True."""
    d = np.diff(np.r_[0, nan.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ---- team-shape filling -------------------------------------------------------------------------------------------
# Idea (Shaw & Glickman 2019; Bialkowski et al. 2014): a block keeps its spacing while it moves. So a missing player
# is moved with the team instead of along a straight line. The team's movement is measured only on real players.

def team_shift(pos, min_n=3, max_step=1.5):
    """Cumulative team displacement c(t) (n, 2) from the outfield tracks pos (n, 10, 2) with NaN where missing.
    Each step is the mean displacement of the players real on both frames, so a missing forward does not drag the
    centre backwards (no composition bias). With fewer than min_n such players the team is held still.
    Steps over max_step m per frame (15 m/s) are teleports from a slot handover and are ignored."""
    d = np.diff(pos, axis=0)
    ok = ~np.isnan(d[..., 0]) & (np.hypot(d[..., 0], d[..., 1]) < max_step)
    cnt = ok.sum(1)
    v = np.where(ok[..., None], d, 0.0).sum(1) / np.maximum(cnt, 1)[:, None]
    v[cnt < min_n] = 0.0
    return np.vstack([np.zeros((1, 2)), np.cumsum(v, 0)])


def team_centre_robust(pos, win=600, min_n=3):
    """Composition-corrected centre c(t) = mean over real players of (p_j - o_j), o_j = rolling mean (win frames)
    of the player's offset from the plain centroid. Frames with fewer than min_n real players copy the neighbour."""
    real = ~np.isnan(pos[..., 0])
    cnt = real.sum(1)
    cen = np.nanmean(np.where(real[..., None], pos, np.nan), axis=1)
    off = pos - cen[:, None, :]
    o = np.stack([pd.DataFrame(off[..., a]).rolling(win, center=True, min_periods=50).mean().to_numpy()
                  for a in (0, 1)], -1)
    o = np.where(np.isnan(o), np.nanmean(off, axis=0, keepdims=True), o)
    c = np.nanmean(np.where(real[..., None], pos - o, np.nan), axis=1)
    c[cnt < min_n] = np.nan
    return pd.DataFrame(c).interpolate(limit_area="inside").ffill().bfill().to_numpy()


def shape_fill(p, c, short=0, gain=1.0):
    """Fill the holes of one track p (n, 2) using the team's position c (n, 2).
    Inside a hole from t1 to t2 the player keeps the team's movement, and the offset to the team goes linearly from
    its value before the hole to its value after it (equivalent to: straight line + team displacement minus its
    linear trend over the hole). At a half's start / end the offset is held. Holes shorter than `short` frames get
    the straight line. gain scales the team movement (1 = move with the team, 0 = straight line)."""
    n = len(p)
    out = np.column_stack([fill_track(p[:, 0])[0], fill_track(p[:, 1])[0]])
    nan = np.isnan(p[:, 0])
    if not nan.any() or nan.all():
        return out
    for s, e in zip(*holes_of(nan)):
        if e - s < short:
            continue
        t = np.arange(s, e)
        if s > 0 and e < n:
            lam = (t - (s - 1)) / (e - (s - 1))
            drift = (c[t] - c[s - 1]) - lam[:, None] * (c[e] - c[s - 1])
        elif s > 0:
            drift = c[t] - c[s - 1]
        else:
            drift = c[t] - c[e]
        out[t] += gain * drift
    return out


def fill_team(pos, gk, method="line", gk_gain=0.0, short=0):
    """Fill the 10 outfield tracks (n, 10, 2) and the keeper (n, 2). Returns the 11 filled tracks as a list.
    method: 'line' (straight), 'shift' (cumulative team displacement), 'robust' (composition-corrected centroid)."""
    tracks = [pos[:, k] for k in range(pos.shape[1])] + [gk]
    if method == "line":
        return [np.column_stack([fill_track(p[:, 0])[0], fill_track(p[:, 1])[0]]) for p in tracks]
    c = team_shift(pos) if method == "shift" else team_centre_robust(pos)
    out = [shape_fill(p, c, short=short) for p in tracks[:-1]]
    out.append(shape_fill(gk, c, short=short, gain=gk_gain))
    lim = np.array([M.LENGTH / 2, M.WIDTH / 2])
    return [np.clip(o, -lim, lim) for o in out]


def parse():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", default="line", choices=["line", "shift", "robust"],
                    help="hole filling: line (straight, default) or team shape (shift / robust centre)")
    ap.add_argument("--gk-gain", type=float, default=0.0, help="share of the team movement given to the keeper")
    ap.add_argument("--short", type=int, default=0, help="holes shorter than this many frames keep the straight line")
    ap.add_argument("--suffix", default=None, help="output name suffix (default: '' for line, '_shape' otherwise)")
    return ap.parse_args()


def main():
    a = parse()
    suffix = a.suffix if a.suffix is not None else ("" if a.fill == "line" else "_shape")
    halves, team, is_gk, _ = M.load()
    acts = step0.apply_halves(halves, team, is_gk, M.FPS)
    team, is_gk = np.asarray(team), np.asarray(is_gk, bool)
    parts, stats = [], {"step0_actions": len(acts), "fill": vars(a), "teams": {}}
    gaps_all = {0: [], 1: []}
    swaps = {0: 0, 1: 0}
    jumps = {0: 0, 1: 0}
    for h in halves:
        n = len(h["video_pos"])
        lo = M.HALVES[h["period"]][0]
        cols = {"period": np.full(n, h["period"]), "video_frame": h["video_pos"],
                "clock_s": (h["video_pos"] - lo) / M.VIDEO_FPS,
                # team 0 (Barça, home) defends the left goal when side0 == -1: then BLUE (left) = home
                "blue_team": np.full(n, 0 if h["side0"] == -1 else 1)}
        bx, by = h["bx"], h["by"]
        cols["ball_x"] = (bx + M.LENGTH / 2) * PX / M.LENGTH
        cols["ball_y"] = (M.WIDTH / 2 - by) * PY / M.WIDTH
        # rows where the ball is not observed after step 0 (its position is gap-filled): the ball model must not trust it
        cols["ball_missing"] = ~np.asarray(h["observed"], bool)
        for t, name in ((0, "home"), (1, "away")):
            gk = np.flatnonzero((team == t) & is_gk)
            out = np.flatnonzero((team == t) & ~is_gk)
            assert len(gk) == 1, gk
            pos, src = link_tracks(h["X"], h["Y"], out)
            swaps[t] += int(((src[1:] != src[:-1]) & (src[1:] >= 0)).sum())   # a track starts taking a new slot
            gkp = np.column_stack([h["X"][:, gk[0]], h["Y"][:, gk[0]]])
            tracks = [pos[:, k] for k in range(N_OUT)] + [gkp]
            filled = fill_team(pos, gkp, a.fill, gk_gain=a.gk_gain, short=a.short)
            for k, (p, f) in enumerate(zip(tracks, filled), start=1):
                fx, fy = f[:, 0], f[:, 1]
                s0, e0 = holes_of(np.isnan(p[:, 0]))
                gaps_all[t] += (e0 - s0).tolist()
                step = np.hypot(np.diff(fx), np.diff(fy))
                jumps[t] += int((step > 3.0).sum())                 # > 3 m in 0.1 s = 30 m/s: a teleport
                cols[f"{name}_{k}_x"] = (fx + M.LENGTH / 2) * PX / M.LENGTH
                cols[f"{name}_{k}_y"] = (M.WIDTH / 2 - fy) * PY / M.WIDTH
                cols[f"{name}_{k}_filled"] = np.isnan(p[:, 0])
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row", np.arange(len(df)))
    df.to_parquet(OUT / f"metrica_virtual{suffix}.parquet", index=False)
    for t, name in ((0, "home"), (1, "away")):
        f = df[[c for c in df.columns if c.startswith(f"{name}_") and c.endswith("_filled")]].to_numpy()
        g = np.array(gaps_all[t]) / M.FPS
        stats["teams"][name] = {
            "filled_share_of_player_frames": round(float(f.mean()), 4),
            "filled_share_keeper": round(float(f[:, -1].mean()), 4),
            "frames_all_11_real": round(float((~f).all(1).mean()), 4),
            "real_players_per_frame_mean": round(float((~f).sum(1).mean()), 2),
            "holes": int(len(g)), "hole_s_median": round(float(np.median(g)), 1) if len(g) else 0,
            "hole_s_p90": round(float(np.percentile(g, 90)), 1) if len(g) else 0,
            "share_of_filled_time_in_holes_over_10s": round(float(g[g > 10].sum() / max(g.sum(), 1e-9)), 3),
            "slot_handovers": swaps[t], "jumps_over_3m_per_0.1s": jumps[t]}
    stats["rows"] = len(df)
    json.dump(stats, open(OUT / f"metrica_adapter{suffix}.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

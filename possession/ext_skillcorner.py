"""External dead-ball data: SkillCorner Open Data (A-League 2024/25) -> Sportec-compatible npz at 25 fps.

Source: https://github.com/SkillCorner/opendata (branch master), licence MIT (file LICENSE,
"Copyright (c) 2020 SkillCorner"). Per match: <id>_match.json, <id>_dynamic_events.csv,
<id>_phases_of_play.csv, <id>_tracking_extrapolated.jsonl (Git LFS; real content served by
media.githubusercontent.com). Broadcast tracking at 10 fps, origin at the pitch centre, metres.
Nothing here touches the provided Barca-Athletic match.

Output npz (one per match, same keys as fetch_sportec.py so dead_bounds.load_sportec loads it):
  L, W; for h in (h1, h2): h_PX, h_PY (T x n float32, m, centre origin, NaN when the player is
  absent from player_data; SkillCorner EXTRAPOLATED off-screen players are kept), h_team (int8 0
  home / 1 away), h_BX, h_BY, h_BZ (float32, ball only where is_detected=True, NaN otherwise, no
  gap filling), h_BS (int8 1 live / 0 dead / -1 unknown), h_BP (int8 1 home / 2 away / -1 unknown,
  from tracking `possession.group`), h_n0 (25 fps frame index of row 0, = round(2.5 * SkillCorner
  frame)). Extra: h_ball_obs (bool), h_stop (int8, dead_v4 codes), h_label_conf (uint8 0-2),
  h_PD (bool, player is_detected), h_pid (int32), h_gk (bool, starting goalkeeper role),
  h_BXe, h_BYe (ball incl. SkillCorner extrapolation), h_cam (T x 8, image_corners_projection
  x_tl, y_tl, x_bl, y_bl, x_br, y_br, x_tr, y_tr), h_has_track (bool, frame has player_data),
  source (str), stoppages_json (str).
Resampling 10 -> 25 fps: linear interpolation between the two nearest valid samples if they are
at most 0.2 s apart (bridges one missing 10 fps frame), NaN otherwise; labels / flags = nearest
10 fps frame.

Dead-ball labels (see LABELS below and the report):
  live  = inside a SkillCorner phase of play (the spec: phases are "only defined when the ball
          is in play"); dead = gap between two consecutive phases of the same period.
  start = phase end a (SkillCorner convention: last on-ball event end + 1.0 s) except for
          out-of-play restarts (throw-in, corner, goal kick, goal): first frame after the last
          on-ball event where the ball (detected, else SkillCorner-extrapolated) is beyond a line.
  end   = start of the next phase b (= the restart kick / throw; checked against ball impulse).
  type  = game_interruption_before of the first event starting in [b - 0.5 s, b + 6 s], else
          game_interruption_after of the last event ending in [a - 4 s, a + 0.5 s], else inconnu.

CLI
  python possession/ext_skillcorner.py run --work <dir> [--matches 1886347 ...] [--keep-raw]
  python possession/ext_skillcorner.py check --work <dir>      # controls + baseline only
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS, SC_FPS = 25, 10
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/"
MATCH_IDS = ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
             "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
             "2013725", "2015213", "2016236", "2017461"]
STOP_NAMES = {0: "vivant", 1: "touche", 2: "corner", 3: "six_metres", 4: "coup_franc", 5: "engagement",
              6: "penalty", 7: "balle_arbitre", 8: "inconnu"}
INTERRUPT = {"throw_in": 1, "corner": 2, "goal_kick": 3, "free_kick": 4, "goal": 5, "penalty": 6}
OUT_TYPES = (1, 2, 3, 5)           # restarts that follow the ball crossing a line
MAX_GAP_S = 0.2                    # interpolation bridge
ANOM_FB, ANOM_DUR_S = 1, 3.0       # restart event <= 0.1 s after the phase start and gap < 3 s
ANOM_MATCH_FRAC = 0.10             # > 10 % anomalous gaps -> match flagged label_quality='douteuse'
NO_EVENT_MIN_S = 5.0               # phase gaps shorter than this with no interruption event -> BS -1
MAX_ROLL_S = 8.0                   # max last touch -> line crossing search window
# label confidence windows (s), fixed a priori and checked by the controls
CONF_START_REFINED, CONF_START_PHASE, CONF_END = 0.6, 1.2, 0.4
SPORTEC_REF = {  # 7 IDSSE matches, sportec_arrets.csv (dead_v4 events), per match / median s
    "touche": (43.57, 13.48), "coup_franc": (25.29, 30.68), "six_metres": (17.0, 22.08),
    "corner": (8.14, 32.16), "engagement": (2.57, 70.30), "penalty": (0.43, 119.0),
    "balle_arbitre": (2.14, 55.88), "inconnu": (0.29, 78.08), "dead_frac": 0.4451}


def runs_of(mask):
    e = np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


# ----------------------------------------------------------------------------------- download
def _curl(url, dst):
    if os.path.exists(dst):
        return dst
    r = subprocess.run(["curl", "-sL", "--fail", "-o", dst + ".part", url])
    if r.returncode:
        raise RuntimeError(f"download failed {url}")
    os.replace(dst + ".part", dst)
    return dst


def download(mid, raw_dir):
    d = os.path.join(raw_dir, mid)
    os.makedirs(d, exist_ok=True)
    for suf in ("match.json", "dynamic_events.csv", "phases_of_play.csv"):
        _curl(f"{RAW}data/matches/{mid}/{mid}_{suf}", os.path.join(d, f"{mid}_{suf}"))
    tj = os.path.join(d, f"{mid}_tracking_extrapolated.jsonl")
    _curl(f"{LFS}data/matches/{mid}/{mid}_tracking_extrapolated.jsonl", tj)
    if os.path.getsize(tj) < 10_000:
        raise RuntimeError(f"{tj}: Git LFS pointer, not data")
    return d


# ----------------------------------------------------------------------------------- parsing
CAM = ("x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left", "x_bottom_right", "y_bottom_right",
       "x_top_right", "y_top_right")


def parse_tracking(path, n_frames, pid_col):
    """Stream the jsonl into 10 fps arrays indexed by SkillCorner frame."""
    n = len(pid_col)
    PX = np.full((n_frames, n), np.nan, np.float32); PY = PX.copy()
    PD = np.zeros((n_frames, n), bool)
    B = np.full((n_frames, 3), np.nan, np.float32)
    bdet = np.zeros(n_frames, np.int8)            # 0 none, 1 detected, 2 extrapolated
    grp = np.full(n_frames, -1, np.int8)
    cam = np.full((n_frames, 8), np.nan, np.float32)
    has = np.zeros(n_frames, bool)
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            f = d["frame"]
            if f >= n_frames:
                continue
            b = d["ball_data"]
            if b.get("x") is not None:
                B[f] = (b["x"], b["y"], b["z"] if b.get("z") is not None else np.nan)
                bdet[f] = 1 if b.get("is_detected") else 2
            g = (d.get("possession") or {}).get("group")
            grp[f] = 1 if g == "home team" else 2 if g == "away team" else -1
            c = d.get("image_corners_projection") or {}
            if c.get("x_top_left") is not None:
                cam[f] = [c[k] for k in CAM]
            pdl = d["player_data"]
            has[f] = len(pdl) > 0
            for q in pdl:
                j = pid_col.get(q["player_id"])
                if j is not None and q.get("x") is not None:
                    PX[f, j], PY[f, j], PD[f, j] = q["x"], q["y"], bool(q.get("is_detected"))
    return dict(PX=PX, PY=PY, PD=PD, B=B, bdet=bdet, grp=grp, cam=cam, has=has)


# ----------------------------------------------------------------------------------- labels
def _itype(s):
    if not isinstance(s, str):
        return None
    for k, v in INTERRUPT.items():
        if s.startswith(k + "_"):
            return v
    return None


def _geo_type(xr, yr, L, W):
    """Stop type from the restart spot only (used for unknown types and the centre / penalty spots)."""
    gx = L / 2 - abs(xr)
    if np.hypot(xr, yr) < 1.5:
        return 5
    if abs(gx - 11.0) < 1.0 and abs(yr) < 1.0:
        return 6
    if abs(abs(yr) - W / 2) < 1.5 and gx > 2.0:
        return 1
    if gx < 2.0 and abs(abs(yr) - W / 2) < 2.0:
        return 2
    if 0 <= gx <= 6.5 and abs(yr) <= 10.0:
        return 3
    return None


def stoppages(ev, ph, periods, B, bdet, L, W):
    """Dead runs at 10 fps: list of dicts (period, s, b, a, type, sources, refined start)."""
    onball = ev[ev.event_type.isin(["player_possession", "on_ball_engagement"])]
    ends = np.sort(onball.frame_end.dropna().astype(int).values)
    ib = ev[ev.game_interruption_before.notna()][["frame_start", "game_interruption_before"]].sort_values("frame_start")
    ia = ev[ev.game_interruption_after.notna()][["frame_end", "game_interruption_after"]].sort_values("frame_end")
    x, y = B[:, 0], B[:, 1]
    out_any = (np.abs(x) > L / 2) | (np.abs(y) > W / 2)
    rows = []
    for per in periods:
        p0, p1 = per["start_frame"], per["end_frame"]
        q = ph[ph.period == per["period"]].sort_values("frame_start")
        fs, fe = q.frame_start.values.astype(int), q.frame_end.values.astype(int)
        if len(q) == 0:
            continue
        if fs[0] > p0:
            rows.append(dict(period=per["period"], s=p0, b=fs[0], a=p0, type=5, src_type="debut_periode",
                             src_start="debut_periode", it_before=None, it_after=None, edge=True))
        gaps = [(fe[k], fs[k + 1]) for k in range(len(q) - 1) if fs[k + 1] - fe[k] > 1]
        ga = np.array([g[0] for g in gaps]); gb = np.array([g[1] for g in gaps])
        # one-to-one event <-> gap matching: a restart event (game_interruption_before, start t) goes to
        # the LAST gap whose restart b <= t + 0.5 s (and t - b <= 6 s), the earliest such event wins;
        # an interruption event (game_interruption_after, end t) goes to the FIRST gap whose start
        # a >= t - 0.5 s (and a - t <= 4 s), the latest such event wins.
        TB, TA = {}, {}
        for t, lab in zip(ib.frame_start.values, ib.game_interruption_before.values):
            kk = np.flatnonzero(gb <= t + 5)
            if len(kk) and t - gb[kk[-1]] <= 60 and kk[-1] not in TB and _itype(lab):
                TB[kk[-1]] = _itype(lab)
        for t, lab in zip(ia.frame_end.values[::-1], ia.game_interruption_after.values[::-1]):
            kk = np.flatnonzero(ga >= t - 5)
            if len(kk) and ga[kk[0]] - t <= 40 and kk[0] not in TA and _itype(lab):
                TA[kk[0]] = _itype(lab)
        ibs = np.sort(ib.frame_start.values.astype(int)); iae = np.sort(ia.frame_end.values.astype(int))
        for k, (a, b) in enumerate(gaps):
            tb, ta = TB.get(k), TA.get(k)
            nxt = ibs[ibs >= b - 5]
            if len(nxt) and nxt[0] - b <= ANOM_FB and (b - a) < ANOM_DUR_S * SC_FPS:
                # anomalous gap: the next phase starts AT the restart reception and the gap is short, i.e.
                # the preceding phase ran through the stoppage -> true dead interval unknown, masked
                # from the last interruption event (<= 60 s before) to the restart
                prev = iae[(iae <= a) & (iae >= a - 600)]
                s0 = int(prev[-1]) if len(prev) else int(a)
                s0 = min(int(a), max(s0, int(gaps[k - 1][1]) if k else int(p0)))
                rows.append(dict(period=per["period"], s=s0, b=int(b), a=int(a), type=0, src_type="trou_anormal",
                                 src_start="inconnu", it_before=tb, it_after=None, last_touch=a, edge=False, unknown=True))
                continue
            typ = tb or ta or 8
            src_type = "reprise" if tb else ("interruption" if ta else "aucun_evenement")
            if src_type == "aucun_evenement" and (b - a) < NO_EVENT_MIN_S * SC_FPS:
                # short phase gap with no interruption event: phase-segmentation artefact or not -> unknown
                rows.append(dict(period=per["period"], s=int(a), b=int(b), a=int(a), type=0, src_type="trou_court_sans_evenement",
                                 src_start="fin_phase", it_before=None, it_after=None, last_touch=a, edge=False, unknown=True))
                continue
            if tb and ta and tb != ta:
                src_type = "reprise_desaccord"
            # restart spot (ball, any source, median over the 1 s before the restart)
            w = slice(max(p0, b - SC_FPS), b + 1)
            okb = ~np.isnan(x[w])
            if okb.sum() >= 3:
                xr, yr = float(np.median(x[w][okb])), float(np.median(y[w][okb]))
                geo = _geo_type(xr, yr, L, W)
                if geo in (5, 6) and typ != geo:          # unambiguous spots override the event label
                    typ, src_type = geo, "geometrie_point"
                elif typ == 8 and geo is not None:
                    typ, src_type = geo, "geometrie"
            lt = ends[(ends <= a) & (ends >= a - 50)]
            lt = int(lt[-1]) if len(lt) else a - SC_FPS
            s, src_start = a, "fin_phase"
            if typ in OUT_TYPES:
                # observed crossing: ball in-bounds at f-1 and beyond a line at f, both samples of the
                # same kind (detected, else SkillCorner-extrapolated), within MAX_ROLL_S of the last touch
                seg = np.arange(max(lt, p0 + 1), min(b, lt + int(MAX_ROLL_S * SC_FPS)))
                for code, name in ((1, "franchissement_detecte"), (2, "franchissement_extrapole")):
                    kind = bdet[seg] == code if code == 1 else bdet[seg] >= 1
                    kprev = bdet[seg - 1] == code if code == 1 else bdet[seg - 1] >= 1
                    hit = seg[out_any[seg] & ~out_any[seg - 1] & kind & kprev]
                    if len(hit):
                        s, src_start = int(hit[0]), name
                        break
            rows.append(dict(period=per["period"], s=int(s), b=int(b), a=int(a), type=int(typ), src_type=src_type,
                             src_start=src_start, it_before=tb, it_after=ta, last_touch=lt, edge=False))
        if fe[-1] < p1:
            rows.append(dict(period=per["period"], s=fe[-1], b=p1 + 1, a=fe[-1], type=8, src_type="fin_periode",
                             src_start="fin_phase", it_before=None, it_after=None, edge=True))
    return rows


# ----------------------------------------------------------------------------------- resampling
def _prev_next_valid(valid):
    n = len(valid)
    idx = np.arange(n)
    prev = np.where(valid, idx, -1); prev = np.maximum.accumulate(prev)
    nxt = np.where(valid, idx, n); nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return prev, nxt


def interp_25(v10, f10):
    """v10: (N,) or (N, k) float at 10 fps; f10: target positions in 10 fps frame units."""
    v = v10 if v10.ndim == 2 else v10[:, None]
    N = len(v)
    out = np.full((len(f10), v.shape[1]), np.nan, np.float32)
    i0 = np.clip(np.floor(f10 + 1e-9).astype(int), 0, N - 1)
    for j in range(v.shape[1]):
        ok = ~np.isnan(v[:, j])
        if not ok.any():
            continue
        prev, nxt = _prev_next_valid(ok)
        p = prev[i0]
        i1 = np.minimum(i0 + 1, N - 1)
        exact = np.abs(f10 - i0) < 1e-6
        qn = np.where(exact & ok[i0], i0, nxt[i1])
        good = (p >= 0) & (qn < N) & ((qn - p) <= MAX_GAP_S * SC_FPS + 1e-9)
        p_, q_ = p[good], qn[good]
        den = np.where(q_ > p_, q_ - p_, 1)
        t = np.where(q_ > p_, (f10[good] - p_) / den, 0.0)
        out[good, j] = (1 - t) * v[p_, j] + t * v[q_, j]
    return out if v10.ndim == 2 else out[:, 0]


# ----------------------------------------------------------------------------------- conversion
def convert(mid, raw_dir, out_dir, log=print):
    d = os.path.join(raw_dir, mid)
    mj = json.load(open(os.path.join(d, f"{mid}_match.json")))
    ev = pd.read_csv(os.path.join(d, f"{mid}_dynamic_events.csv"), low_memory=False)
    ph = pd.read_csv(os.path.join(d, f"{mid}_phases_of_play.csv"))
    L, W = float(mj["pitch_length"]), float(mj["pitch_width"])
    home = mj["home_team"]["id"]
    periods = sorted([p for p in mj["match_periods"] if p["period"] in (1, 2)], key=lambda p: p["period"])
    players = sorted(mj["players"], key=lambda p: (p["team_id"] != home, p["id"]))
    pid_col = {p["id"]: j for j, p in enumerate(players)}
    team_all = np.array([0 if p["team_id"] == home else 1 for p in players], np.int8)
    gk_all = np.array([(p.get("player_role") or {}).get("name") == "Goalkeeper" for p in players])
    pid_all = np.array([p["id"] for p in players], np.int32)
    n10 = max(p["end_frame"] for p in periods) + 2
    tk = parse_tracking(os.path.join(d, f"{mid}_tracking_extrapolated.jsonl"), n10, pid_col)
    B, bdet = tk["B"], tk["bdet"]
    stops = stoppages(ev, ph, periods, B, bdet, L, W)
    # 10 fps label arrays
    BS10 = np.full(n10, -1, np.int8); ST10 = np.zeros(n10, np.int8)
    for per in periods:
        BS10[per["start_frame"]:per["end_frame"] + 1] = 1
    for r in stops:
        if r.get("unknown"):
            continue
        BS10[r["s"]:r["b"]] = 0
        ST10[r["s"]:r["b"]] = r["type"]
    for r in stops:
        if r.get("unknown"):
            BS10[r["s"]:r["b"]] = -1
            ST10[r["s"]:r["b"]] = 0
    out, tab = {}, []
    n_unknown = [0]
    for per in periods:
        h = f"h{per['period']}"
        k0, k1 = int(np.ceil(per["start_frame"] * 2.5)), int(np.floor(per["end_frame"] * 2.5))
        k = np.arange(k0, k1 + 1)
        f10 = k / 2.5
        near = np.clip(np.rint(f10).astype(int), 0, n10 - 1)
        sl = slice(per["start_frame"], per["end_frame"] + 1)
        used = np.flatnonzero(~np.all(np.isnan(tk["PX"][sl]), axis=0))
        PX = interp_25(tk["PX"][:, used], f10); PY = interp_25(tk["PY"][:, used], f10)
        PD = tk["PD"][near][:, used] & ~np.isnan(PX)
        det = bdet == 1
        Bd = np.where(det[:, None], B, np.nan).astype(np.float32)
        BX, BY, BZ = (interp_25(Bd[:, c], f10) for c in range(3))
        BXe, BYe = interp_25(B[:, 0], f10), interp_25(B[:, 1], f10)
        ball_obs = ~np.isnan(BX)
        BS, ST = BS10[near].copy(), ST10[near].copy()
        # confidence (25 fps)
        T = len(k)
        conf = np.where(BS >= 0, 2, 0).astype(np.uint8)
        def lower(c_lo, c_hi, val):
            c_lo, c_hi = max(0, c_lo), min(T, c_hi)
            if c_hi > c_lo:
                conf[c_lo:c_hi] = np.minimum(conf[c_lo:c_hi], val)
        for r in stops:
            if r["period"] != per["period"]:
                continue
            s25, b25 = int(round(r["s"] * 2.5)) - k0, int(round(r["b"] * 2.5)) - k0
            if r.get("unknown"):
                lower(s25, b25, 0)
                n_unknown[0] += 1
                continue
            if r["edge"]:
                lower(s25, b25, 0)
                tab.append(dict(match=mid, half=h, a=s25, b=b25, dur_s=(b25 - s25) / FPS, type=r["type"],
                                type_name=STOP_NAMES[r["type"]], src_type=r["src_type"], src_start=r["src_start"],
                                phase_end_vs_start_s=np.nan, last_touch_vs_start_s=np.nan, edge=True))
                continue
            w = CONF_START_REFINED if r["src_start"].startswith("franchissement") else CONF_START_PHASE
            lower(s25 - int(w * FPS), s25 + int(w * FPS), 1)
            lower(s25 - int(0.2 * FPS), s25 + int(0.2 * FPS), 0)
            lower(b25 - int(CONF_END * FPS), b25 + int(CONF_END * FPS), 1)
            if r["src_type"] in ("aucun_evenement", "reprise_desaccord", "geometrie", "geometrie_point"):
                lower(s25, b25, 1)
            tab.append(dict(match=mid, half=h, a=s25, b=b25, dur_s=(b25 - s25) / FPS, type=r["type"],
                            type_name=STOP_NAMES[r["type"]], src_type=r["src_type"], src_start=r["src_start"],
                            phase_end_vs_start_s=(r["a"] - r["s"]) / SC_FPS,
                            last_touch_vs_start_s=(r["last_touch"] - r["s"]) / SC_FPS, edge=False))
        out.update({f"{h}_PX": PX, f"{h}_PY": PY, f"{h}_team": team_all[used], f"{h}_BX": BX, f"{h}_BY": BY,
                    f"{h}_BZ": BZ, f"{h}_BS": BS, f"{h}_BP": tk["grp"][near], f"{h}_n0": k0,
                    f"{h}_ball_obs": ball_obs, f"{h}_stop": ST, f"{h}_label_conf": conf, f"{h}_PD": PD,
                    f"{h}_pid": pid_all[used], f"{h}_gk": gk_all[used], f"{h}_BXe": BXe, f"{h}_BYe": BYe,
                    f"{h}_cam": interp_25(tk["cam"], f10), f"{h}_has_track": tk["has"][near]})
    src = (f"SkillCorner Open Data (github.com/SkillCorner/opendata, MIT), match {mid} "
           f"{mj['home_team']['short_name']} - {mj['away_team']['short_name']} {mj['date_time'][:10]}, "
           f"10 fps broadcast tracking resampled to 25 fps; labels from phases_of_play + dynamic_events")
    n_anom = sum(1 for r in stops if r["src_type"] == "trou_anormal")
    n_gaps = sum(1 for r in stops if not r["edge"])
    quality = "douteuse" if n_anom > ANOM_MATCH_FRAC * max(1, n_gaps) else "bonne"
    path = os.path.join(out_dir, f"SC_{mid}.npz")
    np.savez_compressed(path, L=L, W=W, source=src, stoppages_json=json.dumps(tab), label_quality=quality,
                        n_masked_gaps=n_anom + sum(1 for r in stops if r["src_type"] == "trou_court_sans_evenement"), **out)
    log(f"{mid} ok: {len(tab)} stoppages, {n_unknown[0]} masked gaps -> BS -1 (anomalous {n_anom}), quality {quality}, T={[out[f'h{p}_BS'].shape[0] for p in (1, 2)]}")
    return path, pd.DataFrame(tab)


# ----------------------------------------------------------------------------------- controls
def event_f1(true, pred, iou=0.5):
    """Same as dead_v4.event_f1: one-to-one matching of runs at IoU >= iou."""
    tr, pr = runs_of(true), runs_of(pred)
    pairs = []
    for i, (a, b) in enumerate(tr):
        for j, (c, d) in enumerate(pr):
            if c < b and d > a:
                pairs.append(((min(b, d) - max(a, c)) / (max(b, d) - min(a, c)), i, j))
    pairs.sort(reverse=True)
    ui, uj, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < iou:
            break
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); tp += 1
    return tp, len(pr) - tp, len(tr) - tp


def _f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def baseline_half(dead, ball_obs, valid, min_s=2.0):
    pred = np.zeros_like(dead)
    for a, b in runs_of(~ball_obs):
        if b - a >= min_s * FPS:
            pred[a:b] = True
    v = valid
    tp, fp, fn = int((pred & dead & v).sum()), int((pred & ~dead & v).sum()), int((~pred & dead & v).sum())
    etp, efp, efn = event_f1(dead, pred)
    return dict(tp=tp, fp=fp, fn=fn, etp=etp, efp=efp, efn=efn)


def _impulse(bx, by, obs, b, w=50):
    """Frame (rel. to b) of the restart kick / throw seen on the detected ball: first frame in
    [b - 2 s, b + 2 s] whose speed over 0.2 s exceeds 6 m/s while the median speed over the preceding
    0.8 s is below 2.5 m/s, with the ball observed on >= 80 % of that window."""
    lo, hi = max(25, b - w), min(len(bx) - 5, b + w)
    v = np.full(len(bx), np.nan)
    v[:-5] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * FPS / 5
    for f in range(lo, hi):
        pre = v[f - 20:f]
        if v[f] > 6 and obs[f - 20:f].mean() >= 0.8 and np.nanmedian(pre) < 2.5:
            return f - b
    return None


def controls_match(path):
    import dead_bounds as db3
    z = np.load(path)
    L, W = float(z["L"]), float(z["W"])
    halves = db3.load_sportec(path)             # compatibility check (raises if incompatible)
    tab = pd.DataFrame(json.loads(str(z["stoppages_json"])))
    tab = tab[~tab.edge.astype(bool)]     # pre-kick-off / post-final-whistle: cropped by load_sportec
    res = dict(n_frames=0, dead=0, obs_live=0, n_live=0, obs_dead=0, n_dead=0, notrack_live=0, notrack_dead=0,
               conf_hist=[0, 0, 0], base=dict(tp=0, fp=0, fn=0, etp=0, efp=0, efn=0))
    geo = []
    for k, h in enumerate(("h1", "h2")):
        bs = z[f"{h}_BS"]; alive = np.flatnonzero(bs == 1); a0, a1 = alive[0], alive[-1] + 1
        dead = bs[a0:a1] == 0; valid = bs[a0:a1] >= 0
        obs = z[f"{h}_ball_obs"][a0:a1]; has = z[f"{h}_has_track"][a0:a1]
        assert len(halves[k]["dead"]) == len(dead)
        res["n_frames"] += int(valid.sum()); res["dead"] += int(dead.sum())
        res["obs_live"] += int((obs & ~dead & valid).sum()); res["n_live"] += int((~dead & valid).sum())
        res["obs_dead"] += int((obs & dead).sum()); res["n_dead"] += int(dead.sum())
        res["notrack_live"] += int((~has & ~dead & valid).sum()); res["notrack_dead"] += int((~has & dead).sum())
        for c in range(3):
            res["conf_hist"][c] += int((z[f"{h}_label_conf"][a0:a1] == c).sum())
        bl = baseline_half(dead, obs, valid)
        for kk in bl:
            res["base"][kk] += bl[kk]
        bx, by = z[f"{h}_BX"], z[f"{h}_BY"]; ob = z[f"{h}_ball_obs"]
        for r in tab[tab.half == h].itertuples():
            a, b = int(r.a), int(r.b)
            g = dict(type=r.type_name, src_start=r.src_start, src_type=r.src_type)
            wa = slice(max(0, a - 10), min(len(bx), a + 11))
            wb = slice(max(0, b - 10), min(len(bx), b + 1))      # ball just before the restart
            oa = ob[wa]
            if oa.any():
                xa, ya = bx[wa][oa], by[wa][oa]
                g["out_at_start"] = bool(((np.abs(xa) > L / 2) | (np.abs(ya) > W / 2)).any())
            obb = ob[wb]
            if obb.any():
                xb, yb = np.nanmedian(bx[wb][obb]), np.nanmedian(by[wb][obb])
                g["d_touchline_restart"] = float(abs(abs(yb) - W / 2))
                g["d_corner_restart"] = float(np.hypot(abs(xb) - L / 2, abs(yb) - W / 2))
                g["in_6m_restart"] = bool(abs(xb) > L / 2 - 7.0 and abs(yb) < 10.5)
                g["d_centre_restart"] = float(np.hypot(xb, yb))
            imp = _impulse(bx, by, ob, b)
            g["impulse_vs_end_s"] = imp / FPS if imp is not None else np.nan
            geo.append(g)
    return res, tab, pd.DataFrame(geo)


def run_controls(npz_dir, out_dir, log=print):
    files = sorted(glob.glob(os.path.join(npz_dir, "SC_*.npz")))
    per, tabs, geos = [], [], []
    for f in files:
        mid = os.path.basename(f)[3:-4]
        r, tab, geo = controls_match(f)
        b = r["base"]
        per.append(dict(match=mid, label_quality=str(np.load(f)["label_quality"]), n_stops=len(tab), dead_frac=r["dead"] / r["n_frames"],
                        ball_obs_live=r["obs_live"] / r["n_live"], ball_obs_dead=r["obs_dead"] / r["n_dead"],
                        no_track_live=r["notrack_live"] / r["n_live"], no_track_dead=r["notrack_dead"] / r["n_dead"],
                        conf0=r["conf_hist"][0] / r["n_frames"], conf1=r["conf_hist"][1] / r["n_frames"],
                        conf2=r["conf_hist"][2] / r["n_frames"],
                        base_f1_frame=_f1(b["tp"], b["fp"], b["fn"]), base_f1_stop=_f1(b["etp"], b["efp"], b["efn"]),
                        **{f"base_{k}": v for k, v in b.items()},
                        **{f"n_{t}": int((tab.type_name == t).sum()) for t in STOP_NAMES.values() if t != "vivant"}))
        geo["match"] = mid
        tabs.append(tab); geos.append(geo)
        log(f"controls {mid}: dead {per[-1]['dead_frac']:.3f} base F1 {per[-1]['base_f1_frame']:.3f}")
    P_all = pd.DataFrame(per); T_all = pd.concat(tabs); G_all = pd.concat(geos)
    good = set(P_all[P_all.label_quality == "bonne"].match)
    P = P_all[P_all.match.isin(good)]; T = T_all[T_all.match.astype(str).isin(good)]; G = G_all[G_all.match.isin(good)]
    excluded = sorted(set(P_all.match) - good)
    nm = len(P)
    types = pd.DataFrame([dict(type=t, n_par_match=(T.type_name == t).sum() / nm,
                               duree_med_s=T[T.type_name == t].dur_s.median(),
                               sportec_n_par_match=SPORTEC_REF.get(t, (np.nan,))[0],
                               sportec_duree_med_s=SPORTEC_REF.get(t, (np.nan, np.nan))[1])
                          for t in STOP_NAMES.values() if t != "vivant"])
    def frac(s):
        s = s.dropna()
        return float(s.mean()) if len(s) else None
    def q(s):
        s = s.dropna()
        return [float(s.quantile(x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)] if len(s) else None
    Gt = {t: G[G.type == t] for t in G.type.unique()}
    geom = dict(
        touche_ball_out_at_start=frac(Gt.get("touche", G.iloc[:0]).get("out_at_start", pd.Series(dtype=float))),
        corner_ball_out_at_start=frac(Gt.get("corner", G.iloc[:0]).get("out_at_start", pd.Series(dtype=float))),
        six_m_ball_out_at_start=frac(Gt.get("six_metres", G.iloc[:0]).get("out_at_start", pd.Series(dtype=float))),
        coup_franc_ball_out_at_start=frac(Gt.get("coup_franc", G.iloc[:0]).get("out_at_start", pd.Series(dtype=float))),
        touche_restart_on_touchline_1p5m=frac(Gt["touche"].d_touchline_restart.dropna() < 1.5) if "touche" in Gt else None,
        touche_restart_d_touchline_q=q(Gt["touche"].d_touchline_restart) if "touche" in Gt else None,
        corner_restart_within_3m_flag=frac(Gt["corner"].d_corner_restart.dropna() < 3) if "corner" in Gt else None,
        six_m_restart_in_box=frac(Gt["six_metres"].in_6m_restart.dropna().astype(float)) if "six_metres" in Gt else None,
        engagement_restart_within_2m_centre=frac(Gt["engagement"].d_centre_restart.dropna() < 2) if "engagement" in Gt else None,
        n_with_ball_at_start=int(G.out_at_start.notna().sum()) if "out_at_start" in G else 0,
        n_with_ball_at_restart=int(G.d_touchline_restart.notna().sum()) if "d_touchline_restart" in G else 0)
    timing = dict(
        restart_impulse_vs_end_s_q10_25_50_75_90=q(G.impulse_vs_end_s), restart_impulse_n=int(G.impulse_vs_end_s.notna().sum()),
        restart_impulse_within_0p4s=frac((G.impulse_vs_end_s.dropna().abs() <= 0.4).astype(float)),
        start_source_counts=T.src_start.value_counts().to_dict(),
        out_types_phase_end_minus_refined_start_s_q=q(T[T.src_start.str.startswith("franchissement")].phase_end_vs_start_s),
        last_touch_minus_start_s_q=q(T.last_touch_vs_start_s),
        type_source_counts=T.src_type.value_counts().to_dict())
    tot = P[[c for c in P.columns if c.startswith("base_") and not c.startswith("base_f1")]].sum()
    base = dict(regle="balle non observee >= 2 s => D (frames load_sportec, BS>=0)",
                f1_frame_global=_f1(tot.base_tp, tot.base_fp, tot.base_fn),
                precision_frame=tot.base_tp / max(1, tot.base_tp + tot.base_fp),
                rappel_frame=tot.base_tp / max(1, tot.base_tp + tot.base_fn),
                f1_arret_iou05_global=_f1(tot.base_etp, tot.base_efp, tot.base_efn),
                f1_frame_par_match_med=float(P.base_f1_frame.median()),
                f1_arret_par_match_med=float(P.base_f1_stop.median()))
    summary = dict(n_matchs=nm, matchs_exclus_labels_douteux=excluded, dead_frac_mean=float(P.dead_frac.mean()), dead_frac_sportec=SPORTEC_REF["dead_frac"],
                   ball_obs_live_mean=float(P.ball_obs_live.mean()), ball_obs_dead_mean=float(P.ball_obs_dead.mean()),
                   no_track_live_mean=float(P.no_track_live.mean()), no_track_dead_mean=float(P.no_track_dead.mean()),
                   conf_frac_mean=[float(P.conf0.mean()), float(P.conf1.mean()), float(P.conf2.mean())],
                   types=types.round(3).to_dict("records"), geometrie=geom, temporel=timing, baseline=base)
    os.makedirs(out_dir, exist_ok=True)
    P_all.to_csv(os.path.join(out_dir, "sc_controles_par_match.csv"), index=False)
    T_all.to_csv(os.path.join(out_dir, "sc_arrets.csv"), index=False)
    G_all.to_csv(os.path.join(out_dir, "sc_geometrie_arrets.csv"), index=False)
    with open(os.path.join(out_dir, "sc_controles.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary, P_all, T_all, G_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--matches", nargs="*", default=MATCH_IDS)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    raw, npz, rep = (os.path.join(a.work, x) for x in ("raw", "npz", "controles"))
    for x in (raw, npz, rep):
        os.makedirs(x, exist_ok=True)
    if a.cmd == "run":
        for mid in a.matches:
            if os.path.exists(os.path.join(npz, f"SC_{mid}.npz")):
                print(mid, "cached", flush=True); continue
            try:
                d = download(mid, raw)
                convert(mid, raw, npz, log=lambda s: print(s, flush=True))
                if not a.keep_raw:
                    os.remove(os.path.join(d, f"{mid}_tracking_extrapolated.jsonl"))
            except Exception as e:  # keep going, report
                print(mid, "FAILED", repr(e), flush=True)
    s, *_ = run_controls(npz, rep)
    print(json.dumps(s, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()

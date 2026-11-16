"""External dead-ball data: Metrica Sports sample data (3 open matches) -> Sportec-compatible npz.

Source: Metrica Sports, "sample-data", https://github.com/metrica-sports/sample-data (commit
e706dd506b360d69d9d123d5b8026e7294b13996, 2021-04-15). NO explicit licence file in the repository
(GitHub licence API: 404). README "Legal stuff": be responsible with the use of the data and
acknowledge the source for any public use. Intended use here: validation / training of the dead-ball
detector, data NOT redistributed, source cited.

Output: one npz per match, same keys as fetch_sportec.py (loadable by dead_bounds.load_sportec):
  L, W (m); for h in (h1, h2): h_PX, h_PY (T x n float32, m, origin at the pitch CENTRE, x along the
  length, y = Metrica y direction (downwards on the Metrica picture; dead_v4 geometry is symmetric
  in y), NaN when the player is not on the pitch / not tracked), h_team (int8 0 home / 1 away),
  h_BX, h_BY (float32, NaN when the provider has no ball), h_BZ (all NaN: Metrica gives no height),
  h_BS (int8 1 live / 0 dead / -1 outside [first kick-off, end of half]), h_BP (int8 1 home / 2 away
  / -1 unknown, from events, approximate), h_n0 (first Metrica frame number of the half), plus
  h_ball_obs (bool: finite ball AND not inside an exactly linear run, i.e. not interpolated between
  keyframes by the provider), h_ball_interp (bool, the linear-run flag), h_stop (int8 dead_v4 codes),
  h_label_conf (uint8 0-2, see LABEL_CONF_DOC), source (str).
Native rate is 25 fps with contiguous frames in all three games: no resampling, no interpolation.

Labels (events -> stoppages): every restart event closes a stoppage at its Start Frame (first live
frame): SET PIECE THROW IN -> 1 touche, CORNER KICK -> 2 corner, FREE KICK* -> 4 coup_franc,
KICK OFF -> 5 engagement, PENALTY -> 6 penalty, and the goal kick, which Metrica does NOT code as
a SET PIECE but as a PASS / BALL LOST / BALL OUT with subtype GOAL KICK -> 3 six_metres. The start is
found by walking back from the restart over the "dead chain" (cards, fault events, out events and
events within 0.5 s of the restart) until the last live-play event; the start is the earliest
interruption in the chain: BALL OUT (End Frame), SHOT *OUT / *GOAL (End Frame), FAULT RECEIVED /
CHALLENGE *FAULT* without ADVANTAGE (Start Frame), *OFFSIDE (End Frame), HAND BALL (Start Frame).
No interruption in the chain -> fallback start = end of the last live event, refined by the first
out-of-pitch ball frame for touche / corner / six_metres (type_source 'geometrie' / 'repli').
The first kick-off of each half opens the live window and END HALF closes it (BS = -1 outside).

CLI
  python possession/ext_metrica.py all --work <dir outside the repo>     # download+convert+check
  python possession/ext_metrica.py download|convert|check --work <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 25
L_PITCH, W_PITCH = 105.0, 68.0
REPO = "https://raw.githubusercontent.com/metrica-sports/sample-data/"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
SOURCE = ("Metrica Sports sample-data (github.com/metrica-sports/sample-data, commit " + COMMIT[:10] +
          "), no explicit licence; README asks to acknowledge the source")
FILES = {1: ["Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
             "Sample_Game_1_RawTrackingData_Away_Team.csv"],
         2: ["Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
             "Sample_Game_2_RawTrackingData_Away_Team.csv"],
         3: ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"]}
STOP_NAMES = {0: "vivant", 1: "touche", 2: "corner", 3: "six_metres", 4: "coup_franc", 5: "engagement",
              6: "penalty", 7: "balle_arbitre", 8: "inconnu"}
OUT_TYPES = (1, 2, 3)                # stoppages that start with the ball crossing a line
CHAIN_RESTART_S = 0.5                # events this close before the restart belong to it
LIN_TOL_M = 0.003                    # |second difference| below this = exactly linear (interpolated)
LIN_MIN = 10                         # min run (frames) of exactly linear ball = provider interpolation
NEAR_S = 1.0                         # half-width of the "boundary" zone for label confidence
LABEL_CONF_DOC = ("2: frame more than 1 s from any boundary of its stoppage, boundary from an event "
                  "(restart frame or out / goal / offside event); 1: within 1 s of such a boundary, or "
                  "start from a foul event (actual whistle 0-3 s later, unknown), or ball-geometry start; "
                  "0: outside the live window (BS=-1), within 1 s of a fallback boundary (no "
                  "interruption event, no ball exit), or stoppage of unknown type")


# ============================================================================ download
def download(work):
    raw = os.path.join(work, "raw")
    for g, names in FILES.items():
        d = os.path.join(raw, f"Sample_Game_{g}")
        os.makedirs(d, exist_ok=True)
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                continue
            url = f"{REPO}{COMMIT}/data/Sample_Game_{g}/{n}"
            r = subprocess.run(["curl", "-sL", "--fail", "-o", p + ".part", url])
            if r.returncode:
                raise RuntimeError(f"download failed {url}")
            os.replace(p + ".part", p)
            print("downloaded", n, os.path.getsize(p), flush=True)
    return raw


# ============================================================================ readers
def _tokens(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return frozenset()
    return frozenset(t.strip() for t in str(s).replace(",", "-").split("-") if t.strip())


def read_csv_game(raw, g):
    """Games 1-2 (Metrica CSV). Returns frames, period, PX, PY (normalised), team, bx, by, events."""
    d = os.path.join(raw, f"Sample_Game_{g}")
    out = {}
    for side, t in (("Home", 0), ("Away", 1)):
        p = os.path.join(d, f"Sample_Game_{g}_RawTrackingData_{side}_Team.csv")
        hdr = pd.read_csv(p, skiprows=2, nrows=0).columns.tolist()
        df = pd.read_csv(p, skiprows=3, header=None, dtype=np.float64)
        pcols = [i for i, h in enumerate(hdr) if h.startswith("Player")]
        out[side] = dict(frame=df[1].values.astype(np.int64), period=df[0].values.astype(np.int64),
                         x=df[pcols].values, y=df[[i + 1 for i in pcols]].values,
                         names=[f"{side}_{hdr[i]}" for i in pcols], t=t,
                         bx=df[len(hdr) - 2].values, by=df[len(hdr) - 1].values)
    h, a = out["Home"], out["Away"]
    assert np.array_equal(h["frame"], a["frame"])
    ball_same = np.array_equal(np.nan_to_num(h["bx"], nan=-9), np.nan_to_num(a["bx"], nan=-9))
    ev = pd.read_csv(os.path.join(d, f"Sample_Game_{g}_RawEventsData.csv"))
    ev = pd.DataFrame(dict(team=ev.Team.map({"Home": 0, "Away": 1}), type=ev.Type, sub=ev.Subtype.fillna(""),
                           period=ev.Period, start=ev["Start Frame"], end=ev["End Frame"],
                           sx=ev["Start X"], sy=ev["Start Y"], ex=ev["End X"], ey=ev["End Y"]))
    return dict(frame=h["frame"], period=h["period"], x=np.hstack([h["x"], a["x"]]), y=np.hstack([h["y"], a["y"]]),
                team=np.r_[np.zeros(len(h["names"]), np.int8), np.ones(len(a["names"]), np.int8)],
                names=h["names"] + a["names"], bx=h["bx"], by=h["by"], events=ev,
                info=dict(ball_identical_home_away=bool(ball_same)))


def read_epts_game(raw, g=3):
    """Game 3 (FIFA EPTS metadata + txt, events JSON), parsed directly (no kloppy)."""
    d = os.path.join(raw, f"Sample_Game_{g}")
    root = ET.parse(os.path.join(d, f"Sample_Game_{g}_metadata.xml")).getroot()
    sc = root.find(".//Score")
    home_id = sc.get("idLocalTeam")
    ch = {e.get("id"): e.get("playerId") for e in root.iter("PlayerChannel")}
    pteam = {e.get("id"): e.get("teamId") for e in root.iter("Player")}
    prm = {e.findtext("Name"): e.findtext("Value") for e in root.iter("ProviderParameter")}
    specs = []
    for s in root.iter("DataFormatSpecification"):
        refs = [e.get("playerChannelId") for e in s.iter("PlayerChannelRef")]
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), [ch[r] for r in refs[0::2]]))
    pids = sorted({p for _, _, ps in specs for p in ps}, key=lambda p: (pteam[p] != home_id, p))
    col = {p: j for j, p in enumerate(pids)}
    n0, n1 = min(s[0] for s in specs), max(s[1] for s in specs)
    T = n1 - n0 + 1
    X = np.full((T, len(pids)), np.nan)
    Y = X.copy()
    bx = np.full(T, np.nan)
    by = bx.copy()
    seen = np.zeros(T, bool)
    starts = np.array([s[0] for s in specs])
    with open(os.path.join(d, f"Sample_Game_{g}_tracking.txt")) as fh:
        for line in fh:
            fr, pl, ba = line.rstrip("\n").split(":")
            fr = int(fr)
            k = np.searchsorted(starts, fr, side="right") - 1
            ps = specs[k][2]
            i = fr - n0
            seen[i] = True
            for p, xy in zip(ps, pl.split(";")):
                u, v = xy.split(",")
                X[i, col[p]], Y[i, col[p]] = float(u), float(v)
            u, v = ba.split(",")[:2]
            bx[i], by[i] = float(u), float(v)
    frames = np.arange(n0, n1 + 1)
    h1e = int(prm["first_half_end"])
    period = np.where(frames <= h1e, 1, 2)
    evj = json.load(open(os.path.join(d, f"Sample_Game_{g}_events.json")))["data"]
    rows = []
    for e in evj:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = sub["name"]
        elif isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        else:
            sub = ""
        st, en = e["start"], e["end"] or {}
        rows.append(dict(team=0 if e["team"]["id"] == home_id else 1, type=e["type"]["name"], sub=sub,
                         period=e["period"], start=st["frame"], end=en.get("frame") or 0,
                         sx=st.get("x"), sy=st.get("y"), ex=en.get("x"), ey=en.get("y")))
    ev = pd.DataFrame(rows).astype(dict(sx=float, sy=float, ex=float, ey=float))
    return dict(frame=frames, period=period, x=X, y=Y,
                team=np.array([0 if pteam[p] == home_id else 1 for p in pids], np.int8),
                names=pids, bx=bx, by=by, events=ev,
                info=dict(all_frames_present=bool(seen.all()), score=(sc.findtext("LocalTeamScore"),
                          sc.findtext("VisitingTeamScore")), n_specs=len(specs)))


# ============================================================================ labels
def _is_restart(r):
    tk = _tokens(r["sub"])
    if r.type == "SET PIECE":
        for key, code in (("THROW IN", 1), ("CORNER KICK", 2), ("FREE KICK", 4), ("KICK OFF", 5), ("PENALTY", 6)):
            if key in tk:
                return code
        return 8
    if r.type in ("PASS", "BALL LOST", "BALL OUT", "SHOT") and "GOAL KICK" in tk:
        return 3
    return 0


def _interruption(r):
    """(frame, cause) if the event stops play, else None."""
    tk = _tokens(r["sub"])
    endf = r.end if r.end and r.end > 0 else r.start
    if r.type == "BALL OUT":
        return endf, "goal" if "GOAL" in tk else "out"
    if r.type == "SHOT" and ("OUT" in tk or "GOAL" in tk):
        return endf, "goal" if "GOAL" in tk else "out"
    if r.type == "FAULT RECEIVED":
        return r.start, "foul"
    if r.type == "CHALLENGE" and "FAULT" in tk and "ADVANTAGE" not in tk:
        return r.start, "foul"
    if "OFFSIDE" in tk:
        return endf, "offside"
    if "HAND BALL" in tk:
        return r.start, "handball"
    return None


def _transparent(r):
    """Events that do not prove the ball was live (inside a dead chain)."""
    return (r.type in ("CARD", "FAULT RECEIVED", "RECOVERY", "CARRY")
            or (r.type == "CHALLENGE" and "FAULT" in _tokens(r["sub"])))


FOUL_GRACE_S = 3.0                   # events up to 3 s after a foul / offside / handball are play-on


def _after_interruption(ev, j, prev_restart, look=12):
    """True if event j follows an interruption that already stopped play: an out / goal event whose
    exit frame is >= its start (ball in flight going out) or a foul-type event at most FOUL_GRACE_S
    before it."""
    t = ev.iloc[j].start
    for k in range(j - 1, max(j - 1 - look, -1), -1):
        q = ev.iloc[k]
        if q.start < prev_restart or t - q.start > 10 * FPS:
            break
        itr = _interruption(q)
        if itr is None:
            continue
        if itr[1] in ("out", "goal") and itr[0] >= t:
            return True
        if itr[1] in ("foul", "offside", "handball") and 0 <= t - itr[0] <= FOUL_GRACE_S * FPS:
            return True
    return False


def build_stoppages(ev, bxm, bym, n0, T):
    """Stoppages of one half. bxm/bym in metres (0..L, 0..W), index = frame - n0."""
    ev = ev.sort_values(["start"], kind="stable").reset_index(drop=True)
    ko = ev[(ev.type == "SET PIECE") & ev["sub"].map(lambda s: "KICK OFF" in _tokens(s))]
    first_ko = int(ko.start.iloc[0]) if len(ko) else int(ev.start.iloc[0])
    eh = ev[ev["sub"].map(lambda s: "END HALF" in _tokens(s))]
    if len(eh):
        r = eh.iloc[-1]
        end_half = int(max(r.start, r.end))
    else:
        end_half = int(max(ev.start.max(), ev.end.max()))
    stops, warn = [], []
    prev_restart = first_ko
    out_pitch = (bxm < 0) | (bxm > L_PITCH) | (bym < 0) | (bym > W_PITCH)
    for i, r in ev.iterrows():
        code = _is_restart(r)
        fr = int(r.start)
        if code == 0 or fr <= first_ko or fr > end_half:
            continue
        if code == 3 and i > 0 and _is_restart(ev.iloc[i - 1]) == 3 and fr - int(ev.iloc[i - 1].start) < FPS:
            continue                                     # duplicate goal-kick rows
        chain, last_live = [], None
        for j in range(i - 1, -1, -1):
            q = ev.iloc[j]
            if q.start < prev_restart:
                break
            itr = _interruption(q)
            if itr is not None:
                chain.append((int(itr[0]), itr[1], j))
                continue
            if _transparent(q) or q.start >= fr - CHAIN_RESTART_S * FPS:
                continue
            if _after_interruption(ev, j, prev_restart):
                continue                                 # play-on after the whistle / ball already going out
            last_live = q
            break
        retaken = "RETAKEN" in _tokens(r["sub"])
        if chain:
            s, cause, _ = min(chain)
            src = "evenement"
        else:
            cause = "retaken" if retaken else "aucun_evenement"
            s = int(max(last_live.start, last_live.end)) if last_live is not None else prev_restart
            src = "repli"
            if code in OUT_TYPES:
                lo = max((int(last_live.start) if last_live is not None else prev_restart) - n0, 0)
                hit = np.flatnonzero(out_pitch[lo:fr - n0])
                if len(hit):
                    s, src = lo + int(hit[0]) + n0, "geometrie"
        if s >= fr:
            warn.append(f"start>=restart at frame {fr} ({STOP_NAMES[code]}), skipped")
            prev_restart = fr
            continue
        stops.append(dict(a=s - n0, b=fr - n0, type=code, type_name=STOP_NAMES[code], cause=cause, start_src=src,
                          restart_frame=fr, restart_team=int(r.team) if r.team == r.team else -1,
                          restart_sub=r["sub"], retaken=retaken, restart_src="explicite"))
        prev_restart = fr
    stops += _implicit_stoppages(ev, stops, first_ko, end_half, bxm, bym, n0, T, warn)
    stops.sort(key=lambda d: d["a"])
    st = pd.DataFrame(stops)
    # overlapping stoppages (e.g. retaken free kick whose fallback start falls before the previous end)
    if len(st):
        for k in range(1, len(st)):
            if st.loc[k, "a"] < st.loc[k - 1, "b"]:
                warn.append(f"overlap at restart {st.loc[k, 'restart_frame']}, start clipped")
                st.loc[k, "a"] = st.loc[k - 1, "b"]
        st = st[st.b > st.a].reset_index(drop=True)
    return st, first_ko - n0, min(end_half - n0 + 1, T), warn


KICK_TYPES = ("PASS", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE")
IMPLICIT_MIN_FOUL_S = 3.0          # a foul followed by a kick sooner than this = advantage, no stoppage
IMPLICIT_MAX_S = 180.0


def _implicit_stoppages(ev, stops, first_ko, end_half, bxm, bym, n0, T, warn):
    """Interruptions (out / goal / foul / offside / handball) not covered by an explicit restart:
    Metrica often codes a goal kick or throw-in as RECOVERY + PASS without SET PIECE / GOAL KICK.
    The restart is the first kick event (PASS, BALL LOST, BALL OUT, SHOT) after the interruption;
    the type comes from the exit side and the restart spot."""
    cov = np.zeros(T + 2, bool)
    starts = np.array(sorted(d["a"] for d in stops)) if stops else np.zeros(0, int)
    for d in stops:
        cov[d["a"]:d["b"] + 1] = True
    out = []
    evr = ev.reset_index(drop=True)
    for i, q in evr.iterrows():
        itr = _interruption(q)
        if itr is None:
            continue
        f, cause = int(itr[0]), itr[1]
        if f <= first_ko or f >= end_half or cov[min(max(f - n0, 0), T)]:
            continue
        r = None
        for j in range(i + 1, len(evr)):
            u = evr.iloc[j]
            if u.start <= f:
                continue
            if u.start - f > IMPLICIT_MAX_S * FPS:
                break
            if u.type in KICK_TYPES and _interruption(u) is None:
                r = u
                break
            if u.type in ("BALL OUT", "SHOT") and _interruption(u) is not None and u.start > f + FPS:
                r = u                                    # restart kick that went straight out
                break
        if r is None:
            continue
        fr = int(r.start)
        if np.any((starts > f - n0) & (starts < fr - n0)):
            continue                                     # another (explicit) stoppage starts in between
        if cause == "foul" and fr - f < IMPLICIT_MIN_FOUL_S * FPS:
            warn.append(f"faute sans arret (avantage probable) frame {f}")
            continue
        if fr - f < FPS:
            warn.append(f"interruption {cause} at {f} followed by a kick after {(fr - f) / FPS:.2f} s, ignored")
            continue
        a, b = f - n0, fr - n0
        w = slice(max(a - 12, 0), min(a + 25, T))
        ox = bool(((bxm[w] < 0) | (bxm[w] > L_PITCH)).any())
        oy = bool(((bym[w] < 0) | (bym[w] > W_PITCH)).any())
        fin = np.flatnonzero(np.isfinite(bxm[b:b + 13]))
        xr, yr = (bxm[b + fin[0]], bym[b + fin[0]]) if len(fin) else (np.nan, np.nan)
        gx = min(xr, L_PITCH - xr)
        if cause in ("out", "goal") and oy and not ox:
            code = 1
        elif cause == "goal" and np.hypot(xr - L_PITCH / 2, yr - W_PITCH / 2) < 2:
            code = 5
        elif cause in ("out", "goal") and ox and gx <= 7.0 and abs(yr - W_PITCH / 2) <= 10.2:
            code = 3
        elif cause in ("out", "goal") and ox and np.hypot(gx, min(yr, W_PITCH - yr)) <= 3.0:
            code = 2
        elif cause in ("foul", "offside", "handball"):
            code = 6 if abs(gx - 11) < 2 and abs(yr - W_PITCH / 2) < 2 else 4
        else:
            code = 8
        out.append(dict(a=a, b=b, type=code, type_name=STOP_NAMES[code], cause=cause, start_src="evenement",
                        restart_frame=fr, restart_team=int(r.team) if r.team == r.team else -1,
                        restart_sub=f"{r.type}:{r['sub']}", retaken=False, restart_src="implicite"))
        cov[a:b + 1] = True
    return out


def frame_labels(st, lo, hi, T):
    BS = np.full(T, -1, np.int8)
    BS[lo:hi] = 1
    stop = np.zeros(T, np.int8)
    conf = np.full(T, 2, np.uint8)
    near = int(NEAR_S * FPS)
    for r in st.itertuples():
        BS[r.a:r.b] = 0
        stop[r.a:r.b] = r.type
        if r.type == 8:
            conf[r.a:r.b] = 0
        # start boundary
        sa = slice(max(r.a - near, 0), min(r.a + near, T))
        if r.start_src == "repli":
            conf[sa] = 0
        elif r.start_src == "geometrie" or r.cause == "foul":
            conf[sa] = np.minimum(conf[sa], 1)
            if r.cause == "foul":
                conf[r.a:min(r.a + 3 * FPS, r.b)] = np.minimum(conf[r.a:min(r.a + 3 * FPS, r.b)], 1)
        else:
            conf[sa] = np.minimum(conf[sa], 1)
        sb = slice(max(r.b - near, 0), min(r.b + near, T))
        if r.restart_src == "implicite":               # restart not coded as a set piece: type uncertain
            conf[r.a:r.b] = np.minimum(conf[r.a:r.b], 1)
        conf[sb] = np.minimum(conf[sb], 0 if (r.retaken or r.restart_src == "implicite") else 1)
    conf[BS == -1] = 0
    return BS, stop, conf


def possession(ev, st, n0, T, BS):
    """Approximate per-frame possession from events (1 home, 2 away, -1 unknown)."""
    bp = np.full(T, -1, np.int8)
    e = ev[ev.type.isin(["PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE"]) &
           ev.team.notna()].sort_values("start", kind="stable")
    idx = np.clip(e.start.values.astype(int) - n0, 0, T - 1)
    lab = (e.team.values.astype(int) + 1).astype(np.int8)
    cur = np.full(T, -1, np.int8)
    cur[idx] = lab                                     # last event of a frame wins
    m = cur >= 0
    pos = np.where(m, np.arange(T), 0)
    np.maximum.accumulate(pos, out=pos)
    bp = np.where(np.maximum.accumulate(m), cur[pos], -1).astype(np.int8)
    for r in st.itertuples():                          # dead: team awarded the restart
        bp[r.a:r.b] = r.restart_team + 1 if r.restart_team >= 0 else -1
    bp[BS == -1] = -1
    return bp


def interp_flag(bxm, bym):
    """True on frames strictly inside an exactly linear run (provider interpolation between keyframes)."""
    T = len(bxm)
    d2 = np.hypot(np.diff(bxm, 2), np.diff(bym, 2))
    lin = np.zeros(T, bool)
    m = np.r_[False, np.nan_to_num(d2, nan=1.0) < LIN_TOL_M, False]
    e = np.diff(np.r_[0, m.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a >= LIN_MIN:
            lin[a:b] = True                            # m[k] = point k is between colinear neighbours
    return lin & np.isfinite(bxm)


# ============================================================================ convert
def convert(work):
    raw = os.path.join(work, "raw")
    os.makedirs(os.path.join(work, "npz"), exist_ok=True)
    tables, infos = [], {}
    for g in (1, 2, 3):
        G = read_csv_game(raw, g) if g < 3 else read_epts_game(raw, g)
        out, info = {}, dict(G["info"])
        for k, h in ((1, "h1"), (2, "h2")):
            m = G["period"] == k
            fr = G["frame"][m]
            n0, T = int(fr[0]), len(fr)
            assert np.all(np.diff(fr) == 1), "non contiguous frames"
            bxm, bym = G["bx"][m] * L_PITCH, G["by"][m] * W_PITCH
            ev = G["events"][G["events"].period == k]
            st, lo, hi, warn = build_stoppages(ev, bxm, bym, n0, T)
            BS, stop, conf = frame_labels(st, lo, hi, T)
            lin = interp_flag(bxm, bym)
            obs = np.isfinite(bxm) & ~lin
            out.update({f"{h}_PX": (G["x"][m] * L_PITCH - L_PITCH / 2).astype(np.float32),
                        f"{h}_PY": (G["y"][m] * W_PITCH - W_PITCH / 2).astype(np.float32),
                        f"{h}_team": G["team"].astype(np.int8),
                        f"{h}_BX": (bxm - L_PITCH / 2).astype(np.float32), f"{h}_BY": (bym - W_PITCH / 2).astype(np.float32),
                        f"{h}_BZ": np.full(T, np.nan, np.float32), f"{h}_BS": BS,
                        f"{h}_BP": possession(ev, st, n0, T, BS), f"{h}_n0": n0,
                        f"{h}_ball_obs": obs, f"{h}_ball_interp": lin, f"{h}_stop": stop, f"{h}_label_conf": conf})
            st.insert(0, "half", h)
            st.insert(0, "mid", f"metrica_g{g}")
            st["a_trim"], st["b_trim"] = st.a - lo, st.b - lo        # indices in dead_bounds.load_sportec halves
            st["dur_s"] = (st.b - st.a) / FPS
            tables.append(st)
            info[h] = dict(n0=n0, T=T, live_window=[int(lo), int(hi)], warnings=warn)
        np.savez_compressed(os.path.join(work, "npz", f"metrica_g{g}.npz"), L=L_PITCH, W=W_PITCH,
                            source=SOURCE + f"; Sample_Game_{g}", player_ids=np.array(G["names"]),
                            label_conf_doc=LABEL_CONF_DOC, stop_type_codes=json.dumps(STOP_NAMES), **out)
        infos[f"metrica_g{g}"] = info
        print(f"metrica_g{g}.npz written", {h: infos[f'metrica_g{g}'][h]['T'] for h in ('h1', 'h2')}, flush=True)
        del G, out
    allst = pd.concat(tables, ignore_index=True)
    allst.to_csv(os.path.join(work, "metrica_arrets.csv"), index=False)
    json.dump(infos, open(os.path.join(work, "convert_info.json"), "w"), indent=1, default=str)
    return allst, infos


# ============================================================================ checks
def runs_of(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def baseline_eval(true, pred):
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    tr, pr = runs_of(true), runs_of(pred)
    used, match = set(), 0
    for a, b in tr:
        best, bj = 0.0, None
        for j, (c, d) in enumerate(pr):
            if d <= a or c >= b or j in used:
                continue
            iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou > best:
                best, bj = iou, j
        if best >= 0.5:
            used.add(bj)
            match += 1
    p = match / max(len(pr), 1)
    r = match / max(len(tr), 1)
    return dict(tp=tp, fp=fp, fn=fn, f1_frame=f1, n_true=len(tr), n_pred=len(pr), n_match=match,
                prec_stop=p, rec_stop=r, f1_stop=2 * p * r / max(p + r, 1e-9))


def baseline_pred(obs, BS, min_s=2.0):
    pred = np.zeros(len(obs), bool)
    for a, b in runs_of(~obs):
        if b - a >= min_s * FPS:
            pred[a:b] = True
    return pred & (BS >= 0)


SPORTEC_REF = dict(per_match={"touche": 43.6, "coup_franc": 25.3, "six_metres": 17.0, "corner": 8.1,
                              "engagement": 2.6, "balle_arbitre": 2.1, "penalty": 0.4, "inconnu": 0.3},
                   median_s={"touche": 13.48, "coup_franc": 30.68, "six_metres": 22.08, "corner": 32.16,
                             "engagement": 70.3, "balle_arbitre": 55.88, "penalty": 119.0},
                   dead_share=0.445, stops_per_match=99.4,
                   note="7 Sportec matches, dead_v4 typology (sportec_arrets.csv) and LOMO true_share")


def check(work):
    import dead_bounds as db3
    st = pd.read_csv(os.path.join(work, "metrica_arrets.csv"))
    res = dict(sportec_ref=SPORTEC_REF, matchs={}, geometrie={}, precision_bornes={}, baseline={})
    geo_rows, prec_rows = [], []
    for g in (1, 2, 3):
        mid = f"metrica_g{g}"
        z = np.load(os.path.join(work, "npz", f"{mid}.npz"))
        H = db3.load_sportec(os.path.join(work, "npz", f"{mid}.npz"))   # compatibility test
        assert all(len(h["dead"]) == len(h["bx"]) == h["px"].shape[0] for h in H)
        m = dict(load_sportec_ok=True, shapes={h: list(z[f"{h}_PX"].shape) for h in ("h1", "h2")})
        s = st[st.mid == mid]
        m["n_arrets"] = int(len(s))
        m["par_type"] = s.type_name.value_counts().to_dict()
        m["duree_mediane_s"] = s.groupby("type_name").dur_s.median().round(2).to_dict()
        m["cause_debut"] = s.groupby(["type_name", "start_src"]).size().rename("n").reset_index().to_dict("records")
        tot = dict(live=0, dead=0, obs_live=0, obs_dead=0, fin_live=0, fin_dead=0, interp=0, fin=0,
                   conf={0: 0, 1: 0, 2: 0})
        bl = {k: dict(tp=0, fp=0, fn=0, n_true=0, n_pred=0, n_match=0) for k in ("ball_obs", "ball_finite")}
        for h in ("h1", "h2"):
            BS, obs = z[f"{h}_BS"], z[f"{h}_ball_obs"]
            fin = np.isfinite(z[f"{h}_BX"])
            bx, by = z[f"{h}_BX"] + L_PITCH / 2, z[f"{h}_BY"] + W_PITCH / 2
            lv, dd = BS == 1, BS == 0
            tot["live"] += int(lv.sum()); tot["dead"] += int(dd.sum())
            tot["obs_live"] += int((obs & lv).sum()); tot["obs_dead"] += int((obs & dd).sum())
            tot["fin_live"] += int((fin & lv).sum()); tot["fin_dead"] += int((fin & dd).sum())
            tot["interp"] += int(z[f"{h}_ball_interp"].sum()); tot["fin"] += int(fin.sum())
            for c in (0, 1, 2):
                tot["conf"][c] += int((z[f"{h}_label_conf"][BS >= 0] == c).sum())
            for key, o in (("ball_obs", obs), ("ball_finite", fin)):
                r = baseline_eval(dd, baseline_pred(o, BS))
                for k in bl[key]:
                    bl[key][k] += r[k]
            sh = s[s.half == h]
            T = len(BS)
            speed = np.r_[np.hypot(np.diff(bx), np.diff(by)) * FPS, np.nan]
            for r in sh.itertuples():
                # ---- start of out stoppages: ball beyond a line around the start
                w = slice(max(r.a - 12, 0), min(r.a + 25, T))
                xx, yy = bx[w], by[w]
                outx = (xx < 0) | (xx > L_PITCH)
                outy = (yy < 0) | (yy > W_PITCH)
                anyfin = bool(np.isfinite(xx).any())
                # first geometric exit near the event start (+-3 s): offset start
                w3 = slice(max(r.a - 75, 0), min(r.a + 75, T))
                o3 = (bx[w3] < 0) | (bx[w3] > L_PITCH) | (by[w3] < 0) | (by[w3] > W_PITCH)
                ex = np.flatnonzero(o3)
                exit_off = (ex[np.argmin(np.abs(ex + w3.start - r.a))] + w3.start - r.a) / FPS if len(ex) else np.nan
                # ---- restart point: first finite ball in [b, b+0.5 s]
                k = r.b + int(np.argmax(np.isfinite(bx[r.b:r.b + 13]))) if np.isfinite(bx[r.b:r.b + 13]).any() else None
                xr, yr = (bx[k], by[k]) if k is not None else (np.nan, np.nan)
                d_touch = min(abs(yr), abs(W_PITCH - yr))
                d_flag = min(np.hypot(min(xr, L_PITCH - xr), min(yr, W_PITCH - yr)), 99)
                gk_box = (min(xr, L_PITCH - xr) <= 6.0 + 1.0) and abs(yr - W_PITCH / 2) <= 9.16 + 1.0
                d_centre = np.hypot(xr - L_PITCH / 2, yr - W_PITCH / 2)
                d_spot = min(np.hypot(xr - 11, yr - W_PITCH / 2), np.hypot(xr - (L_PITCH - 11), yr - W_PITCH / 2))
                # ---- restart kick: first frame >= b-1 s with speed > 3 m/s after a still ball
                ws = slice(max(r.b - 25, r.a), min(r.b + 50, T))
                sp = speed[ws]
                kick = np.flatnonzero(sp > 3.0)
                kick_off = (kick[0] + ws.start - r.b) / FPS if len(kick) else np.nan
                # ---- independent check with the PLAYER tracking (not keyed on events): distance from the
                # restart spot to the nearest player over [b-2 s, b+2 s]; offset of its minimum vs b
                px, py = z[f"{h}_PX"] + L_PITCH / 2, z[f"{h}_PY"] + W_PITCH / 2
                if np.isfinite(xr):
                    wp = slice(max(r.b - 50, 0), min(r.b + 51, T))
                    with np.errstate(all="ignore"):
                        dn = np.nanmin(np.hypot(px[wp] - xr, py[wp] - yr), axis=1)
                    d_near_b = float(dn[r.b - wp.start]) if np.isfinite(dn[r.b - wp.start]) else np.nan
                    near_off = (int(np.nanargmin(dn)) + wp.start - r.b) / FPS if np.isfinite(dn).any() else np.nan
                    if r.type == 1:   # thrower: nearest player to the spot outside / on the touchline
                        jn = int(np.nanargmin(np.hypot(px[r.b] - xr, py[r.b] - yr))) if np.isfinite(px[r.b]).any() else None
                        thrower_line = (min(abs(py[r.b, jn]), abs(W_PITCH - py[r.b, jn])) if jn is not None else np.nan)
                    else:
                        thrower_line = np.nan
                else:
                    d_near_b = near_off = thrower_line = np.nan
                geo_rows.append(dict(d_nearest_player_at_restart=d_near_b, nearest_player_min_vs_restart_s=near_off,
                                     thrower_d_touchline=thrower_line,
                                     mid=mid, half=h, type_name=r.type_name, cause=r.cause, start_src=r.start_src,
                                     ball_fin_start=anyfin, out_x=bool(outx.any()), out_y=bool(outy.any()),
                                     exit_vs_start_s=exit_off, restart_ball_fin=k is not None,
                                     d_touchline=d_touch, d_flag=d_flag, gk_box=gk_box, d_centre=d_centre,
                                     d_spot=d_spot, kick_vs_end_s=kick_off,
                                     ball_obs_frac_dead=float(obs[r.a:r.b].mean())))
        m["part_frames_mortes"] = tot["dead"] / max(tot["live"] + tot["dead"], 1)
        m["balle_obs_vivant"] = tot["obs_live"] / max(tot["live"], 1)
        m["balle_obs_mort"] = tot["obs_dead"] / max(tot["dead"], 1)
        m["balle_finie_vivant"] = tot["fin_live"] / max(tot["live"], 1)
        m["balle_finie_mort"] = tot["fin_dead"] / max(tot["dead"], 1)
        m["part_balle_interpolee"] = tot["interp"] / max(tot["fin"], 1)
        n = sum(tot["conf"].values())
        m["label_conf_part"] = {str(c): v / n for c, v in tot["conf"].items()}
        m["minutes_fenetre"] = (tot["live"] + tot["dead"]) / FPS / 60
        res["matchs"][mid] = m
        for key in bl:
            b = bl[key]
            f1 = 2 * b["tp"] / max(2 * b["tp"] + b["fp"] + b["fn"], 1)
            p, rr = b["n_match"] / max(b["n_pred"], 1), b["n_match"] / max(b["n_true"], 1)
            res["baseline"].setdefault(key, {})[mid] = dict(f1_frame=f1, prec_frame=b["tp"] / max(b["tp"] + b["fp"], 1),
                                                            rec_frame=b["tp"] / max(b["tp"] + b["fn"], 1),
                                                            f1_stop_iou50=2 * p * rr / max(p + rr, 1e-9),
                                                            prec_stop=p, rec_stop=rr, n_true=b["n_true"], n_pred=b["n_pred"])
    G = pd.DataFrame(geo_rows)
    G.to_csv(os.path.join(work, "metrica_controles_geometrie.csv"), index=False)
    out = {}
    for t, d in G.groupby("type_name"):
        e = dict(n=len(d))
        if t in ("touche", "corner", "six_metres"):
            dd = d[d.ball_fin_start]
            e["balle_finie_debut"] = float(d.ball_fin_start.mean())
            e["hors_lignes_debut(-0.5s,+1s)"] = float((dd.out_x | dd.out_y).mean()) if len(dd) else None
            e["sortie_bon_cote"] = float((dd.out_y if t == "touche" else dd.out_x).mean()) if len(dd) else None
        e["balle_finie_reprise(+0.5s)"] = float(d.restart_ball_fin.mean())
        dr = d[d.restart_ball_fin]
        if t == "touche":
            e["reprise_a_<=1.5m_ligne_touche"] = float((dr.d_touchline <= 1.5).mean())
            e["d_ligne_touche_med_m"] = float(dr.d_touchline.median())
        if t == "corner":
            e["reprise_a_<=2m_poteau_corner"] = float((dr.d_flag <= 2.0).mean())
            e["reprise_a_<=3m_poteau_corner"] = float((dr.d_flag <= 3.0).mean())
            e["d_poteau_med_m"] = float(dr.d_flag.median())
        if t == "touche":
            e["lanceur_a_<=1.5m_ligne"] = float((dr.thrower_d_touchline.dropna() <= 1.5).mean())
        e["joueur_le_plus_proche_reprise_med_m"] = float(dr.d_nearest_player_at_restart.median())
        e["joueur_a_<=1.5m_reprise"] = float((dr.d_nearest_player_at_restart <= 1.5).mean())
        if t == "penalty":
            e["d_point_m"] = float(dr.d_spot.median())
        if t == "six_metres":
            e["reprise_dans_surface_but(+1m)"] = float(dr.gk_box.mean())
        if t == "engagement":
            e["reprise_a_<=1.5m_centre"] = float((dr.d_centre <= 1.5).mean())
        if t == "penalty":
            e["reprise_a_<=1m_point"] = float((dr.d_spot <= 1.0).mean())
        out[t] = e
    res["geometrie"] = out
    for t, d in G.groupby("type_name"):
        q = d.exit_vs_start_s.dropna()
        k = d.kick_vs_end_s.dropna()
        res["precision_bornes"][t] = dict(
            debut_sortie_geo_moins_evt_s=(dict(n=len(q), med=float(q.median()), p10=float(q.quantile(.1)),
                                               p90=float(q.quantile(.9))) if len(q) and t in ("touche", "corner", "six_metres") else None),
            joueur_min_moins_reprise_s=(lambda u: dict(n=len(u), med=float(u.median()), p10=float(u.quantile(.1)),
                                                       p90=float(u.quantile(.9))) if len(u) else None)(
                d.nearest_player_min_vs_restart_s.dropna()),
            frappe_moins_reprise_s=dict(n=len(k), med=float(k.median()), p10=float(k.quantile(.1)),
                                        p90=float(k.quantile(.9))) if len(k) else None)
    res["identite_matchs"] = match_identity(work)
    res["provenance_balle"] = ball_provenance(work)
    json.dump(res, open(os.path.join(work, "metrica_controles.json"), "w"), indent=1, default=float)
    return res


def ball_provenance(work):
    """Is the Metrica ball a tracker output? Share of ball frames inside exactly linear runs, share
    of ball frames sitting on a player (< 5 cm), share of event frames where the ball is present."""
    out = {}
    for g in (1, 2, 3):
        z = np.load(os.path.join(work, "npz", f"metrica_g{g}.npz"))
        n = dict(fin=0, lin=0, on=0, live=0, live_nan=0, dead=0, dead_fin=0)
        for h in ("h1", "h2"):
            bx, by, BS = z[f"{h}_BX"], z[f"{h}_BY"], z[f"{h}_BS"]
            fin = np.isfinite(bx)
            with np.errstate(all="ignore"):
                d = np.hypot(z[f"{h}_PX"] - bx[:, None], z[f"{h}_PY"] - by[:, None])
                d = np.where(np.isfinite(d), d, np.inf).min(axis=1)
            n["fin"] += int(fin.sum()); n["lin"] += int(z[f"{h}_ball_interp"].sum())
            n["on"] += int((fin & (d < 0.05)).sum())
            n["live"] += int((BS == 1).sum()); n["live_nan"] += int(((BS == 1) & ~fin).sum())
            n["dead"] += int((BS == 0).sum()); n["dead_fin"] += int(((BS == 0) & fin).sum())
        out[f"metrica_g{g}"] = dict(part_lineaire_exacte=n["lin"] / n["fin"], part_sur_un_joueur_5cm=n["on"] / n["fin"],
                                    part_vivant_sans_balle=n["live_nan"] / n["live"],
                                    part_mort_avec_balle=n["dead_fin"] / n["dead"])
    return out


def match_identity(work):
    """Is Sample_Game_3 a different match from games 1-2? Score from events, half lengths, events."""
    raw = os.path.join(work, "raw")
    out = {}
    for g in (1, 2, 3):
        ev = read_csv_game(raw, g)["events"] if g < 3 else read_epts_game(raw, g)["events"]
        tk = ev["sub"].map(_tokens)
        goal = ((ev.type == "SHOT") | (ev.type == "BALL OUT")) & tk.map(lambda t: "GOAL" in t)
        z = np.load(os.path.join(work, "npz", f"metrica_g{g}.npz"))
        ko = ev[(ev.type == "SET PIECE") & tk.map(lambda t: "KICK OFF" in t)]
        out[f"metrica_g{g}"] = dict(
            buts_domicile_exterieur=[int((goal & (ev.team == 0)).sum()), int((goal & (ev.team == 1)).sum())],
            minutes_buts=[round((int(f) - int(ko[ko.period == p].start.min())) / FPS / 60, 2)
                          for f, p in zip(ev[goal].start, ev[goal].period)],
            n_evenements=int(len(ev)), frames_par_mi_temps=[int(len(z["h1_BS"])), int(len(z["h2_BS"]))],
            n_joueurs=[int((z["h1_team"] == 0).sum()), int((z["h1_team"] == 1).sum())])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "convert", "check", "all"])
    ap.add_argument("--work", required=True, help="working dir OUTSIDE the repo (raw/, npz/, tables)")
    a = ap.parse_args()
    if a.cmd in ("download", "all"):
        download(a.work)
    if a.cmd in ("convert", "all"):
        convert(a.work)
    if a.cmd in ("check", "all"):
        r = check(a.work)
        print(json.dumps(r["baseline"], indent=1, default=float))


if __name__ == "__main__":
    main()

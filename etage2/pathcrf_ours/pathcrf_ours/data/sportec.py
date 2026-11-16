"""Sportec / IDSSE open data: tracking at 25 Hz in metres, episodes (in-play stretches) and phases.

Coordinates after loading: pitch 105 x 68 m, origin at the bottom-left corner, home team attacks
left -> right in BOTH halves (kloppy's STATIC_HOME_AWAY orientation), so the home keeper sits near
x = 0. Player columns are named home_<shirt>_x / _y and away_<shirt>_x / _y.

Episode = a stretch of consecutive in-play frames (ball_state == "alive") inside one period; a gap
of 5 or more frame ids starts a new one. Dead-ball frames get episode_id 0.
Phase = a stretch during which the set of players on the pitch does not change (cut at every
player's first frame, every player's last frame + 1, and every period start). Phases with fewer
than 100 in-play frames are dropped (phase_id 0).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import PITCH_X, PITCH_Y

PLAYER_X = re.compile(r"^(home|away)_\d+_x$")


# ------------------------------------------------------------------------------------------- files
def match_ids(data_dir: Path) -> List[str]:
    files = sorted((Path(data_dir) / "metadata").glob("*matchinformation*.xml"))
    return [re.search(r"DFL-MAT-(\w+)\.xml$", f.name).group(1) for f in files]


def _find(data_dir: Path, sub: str, key: str, match_id: str) -> Path:
    hits = [f for f in (Path(data_dir) / sub).iterdir() if key in f.name and match_id in f.name]
    assert len(hits) == 1, (sub, key, match_id, hits)
    return hits[0]


def load_lineup(meta_path: Path) -> pd.DataFrame:
    root = ET.parse(meta_path).getroot()
    rows = []
    for team in root.findall(".//Team"):
        side = "away" if team.attrib.get("Role") == "guest" else "home"
        players = team.find("Players")
        if players is None:
            continue
        for p in players.findall("Player"):
            shirt = int(p.attrib["ShirtNumber"])
            rows.append(
                {
                    "team_id": team.attrib.get("TeamId"),
                    "home_away": side,
                    "person_id": p.attrib.get("PersonId"),
                    "shirt": shirt,
                    "object_id": f"{side}_{shirt}",
                    "position": p.attrib.get("PlayingPosition"),
                    "starting": p.attrib.get("Starting") == "true",
                }
            )
    return pd.DataFrame(rows).sort_values(["home_away", "shirt"], ignore_index=True)


def load_tracking(match_id: str, data_dir: Path, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Raw 25 Hz tracking as a wide DataFrame (one row per frame). Cached as parquet when cache_dir is set."""
    cache = Path(cache_dir) / f"{match_id}.parquet" if cache_dir else None
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)

    from kloppy import sportec
    from kloppy.domain import Dimension, MetricPitchDimensions, Orientation

    meta = _find(data_dir, "metadata", "matchinformation", match_id)
    raw = _find(data_dir, "tracking", "positions", match_id)
    lineup = load_lineup(meta)

    ds = sportec.load_tracking(raw_data=str(raw), meta_data=str(meta), coordinates="sportec", only_alive=False)
    dims = MetricPitchDimensions(standardized=True, x_dim=Dimension(0, PITCH_X), y_dim=Dimension(0, PITCH_Y))
    ds = ds.transform(to_orientation=Orientation.STATIC_HOME_AWAY, to_pitch_dimensions=dims)
    df: pd.DataFrame = ds.to_df()

    rename = {}
    for pid, oid in zip(lineup["person_id"], lineup["object_id"]):
        for t in ("x", "y", "d", "s"):
            rename[f"{pid}_{t}"] = f"{oid}_{t}"
    df = df.rename(columns=rename)
    xcols = [c for c in df.columns if PLAYER_X.match(c)]
    df = df.dropna(subset=xcols, how="all").copy()
    df["timestamp"] = df["timestamp"].dt.total_seconds()
    df = df.reset_index(drop=True)
    df[["timestamp", "ball_x", "ball_y"]] = df[["timestamp", "ball_x", "ball_y"]].round(2)
    keep = ["period_id", "timestamp", "frame_id", "ball_state", "ball_owning_team_id", "ball_x", "ball_y"]
    if "ball_z" in df.columns:
        keep.append("ball_z")
    players = sorted({c[:-2] for c in xcols})
    keep += [f"{p}_{t}" for p in players for t in ("x", "y")]
    df = df[keep]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df


def load_synced_events(match_id: str, synced_dir: Path) -> pd.DataFrame:
    """ELASTIC-synchronised events (frame_id, period_id, episode_id, timestamp, player_id, spadl_type, success, score)."""
    ev = pd.read_parquet(Path(synced_dir) / f"{match_id}.parquet")
    return ev.dropna(subset=["frame_id"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------- structure
def players_of(tracking: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = tracking.dropna(axis=1, how="all").columns
    home = sorted({c[:-2] for c in cols if re.match(r"^home_\d+_x$", c)})
    away = sorted({c[:-2] for c in cols if re.match(r"^away_\d+_x$", c)})
    return home, away


def label_episodes(tracking: pd.DataFrame) -> pd.DataFrame:
    """Adds `episode_id` (0 = dead ball). Must be called on the frame-sorted full match."""
    tracking = tracking.sort_values(["period_id", "timestamp"], ignore_index=True).copy()
    tracking["episode_id"] = 0
    n_prev = 0
    for period in tracking["period_id"].unique():
        p = tracking[tracking["period_id"] == period]
        alive = p[p["ball_state"] == "alive"]
        if alive.empty:
            continue
        gaps = np.diff(alive["frame_id"].to_numpy(), prepend=-5)
        ids = (gaps >= 5).astype(int).cumsum() + n_prev
        tracking.loc[alive.index, "episode_id"] = ids
        n_prev = int(ids.max())
    return tracking


def label_phases(tracking: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adds `phase_id` (0 = unused) and returns the phase table (start/end frame, active players)."""
    t = tracking.set_index("frame_id") if "frame_id" in tracking.columns else tracking
    home, away = players_of(t)
    first, last_plus = [], []
    for p in home + away:
        x = t[f"{p}_x"].dropna()
        if not x.empty:
            first.append(int(x.index[0]))
            last_plus.append(int(x.index[-1]) + 1)
    period_starts = t.reset_index().groupby("period_id")["frame_id"].first().astype(int).tolist()
    cuts = np.sort(np.unique(first + last_plus + period_starts))

    rows = []
    for start, nxt in zip(cuts[:-1], cuts[1:]):
        end = int(nxt) - 1
        alive = t[t["ball_state"] == "alive"].loc[start:end]
        if len(alive) < 100:
            continue
        snap = alive.dropna(how="all", axis=1)
        act_home = [c[:-2] for c in snap.columns if re.match(r"^home_\d+_x$", c)]
        act_away = [c[:-2] for c in snap.columns if re.match(r"^away_\d+_x$", c)]
        rows.append(
            {
                "period_id": int(alive["period_id"].iloc[0]),
                "start_frame_id": int(start),
                "end_frame_id": end,
                "active_players": act_home + act_away,
            }
        )
    phases = pd.DataFrame(rows)
    phases.index = pd.RangeIndex(1, len(phases) + 1, name="phase")

    out = tracking.copy()
    fid = out["frame_id"] if "frame_id" in out.columns else out.index.to_series()
    out["phase_id"] = 0
    for phase, row in phases.iterrows():
        out.loc[fid.between(row["start_frame_id"], row["end_frame_id"]).to_numpy(), "phase_id"] = phase
    return out, phases


def detect_keepers(tracking: pd.DataFrame) -> Tuple[str, str]:
    """(left_gk, right_gk): per team the player whose mean x is farthest from the centre line."""
    home, away = players_of(tracking)
    hx = tracking[[f"{p}_x" for p in home]].mean()
    ax = tracking[[f"{p}_x" for p in away]].mean()
    home_gk = (hx - PITCH_X / 2).abs().idxmax()[:-2]
    away_gk = (ax - PITCH_X / 2).abs().idxmax()[:-2]
    if tracking[f"{home_gk}_x"].mean() < tracking[f"{away_gk}_x"].mean():
        return home_gk, away_gk
    return away_gk, home_gk


def team_of(player: str) -> str:
    return player.split("_")[0]

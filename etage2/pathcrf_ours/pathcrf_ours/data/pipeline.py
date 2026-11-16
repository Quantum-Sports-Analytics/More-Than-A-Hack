"""End-to-end preprocessing of one Sportec match into the "processed" table used for training/inference.

    raw XML --kloppy--> tracking (25 Hz)
        -> episodes (in-play stretches) -> touches (events + RDP corners) -> per-frame player label
        -> kinematics (vx, vy, speed, accel) -> phases -> processed parquet

Cached under <cache_dir>/processed/<match>.parquet and <cache_dir>/touches/<match>.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import DataConfig
from .features import add_kinematics, processed_columns
from .sportec import label_episodes, label_phases, load_synced_events, load_tracking
from .touches import frame_labels, match_touches


def process_match(match_id: str, data_dir: Path, synced_dir: Path, cache_dir: Path, cfg: DataConfig, force: bool = False) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    out = cache_dir / "processed" / f"{match_id}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    tracking = load_tracking(match_id, data_dir, cache_dir / "tracking")
    tracking = label_episodes(tracking)
    events = load_synced_events(match_id, synced_dir)
    if "episode_id" not in events.columns or events["episode_id"].isna().any():
        fid2ep = tracking.set_index("frame_id")["episode_id"]
        events["episode_id"] = events["frame_id"].map(fid2ep).fillna(0).astype(int)

    touches, affected = match_touches(tracking, events, cfg)
    tracking["player_id"] = frame_labels(tracking, touches)
    tracking = add_kinematics(tracking, cfg)
    tracking, _ = label_phases(tracking)
    tracking = tracking[processed_columns(tracking)].copy()

    (cache_dir / "processed").mkdir(parents=True, exist_ok=True)
    (cache_dir / "touches").mkdir(parents=True, exist_ok=True)
    tracking.to_parquet(out)
    touches.to_parquet(cache_dir / "touches" / f"{match_id}.parquet")
    (cache_dir / "touches" / f"{match_id}_sanitized_episodes.json").write_text(json.dumps(affected))
    return tracking


def load_matches(match_ids: List[str], data_dir: Path, synced_dir: Path, cache_dir: Path, cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    return {m: process_match(m, data_dir, synced_dir, cache_dir, cfg) for m in match_ids}

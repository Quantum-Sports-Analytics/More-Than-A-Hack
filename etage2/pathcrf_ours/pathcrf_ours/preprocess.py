"""Preprocess Sportec matches into the processed parquet tables.

    .venv/bin/python -m pathcrf_ours.preprocess            # all 7 matches
    .venv/bin/python -m pathcrf_ours.preprocess J03WR9     # one match
"""
from __future__ import annotations

import argparse
import time

from .config import DataConfig
from .data.pipeline import process_match
from .data.sportec import match_ids
from .paths import CACHE, SPORTEC_DIR, SYNCED_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matches", nargs="*", help="match ids; default all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sanitize", action="store_true", help="keep touches after an out/goal (official behaviour)")
    a = ap.parse_args()
    cfg = DataConfig(sanitize_after_out=not a.no_sanitize)
    ids = a.matches or match_ids(SPORTEC_DIR)
    for m in ids:
        t0 = time.time()
        df = process_match(m, SPORTEC_DIR, SYNCED_DIR, CACHE, cfg, force=a.force)
        n_ep = df.loc[df["episode_id"] > 0, "episode_id"].nunique()
        print(f"{m}: {len(df)} frames, {n_ep} episodes, {df['player_id'].notna().mean():.1%} frames labelled, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

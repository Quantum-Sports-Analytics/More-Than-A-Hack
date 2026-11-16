"""Phase B caches: SkillCorner processed tables and damaged draws of every training match.

    python -m pathcrf_ours.preprocess_b --variants 3 --dial 1.0 --workers 8
    python -m pathcrf_ours.preprocess_b --keys sc_1886347,J03WR9 --variants 1 --force

Writes cache/processed/sc_<id>.parquet (clean SkillCorner, same columns as the Sportec tables) and
cache/damaged/<key>_v<k>.parquet (k = 1..variants; the clean table with Metrica-like player damage, the
`<player>_present` flags and recomputed kinematics; float32). Draw k of match m uses the seed
(seed + 1000 * k + hash(m)), so the caches are reproducible. A damage_stats.json sits next to them.
"""
from __future__ import annotations

import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig
from .data.damage import DamageConfig, damage_match
from .data.pipeline import load_matches
from .data.skillcorner import all_match_ids, is_skillcorner, process_skillcorner
from .data.sportec import match_ids
from .paths import CACHE, SPORTEC_DIR, SYNCED_DIR


def _job(args: Tuple[str, int, float, int, bool]) -> dict:
    key, variant, dial, seed, force = args
    cfg = DataConfig()
    out = CACHE / "damaged" / f"{key}_v{variant}.parquet"
    if variant == 0:
        if is_skillcorner(key):
            df = process_skillcorner(key[3:], cfg, CACHE, force=force)
        else:
            df = load_matches([key], SPORTEC_DIR, SYNCED_DIR, CACHE, cfg)[key]
        return {"key": key, "variant": 0, "rows": len(df)}
    if out.exists() and not force:
        return {"key": key, "variant": variant, "cached": True}
    if is_skillcorner(key):
        proc = process_skillcorner(key[3:], cfg, CACHE)
    else:
        proc = load_matches([key], SPORTEC_DIR, SYNCED_DIR, CACHE, cfg)[key]
    s = (seed + 1000 * variant + zlib.crc32(key.encode())) % (2**32)
    dmg, stats = damage_match(proc, DamageConfig(dial=dial, seed=s), cfg, np.random.default_rng(s))
    for c in dmg.columns:
        if dmg[c].dtype == np.float64:
            dmg[c] = dmg[c].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    dmg.to_parquet(out)
    return {"key": key, "variant": variant, "seed": s, **stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=None, help="comma-separated match keys; default = all Sportec + all SkillCorner")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--dial", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    keys: List[str] = a.keys.split(",") if a.keys else list(match_ids(SPORTEC_DIR)) + [f"sc_{m}" for m in all_match_ids()]
    # clean tables first (SkillCorner processing writes the cache the damaged jobs read)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, [(k, 0, a.dial, a.seed, a.force) for k in keys]):
            print("clean", r, flush=True)
        jobs = [(k, v, a.dial, a.seed, a.force) for k in keys for v in range(1, a.variants + 1)]
        results = []
        for r in pool.imap_unordered(_job, jobs):
            print("damaged", r, flush=True)
            results.append(r)
    stats_f = CACHE / "damaged" / "damage_stats.json"
    stats_f.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(stats_f.read_text()) if stats_f.exists() else []
    stats_f.write_text(json.dumps(old + results, indent=1))


if __name__ == "__main__":
    main()

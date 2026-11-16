"""Shared paths, constants and geometry for the ball-in-the-air work.

Axes used everywhere in this folder (SkillCorner's convention): metres, centre spot at (0, 0), x along the pitch,
y across it. For Metrica, y grows towards the tactical camera (tracking y01 = 1 is the near touchline).
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
CACHE = HERE / "cache"
FIG = HERE / "report_figures"

DATA = Path(os.environ.get("MTAH_DATA_DIR") or (ROOT / "data.nosync" if (ROOT / "data.nosync").exists() else ROOT / "data"))
SC_DIR = Path(os.environ.get("SKILLCORNER_DIR",
                             Path.home() / "Documents/Quantum Football Event Detection.nosync/skillcorner-opendata"))
SC_SPLITS = Path(os.environ.get("SKILLCORNER_SPLITS",
                                Path.home() / "Documents/Quantum Football Event Detection.nosync/data/splits.json"))

FPS = 10                       # every model in this folder works at 10 frames per second
G = 9.81                       # m/s²
Z_AIR = 0.5                    # m: the label threshold asked for
Z_HIGH = 1.5                   # m: second, stricter label (header height in the SkillCorner project)
MIN_SPELL_S = 0.3              # s: air stretches and ground gaps shorter than this are noise


def runs(mask):
    """[start, end) of each run of True."""
    d = np.diff(np.r_[0, np.asarray(mask, bool).astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_runs(mask, min_len):
    """Close False gaps shorter than min_len between two True runs, then drop True runs shorter than min_len."""
    m = np.asarray(mask, bool).copy()
    r = runs(m)
    for (s0, e0), (s1, e1) in zip(r[:-1], r[1:]):
        if s1 - e0 < min_len:
            m[e0:s1] = True
    for s, e in runs(m):
        if e - s < min_len:
            m[s:e] = False
    return m


def project_to_grass(x, y, z, cam):
    """Where a camera at cam = (cx, cy, cz) sees a ball at (x, y, z) on the grass: the line of sight through the
    ball meets z = 0 at cam + (ball - cam) * cz / (cz - z)."""
    cx, cy, cz = cam
    z = np.minimum(np.nan_to_num(z), cz - 1.0)
    k = cz / (cz - z)
    return cx + (x - cx) * k, cy + (y - cy) * k


def load_camera():
    """Metrica tactical camera in the axes above (written by camera.py)."""
    d = json.loads((OUT / "camera.json").read_text())
    return np.array(d["camera_centred_m"], float)


def splits():
    return json.loads(SC_SPLITS.read_text())

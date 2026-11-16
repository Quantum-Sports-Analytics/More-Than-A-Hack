"""Export of the fused ball (balle_fusionnee.npz) as the per-frame CSV read by etage2 (airball/metrica_io.py).

One row per graded video frame (both halves): frame, video time, half, ball in metres, in normalised pitch
coordinates, in tactical-video pixels (provided homography), source and video confidence.
source: aucune (no ball), video (detected), video_interpolee (bridged), atd (tracked ball only).
Usage: python possession/export_ball_csv.py --ball outputs/intermediaires/balle_fusionnee.npz --data data \
           --out data/tactical/detections_balle_BAR-ATH.csv
"""
import argparse

import numpy as np
import pandas as pd

HALVES = ((11185, 90652), (110808, 190198))
LENGTH, WIDTH = 105.3, 68.0
SOURCES = np.array(["aucune", "video", "video_interpolee", "atd"], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.ball)
    hm = pd.read_csv(f"{a.data}/tactical/BAR-ATH_tactical_homography.csv").iloc[:, 1:10].values.reshape(-1, 3, 3)
    parts = []
    for half, (lo, hi) in enumerate(HALVES, start=1):
        f = np.arange(lo, hi)
        x, y = z["x"][f].astype(np.float64), z["y"][f].astype(np.float64)
        xn, yn = x / LENGTH, y / WIDTH
        p = np.einsum("nij,nj->ni", hm[f], np.stack([xn, yn, np.ones(len(f))], 1))
        src = z["source"][f]
        parts.append(pd.DataFrame({
            "frame": f, "temps_video_s": np.round(f / 25.0, 2), "mi_temps": half,
            "x_m": np.round(x, 2), "y_m": np.round(y, 2), "x_norm": np.round(xn, 5), "y_norm": np.round(yn, 5),
            "u_px": np.round(p[:, 0] / p[:, 2], 1), "v_px": np.round(p[:, 1] / p[:, 2], 1),
            "source": SOURCES[src], "conf_video": np.where(src == 1, np.round(z["video_conf"][f].astype(np.float64), 3), np.nan)}))
    pd.concat(parts, ignore_index=True).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

"""ATD ball projected to tactical-video pixels (crop centres for detect_ball_gpu.py).

Reads the provided ATD (normalised pitch coordinates) and the provided per-frame homography, and writes a
(199319, 2) float32 array of pixel positions (NaN when the ATD ball is missing or projects off-image).
Usage: python possession/atd_ball_px.py --data data --out atd_ball_px.npy
"""
import argparse

import numpy as np
import pandas as pd

N = 199319


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="atd_ball_px.npy")
    a = ap.parse_args()
    bx = np.full(N, np.nan, np.float64)
    by = bx.copy()
    with open(f"{a.data}/tactical/BAR-ATH_tactical_atd.txt") as fh:
        for line in fh:
            fid, _, ball = line.rstrip("\n").split(":")
            i = int(fid) - 1                      # ATD frame_id 1..N -> video frame 0..N-1
            x, y = ball.split(",")[:2]
            if x != "NaN":
                bx[i], by[i] = float(x), float(y)
    hm = pd.read_csv(f"{a.data}/tactical/BAR-ATH_tactical_homography.csv").iloc[:, 1:10].values.reshape(-1, 3, 3)
    p = np.einsum("nij,nj->ni", hm, np.stack([bx, by, np.ones(N)], 1))
    px = (p[:, :2] / p[:, 2:3]).astype(np.float32)
    ok = np.isfinite(px).all(1) & (px[:, 0] > -50) & (px[:, 0] < 1970) & (px[:, 1] > -50) & (px[:, 1] < 1130)
    px[~ok] = np.nan
    np.save(a.out, px)
    print("frames with an ATD ball pixel:", round(float(ok.mean()), 3))


if __name__ == "__main__":
    main()

"""Compares what reproduce.sh wrote with the copies committed in the repository. Exit code 1 if the submission differs."""
import argparse
import filecmp
import os
import sys

import numpy as np
import pandas as pd


def npz(a, b, keys=None):
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    worst, exact = 0.0, True
    for k in keys or y.files:
        u, v = x[k], y[k]
        if u.dtype.kind in "USO" or u.shape == ():
            exact &= str(u) == str(v)
            continue
        if u.shape != v.shape:
            return "DIFFERENT shape of " + k
        if not np.array_equal(u, v, equal_nan=True):
            exact = False
            if not np.array_equal(np.isnan(u.astype(float)), np.isnan(v.astype(float))):
                return "DIFFERENT (missing values) in " + k
            worst = max(worst, float(np.nanmax(np.abs(u.astype(float) - v.astype(float)))))
    return "identical" if exact else f"equal within {worst:.1e}"


def parquet(a, b):
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return "DIFFERENT shape"
    worst, exact = 0.0, True
    for c in y.columns:
        u, v = x[c].to_numpy(), y[c].to_numpy()
        if u.dtype.kind == "f":
            if not np.array_equal(u, v, equal_nan=True):
                exact = False
                worst = max(worst, float(np.nanmax(np.abs(u - v))))
        elif not (u == v).all():
            return "DIFFERENT in " + c
    return "identical" if exact else f"equal within {worst:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pcrf-ref", default="etage2/pathcrf_test/outputs/metrica_pcrf_phasebb_best_acc_v6.parquet",
                    help="committed PathCRF table to compare with (default: the older epoch-5 / Viterbi submission)")
    a = ap.parse_args()
    w, o = a.work, "outputs"
    def ref_missing(*paths):
        return next((p for p in paths if not os.path.exists(p)), None)

    checks = [
        ("ATD ball in pixels", [f"{o}/intermediaires/atd_ball_px.npy"], lambda: "identical" if np.array_equal(np.load(f"{w}/atd_ball_px.npy"), np.load(f"{o}/intermediaires/atd_ball_px.npy"), equal_nan=True) else "DIFFERENT"),
        ("fused ball: positions, sources", [f"{o}/intermediaires/balle_fusionnee.npz"], lambda: npz(f"{w}/balle_fusionnee.npz", f"{o}/intermediaires/balle_fusionnee.npz", ["x", "y", "xs", "ys", "source", "video_x", "video_y", "video_conf"])),
        ("fused ball: velocities", [f"{o}/intermediaires/balle_fusionnee.npz"], lambda: npz(f"{w}/balle_fusionnee.npz", f"{o}/intermediaires/balle_fusionnee.npz", ["vx", "vy", "atd_x", "atd_y"])),
        ("D v6 input features", ["models/etage1/bar_X5.npz"], lambda: npz(f"{w}/etage1/bar_X5.npz", "models/etage1/bar_X5.npz")),
        ("D v6 network outputs", ["models/etage1/bar_v6_pred.npz"], lambda: npz(f"{w}/etage1/bar_v6_pred.npz", "models/etage1/bar_v6_pred.npz")),
        ("D v6 stoppages (dead, type)", [f"{o}/etage1/dead_v6.npz"], lambda: npz(f"{w}/etage1/dead_v6.npz", f"{o}/etage1/dead_v6.npz", ["dead", "stop_type", "graded"])),
        ("TV signal", [f"{o}/etage1/tv_signal.npz"], lambda: npz(f"{w}/etage1/tv_signal.npz", f"{o}/etage1/tv_signal.npz")),
        ("stage 1 stoppage list", [f"{o}/etage1/stage1_v6_runs.csv"], lambda: "identical" if filecmp.cmp(f"{w}/etage1/stage1_v6_runs.csv", f"{o}/etage1/stage1_v6_runs.csv", shallow=False) else "DIFFERENT"),
        ("stage 1 D / L file", [f"{o}/etage1/stage1_v6_dead.csv"], lambda: "identical" if filecmp.cmp(f"{w}/etage1/stage1_v6_dead.csv", f"{o}/etage1/stage1_v6_dead.csv", shallow=False) else "DIFFERENT"),
        ("slot adapter table", ["etage2/pathcrf_test/outputs/metrica_virtual.parquet"], lambda: parquet(f"{w}/metrica_virtual.parquet", "etage2/pathcrf_test/outputs/metrica_virtual.parquet")),
        ("PathCRF probabilities", [a.pcrf_ref], lambda: parquet(f"{w}/metrica_pcrf.parquet", a.pcrf_ref)),
    ]
    rows = []
    for name, refs, fn in checks:
        gone = ref_missing(*refs)
        rows.append((name, "no reference file in this repository (not compared)" if gone else fn()))
    same = filecmp.cmp(f"{w}/submission.csv", a.ref, shallow=False)
    if not same:
        x, y = pd.read_csv(f"{w}/submission.csv")["label"].to_numpy(), pd.read_csv(a.ref)["label"].to_numpy()
        rows.append(("SUBMISSION", f"DIFFERENT: {(x != y).sum()} frames of {len(y)}"))
    else:
        rows.append(("SUBMISSION", "identical, byte for byte"))
    for name, res in rows:
        print(f"  {name:32s} {res}")
    lab = pd.read_csv(f"{w}/submission.csv")
    ok = len(lab) == 199319 and (lab["frame"].to_numpy() == np.arange(199319)).all() and set(lab["label"]) <= set("HADX")
    print(f"  {'file format (199319 rows, H/A/D/X)':32s} {'ok' if ok else 'INVALID'}")
    print("\nsubmission:", f"{w}/submission.csv", "| reference:", a.ref)
    sys.exit(0 if same and ok else 1)


if __name__ == "__main__":
    main()

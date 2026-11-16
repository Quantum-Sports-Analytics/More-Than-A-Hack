"""Stage-1 table (stage1.py) -> per-frame `frame,label` file with D (dead) / L (live), as read by
possession/v4_assemble.py --dead-csv.
Usage: python possession/export_dead_csv.py --stage1 outputs/etage1/stage1_v6.csv --out outputs/etage1/stage1_v6_dead.csv
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = pd.read_csv(a.stage1)
    pd.DataFrame({"frame": s["frame"], "label": np.where(s["dead"] == 1, "D", "L")}).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

"""One PathCRF probability table -> the 25 fps submission file, with the stage-1 dead ball.

Same decoding as score_all_v6.py for a single model at its fixed cost, written to stand alone (numpy and pandas
only) so the submitted file can be regenerated from one table:
  1. live stretches = 10 fps rows whose video frame is neither D nor X in the stage-1 file;
  2. P(blue) = P(home) seen from the team defending the left goal in that half; two-state Viterbi with a fixed
     price per team change inside each live stretch (cost 30 = the PathCRF convention, chosen beforehand);
  3. back to 25 fps (nearest 10 fps row), D on the stage-1 dead frames, D before the second-half kick-off
     (frames 110808..113741, published in the organisers' FAQ), X outside the graded frames.

  python pathcrf_test/make_submission.py --pcrf pathcrf_test/outputs/metrica_pcrf_phasebb_best_acc_v6.parquet \
      --dead-csv ../outputs/soumissions/v6.csv --cost 30 --out v6dead_pathcrf_phase_b_b.csv

Path route (2026-11-15, spellF1 option 1 + dense inference stride): with --path-team the label is the sender team of
the CRF's own decoded path (column team_pcrf_<dead-tag>, written by run_metrica --team-switch 3 --stride-div 4),
no second Viterbi; steps 1 and 3 unchanged, rows without a decoded team inside a live stretch stay X:
  python pathcrf_test/make_submission.py --pcrf pathcrf_test/outputs/metrica_pcrf_phasebb_ts3_s4.parquet \
      --dead-csv ../outputs/soumissions/v6.csv --path-team --out v6dead_pathcrf_path_ts3_s4.csv
"""
import argparse

import numpy as np
import pandas as pd

N_FRAMES = 199319
HALVES = {1: (11185, 90651), 2: (110808, 190197)}       # first and last graded frame of each half
KICKOFF_2ND = 113742


def viterbi(p, cost):
    """Most likely BLUE (0) / RED (1) sequence: emissions -log p, `cost` per switch (as pipeline/train_possession.py)."""
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    e = np.column_stack([-np.log(p), -np.log(1 - p)])
    n = len(p)
    back = np.zeros((n, 2), np.int8)
    acc = e[0].copy()
    for i in range(1, n):
        stay, switch = acc, acc[::-1] + float(cost)
        choose = switch < stay
        back[i] = np.where(choose, [1, 0], [0, 1])
        acc = np.minimum(stay, switch) + e[i]
    out = np.zeros(n, np.int8)
    out[-1] = int(np.argmin(acc))
    for i in range(n - 1, 0, -1):
        out[i - 1] = back[i, out[i]]
    return out


def decode_live(period, dead10, p, cost):
    colour = np.full(len(p), -1)
    for h in (1, 2):
        ix = np.flatnonzero(period == h)
        d = np.diff(np.r_[0, (~dead10[ix]).astype(int), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            colour[ix[s:e]] = viterbi(p[ix[s:e]], cost)
    return colour


def to_codes25(period, colour, blue_team, dead10, dead25):
    team = np.where(colour == 0, blue_team, 1 - blue_team)
    code10 = np.where(dead10, "D", np.where(colour < 0, "X", np.where(team == 0, "H", "A")))
    codes25 = np.full(N_FRAMES, "X", dtype=object)
    for p, (lo, hi) in HALVES.items():
        ix = np.flatnonzero(period == p)
        fr = np.arange(lo, hi + 1)
        k = np.clip(np.round((fr - lo) / 2.5).astype(int), 0, len(ix) - 1)
        codes25[lo:hi + 1] = code10[ix[k]]
    codes25[dead25] = "D"
    codes25[110808:KICKOFF_2ND] = "D"
    return codes25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcrf", required=True, help="metrica_pcrf_<tag>.parquet written by pathcrf_ours.run_metrica")
    ap.add_argument("--dead-csv", required=True, help="25 fps frame,label file whose D / X frames come from stage 1")
    ap.add_argument("--cost", type=float, default=30.0)
    ap.add_argument("--path-team", action="store_true", help="labels from the CRF path's team (team_pcrf_<dead-tag>) instead of the Viterbi on p_home")
    ap.add_argument("--dead-tag", default="v6", help="suffix of the alive_/p_home_pcrf_/team_pcrf_ columns in the parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = pd.read_csv(a.dead_csv)["label"].to_numpy().astype(str)
    assert len(lab) == N_FRAMES
    dead25, x25 = lab == "D", lab == "X"
    r = pd.read_parquet(a.pcrf)
    period, blue = r["period"].to_numpy(), r["blue_team"].to_numpy()
    vf = np.clip(np.round(r["video_frame"].to_numpy()).astype(int), 0, N_FRAMES - 1)
    dead10 = dead25[vf] | x25[vf]
    if a.path_team:
        team = r[f"team_pcrf_{a.dead_tag}"].to_numpy(dtype=float)          # 0 home, 1 away, NaN = no decoded step
        colour = np.where(np.isnan(team), -1, np.where(team == blue, 0, 1))
    else:
        col = [c for c in r.columns if c.startswith("p_home_pcrf")][0]
        ph = r[col].to_numpy()
        ph = np.where(np.isnan(ph), 0.5, ph)
        p_blue = np.where(blue == 0, ph, 1 - ph)
        colour = decode_live(period, dead10, p_blue, a.cost)
    codes = to_codes25(period, colour, blue, dead10, dead25)
    codes[x25] = "X"
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": codes}).to_csv(a.out, index=False)
    print("wrote", a.out, dict(zip(*np.unique(codes.astype(str), return_counts=True))))


if __name__ == "__main__":
    main()

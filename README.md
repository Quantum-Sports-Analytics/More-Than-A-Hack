# More than a Hack 2027 — possession on Barça–Athletic

Team **Quantum Sport Analytics**: Idriss Ben Mrad, Augustin Clot, Mathieu Bouteiller.

This repository turns the organisers' data bundle into our submission file: one label per frame of the tactical
video (`H`, `A`, `D` or `X`, 199,319 rows). One command rebuilds the file and checks it byte for byte against the
copy we submitted. The method is described in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Reproduce the submission

Needs Python 3.13 and about 12 minutes on a laptop CPU. No GPU.

```bash
git clone <this repository> && cd <this repository>
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put the organisers' files here (they are not in this repository and must not be committed)
mkdir -p data/tactical data/broadcast
#   data/tactical/BAR-ATH_tactical_atd.txt
#   data/tactical/BAR-ATH_tactical_atd.xml
#   data/tactical/BAR-ATH_tactical_homography.csv
#   data/broadcast/BAR-ATH_broadcast_atd.txt
#   data/broadcast/BAR-ATH_broadcast_atd.xml
#   data/broadcast/BAR-ATH_broadcast_homography.csv

./reproduce.sh
```

The script writes everything under `build/repro/` and ends with:

```
  SUBMISSION                       identical, byte for byte
  file format (199319 rows, H/A/D/X) ok
OK    submission.csv
```

- The rebuilt file is `build/repro/submission.csv`.
- The file we submitted is `etage2/pathcrf_test/outputs/v6dead_pathcrf_phase_b_b_ep10_ts3_s4_path.csv`.
- The last lines come from the organisers' own validator (`etage2/evaluation/validate_submission.py`).
- The intermediate comparisons print "no reference file in this repository": the intermediate tables are
  frame-level exports of the match data, so we do not publish them. Only the final file is compared.

## What the script runs

| step | script | what it does |
|---|---|---|
| 1 | `possession/atd_ball_px.py` | the organisers' ball track, projected to pixels with the homography |
| 2 | `possession/video_ball.py` | fused ball: our video detections + the organisers' ball, one track |
| 3 | `possession/dead_v6_infer.py` | dead-ball detector (temporal network, 3 seeds) |
| 4 | `possession/tv_signal.py`, `stage1.py`, `export_dead_csv.py` | TV signal from the two tracking files, then stage 1: stoppages merged with it |
| 5 | `possession/export_ball_csv.py` | ball file read by stage 2 |
| 6 | `etage2/pathcrf_test/metrica_adapter.py` | 46 tracking slots → 11 players per team |
| 7 | `etage2/pathcrf_ours/pathcrf_ours/run_metrica.py` | PathCRF on the live stretches |
| 8 | `etage2/pathcrf_test/make_submission.py` | team per frame + stage-1 dead ball → `frame,label` |
| 9 | `tools/check_repro.py`, organisers' validator | comparison with the submitted file, format check |

## What is provided as an intermediate file

One input is not recomputed by `reproduce.sh`. It is the output of our own code on the provided video, and the
code is in this repository.

| file | why it is provided | how to recompute it |
|---|---|---|
| `outputs/intermediaires/rfdetr_parts/` | ball detections on the tactical video: about 6 hours on a GPU, and not guaranteed bit-exact across GPUs | `possession/detect_ball_gpu.py`, see `outputs/intermediaires/README_rfdetr.md` |

## Trained weights

All weights were trained on external data only: Sportec / IDSSE (CC-BY 4.0), SkillCorner open data, and the Metrica
Sports sample data (github.com/metrica-sports/sample-data, acknowledged here as its README asks). Nothing was trained, fitted or tuned on the
ground truth or on the provided video.

| weights | model | training data | training code |
|---|---|---|---|
| `models/etage1/tcn_all_s{0,1,2}.pt` | dead-ball network (MS-TCN, 3 seeds) | Sportec / IDSSE (CC-BY 4.0), SkillCorner open data, Metrica sample data | `possession/dead_v6_tcn.py` |
| `models/etage1/model_v4_final.pkl` | our earlier dead-ball model (gradient boosting); loaded by the decoder with weight 0, so it does not change the output | Sportec / IDSSE | `possession/dead_v4.py` |
| `etage2/pathcrf_ours/outputs/runs/phasebb/model/best_acc_ep10.pt` | PathCRF, our re-implementation, with the ball as input | 6 Sportec / IDSSE matches + 17 SkillCorner open-data matches | `etage2/pathcrf_ours/pathcrf_ours/train.py` |
| RF-DETR Medium (downloaded by `detect_ball_gpu.py`) | ball detector, used as released | COCO (pre-trained by Roboflow, Apache-2.0) | none |

## Layout

- `possession/` — stage 1: ball fusion, dead-ball detector, TV signal.
- `etage2/` — stage 2: slot adapter, PathCRF (`pathcrf_ours/`), submission writer, the organisers' scorer and validator.
- `models/`, `outputs/` — weights, decoder settings, the ball detections.
- `tools/check_repro.py` — comparison run at the end of `reproduce.sh`.

`etage2/pipeline/keeper_slots.py`, `etage2/airball/` and `etage2/keeper_rule/` hold the helpers the slot adapter
imports: reading the tracking, finding each team's side, and the goalkeeper rule.

## Data

The match data is licensed for the competition only and is not in this repository. `data/` is git-ignored.

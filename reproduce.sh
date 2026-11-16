#!/usr/bin/env bash
# Regenerates the submission (PathCRF Phase B.B epoch 10, team taken from the CRF path with a team-switch penalty
# λ = 3 and a 2.4 s inference stride, + dead ball D v6) from the provided match data, end to end,
# and checks every intermediate file against the copy committed in this repository.
#
#   pip install -r requirements.txt          (Python 3.13; versions of the check of 2026-11-13)
#   ./reproduce.sh                           (CPU only, about 12 minutes on a laptop)
#   PYTHON=/path/to/python WORK=/tmp/repro ./reproduce.sh
#   CKPT=outputs/runs/<run>/model/<file>.pt ./reproduce.sh   (another PathCRF checkpoint, path inside etage2/pathcrf_ours;
#                                            the comparison of step 9 then only makes sense against that checkpoint's files)
#
# Needs data/tactical/ and data/broadcast/ with the organisers' files (ATD txt and xml, homography csv). Nothing is written outside
# $WORK (default build/repro).
#
# One input is taken as a provided intermediate file and is NOT recomputed here:
#   outputs/intermediaires/rfdetr_parts/   RF-DETR ball detections (GPU, ~6 h: possession/detect_ball_gpu.py,
#                                          see outputs/intermediaires/README_rfdetr.md)
# and the trained weights (models/etage1/, etage2/pathcrf_ours/outputs/runs/phasebb/) are used as they are.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; esac   # absolute: steps 6-7 run from etage2/
# PathCRF settings of the submission, all chosen on SkillCorner only (out of fold, 20 matches):
CKPT="${CKPT:-outputs/runs/phasebb/model/best_acc_ep10.pt}"   # Phase B.B epoch 10 (best validation edge accuracy)
TEAM_SWITCH=3   # penalty inside the CRF decode each time the ball goes to the other team
STRIDE_DIV=4    # a 10 s window every 2.4 s (window / 4) instead of every 5 s
WORK="${WORK:-$ROOT/build/repro}"
REF="$ROOT/etage2/pathcrf_test/outputs/v6dead_pathcrf_phase_b_b_ep10_ts3_s4_path.csv"
PCRF_REF="$ROOT/etage2/pathcrf_test/outputs/metrica_pcrf_phasebb_ep10_ts3_s4_v6.parquet"
PCRF_OUT="$ROOT/etage2/pathcrf_test/outputs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONWARNINGS="ignore"
cd "$ROOT"

for f in tactical/BAR-ATH_tactical_atd.txt tactical/BAR-ATH_tactical_atd.xml tactical/BAR-ATH_tactical_homography.csv \
         broadcast/BAR-ATH_broadcast_atd.txt broadcast/BAR-ATH_broadcast_atd.xml broadcast/BAR-ATH_broadcast_homography.csv; do
  [ -f "data/$f" ] || { echo "missing data/$f (organisers' data bundle)"; exit 1; }
done
mkdir -p "$WORK/etage1" "$WORK/data/tactical"
step() { echo; echo "== [$1/9] $2"; }

step 1 "ATD ball projected to pixels"
"$PY" possession/atd_ball_px.py --data data --out "$WORK/atd_ball_px.npy"

step 2 "fused ball (RF-DETR detections + ATD)"
"$PY" possession/video_ball.py --parts outputs/intermediaires/rfdetr_parts --data data --out "$WORK/balle_fusionnee.npz" | tail -2

step 3 "dead ball D v6: features, MS-TCN (3 seeds), decoding"
"$PY" possession/dead_v6_infer.py --data data --ball "$WORK/balle_fusionnee.npz" --models models/etage1 \
      --recipe outputs/etage1/decode_v6.json --out-dir "$WORK/etage1" | tail -1

step 4 "TV signal (broadcast tracking aligned on the tactical tracking, ~6 min) and stage 1 assembly"
"$PY" possession/tv_signal.py --data data --out "$WORK/etage1/tv_signal.npz" | tail -1
"$PY" possession/stage1.py --dead "$WORK/etage1/dead_v6.npz" --decode-json outputs/etage1/decode_v6.json \
      --tv "$WORK/etage1/tv_signal.npz" --out "$WORK/etage1/stage1_v6" > "$WORK/etage1/stage1_v6_resume.log"
"$PY" possession/export_dead_csv.py --stage1 "$WORK/etage1/stage1_v6.csv" --out "$WORK/etage1/stage1_v6_dead.csv"

step 5 "ball file read by stage 2"
for f in data/tactical/*; do ln -sf "$ROOT/$f" "$WORK/data/tactical/"; done
"$PY" possession/export_ball_csv.py --ball "$WORK/balle_fusionnee.npz" --data data \
      --out "$WORK/data/tactical/detections_balle_BAR-ATH.csv"

step 6 "slot adapter: 46 tracking slots -> 11 players per team"
(cd etage2 && MTAH_DATA_DIR="$WORK/data" "$PY" pathcrf_test/metrica_adapter.py --suffix _repro > "$WORK/metrica_adapter.log")
mv "$PCRF_OUT/metrica_virtual_repro.parquet" "$WORK/metrica_virtual.parquet"
rm -f "$PCRF_OUT/metrica_adapter_repro.json"

step 7 "PathCRF Phase B.B on the live stretches of stage 1"
cp "$WORK/metrica_virtual.parquet" "$PCRF_OUT/metrica_virtual_repro.parquet"
(cd etage2/pathcrf_ours && "$PY" -m pathcrf_ours.run_metrica --weights "$CKPT" \
      --virtual repro --tag repro --dead-csv "$WORK/etage1/stage1_v6_dead.csv" --dead-tag v6 \
      --team-switch "$TEAM_SWITCH" --stride-div "$STRIDE_DIV" > "$WORK/run_metrica.log")
head -1 "$WORK/run_metrica.log"
mv "$PCRF_OUT/metrica_pcrf_repro.parquet" "$WORK/metrica_pcrf.parquet"
rm -f "$PCRF_OUT/metrica_virtual_repro.parquet" "$PCRF_OUT/run_metrica_log_repro.json"

step 8 "decoding and submission file"
"$PY" etage2/pathcrf_test/make_submission.py --pcrf "$WORK/metrica_pcrf.parquet" \
      --dead-csv "$WORK/etage1/stage1_v6_dead.csv" --path-team --dead-tag v6 --out "$WORK/submission.csv"

step 9 "comparison with the committed files"
"$PY" tools/check_repro.py --work "$WORK" --ref "$REF" --pcrf-ref "$PCRF_REF"

echo; echo "== organisers' validator (etage2/evaluation/validate_submission.py)"
"$PY" etage2/evaluation/validate_submission.py "$WORK/submission.csv" | head -3

# Technical report — team Quantum Sport Analytics

Idriss Ben Mrad, Augustin Clot, Mathieu Bouteiller. More than a Hack 2027, phase 1: possession on Barça–Athletic.

## 1. Approach

We split the question in two, because the two halves fail for different reasons and can be trained on different
external data.

- **Stage 1 answers "is the ball in play?"** It marks every stoppage as `D`.
- **Stage 2 answers "which team has it?"** It runs only on the stretches stage 1 left in play, and labels them `H`
  or `A`.

Every model was trained and every setting was chosen on external matches: Sportec / IDSSE (7 matches), SkillCorner
open data (20 matches) and the Metrica sample data. Nothing was trained, fitted or tuned on the ground truth or on
the provided video.

### Stage 1: dead ball

1. **A better ball.** The provided ball track is missing on about 43 % of frames. We run a pre-trained detector
   (RF-DETR Medium, COCO weights, used as released) on the tactical video: the full frame, plus zoomed crops around
   the provided ball and around the best full-frame candidates. A tracker (`possession/video_ball.py`) then picks
   one path through the candidates by dynamic programming, removes fixed look-alikes such as spare balls and pitch
   marks, and falls back on the provided ball where the video has nothing. The fused ball covers about 89 % of the
   graded frames.
2. **A temporal network.** Five times per second we compute 217 hand-made features (ball position and speed
   relative to the lines, player spread, speeds, who is near the ball) and a small set encoder over the players of
   each team. A multi-stage temporal convolutional network (MS-TCN, 3 stages of 9 dilated layers) reads the whole
   half and outputs the probability that the ball is dead, where a stoppage starts and ends, and its type (throw-in,
   corner, goal kick, free kick, kick-off, penalty, drop ball). We average three seeds.
3. **Decoding.** A stoppage opens when the probability passes 0.95, stays open while it is above 0.5, and must last
   3 s. Its two ends are then moved to the peaks of the start and end outputs.
4. **TV signal.** The broadcast tracking shows only what the TV director shows. We align it in time with the
   tactical tracking by matching player positions, which tells us for each tactical frame whether the TV was live,
   off the pitch or on a replay. Replays and off-pitch shots mostly happen during stoppages, so this signal merges
   and extends stoppages the network already found. It uses no learning.

Trained without SkillCorner and tested on 19 SkillCorner matches, the detector finds stoppages with an F1 of 0.898
(a stoppage counts when it overlaps the true one with IoU ≥ 0.5), with 6.1 false stoppages per match.

### Stage 2: possession

1. **From 46 slots to 22 players.** The provided tracking has 46 slots whose identities come and go. A slot adapter
   links them into 11 tracks per team (assignment by position, gap filling, a rule that keeps one goalkeeper per
   team) and flags which positions are observed and which are filled.
2. **PathCRF.** We re-implemented PathCRF (Kim et al., 2026) in our own code. Every 0.2 s the network picks one edge
   between two players: a player to himself means he holds the ball, player A to player B means the ball is
   travelling from A to B. A conditional random field scores whole sequences of edges, so impossible chains are
   ruled out. The team in possession is the team of the sender. We trained it in three steps:
   - *Phase A*: the paper's recipe, players only, 5 clean Sportec / IDSSE matches, from random weights.
   - *Phase B*: the same network continued on 6 Sportec / IDSSE and 17 SkillCorner matches, with the players
     damaged in training (holes, identity swaps, straight-line fills, goalkeeper errors) and a "present" flag per
     player.
   - *Phase B.B*: Phase B continued with the ball as extra input for every player (distance and direction to the
     ball, ball speed, a "ball missing" flag), the ball being damaged in training too. This is the submitted model.
3. **Decoding.** Inside the CRF, each change of team costs a fixed penalty (λ = 3), and the 10 s inference windows
   start every 2.4 s instead of every 5 s. Both were chosen on SkillCorner matches only. λ was swept from −2 to 5 on
   the 20 matches, each scored by a model that had not seen it (two-fold training); λ = 3 was best in every fold.
   The denser windows were first chosen on the 3 SkillCorner validation matches, then confirmed on the same 20
   matches. Together they take the out-of-fold spellF1 from 0.756 to 0.782. The checkpoint (epoch 10) is the one
   with the best edge accuracy on the SkillCorner validation matches.
4. **Assembly.** Stage 1 decides `D` frame by frame; stage 2 fills the rest with `H` or `A`; frames outside the two
   halves are `X`.

### Why this rather than something else

- Possession is a sequence problem: the ranking metric counts whole possessions, so one wrong flip in the middle
  of a possession costs two possessions. A model that scores whole sequences fits that better than a per-frame
  classifier.
- The published PathCRF weights never see the ball and assume clean tracking. Retraining with damaged players and
  with the ball gave the largest gain on held-out external matches (edge accuracy 0.71 → 0.80 on the paper's test
  match).
- Splitting off the dead ball lets us use all three external sources for it, including those without reliable
  possession labels.

## 2. Assets used, and assets dropped

| asset | use |
|---|---|
| Tactical tracking | players for both stages; the provided ball as a fallback and as a guide for the detector's crops |
| Tactical homography | pixels ↔ pitch metres |
| Tactical video | ball detection with a pre-trained detector |
| Broadcast tracking and homography | TV signal (live / off pitch / replay) for stage 1 |
| Broadcast video | **not used** |
| Smart tagging | **dropped** |

What we tried and dropped:

- **Smart tagging.** We tried to use set-piece tags to anchor stoppages and to tell who restarts. The time windows
  are wide (35 s for throw-ins and goal kicks) and some tags do not match what happens in the video, so they added
  as many errors as they removed.
- **A touch model fused with PathCRF.** We tried a classifier of ball touches, combined with PathCRF by a learned
  fusion. It is not in the submitted chain.
- **A second, more permissive path for short stoppages** (quick throw-ins and free kicks). On external matches it
  found a few more stoppages and added as many false ones; the possession score did not move. It is in the code,
  switched off.
- **Other possession models**: the authors' PathCRF weights, a transformer that imputes the ball (TranSPORTmer
  style), gradient boosting, and blends of these with PathCRF. All scored below Phase B.B alone.
- **A ball-in-the-air model and a camera calibration** for ball height. Not used in the submitted chain.

## 3. Possession rules implemented

- **A team keeps the ball while it travels.** During a pass the label is the passer's team until an opponent
  becomes the sender. A loose ball therefore stays with the last team that played it.
- **A challenge changes possession only when the other team comes out with the ball.** The team-switch penalty in
  the CRF removes most one-step flips during duels. The price is that very short possessions (a tackle, a blocked
  clearance) are often merged into their neighbours.
- **Restarts.** From the moment stage 1 opens a stoppage until it closes it, the label is `D`. The restart itself
  (throw-in, free kick, goal kick) belongs to the team PathCRF sees sending the ball once play resumes.
- **Goalkeeper holding the ball** is in play and belongs to the goalkeeper's team.
- **Outside the two halves** the label is `X`. The start of the second half is `D` until the kick-off frame, which
  is a constant in the code.

## 4. Limitations

- **Short possessions.** The model finds few possessions shorter than 2 s. This is the main source of error we
  know of, and the touch model was our attempt at it.
- **Quick restarts.** A throw-in or free kick taken within about 3 s is usually not marked as a stoppage.
- **False stoppages.** A goalkeeper holding the ball for a long time, or a long pause in play, can be marked `D`.
- **Tracking identities.** Every stage-2 input goes through the slot adapter, which makes mistakes of its own when
  slots swap players.
- **Damage statistics.** To make external matches look like the provided tracking, the damage we apply in training
  (how long players disappear, how often slots swap, how often the ball is missing) follows statistics measured on
  the provided tracking. These statistics use no labels and no ground truth; we state it here so that it can be
  judged.
- **One match of the target kind.** All settings come from external matches recorded by other systems. We could
  not measure how well they transfer other than by inspection.
- **Reproducibility of the ball detections.** They are provided as an intermediate file (see the README): the
  detection takes about 6 hours on a GPU and is not guaranteed bit-exact across GPUs. Everything else is recomputed.

With more time: train the possession model to output touches directly, so that short possessions are learned
rather than fused in afterwards; and replace the slot adapter with a re-identification step.

## 5. Reproduction

```bash
pip install -r requirements.txt        # Python 3.13
# organisers' files in data/tactical/ and data/broadcast/
./reproduce.sh                         # about 12 minutes on a laptop CPU
```

The script rebuilds the submission from the provided data and the files in this repository, compares it byte for
byte with the file we submitted, and runs the organisers' validator. Details, the list of weights and the
intermediate file are in [README.md](README.md).

## References

- Kim et al., *PathCRF*, KDD 2026 (the possession model we re-implemented).
- Y. Abu Farha, J. Gall, *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*, CVPR 2019.
- RF-DETR, Roboflow (pre-trained COCO weights, Apache-2.0).
- Data: Sportec / IDSSE open tracking data (CC-BY 4.0); SkillCorner open data; Metrica Sports sample data
  (github.com/metrica-sports/sample-data), which we acknowledge as a source as its README asks. That repository has
  no formal licence file: it is one of the three sources of the dead-ball network, and we flag it here so the
  organisers can tell us if this use is not acceptable.

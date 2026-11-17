# BSD Team technical documentation

More than a Hack 2027 Phase 1 (FC Barcelona `H` vs Athletic Club `A`).
File `BSD.csv` for submission.
SHA-256 `aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450`.
Code `bsd/`.

## 1. Approach & why

The spellF1 metric prefers complete possessions with correct boundaries. It is determined by:
- **Alive/Dead boundaries**: every stoppage breaks up a possession.
- **Hand-overs**: one flicker frame gives two false possessions.

Thus, a **sequence model** is preferred: per-frame evidence for `H` / `A` / `D`,
then a decoder that only switches the label when the evidence for the switch exceeds a cost.

| approach | why not / why |
|---|---|
| Hand-crafted state machine over tracking and tagging | Dozens of thresholds required, which we cannot legally tune on this match. Our hand-crafted state machine (`bsd/heuristic.py`) fails clearly on our bench (spellF1 0.26 vs 0.54). |
| Custom ball/player detection from video | Hours per run. No way to know if it's worth it before the deadline. |
| Copying Smart Tagging intervals as labels | Tagging intervals are phases, not possessions. They start/end seconds away from possession boundaries. |
| **Learned per-frame evidence + Viterbi over {H, A, D}** | Matches spellF1's view of segments. Missing ball as weak evidence instead of failure. Only 5 decoder parameters. Legitimate training/validation set from other games with labelled data. **Chosen.** |

### Pipeline

1. **Tracking** (`bsd/io_epts.py`): player positions per slot. Slot team (from XML). Ball
   (metres: 105.3 × 68.0). `i` row = `frame_id − 1` = video frame `i`. Only decode the two
   graded halves (`games.half_video_blocks`). All other frames are `X`.
2. **Features** (`bsd/features.py`, 61 columns. All team-level features. No per-player features.):
   - **ball**: visible, out of play, distance to closest line, speed, static, last seen frames
     ago;
   - **players near ball**: closest distance to home/away, count < 2 m, count < 5 m;
   - **team shape**: centroid depth/width in its own attacking frame, forward velocity, tracked
     count;
   - **pace**: overall player speed, trend in player speed;
   - **hand-over cues**: ball acceleration, ball direction change, closest team to ball change,
     closest player < 2 m;
   - **context**: 1 s, 3 s rolling means.
3. **Per-frame classifier** (`bsd/model.py`): scikit-learn `HistGradientBoostingClassifier`.
   - Trained on degraded sample games (§3, §5). Every frame is also mirrored with swapped teams,
     making the evidence perfectly team-symmetric.
   - Output: evidence = log(p / prior).
4. **Hand-over detector** (`bsd/handover.py`): another classifier of the same type predicting
   "possession switches team within 0.5 s of this frame". Cost of switching H↔A at this frame:
   `max(0.5, 10 − 16·logit(p))`. Switch is cheap when hand-over is likely. Expensive otherwise.
5. **Smart Tagging** (`bsd/tagging.py`): small fixed-weight evidence (§4).
6. **Decoder** (`bsd/decode.py`): Viterbi over {H, A, D} per graded half. Maximise evidence -
   transition costs. Costs:
   - H↔A switch: per frame, from step 4;
   - alive → dead: 40;
   - dead → alive: 80.

Decoder costs are only chosen on mean spellF1 on a leave-one-game-out bench over the training
games. The hand-over detector improves the bench score from 0.544 to 0.598. It wins in all 6
held-out cases. No number fitted to the labels of this match.

## 2. Assets used/dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD (tracking, `*_atd.txt` + `.xml`) | **yes** | All per-frame features. Also label-free defect statistics (ball gap lengths, missing rate, tracked-player counts) which we degrade the training games to (§5). |
| Tactical Smart Tagging | **yes** | Small fixed-weight evidence (§4). |
| Tactical video | dropped | Could not build/validate our own detection before deadline. |
| Tactical homography | dropped | Only needed for video detection projection. ATD is already in pitch coordinates. |
| Broadcast video, ATD, homography | dropped | On its own unsynchronised timeline. Not worth aligning it to graded frames before deadline. |
| Public 5-minute GT | read only | Read to understand label conventions. Scored once for correctness. Nothing tuned on it. |
| DFL/IDSSE open Bundesliga data (Bassek et al., Scientific Data 2025, CC BY 4.0) | dropped | Tried as extra training data. Did not improve bench. Not used in submission. |
| Training tweaks (model ensembles, more boosting rounds, more data copies) | dropped | Tried. None improved bench enough. |

## 3. Training data

The only labelled data in the submission is **Metrica Sports public sample data**.
<https://github.com/metrica-sports/sample-data>. Pinned to commit
`e706dd506b360d69d9d123d5b8026e7294b13996`.

- **What**: 3 anonymised professional matches with tracking (players + ball, 25 fps) and event
  data. None of it is about the provided game.
- **Who labelled**: Metrica Sports produced the event data (1,745, 1,935, 3,620 events in
  games 1–3). We did not hand-label anything.
- **Our labels**: we derive the per-frame H/A/D labels from those events by code
  (`bsd/events.py`, rules in §4). No human chose any label.
- **How much**: 428,169 labelled frames (~4.8 h).
- **Provided game**: no hand labels. No event data. No corrected tracking. No rules/constants
  tied to specific frames of this match.

## 4. Possession rules

### Training labels (`bsd/events.py`)

- **Owner**: each possession-establishing event (pass, carry, recovery, shot, set piece) sets
  the owner from the start frame.
- **Loose balls / challenges**: losing the ball or a challenge without control doesn't change
  possession. Thus, loose/contested ball stays with the last team. Short control by other team (a
  recovery, however brief) is a possession.
- **Hand-over**: new team owns from their first control, not last pass of losing team.
- **Dead spell start**:
  - Out of play ball: first frame where ball is beyond the line.
  - Fouls: at whistle, even if ball still moving.
  - Goals.
- **Dead spell end**: restart touch (next set piece, goal kick). If no restart logged, ends at
  next control event.
- **Advantage**: foul followed directly by control event: no dead spell.
- **Restarts**: restarting team owns from restart frame. No neutral stretch.
- **Kick-off**: alive, owned by kicking team.

### Decoder application

- Loose ball or one-touch duel: don't change possession unless evidence for other team persists.
- Short stoppage: must be clearly dead to label `D`. Cost of entering/leaving: 120 nats.

### Smart Tagging rules (fixed weights)

| rule | bonus |
|---|---|
| Set piece | +2.0 on D 3 s before. +1.0 on tagged team first 2 s. |
| Goal | +2.0 on D until next kick-off (capped at 90 s). |
| Other team-tagged phase | +0.3 on team. |
| Before first `Kick Off (Start)` of period | +3.0 on D. |

## 5. Missing / noisy tracking

The ATD has 3 defects. Players are tracklets, not identities. Slots swap/fragment. Ball is
missing on ~41 % of graded frames. Long gaps. Positions are noisy. We handle them as follows:

- **Slot-agnostic features**: all features are team-level aggregates. No per-slot tracks. Thus,
  no identity swaps. If slot jumps faster than 12 m/s, it's an identity swap, not motion.
- **Missing ball as weak evidence**: we leave ball features as missing values (no filling in).
  Classifier handles them natively. Decoder carries the last confident label through a gap,
  unless evidence for switch outweighs cost.
- **Degrading training data** (`bsd/degrade.py`): sample games have clean tracking. Before
  training we make it look like this match's tracking:
  - hide ball in visible/missing runs resampled from this match's own gap length distribution
    (label-free statistic).
  - drop player observations to match this match's tracked count distribution.
  - add smooth position jitter.
  - shuffle slot identities.
- **No "missing ball means dead" shortcut**: ball is mostly missing during stoppages in sample
  games. Here, missing ball is mostly occlusion. We reweight the training rows so the ball is
  visible equally often under every label.

## 6. Limitations & what we'd do with more time

- **Under-segmentation**: we predict fewer, longer possessions than labels (bench: 259 vs 349 per
  game). Miss short duel possessions and quick hand-overs. Next up: duration-aware states.
- **Residual "ball missing ⇒ dead" cue**: most of it is removed by reweighting. But long
  occlusions may still be labelled dead.
- **Optimistic bench**: we tuned decoder costs on the same held-out cases that we report.
  Sample games have cleaner tracking than this match's.
- **Label definition gap**: our training labels are from event logs. Ground truth is from
  corrected tracking. Thus, different timing of boundaries.
- **Small training set**: 3 matches. Next up: more labelled games. Or self-training on this
  match's unlabelled frames.
- **Unvalidated Smart Tagging weights**: no tagging in sample games. Set-piece tags start a few
  seconds early. Short spurious H/A spells remain in the pre-kickoff stretch of the second half.
- **No video**: re-detecting ball during ~41 % of frames where it's missing is the largest
  untapped source of information.

## 7. Reproduction

See [REPRODUCE.md](../REPRODUCE.md) for details.

```bash
python3.14 -m venv venv && source venv/bin/activate
pip install -r requirements-bsd.txt
# unpack match bundle into data/ (DATA.md)
python -m bsd.fetch_sample
python -m bsd.run --check-sha aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450
```

Deterministic run. Every random draw seeded. Single-threaded training. All packages pinned in
`requirements-bsd.txt`. Sample data pinned to a commit. `--check-sha` fails if output differs.
Verified on macOS arm64, Python 3.14.6, clean checkout. Takes ~4 minutes.

1353 words





# Team BSD: technical document

More than a Hack 2027, Phase 1 (FC Barcelona `H` vs Athletic Club `A`). Submission file `BSD.csv`,
SHA-256 `aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450`. Code: `bsd/`.

## 1. Approach, and why

spellF1 rewards whole possessions with the right boundaries. Two things decide it:
- **alive/dead boundaries**, because every stoppage splits a possession;
- **hand-overs**, because one flickering frame creates two false possessions.

That favours a **sequence model**: per-frame evidence for `H` / `A` / `D`, then a decoder that
changes the label only when the evidence for the change outweighs a cost.

| approach | why not / why |
|---|---|
| Hand-written state machine over tracking and tagging | Needs dozens of thresholds that cannot legally be tuned on this match. Our hand-set version (`bsd/heuristic.py`) loses clearly on our bench (spellF1 0.26 vs 0.54). |
| Our own ball/player detection from video | Hours per run, and no way to measure its payoff before the deadline. |
| Copying Smart Tagging intervals as labels | Tagging intervals are phases, not possessions, and start or end seconds away from possession boundaries. |
| **Learned per-frame evidence + Viterbi over {H, A, D}** | Fits spellF1's segment view, treats a missing ball as weak evidence rather than a failure, and has only 5 decoder parameters. Labelled data from other games gives a legitimate training and validation set. **Chosen.** |

### Pipeline

1. **Tracking** (`bsd/io_epts.py`): player positions per slot, slot team (from the XML) and the
   ball, in metres (105.3 × 68.0). Row `i` = `frame_id − 1` = video frame `i`. Only the two graded
   halves (`games.half_video_blocks`) are decoded; every other frame is written `X`.
2. **Features** (`bsd/features.py`, 61 columns, all team-level, never per player):
   - **ball:** visible, out of play, distance to the nearest line, speed, static, frames since
     last seen;
   - **players near the ball:** nearest home/away distance, counts within 2 m and 5 m;
   - **each team's shape:** centroid depth and width in its own attacking frame, forward velocity,
     tracked count;
   - **pace:** overall player speed and its trend;
   - **hand-over cues:** ball acceleration and direction change, change of the team nearest the
     ball, closest player within 2 m;
   - **context:** 1 s and 3 s rolling means.
3. **Per-frame classifier** (`bsd/model.py`): a scikit-learn `HistGradientBoostingClassifier`.
   - It is trained on degraded sample games (§3, §5), with every frame also mirrored with the
     teams swapped, so the evidence is exactly team-symmetric.
   - Its output is evidence = log(p / prior).
4. **Hand-over detector** (`bsd/handover.py`): a second classifier of the same kind predicts
   "possession changes team within 0.5 s of this frame". Its probability `p` sets the cost of an
   H↔A switch at that frame: `max(0.5, 10 − 16·logit(p))`. Switching is cheap where a hand-over is
   likely and expensive elsewhere.
5. **Smart Tagging** (`bsd/tagging.py`): small fixed-weight evidence (§4).
6. **Decoder** (`bsd/decode.py`): Viterbi over {H, A, D} on each graded half, maximising evidence
   minus transition costs. Costs:
   - H↔A switch: per frame, from step 4;
   - alive → dead: 40;
   - dead → alive: 80.

All decoder costs were chosen only by mean spellF1 on a leave-one-game-out bench over the
training games. The hand-over detector raised that bench score from 0.544 to 0.598 and won on
all 6 held-out cases. No number was fitted to this match's labels.

## 2. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD (tracking, `*_atd.txt` + `.xml`) | **yes** | All per-frame features. Also the label-free defect statistics (ball gap lengths, missing rate, tracked-player counts) that the training games are degraded to match (§5). |
| Tactical Smart Tagging | **yes** | Small fixed-weight evidence (§4). |
| Tactical video | dropped | Our own detection could not be built and validated before the deadline. |
| Tactical homography | dropped | Only needed to project video detections; the ATD is already in pitch coordinates. |
| Broadcast video, ATD and homography | dropped | On its own unsynchronised timeline; aligning it to the graded frames was not worth it before the deadline. |
| Public 5-minute GT | read only | Read to understand the label conventions, and scored once as a correctness check. Nothing was tuned on it. |
| DFL/IDSSE open Bundesliga data (Bassek et al., Scientific Data 2025, CC BY 4.0) | dropped | Tried as extra training data; it did not improve the bench, so it is not used in the submission. |
| Training tweaks (model ensembles, more boosting rounds, more data copies) | dropped | Tried; none improved the bench enough. |

## 3. Training data

The only labelled data in the submission are the **Metrica Sports public sample data**,
<https://github.com/metrica-sports/sample-data>, pinned at commit
`e706dd506b360d69d9d123d5b8026e7294b13996`.

- **What:** three anonymised professional matches with tracking (players and ball, 25 fps) and
  event data. None of it is about the provided game.
- **Who labelled it:** Metrica Sports produced the event data (1,745, 1,935 and 3,620 events in
  games 1–3). We hand-labelled nothing.
- **Our labels:** per-frame H/A/D labels are derived from those events by code (`bsd/events.py`,
  rules in §4). No person chose any label.
- **How much:** 428,169 labelled frames (about 4.8 h).
- **Provided game:** no hand labels, no event data, no corrected tracking, and no rules or
  constants tied to specific frames of this match.

## 4. Possession rules

### Training labels (`bsd/events.py`)

- **Owner.** Each possession-establishing event (pass, carry, recovery, shot, set piece) sets the
  owner from its start frame.
- **Loose balls and challenges.** A lost ball, or a challenge without control, does not switch
  possession, so a loose or contested ball stays with the last team. A short control by the other
  team (a recovery, however brief) is a possession.
- **Hand-over.** The new team owns from its own first control, not from the losing team's last
  pass.
- **Dead spell start.**
  - Ball out of play: the first frame the ball is beyond the line.
  - Fouls: at the whistle, even while the ball still moves.
  - Goals.
- **Dead spell end.** At the restart touch (the next set piece or goal kick). If no restart was
  logged, the spell ends at the next control event.
- **Advantage.** A foul followed directly by a control event has no dead spell.
- **Restarts.** The restarting team owns from the restart frame; there is no neutral stretch.
- **Kick-off.** Alive and owned by the kicking team.

### How the decoder applies them

- A loose ball or a one-touch duel does not switch possession unless the other team's evidence
  persists.
- A short stoppage must be clearly dead to be labelled `D`: entering and leaving costs 120 nats.

### Smart Tagging rules (fixed weights)

| rule | bonus |
|---|---|
| Set piece | +2.0 on D over the 3 s before it, +1.0 on the tagged team over its first 2 s |
| Goal | +2.0 on D until the next kick-off (capped at 90 s) |
| Other team-tagged phase | +0.3 on that team |
| Before the first `Kick Off (Start)` of a period | +3.0 on D |

## 5. Missing and noisy tracking

The ATD has three defects. Players are tracklets, not identities: slots swap and fragment. The ball
is missing on about 41 % of graded frames, in long gaps. Positions are noisy. We handle them as
follows:

- **Slot-agnostic features.** Every feature is a team-level aggregate, never a per-slot track, so
  identity swaps do not matter. A slot that jumps faster than 12 m/s is treated as an identity
  swap, not motion.
- **Missing ball as weak evidence.** Ball features are left as missing values, never filled in,
  and the classifier handles them natively. The decoder carries the last confident label through a
  gap unless the other evidence outweighs the switch cost.
- **Degrading the training data** (`bsd/degrade.py`). The sample games' tracking is clean, so
  before training it is made to look like this match's:
  - the ball is hidden in visible/missing runs resampled from this match's own gap-length
    distribution (a label-free statistic);
  - player observations are dropped to this match's tracked-count distribution;
  - smooth position jitter is added;
  - slot identities are shuffled.
- **No "missing ball means dead" shortcut.** In the sample games the ball is mostly missing during
  stoppages. Here a missing ball is mostly occlusion. Training rows are reweighted so the ball is
  visible equally often under every label.

## 6. Limitations, and what we would do with more time

- **Under-segmentation.** We predict fewer, longer possessions than the labels (bench: 259 vs 349
  per game). Short duel possessions and quick hand-overs are missed. Duration-aware states would be
  next.
- **Residual "ball missing ⇒ dead" cue.** The reweighting removes most of it, but long occlusions
  may still be labelled dead.
- **Optimistic bench.** The decoder costs were tuned on the same held-out cases that are
  reported, and the sample games are cleaner than this match's tracking.
- **Label definition gap.** Our training labels come from event logs, while the ground truth comes
  from corrected tracking, so boundary timing differs.
- **Small training set.** Three matches. More labelled games, or self-training on this match's
  unlabelled frames, would be next.
- **Smart Tagging weights are unvalidated**, because the sample games have no tagging.
  Set-piece tags start a few seconds early, and short spurious H/A spells remain in the
  second-half pre-kickoff stretch.
- **No video.** Re-detecting the ball during the ~41 % of frames where it is missing is the
  largest untapped source of information.

## 7. Reproduction

Details in [REPRODUCE.md](../REPRODUCE.md):

```bash
python3.14 -m venv venv && source venv/bin/activate
pip install -r requirements-bsd.txt
# unpack the match bundle into data/ (DATA.md)
python -m bsd.fetch_sample
python -m bsd.run --check-sha aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450
```

The run is deterministic: every random draw is seeded, training is single-threaded, all packages
are pinned in `requirements-bsd.txt`, and the sample data are pinned to a commit. `--check-sha`
fails if the output differs. It was verified on macOS arm64 with Python 3.14.6, from a clean
checkout. It takes about 4 minutes.

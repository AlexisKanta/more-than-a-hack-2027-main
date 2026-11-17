# Design: frame-level possession via a learned sequence model

Date: 2026-11-16 · Competition: More than a Hack 2027, Phase 1 · Deadline: 2026-11-17

## Intent

Produce the highest possible **spellF1** on the private full-match ground truth for BAR-ATH
(FC Barcelona `H` vs Athletic Club `A`). The user's goal is to place as high as possible, so every
design decision is judged by "does it recover more whole possessions with the right boundaries?",
not by per-frame accuracy.

spellF1 scores possessions (maximal runs of `H` or of `A`, split by `D`) matched one-to-one by
temporal IoU, averaged over IoU 0.3–0.7. Two things therefore dominate:

1. **Alive/dead boundaries** — every stoppage splits a possession; missing or inventing one breaks
   two matches.
2. **Hand-overs** — a switch only when the other team gains *control*; flicker creates fake
   possessions and destroys the score.

## Goals

- Beat the published floor (spellF1 0.2433 full match) by as much as possible.
- A single command regenerates the submission byte-for-byte from the provided data plus public
  sample data.
- A short technical document (approach, assets used/dropped, training data, possession rules,
  limitations, reproduction).
- **Always hold a valid, validated submission file** from the first milestone onward, so a late
  failure never leaves us with nothing.

## Non-goals

- Re-detecting the ball or players from video; re-identification; using the broadcast feed.
- Predicting dead-ball ownership (the answer is always `D`).
- Optimising macroF1 beyond its role as tie-break.

## Anti-goals (would be a failure even if the score is good)

- Any **numeric** choice tuned on the public 5-minute ground truth (frames 11185–18684): no
  thresholds, costs, weights or model selection from its scores. It may be *read* to understand the
  label convention (see §2), which CHALLENGE.md and EVALUATION.md explicitly allow.
- Hand labels, hand corrections, or rules/constants tied to specific frames of this match (e.g.
  hard-coding the second-half kickoff frame). The published half boundaries
  (`games.half_video_blocks`) are game constants DATA.md publishes for use; they may set the decode
  windows and per-half direction inference, and mark ungraded frames `X`.
- Any human-produced data about this match (event data etc.).
- Match data, derived frame-level exports, or submission CSVs committed to the public repo; match
  data sent to any cloud service or hosted LLM.
- Non-determinism: unseeded randomness, or output depending on library versions not pinned.

## Constraints

- Deadline 2026-11-17 (cutoff time per organisers' email).
- Local Python: `more-than-a-hack-2027-main/venv/` (Python 3.14). numpy/pandas are present; adding
  pinned dependencies (e.g. scikit-learn, scipy) is allowed.
- Rules require the organisers can inspect everything and the document explains it. External data
  must be about other games — the Metrica sample games qualify.

## What the data looks like (measured, label-free)

- Ball visible on 59.3% of graded frames. Missing-ball stretches are **long**: 150 gaps, median
  ~268 frames (~11 s), p90 ~845 frames (~34 s), max 5537 frames (~3.7 min). During gaps,
  possession must be inferred from the **players' collective shape and motion**, not the ball.
- Ball coordinates leave `[0,1]` on ~4% of visible graded frames (x −0.07…1.06, y −0.11…1.34), so
  out-of-play is directly observable when the ball is tracked.
- The ATD's ball **speed field looks wrong** (median 1.3 m/s, p99 5.9 m/s; real passes are
  10–25 m/s). Compute ball velocity from positions instead.
- Median 20 tracked objects (5th percentile 11). Slots are tracklets with team membership only.
- Mean player speed varies 0.7–3.1 m/s (p10–p90): stoppages should show as collective slowdowns.
- Smart Tagging: 638 intervals, set pieces with `Type` (throw-ins have `Team = N/A`), goals, and
  team-tagged phases. Automatic, with errors.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | Hand-written state machine over tracking + tagging | Fast, explainable, but dozens of thresholds that cannot legally be tuned on this match |
| **B** | **Probabilistic sequence model (HMM-style) over H/A/D** | Matches spellF1's segment view, handles missing ball as weak evidence; few key parameters; more upfront work |
| C | Supervised model trained on other matches | Only option with a legitimate validation set; domain gap (clean tracking there) |

Dropped: own video ball detection (hours per run, unmeasurable payoff before the deadline); copying
Smart Tagging intervals as labels (phases ≠ possessions).

**Chosen (by the user): B, with C's sample games as the training/validation bench.** A per-frame
evidence model is learned on Metrica's public sample games, and the sequence decoder's switching
costs are tuned on those games for spellF1. Nothing is tuned on BAR-ATH labels.

## Design

### 1. Data layer
- One fast streaming parser for FIFA/EPTS ATD (`frame:players:ball`, groups `x,y,speed`, `NaN` when
  missing) producing arrays: players `(frames, slots, 2)` positions, slot team (from metadata XML),
  ball `(frames, 2)`. Coordinates converted to metres (BAR-ATH: 105.3 × 68.0; sample games: per
  their metadata). Cached as `.npz` in a gitignored cache folder.
- The same parser reads **Sample Game 3** (EPTS). A second loader reads Sample Games 1–2 (Metrica
  CSV, one file per team) into the same arrays.
- Sample games are downloaded by a script from
  `https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{1,2,3}/…`
  into the gitignored data folder, pinned to a specific commit hash for reproducibility.

### 2. Labels for the sample games (training/validation only)
Derived mechanically from their event data, mirroring the competition definition (one team has the
ball while alive; loose balls stay with the last team; dead between out-of-play and restart):
- **Dead** from the end of a `BALL OUT` event (and fouls / goals / end of period) until the start of
  the next `SET PIECE` (kick-off, throw-in, corner, free kick) or goal-kick pass.
- **Alive** otherwise; team = team of the most recent possession-establishing event (pass,
  carry/reception, recovery, set piece). Single touches that are immediately lost (`BALL LOST`
  /`CHALLENGE` without follow-up) do not switch possession — the exact rule should follow
  CHALLENGE.md's "controlled possession" description.
- **Convention check, early in milestone 2, before any training or tuning.** CHALLENGE.md calls the
  public 5 minutes "the reference" for what "dead" and "owning" mean. Read it alongside the BAR-ATH
  tracking and video for those minutes to settle the qualitative conventions: does dead start when
  the ball crosses the line or later; does a restart become alive at the restart touch; does a
  contested single touch switch possession. Encode the answers as general rules in the sample-game
  label builder. No number (threshold, cost, length) is fitted to the public window.
- If the public window doesn't settle a convention, default to CHALLENGE.md's wording: possession
  switches only on controlled possession (a reception followed by a further action by the same
  team, or a recovery followed by control), and loose/contested balls stay with the last team.

### 3. Degrading the sample games to look like BAR-ATH
Sample tracking is clean; BAR-ATH's is not. Before computing features on sample games:
- Remove the ball in **long stretches** whose lengths are drawn from BAR-ATH's own measured
  missing-ball gap distribution (a label-free statistic), to the same ~41% missing rate.
- Drop player observations to match BAR-ATH's tracked-object count distribution; add small
  position jitter; shuffle slot assignment across time (since BAR-ATH slots are not identities).
- Seeded; several degraded copies per game for training.

### 4. Per-frame features (identical code for all games)
Team-level, slot-agnostic, all in metres:
- **Ball (when visible):** position; inside/outside pitch (with margin); velocity from positions
  (smoothed); distance from nearest home and nearest away player; counts of each team within a few
  radii; frames since ball last seen; last seen position.
- **Collective shape and motion (always available):** each team's centroid, spread/compactness,
  mean speed, mean velocity along the attacking direction, defensive-line depth; between-team
  relations (which team is deeper/higher, who is moving forward). These carry possession through
  long ball gaps.
- **Stoppage cues:** overall player speed and its recent trend; ball stationary; ball out of bounds;
  players clustered as at a set piece.
- **Temporal context:** rolling summaries over a few seconds before and after each frame.
- **Attacking direction** per team per half, inferred from the data (e.g. average team positions),
  never hard-coded.

### 5. Per-frame evidence model
A classifier (gradient-boosted trees or logistic regression; builder's choice) trained on degraded
sample games to output `P(H), P(A), P(D)` per frame. Home/away is symmetric, so train on
"team-of-interest vs other" with mirrored copies to double data and avoid side bias.

### 6. Sequence decoder
Viterbi decoding over each published graded half (`games.half_video_blocks`) over states `{H, A, D}` (optionally with duration-aware sub-states):
- Emission score = classifier log-probability (divided by class prior, hybrid-HMM style).
- Transition costs: `H↔A` switch cost, alive→dead and dead→alive costs, optional minimum durations.
- These few parameters are tuned on held-out sample games to maximise spellF1, using the official
  `evaluation/possession_metrics.py` functions so the objective is exactly the competition's.

### 7. Smart Tagging (BAR-ATH only)
Not available for sample games, so its contribution cannot be validated. It enters as a modest,
fixed-weight adjustment to the decoder's emissions/transitions:
- Set-piece start → strong evidence that the preceding frames are `D` and the ball becomes alive
  at the restart; team tag (when not `N/A`) favours that team just after.
- Goal → `D` until the following kick-off.
- Phase intervals with a team tag → mild evidence for that team.
The weight is chosen from first principles and kept conservative; the technical doc states this.
Variant with tagging off is also produced for comparison of possession counts (not scores).

### 8. Validation protocol
- Leave-one-game-out across the three sample games (each game held out in turn, trained on the
  other two, degraded the same way). Report spellF1 and macroF1 with the official scorer functions.
- All model/feature/decoder choices are made on this bench only.
- Sanity checks on BAR-ATH that use no labels: possession count and mean duration vs typical
  football values; dead-ball share; visual spot-checks of transitions against the video
  (watching to debug is allowed; turning observations into frame-specific rules is not).
- Public 5 minutes: run once at the end through `run_evaluation.py` to confirm alignment and
  labels mean what we think. Not used to choose between versions.

### 9. Output
- Labels for every frame; `X` outside the published graded halves; within halves, the decoder's
  `H/A/D`. The second half's pre-kickoff stretch must come out `D` from the model's own cues
  (kick-off tag, static players at the centre circle), never from a hard-coded frame. `X` is not used inside halves in v1 (the scorer forward-fills it anyway; a guess at
  hand-overs is better than abstaining).
- Written to `evaluation/results/BSD.csv`, then `validate_submission.py` is run on that
  exact file.

### 10. Repository and deliverables
- Code lives in a new package inside `more-than-a-hack-2027-main/` (the organisers' repo, whose
  `.gitignore` already protects data and results), which becomes our public repository.
- Pinned `requirements` for our extra dependencies; one entry command that downloads sample data
  (pinned commit), trains, decodes BAR-ATH, writes and validates the CSV.
- Determinism, aimed at **any machine**, not just ours (the organisers may re-run it): fixed seeds;
  single-threaded training and inference; classifier probabilities rounded (e.g. to 1e-6) before
  decoding so floating-point noise can't flip near-ties; exact Python and package versions pinned
  and stated; output SHA-256 recorded in the README and checked by the run command. If time allows,
  confirm the hash in a second fresh environment.
- Technical document (Markdown → PDF) covering the SUBMISSION.md checklist, crediting Metrica's
  sample data.

### 11. Milestones (each ends with a valid submission on disk)
1. **Parser + features on BAR-ATH**, plus a stronger *unlearned* decoder (hand-set evidence:
   nearest-team-to-ball + out-of-bounds + slowdown cues, Viterbi with sensible switch costs).
   Submit-ready fallback.
2. **Sample-game bench**: loaders, event-derived labels, degradation, leave-one-out scoring.
   Re-score milestone 1's decoder on it.
3. **Learned evidence model** + decoder tuning on the bench; adopt only if it beats milestone 1
   on the bench.
4. **Smart Tagging** integration (conservative weights).
5. **Repo, reproducibility check (clean run reproduces the same file hash), technical doc,
   final validation.**

## Decided by the user
- Goal: place as high as possible on spellF1.
- Approach: B (sequence model) with Metrica sample games as the training/validation bench.
- Team size and time budget are not to be planned around.
- Team name: **BSD** → deliverables `BSD.csv` and `BSD.pdf` (or `BSD.md`).
- **No pushing to git.** Milestone 5 ends at "ready to push": the repo may be initialised and
  checked locally (no data or CSVs tracked), but nothing is pushed or published by Claude.

## Builder's calls (made in this design, open to correction)
- Package location inside the organisers' repo, which becomes the public repo.
- Classifier family, exact feature list and radii, number of degraded copies.
- Whether to use duration-aware states.
- Tagging weights (conservative, first-principles).
- No in-half `X` in v1.
- Public 5 minutes read early for label *conventions* only (docs explicitly permit this).
- Half boundaries used as decode windows (published game constants, not labels).
- Reproducibility targeted cross-machine (single-threaded, rounded probabilities, pinned versions).

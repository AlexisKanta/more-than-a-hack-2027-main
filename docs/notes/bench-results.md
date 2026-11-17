# Bench results: heuristic vs learned evidence, decoder tuning (Task 9)

All numbers come from the leave-one-game-out bench on the three Metrica sample games
(`python -m bsd.bench --mode learned`). Nothing here was tuned on or scored against BAR-ATH or
its public ground-truth window.

**These are in-sample tuned figures.** For each emitter the decoder costs are the grid point
with the best mean spellF1 over the same 6 held-out cases that are reported, so each number is
the maximum of a 480-720-point search and is optimistic by the selection effect (the emitters
themselves are trained out of fold; only the 3 decoder costs are tuned in sample). The grid is
flat near the optimum (see Observations), so the bias is probably small, but these are not
clean held-out estimates.

## Protocol (current, after ruling R13)

- Held-out game `g` in {1, 2, 3}, degraded with seed `s` in {0, 1} (`bsd.degrade`): 6 cases.
- `bsd.degrade` never invents ball positions: a frame without a source ball has no ball after
  degrading (in the sample games that is ~95 % of dead frames), on top of the label-free
  BAR-ATH-like visibility mask. Training and bench test data use the same degraded output.
- Learned emitter of fold `g`: trained once on the other two games, each degraded with seeds
  100, 101, 102 (distinct from the held-out seeds) and mirrored (home/away columns and H/A labels
  swapped); X-label frames dropped; every 3rd frame. Rows are weighted so that the ball is
  visible on 59.3 % (BAR-ATH's in-period rate, `1 - gap_profile()["missing_rate"]`) of each
  label's rows (`bsd.model.visibility_weights`), so a missing ball is no dead-ball cue.
  `HistGradientBoostingClassifier` (150 rounds, 31 leaves, min 100 samples/leaf, L2 1.0, no
  early stopping, `random_state=0`, one thread). Evidence = `log(p / prior)`, with the prediction
  averaged with the swapped prediction of the mirrored frame (exact team symmetry), rounded to
  6 decimals.
- Heuristic emitter: `bsd.heuristic.heuristic_logp`, unchanged (no training).
- Each emitter's held-out evidence is computed once and cached; the decoder grid
  `switch_ha` {5, 10, 20, 40, 80} x `alive_to_dead` {5, 10, 20, 40} x `dead_to_alive`
  {5, 10, 20, 40} is decoded from it. An axis whose best value lies on the grid edge is extended
  once (double the top / halve the bottom) and the search repeated. Objective: mean held-out
  spellF1 over the 6 cases (ties: mean macroF1, then the earlier grid point).

## Results after R13 (current; 9 min 11 s)

### Heuristic, tuned: switch_ha 5, alive_to_dead 5, dead_to_alive 10

First pass hit the bottom edge of `switch_ha` and `alive_to_dead`; both were extended to 2.5
once, and 5 stayed best.

| game | seed | spellF1 | macroF1 | n poss | mean poss s | GT n poss | GT mean poss s |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.2542 | 0.5073 | 248 | 21.33 | 353 | 9.84 |
| 2 | 0 | 0.2621 | 0.4681 | 233 | 22.89 | 318 | 11.54 |
| 3 | 0 | 0.2322 | 0.5289 | 293 | 18.46 | 377 | 10.87 |
| 1 | 1 | 0.2634 | 0.5029 | 268 | 20.00 | 353 | 9.84 |
| 2 | 1 | 0.2555 | 0.4656 | 244 | 21.78 | 318 | 11.54 |
| 3 | 1 | 0.2728 | 0.5309 | 280 | 19.41 | 377 | 10.87 |
| **mean** | | **0.2567** | **0.5006** | 261.0 | 20.64 | 349.3 | 10.75 |

### Learned, tuned: switch_ha 20, alive_to_dead 20, dead_to_alive 40

First pass hit the top edge of `dead_to_alive`; it was extended to 80 once, and 40 stayed best.

| game | seed | spellF1 | macroF1 | n poss | mean poss s | GT n poss | GT mean poss s |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.5246 | 0.8352 | 241 | 14.82 | 353 | 9.84 |
| 2 | 0 | 0.5800 | 0.8349 | 233 | 15.19 | 318 | 11.54 |
| 3 | 0 | 0.5309 | 0.8446 | 283 | 14.81 | 377 | 10.87 |
| 1 | 1 | 0.5498 | 0.8383 | 256 | 14.02 | 353 | 9.84 |
| 2 | 1 | 0.5587 | 0.8285 | 239 | 14.88 | 318 | 11.54 |
| 3 | 1 | 0.5201 | 0.8418 | 289 | 14.39 | 377 | 10.87 |
| **mean** | | **0.5440** | **0.8372** | 256.8 | 14.69 | 349.3 | 10.75 |

## Decision (current)

**Adopted: learned** (mean spellF1 0.5440 vs 0.2567, macroF1 0.8372 vs 0.5006; the learned
emitter wins on every one of the 6 cases). Recorded in `bsd/decode_params.json` (committed,
read by `bsd.run`; copy in `cache/decode_params.json`), so `python -m bsd.run` defaults to
`--mode learned`.

## Observations (current)

- Removing the fake ball cost the learned emitter little (spellF1 0.5578 -> 0.5440, macroF1
  0.8531 -> 0.8372): its gain over the heuristic does not come mainly from the interpolated
  ball. The heuristic lost more (0.3049 -> 0.2567, macroF1 0.6011 -> 0.5006), probably because its
  bench test data now lacks the ball on almost every dead frame, so its ball-based dead cues
  (`ball_out`, static ball near a line) rarely fire.
- Sensitivity, learned (best spellF1 for each `switch_ha`): 5: 0.5296, 10: 0.5402, 20: 0.5440,
  40: 0.5406, 80: 0.5406. Flat around 10-40.
- Only the sum `alive_to_dead + dead_to_alive` matters to first order (every D spell inside a
  period is entered and left once): (20, 40, 20) and (20, 20, 40) both score 0.5440; the
  adopted one won the earlier-grid-point tie-break. The learned optimum sum is 60 nats.
- Both emitters still under-segment: ~257-261 predicted possessions per game vs ~349 in the
  labels (learned mean length 14.7 s vs 10.8 s).
- Residual risk: the visibility weights are large for visible dead rows (the ball is visible
  on only 0.2-4.9 % of dead frames per degraded game, so their weight is ~12-300). The unit test
  shows the weighted model keeps a small part of the "no ball => dead" cue on synthetic data;
  the held-out test data still has the ball missing on ~98 % of dead frames, so any such
  residual is still rewarded by the bench.

## BAR-ATH run, Stage 1 (current, label-free)

`python -m bsd.run` (default mode learned with the `handover` block of `bsd/decode_params.json`:
hand-over features, detector, alpha 16, floor 0.5; costs 10 / 40 / 80; tagging on; 4 min 14 s):
validator OK, 223 possessions, mean 18.78 s, D share 34.8 % / 33.3 % per half. Two runs
gave the same SHA-256 `aaf42ebbf0d41f0285877869f1fda9e4d5ee51068a98713b204ee981a4c3b450`.
Shipped v1 (tag `v1-submitted`): 256 possessions, mean 16.30 s, D share 34.8 % / 33.8 %. The
sample-game bench predicted about the same count as v1 (259 vs 257), so this is fewer and longer
possessions on BAR-ATH, not more fragments. Nothing was tuned on these numbers.

## BAR-ATH run (v1 record, label-free)

`python -m bsd.run --no-tagging` (default mode learned; final model: all 3 games x seeds
100-102, mirrored, visibility-weighted; 2 min 42 s): validator OK, 273 possessions, mean
15.35 s, D share 35.7 % / 32.5 % per half. SHA-256
`ae565dd42f3a591e998aace0c4e7c44ff542c78b8563870cfcf6a92564e0dd1b`.

## Stage 1 (2026-11-17): hand-over detector -- PASSED the gate

`python -m bsd.experiments stage1` (record: `bsd/experiments_stage1.json`; per-case CSVs
`cache/stage1_<variant>.csv`, full grids `cache/stage1_<variant>_grid.csv`). 7 min 52 s wall
time on 6 worker processes (a single run: 207 s training all folds of the incumbent emitter,
the hand-over emitter and the k_s 0.5 detector in parallel, then ~1-2 min per detector variant
for decoding and the k_s 1.0 detector). No variant came near the 12-min limit, so the grid was
not narrowed.

Protocol: the same 6 cases and folds as above (held-out game 1-3 x seed 0, 1; each fold trained
on the other two games x seeds 100-102, mirrored, visibility-weighted). The detector is
`bsd.handover.train_detector` on `frame_features(handover=True)` with change labels from
`bsd.handover.change_labels` on the same training pool and visible rate. With a detector the
H<->A cost of frame t is `max(floor, switch_ha - alpha * logit(p_change(t)))`. Decoder grid:
`switch_ha` {10, 20, 40} x (`alive_to_dead`, `dead_to_alive`) {(10, 20), (20, 40), (40, 80)} x
`alpha` {0, 1, 2, 4, 8} x `floor` {1, 5}. alpha = 0 is decoded once, whatever the floor.
Without a detector only alpha = 0 is searched. An axis whose best value is on its edge is
extended once, like `bench.tune_decode`, by doubling or halving (floor only when the best alpha
is > 0). Objective and tie-breaks as above.

| variant | emitter features | detector | spellF1 | macroF1 | n poss | mean poss s | tuned: switch_ha, D pair, alpha, floor | points |
|---|---|---|---|---|---|---|---|---|
| incumbent | default | -- | 0.5440 | 0.8372 | 256.8 | 14.69 | 20, (20, 40), 0, -- | 9 |
| (a) a_features | + hand-over | -- | 0.5644 | 0.8421 | 262.8 | 14.32 | 20, (20, 40), 0, -- | 9 |
| (b) b_detector | default | k_s 0.5 | 0.5737 | 0.8442 | 233.7 | 16.15 | 20, (40, 80), 16, 0.5 | 256 |
| **(c) c_both** | + hand-over | k_s 0.5 | **0.5976** | **0.8480** | 259.0 | 14.55 | 10, (40, 80), 16, 0.5 | 192 |
| (d) d_k1 | + hand-over | k_s 1.0 | 0.5903 | 0.8455 | 273.0 | 13.76 | 5, (20, 40), 16, 5 | 192 |
| (e) e_through_dead | -- | -- | not run | | | | | |

The incumbent re-bench reproduces the committed record exactly (0.5440 / 0.8372 at
(20, 20, 40)). (d) is (c) with k_s 1.0. It scored 0.0073 below (c), which is less than +0.005,
so the stage stopped early and (e) (through_dead_s 2.0) was not run. Edge extensions (first-pass
edges): (b) switch_ha 5, D pair (80, 160), alpha 16, floor 0.5; (c) switch_ha 5, alpha 16,
floor 0.5; (d) switch_ha 5, alpha 16, floor 10. (c)'s first-pass best (original grid) was switch_ha 10, D pair (20, 40),
alpha 8, floor 1 at 0.5954. The D axis was not extended, so the adopted D pair (40, 80) sits on
the top edge of the unextended 3-value D axis; the adopted point also sits on the extended edges
alpha 16 and floor 0.5. The best alpha is the extended edge 16 in all three detector variants
(and (d)'s switch_ha is the extended edge 5), but the curve is flat there. Best spellF1 by alpha for (c): 0: 0.5644, 1: 0.5719, 2: 0.5824,
4: 0.5942, 8: 0.5971, 16: 0.5976. For (b): 0: 0.5440, 4: 0.5686, 8: 0.5729, 16: 0.5737.

Per case, incumbent -> (c) spellF1 (macroF1):

| game | seed | incumbent | c_both |
|---|---|---|---|
| 1 | 0 | 0.5246 (0.8352) | 0.5626 (0.8425) |
| 2 | 0 | 0.5800 (0.8349) | 0.6489 (0.8477) |
| 3 | 0 | 0.5309 (0.8446) | 0.5589 (0.8542) |
| 1 | 1 | 0.5498 (0.8383) | 0.6151 (0.8458) |
| 2 | 1 | 0.5587 (0.8285) | 0.6311 (0.8435) |
| 3 | 1 | 0.5201 (0.8418) | 0.5691 (0.8544) |

**Gate (best variant c_both vs incumbent, same 6 cases): PASSED.** Mean spellF1 +0.0536
(>= 0.010), better on 6 of 6 cases (>= 4), mean macroF1 +0.0108 (drop <= 0.010). Winner config
(`bsd/experiments_stage1.json` `winner`): hand-over emitter features, detector k_s 0.5,
through_dead_s none, alpha 16, floor 0.5, switch_ha 10, alive_to_dead 40, dead_to_alive 80.
Adopted in `bsd.run` and `bsd/decode_params.json` (see "BAR-ATH run" above).

Caveats: the decoder costs are tuned in sample on the 6 reported cases, as above. The detector
variants searched 192-256 points and the incumbent 9, so their selection bias is larger. Even
inside the original grid (alpha <= 8, floor {1, 5}), (c) reaches 0.5954 (+0.051 over the
incumbent; 0.5971 needs the extended floor 0.5), and the margin is far above this bias and the ~0.04 noise band. The hand-over features alone, (a), already give +0.020 with
the same 9-point search as the incumbent.

## Stage 4 (2026-11-17): cheap training tweaks -- FAILED the gate (incumbent stays)

`python -m bsd.experiments stage4` (record: `bsd/experiments_stage4.json`; per-case CSVs
`cache/stage4_<variant>.csv`, full grids `cache/stage4_<variant>_grid.csv`, log
`cache/stage4_log.txt`). 29 min 35 s wall time on 6 worker
processes: 1012 s training every fold of the incumbent and (a)-(d) in parallel, ~1 min of
decoding per variant, then 456 s for (e)'s training and 256-point decode. A first, identical run
(22 min 24 s, `cache/stage4_run1_log.txt`) differed only in the run-time estimate, see below.

Protocol: the incumbent is the committed Stage 1 model (hand-over emitter features, detector
k_s 0.5, decoder costs re-tuned). It runs on the same 6 cases and folds as Stage 1: held-out
game 1-3 x seed 0, 1, and each fold trained on the other two games, mirrored and
visibility-weighted. Every variant is re-tuned on the Stage 1 grid with the detector on (alpha
axis included), with the same extend-once rule, so 192 points each. The variants change only
the training:

- (a) `a_ensemble`: `model.EnsembleEmitter` of 3 H/A/D emitters. Member m has random_state m
  and training degrade seeds 100-102, 103-105 or 106-108. It averages the members'
  mirror-averaged probabilities, then takes log(p / prior). The detector is the incumbent's.
- (b) `b_hgb`: emitter `hgb` override `max_iter` 300, `learning_rate` 0.05. The detector is the
  incumbent's.
- (c) `c_stride1`: emitter trained on every frame (`stride` 1 instead of 3). The detector is the
  incumbent's.
- (d) `d_seeds`: training degrade seeds 100-105 instead of 100-102, for both the emitter and the
  detector.
- (e) `e_combined`: the best two of (a)-(d), i.e. (d) + (a). It has 3 members, each trained on 6
  seeds (100-105, 106-111, 112-117), and the detector on 100-105.

Held-out seeds stay 0 and 1, which never overlap the training seeds.

`bsd.run` time estimate ("est. run"): today's run (254 s) plus today's final-model training
time (emitter 176 s, detector 60 s), each scaled by the variant's bench cost ratio (per-fold
training CPU seconds, variant / incumbent). Ratios are used because the bench's 6 parallel
workers run each job about 1.2x slower than a lone process: the incumbent's bench fold costs
142 + 54 s on 2 games, versus 176 + 60 s alone on 3 games. A first run of the driver added bench
seconds x 1.5 (fold to final data size) directly. That overestimated every variant: it gave (a)
684 s, (d) 570 s and (e) 1489 s a priori, so (e) was skipped. The estimate was corrected and the
stage re-run. The (a)-(d) scores were identical in both runs (deterministic training). In the
re-run, (e)'s a-priori estimate was 1111 s, so it ran. With run 1's timings the same formula
gives 1268 s. The skip decision for (e) is therefore within timing noise of the 1200 s limit.
The estimates are rough (one machine, one timing of today's training).

| variant | spellF1 | macroF1 | n poss | mean poss s | tuned: switch_ha, D pair, alpha, floor | train CPU per fold (emitter + detector) | est. run |
|---|---|---|---|---|---|---|---|
| incumbent (Stage 1 c_both) | 0.5976 | 0.8480 | 259.0 | 14.55 | 10, (40, 80), 16, 0.5 | 158 + 51 s | 254 s |
| (a) a_ensemble | 0.6010 | **0.8557** | 253.0 | 14.65 | 10, (20, 40), 16, 1 | 476 + 51 s | 609 s |
| (b) b_hgb | 0.5956 | 0.8471 | 259.3 | 14.50 | 10, (20, 40), 16, 1 | 306 + 51 s | 419 s |
| (c) c_stride1 | 0.5961 | 0.8483 | 255.5 | 14.68 | 20, (20, 40), 16, 0.5 | 453 + 51 s | 583 s |
| **(d) d_seeds** | **0.6057** | 0.8532 | 270.2 | 13.83 | 5, (20, 40), 16, 1 | 291 + 99 s | 460 s |
| (e) e_combined (a + d) | 0.6029 | 0.8520 | 263.5 | 14.22 | 5, (40, 80), 8, 1 | 873 + 99 s | 1110 s |

The incumbent re-bench reproduces the committed record exactly (0.5976 / 0.8480 at 10, (40, 80),
16, 0.5). (a)-(d) are separate tweaks and all four ran, so the +0.005 early-stop rule did not
apply to them. (e) scored 0.6029, which is 0.0028 below the best of (a)-(d), (d) at 0.6057. It
added less than +0.005, so it is recorded as an early stop (`stopped_early_after`). (e)'s
search extended the D axis to (80, 160) and floor to 0.5 (256 points); its best alpha is 8
(flat to 16).

Per case, spellF1 (macroF1):

| game | seed | incumbent | a_ensemble | b_hgb | c_stride1 | d_seeds | e_combined |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.5626 (0.8425) | 0.5711 (0.8476) | 0.5612 (0.8389) | 0.5858 (0.8487) | 0.5712 (0.8417) | 0.5707 (0.8407) |
| 2 | 0 | 0.6489 (0.8477) | 0.6397 (0.8521) | 0.6426 (0.8438) | 0.6254 (0.8394) | 0.6777 (0.8538) | 0.6524 (0.8506) |
| 3 | 0 | 0.5589 (0.8542) | 0.5687 (0.8602) | 0.5717 (0.8559) | 0.5580 (0.8540) | 0.5657 (0.8591) | 0.5610 (0.8582) |
| 1 | 1 | 0.6151 (0.8458) | 0.6182 (0.8651) | 0.5978 (0.8493) | 0.6083 (0.8514) | 0.6101 (0.8593) | 0.6203 (0.8611) |
| 2 | 1 | 0.6311 (0.8435) | 0.6389 (0.8485) | 0.6304 (0.8408) | 0.6255 (0.8387) | 0.6441 (0.8461) | 0.6330 (0.8450) |
| 3 | 1 | 0.5691 (0.8544) | 0.5693 (0.8608) | 0.5703 (0.8537) | 0.5737 (0.8573) | 0.5656 (0.8595) | 0.5798 (0.8565) |

**Gate (best variant d_seeds vs incumbent, same 6 cases): FAILED.** Mean spellF1 +0.0081
(needs >= 0.010). It is better on 4 of 6 cases (needs >= 4; worse on game 1 and game 3 at seed
1). Mean macroF1 +0.0052 (a drop of at most 0.010 is allowed). Only the spellF1 margin fails.
`winner` is null in the record, and the Stage 1 model stays in `bsd.run` and
`bsd/decode_params.json`. Nothing was changed there.

Observations:
- More training degrade seeds (d) gives the largest spellF1 gain, and the ensemble (a) the
  largest macroF1 gain (+0.0077). Both are below the 0.010 spellF1 margin, and inside the ~0.04
  noise band and the selection bias of a 192-point in-sample decoder search. Both differ from
  the incumbent only in their training, and both re-tuned to the smaller D pair (20, 40).
- Combining them (e) did not add up. It is better on all 6 cases, but by only +0.0053 on
  average, and it costs a ~18.5 min `bsd.run` against ~7.7 min for (d) alone.
- Longer boosting (b) and stride 1 (c) do not help (-0.002 and -0.0015). Neighbouring frames
  are near-duplicates, as `STRIDE` assumes, and 150 rounds at 0.1 are not under-fitted.
- In (a)-(d) the best alpha is the extended edge 16 (the curve is flat from 8). The best
  switch_ha 5 of (d) and (e) is also an extended edge.
- Caveats: the decoder costs are tuned in sample on the 6 reported cases. The best variant is selected
  across 5 variants, so its margin is biased upward. That makes the failed gate more, not
  less, robust.

## Stage 2 (2026-11-17): public DFL Bundesliga games in the training pool -- FAILED the gate (incumbent stays)

`python -m bsd.experiments stage2 --skip c_dfl_seeds3 --secondary-n 3` (record:
`bsd/experiments_stage2.json`; per-case CSVs `cache/stage2_<variant>.csv`, full grids
`cache/stage2_<variant>_grid.csv`, secondary scores `cache/stage2_<variant>_secondary.csv`, log
`cache/stage2_log.txt`). 31 min 49 s (1909 s) wall time on 4 worker processes.

Data: the 7 DFL/IDSSE open-data matches (Bassek et al., CC BY 4.0; `bsd.fetch_dfl`,
`bsd.io_dfl`), 2 periods each, 25 fps, ~137k-147k frames, one slot per player. Each DFL game is
degraded with `bsd.degrade` exactly like a Metrica game (the BAR-ATH-like ball mask provides all
ball missingness, since the DFL ball is present on every frame), featurised with
`frame_features(handover=True)` and labelled from the provider's per-frame flags.

**Label-convention caveat.** DFL labels are the data provider's per-frame BallStatus /
BallPossession flags, not event-derived. They carry more dead time (D share 41-52 % of frames vs
30-40 % on Metrica) and shorter possession spells (mean 5.5-9.2 s vs ~10 s on Metrica; ~500 GT
spells per DFL game) than the Metrica labels built from events (`bsd.events`, R5-R7). The
BAR-ATH GT is built from corrected tracking, so it may be closer to the DFL flags than to the
Metrica labels, but that cannot be checked here; the primary (gate) bench scores against the
Metrica convention only, so a model that learns the DFL convention is penalised there.

Protocol:
- Incumbent: the committed Stage 1 model (hand-over emitter features, detector k_s 0.5),
  re-tuned on the Stage 1 grid like every variant.
- Primary (gate): the usual 6 Metrica cases (held-out game 1-3 x seed 0, 1). A candidate's fold
  trains the emitter and the detector on the 2 other Metrica games x seeds 100-102 plus every
  DFL game (change labels from the DFL labels with the same k_s 0.5), mirrored and
  visibility-weighted as before.
- Secondary: a DFL game held out at degrade seed 0. Incumbent = trained on the 3 Metrica games x
  seeds 100-102 (the `bsd.run` training set); candidate = the same plus the other 6 DFL games.
  Decoded at the parameters each model tuned on the primary (no tuning on DFL). The candidate's
  mean spellF1 must not be more than 0.010 below the incumbent's; this is required on top of the
  gate.
- Variants: (a) `a_dfl` all 7 DFL games, 1 training degrade seed (100) each; (b) `b_dfl_w05` as
  (a) with DFL rows weighted 0.5 (`weights` of `train_emitter` / `train_detector`, multiplied on
  top of the visibility weights; default None = unchanged model); (c) `c_dfl_seeds3` DFL with
  seeds 100-102 -- not run: dropped for time, and the early-stop rule would also have dropped it ((b) added +0.0032 < +0.005 over (a)).
- Time budget: (c) was dropped and the secondary shrunk to the first 3 DFL games (sorted match
  ids J03WMX, J03WN1, J03WOH) to finish inside the task's time box; the full protocol (all 7) is the
  driver's default (`--secondary-n` unset).
- Training rows are pre-strided (every 3rd frame) in the parent and fitted with stride 1: the
  same rows as the default fits, less data shipped to the workers.
- `bsd.run` estimate: today's run (254 s) plus today's final-model training times scaled by the
  cost ratio of the final model (3 Metrica games x 3 seeds + the DFL rows; derived from the
  per-fold training CPU seconds) plus the DFL featurisation the final run would add.

Results (primary = the 6 Metrica cases; secondary = 3 held-out DFL games at seed 0; train CPU is per fold of the
primary, the incumbent's measured by the shared `Evidence` path):

| variant | primary spellF1 | macroF1 | n poss | mean poss s | secondary spellF1 (macroF1) | tuned: switch_ha, D pair, alpha, floor | train CPU per fold (emitter + detector) | est. run |
|---|---|---|---|---|---|---|---|---|
| incumbent | 0.5976 | 0.8480 | 259.0 | 14.55 | 0.3275 (0.6331) | 10, (40, 80), 16, 0.5 | 128 + 42 s | 254 s (today) |
| a_dfl | 0.5988 | 0.8542 | 263.7 | 13.87 | 0.4902 (0.8769) | 5, (20, 40), 8, 1 | 295 + 99 s | 470 s |
| b_dfl_w05 | 0.6020 | 0.8556 | 261.2 | 13.99 | 0.4824 (0.8745) | 10, (20, 40), 8, 0.5 | 297 + 103 s | 475 s |

Primary per case, spellF1 (macroF1):

| game | seed | incumbent | a_dfl | b_dfl_w05 |
|---|---|---|---|---|
| 1 | 0 | 0.5626 (0.8425) | 0.5587 (0.8462) | 0.5785 (0.8497) |
| 2 | 0 | 0.6489 (0.8477) | 0.6483 (0.8516) | 0.6598 (0.8533) |
| 3 | 0 | 0.5589 (0.8542) | 0.5688 (0.8526) | 0.5611 (0.8546) |
| 1 | 1 | 0.6151 (0.8458) | 0.6036 (0.8698) | 0.6197 (0.8742) |
| 2 | 1 | 0.6311 (0.8435) | 0.6329 (0.8511) | 0.6279 (0.8497) |
| 3 | 1 | 0.5691 (0.8544) | 0.5802 (0.8540) | 0.5650 (0.8519) |

Secondary per held-out DFL game (seed 0), spellF1 (macroF1), n poss / GT n poss:

| DFL game | incumbent | a_dfl | b_dfl_w05 |
|---|---|---|---|
| J03WMX | 0.3278 (0.6617), 276 / 511 | 0.4563 (0.8871), 278 / 511 | 0.4432 (0.8877), 267 / 511 |
| J03WN1 | 0.3162 (0.6166), 288 / 500 | 0.4923 (0.8589), 293 / 500 | 0.4833 (0.8502), 297 / 500 |
| J03WOH | 0.3386 (0.6210), 223 / 402 | 0.5221 (0.8847), 250 / 402 | 0.5206 (0.8858), 235 / 402 |

The incumbent re-bench reproduces the committed record exactly (0.5976 / 0.8480 at 10, (40, 80), 16, 0.5).
(b) added +0.0032 over (a), less than +0.005, so it is recorded as an early stop (`stopped_early_after`).

**Gate (best variant b_dfl_w05 vs incumbent, same 6 cases): FAILED.** Mean spellF1 +0.0044 (needs >= 0.010).
Better on 4 of 6 cases (needs >= 4). Mean macroF1 +0.0076 (a drop of at most 0.010 is allowed). Only the spellF1
margin fails. (a) a_dfl: +0.0011, 3 of 6 better, macroF1 +0.0062, which also fails.
**Secondary: OK for both** (b +0.1548, a +0.1627 mean spellF1 over the incumbent on the 3 held-out DFL games, all
3 games better). Adoption needs both, so `winner` is null. The Stage 1 model stays in `bsd.run` and
`bsd/decode_params.json`, and nothing there was changed.

Observations:
- Adding the DFL games helps macroF1 on the Metrica cases (+0.006 to +0.008, mostly at seed 1), but spellF1 barely
  moves (+0.001 / +0.004). That is far inside the ~0.04 noise band and the selection bias of a 192-point in-sample
  decoder search. Both variants re-tuned to the smaller D pair (20, 40) and alpha 8.
- Down-weighting DFL to 0.5 (b) is slightly better on the primary and slightly worse on the secondary than weight
  1.0 (a). That fits a label-convention mismatch: the more the model learns the DFL flags, the less it fits the
  Metrica event-derived labels it is graded against, and vice versa.
- The secondary gain is large (+0.15 spellF1, +0.25 macroF1) and expected. The incumbent never saw the DFL
  convention (more dead time, ~500 short spells per game): it predicts ~220-290 possessions against 400-511 GT
  spells, and its macroF1 there is only 0.62-0.66. The candidates' possession counts stay about the same (235-297,
  decoded with Metrica-tuned costs), so the gain is mostly in the per-frame H/A/D split (dead-ball calls), not in
  segmentation. It shows the models learn the DFL labels; it says little about BAR-ATH.
- Cost: the DFL rows more than double the per-fold training time (emitter 128 -> ~296 s, detector 42 -> ~100 s).
  The estimated `bsd.run` is ~8 min (470-475 s, including the DFL featurisation of the 7 games), well under the
  20 min limit. It would also need the DFL fetch (2.5 GB of XML in `data/dfl/`) in the reproduction.
- Caveats: decoder costs are tuned in sample on the 6 reported cases. The secondary uses 3 of the 7 DFL games (time
  box) and the incumbent's secondary model is the `bsd.run` training set. Labels on the two sides follow different
  conventions (see the caveat above), so the primary and secondary numbers are not comparable with each other.

## Before R13 (record)

Same protocol, except that `bsd.degrade.fill_ball` linearly interpolated the sample ball across
its gaps (mostly dead-ball time) before the visibility mask, and training rows were unweighted.
8 min 44 s.

### Results (before R13)

Reference: the hand-set heuristic with its hand-set costs (25 / 12.5 / 12.5) scored mean spellF1
0.2782, macroF1 0.6078, ~196 possessions per game (`cache/bench_heuristic.csv`).

#### Heuristic, tuned: switch_ha 10, alive_to_dead 10, dead_to_alive 10

Interior optimum, no extension.

| game | seed | spellF1 | macroF1 | n poss | mean poss s | GT n poss | GT mean poss s |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.2801 | 0.5838 | 254 | 19.42 | 353 | 9.84 |
| 2 | 0 | 0.3094 | 0.5756 | 234 | 21.33 | 318 | 11.54 |
| 3 | 0 | 0.3019 | 0.6188 | 267 | 19.22 | 377 | 10.87 |
| 1 | 1 | 0.3266 | 0.6100 | 268 | 18.14 | 353 | 9.84 |
| 2 | 1 | 0.3011 | 0.5610 | 232 | 21.73 | 318 | 11.54 |
| 3 | 1 | 0.3106 | 0.6575 | 276 | 18.12 | 377 | 10.87 |
| **mean** | | **0.3049** | **0.6011** | 255.2 | 19.66 | 349.3 | 10.75 |

#### Learned, tuned: switch_ha 20, alive_to_dead 80, dead_to_alive 10

First pass hit edges on the two dead-ball axes; both were extended to 80 once (750 decoded
points x cases in total).

| game | seed | spellF1 | macroF1 | n poss | mean poss s | GT n poss | GT mean poss s |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.5227 | 0.8427 | 234 | 14.26 | 353 | 9.84 |
| 2 | 0 | 0.6051 | 0.8514 | 236 | 14.39 | 318 | 11.54 |
| 3 | 0 | 0.5221 | 0.8518 | 265 | 15.21 | 377 | 10.87 |
| 1 | 1 | 0.5680 | 0.8603 | 247 | 13.87 | 353 | 9.84 |
| 2 | 1 | 0.5978 | 0.8538 | 236 | 14.38 | 318 | 11.54 |
| 3 | 1 | 0.5309 | 0.8583 | 283 | 14.01 | 377 | 10.87 |
| **mean** | | **0.5578** | **0.8531** | 250.2 | 14.35 | 349.3 | 10.75 |

#### Decision (before R13)

**Adopted: learned** (mean spellF1 0.5578 vs 0.3049, macroF1 0.8531 vs 0.6011; the learned
emitter wins on every one of the 6 cases). Recorded in `bsd/decode_params.json` (committed,
read by `bsd.run`; copy in `cache/decode_params.json`), so `python -m bsd.run` defaults to
`--mode learned`.

#### Observations (before R13)

- Sensitivity, learned (best spellF1 for each `switch_ha`): 5: 0.5445, 10: 0.5574, 20: 0.5578,
  40: 0.5508, 80: 0.5489. The optimum is flat around 10-20; many points are within 0.001 of it.
- Only the *sum* `alive_to_dead + dead_to_alive` matters to first order: inside a period every
  D spell is entered and left once, so (20, 10, 80) and (20, 80, 10) score 0.5578 both, and the
  choice between them was made by a 0.0001 macroF1 tie-break. The learned optimum is a sum of
  ~90 nats, much higher than the heuristic's 20: the classifier's D evidence is strong and
  needs a bigger entry+exit cost to stop short spurious dead spells.
- Both emitters still under-segment: ~250 predicted possessions per game vs ~349 in the labels
  (learned mean length 14.4 s vs 10.8 s). Lowering `switch_ha` adds possessions but loses
  spellF1 (switch_ha 5: 322 possessions, 0.5445).
- **Caveat on the learned numbers.** In the sample games the ball is missing on ~95 % of dead
  frames; `bsd.degrade.fill_ball` linearly interpolates it and only then applies the BAR-ATH
  visibility mask. The visible "ball" during sample-game stoppages is therefore a jittered
  straight constant-speed path that usually lies away from all players -- a pattern the
  classifier can learn as "dead" and which does not exist in BAR-ATH. The bench D accuracy (and
  so macroF1 0.85) is probably optimistic for BAR-ATH; the H/A part of the gain, which drives
  spellF1, is less exposed to this.

#### BAR-ATH run (before R13, label-free)

`python -m bsd.run --mode learned --no-tagging` (final model: all 3 games x seeds 100-102,
mirrored; 2 min 34 s): validator OK, 269 possessions, mean 14.28 s, D share 38.9 % / 40.2 %
per half. SHA-256 `7632e1cf89f3d73255be76f64f3baefc0aacc9277ea880c62a9868c15b91af76`.

### Smart Tagging (Task 10, label-free BAR-ATH runs, learned mode)

`bsd/tagging.py` adds fixed-weight evidence (see its docstring) plus R14 (+3.0 nats on D before
the first kick-off set piece of each period) to the log-scores before decoding. Tagging is on by
default; `--no-tagging` disables it.

| run | possessions | mean | D share H1 / H2 | H2 first 5 min | SHA-256 |
|---|---|---|---|---|---|
| no tagging | 273 | 15.35 s | 35.7 % / 32.5 % | H 58.9 / A 27.7 / D 13.4 | `ae565dd42f3a591e998aace0c4e7c44ff542c78b8563870cfcf6a92564e0dd1b` |
| tagging, original R14 (superseded: bug) | 257 | 15.74 s | 38.8 % / 33.8 % | H 48.0 / A 19.5 / D 32.6 | `0c9f62001d0c5f17c832211d92c7af34a83a87af73b3172227deba0127472095` |
| tagging, R14b (kept) | 256 | 16.30 s | 34.8 % / 33.8 % | H 48.0 / A 19.5 / D 32.6 | `c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee` |

Possession count and D share are nowhere near 2x the no-tagging run, so tagging is kept.

R14b correction: the original R14 picked a kick-off after a goal in half 1 (the match-start tag begins just before the period) and put +3.0 on D for ~25 minutes. R14b uses only `Kick Off (Start)` tags overlapping the period and no bonus when it starts at or before the period's first frame.

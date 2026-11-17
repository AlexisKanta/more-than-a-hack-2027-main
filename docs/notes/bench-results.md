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

## BAR-ATH run (current, label-free)

`python -m bsd.run --no-tagging` (default mode learned; final model: all 3 games x seeds
100-102, mirrored, visibility-weighted; 2 min 42 s): validator OK, 273 possessions, mean
15.35 s, D share 35.7 % / 32.5 % per half. SHA-256
`ae565dd42f3a591e998aace0c4e7c44ff542c78b8563870cfcf6a92564e0dd1b`.

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

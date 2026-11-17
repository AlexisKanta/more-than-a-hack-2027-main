# Bench results: heuristic vs learned evidence, decoder tuning (Task 9)

All numbers come from the leave-one-game-out bench on the three Metrica sample games
(`python -m bsd.bench --mode learned`, 8 min 44 s on one core). Nothing here was tuned on or
scored against BAR-ATH or its public ground-truth window.

## Protocol

- Held-out game `g` in {1, 2, 3}, degraded with seed `s` in {0, 1} (`bsd.degrade`): 6 cases.
- Learned emitter of fold `g`: trained once on the other two games, each degraded with seeds
  100, 101, 102 (distinct from the held-out seeds) and mirrored (home/away columns and H/A labels
  swapped); X-label frames dropped; every 3rd frame. `HistGradientBoostingClassifier`
  (150 rounds, 31 leaves, min 100 samples/leaf, L2 1.0, no early stopping, `random_state=0`,
  one thread). Evidence = `log(p / prior)`, with the prediction averaged with the swapped
  prediction of the mirrored frame (exact team symmetry), rounded to 6 decimals.
- Heuristic emitter: `bsd.heuristic.heuristic_logp`, unchanged (no training).
- Each emitter's held-out evidence is computed once and cached; the decoder grid
  `switch_ha` {5, 10, 20, 40, 80} x `alive_to_dead` {5, 10, 20, 40} x `dead_to_alive`
  {5, 10, 20, 40} is decoded from it. An axis whose best value lies on the grid edge is extended
  once (double the top / halve the bottom) and the search repeated. Objective: mean held-out
  spellF1 over the 6 cases (ties: mean macroF1, then the earlier grid point).

## Results at each emitter's tuned decoder

Reference: the hand-set heuristic with its hand-set costs (25 / 12.5 / 12.5) scored mean spellF1
0.2782, macroF1 0.6078, ~196 possessions per game (`cache/bench_heuristic.csv`).

### Heuristic, tuned: switch_ha 10, alive_to_dead 10, dead_to_alive 10

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

### Learned, tuned: switch_ha 20, alive_to_dead 80, dead_to_alive 10

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

## Decision

**Adopted: learned** (mean spellF1 0.5578 vs 0.3049, macroF1 0.8531 vs 0.6011; the learned
emitter wins on every one of the 6 cases). Recorded in `bsd/decode_params.json` (committed,
read by `bsd.run`; copy in `cache/decode_params.json`), so `python -m bsd.run` defaults to
`--mode learned`.

## Observations

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

## BAR-ATH run (label-free)

`python -m bsd.run --mode learned --no-tagging` (final model: all 3 games x seeds 100-102,
mirrored; 2 min 34 s): validator OK, 269 possessions, mean 14.28 s, D share 38.9 % / 40.2 %
per half. SHA-256 `7632e1cf89f3d73255be76f64f3baefc0aacc9277ea880c62a9868c15b91af76`.

# BSD improvements: design

Date: 2026-11-17 (submission deadline day). Builds on the shipped entry described in
`docs/superpowers/specs/2026-11-16-possession-hmm-design.md`, `docs/BSD.md` and
`docs/notes/bench-results.md`.

## Intent

Raise team BSD's full-match **spellF1** on BAR-ATH (the ranking metric). The shipped entry's main
measured weakness is **under-segmentation**: on the leave-one-game-out (LOGO) bench it predicts
~257 possessions per sample game against ~349 in the labels (mean 14.7 s vs 10.8 s). Its fixed
H↔A switch cost merges possessions across real hand-overs. Its second weakness is a **tiny
training set**: three anonymised Metrica sample games.

The user asked for four ideas to be implemented **one at a time, in this order, each fully refined
before the next starts**, through subagent-driven development.

## Goals

1. **Stage 1, hand-over detector.** Cut under-segmentation by making H↔A switches cheap where a
   hand-over is likely and expensive elsewhere.
2. **Stage 2, more training games.** Add public tracking+event datasets from other matches.
3. **Stage 3, second-stage model.** A stacked model that reads first-stage outputs over a time
   window.
4. **Stage 4, cheap tweaks.** Seed ensembles and training-budget changes.

## Non-goals

- Self-training or pseudo-labels on BAR-ATH frames. We can't validate them honestly.
- Video or ball re-detection.
- Changing Smart Tagging weights (no tags in training data to validate against).
- Pushing to any remote. Publishing the repo is the user's job.

## Anti-goals (a failure even if it "works")

- Any choice informed by the public 5-minute GT (`evaluation/gt/BAR-ATH_possession_gt_public.csv`)
  or by looking at BAR-ATH outputs frame by frame. Variants are chosen **only** on the bench.
- Losing a submit-ready state. Every stage must end with the repo regenerating a valid
  `evaluation/results/BSD.csv` from one command.
- Breaking exact reproducibility (deterministic, pinned, SHA recorded).
- Adopting a change whose bench gain is selection noise.
- Printing raw BAR-ATH data into an assistant/subagent context (competition terms). Scripts may
  print only aggregates (means, counts, scores, SHAs).

## Constraints (carried over, binding)

- Rules (`RULES.md`, `FAQ.md`): labels from an algorithm only; no human data about BAR-ATH; no
  frame-specific constants for this game (`tests/test_no_literals.py` must stay green); external
  data must be about other games, publicly accessible, and explained in the technical document
  (what was labelled, how, by whom, how much).
- External datasets need a licence or terms that allow this use. Acknowledge each source in
  `docs/BSD.md`. A source that needs a login or manual form to download is excluded, because the
  entry must regenerate from one command.
- Runtime: Python 3.14.6 in `venv/`, pinned `requirements-bsd.txt`. Any new dependency is pinned
  there (exact version) and justified. `kloppy==3.18.0` is acceptable if a loader needs it.
- Determinism: seeded RNG, single-threaded fits, rounded evidence, as today.
- Git: work on local branch `bsd-improve` (forked from `bsd-possession` at `336ec0e`). Never push.
  Never commit match data, downloaded datasets, CSVs or caches. `data/` and `cache/` stay
  gitignored; new dataset folders go under `data/` too.
- The shipped entry stays recoverable: commit `336ec0e`, SHA-256
  `c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee`. Tag it locally as
  `v1-submitted`.

## How a stage works (the same for all four)

1. **Implement** behind a switch, so the incumbent pipeline is reproducible unchanged until the
   stage is adopted (existing tests and SHAs must not move before adoption).
2. **Refine.** Try variants of the idea on the bench. A stage has at most **5 variants**. Stop early
   when a new variant adds less than **+0.005** mean spellF1 over the stage's best so far.
3. **Gate.** Re-bench the incumbent and the stage's best variant on the **same cases** (same
   folds, seeds and decoder tuning procedure). Adopt only if all three hold:
   - mean spellF1 improves by **≥ 0.010**;
   - spellF1 improves on **at least 4 of the 6** Metrica cases (paired);
   - mean macroF1 drops by no more than **0.010**.

   Otherwise the stage is recorded as rejected, with its numbers, and the incumbent stays.
4. **On adoption:** make it the default in `bsd.run`, retune and commit `bsd/decode_params.json`,
   regenerate `BSD.csv`, validate it, run it twice to confirm an identical SHA, and update
   `docs/notes/bench-results.md` (variants tried, all numbers), `docs/BSD.md`, `REPRODUCE.md`,
   `README.md` and the SHA in the parent `../CLAUDE.md`.
5. The next stage's incumbent is whatever came out of this one.

**Bench scoreboard.** The 6 Metrica cases (games 1–3 × held-out seeds 0, 1) stay the primary
scoreboard for every stage, so numbers stay comparable across stages. Stage 2 also reports LOGO
over all games (details there). Bench numbers stay in-sample-tuned for the decoder, as documented
today. The gate's paired-case rule and 0.010 margin exist because of that.

## Stage 1: hand-over detector

**What it adds.** A second per-frame classifier, `P(change | features)`, where "change" means an H↔A
possession change in the training labels within **±k frames** (default k = 0.5 s at the game's fps).
Its output lowers the H↔A transition cost locally.

**Features.** Extend `bsd/features.py` with hand-over cues computed from tracking alone:
- ball speed and acceleration magnitude;
- ball direction change over short windows;
- per team, the distance from the ball to that team's nearest player;
- which team is nearest, and whether that changes within a short centred window.

All are NaN-safe when the ball is missing and respect `mirror` naming. These features may also feed
the H/A/D emitter (that is one of the variants).

**Model.** Same training regime as the emitter: degraded sample games, mirrored, visibility
weights, `HistGradientBoostingClassifier`, deterministic. The change probability is mirror-symmetric
(average the prediction with the mirrored frame's).

**Decoder.** Generalise `bsd/decode.py` so H↔A costs can vary per frame:
`switch_ha(t) = max(floor, switch_ha − alpha · logit(p_change(t)))`. The exact functional form is
the builder's call; it must be monotone in `p_change`. When no per-frame costs are given,
`viterbi` must behave bit-identically to today. D-transition costs stay constant.

**Variants to refine over (pick the order):**
- (a) new features into the emitter only;
- (b) detector → per-frame switch cost;
- (c) both;
- (d) different k or label definitions. For example, count H→D→A across a short dead spell as a
  change too.

**Tuning.** Decoder tuning adds `alpha` (and `floor` if used) to the grid. The builder may narrow
the existing axes around the incumbent's optimum to keep bench time reasonable (target: one bench
run ≤ 30 min).

**Performance.** `viterbi` is a pure-Python loop. The builder may vectorise it (numpy over the 3
states per frame) if, and only if, the output stays identical on the existing tests. The same
applies to any speed-up.

## Stage 2: more training games

**Candidate sources.** The builder verifies, in this order, availability, licence/terms, presence
of both tracking and event data, and a scriptable download:
1. DFL / IDSSE open data: Bundesliga matches with synchronised tracking and official event data,
   published with a Scientific Data paper (Bassek et al.), hosted on figshare.
2. SkillCorner open data, only if a release includes event or possession data alongside tracking.
3. Anything else found while checking, under the same tests. PFF FC World Cup 2022 is excluded if
   it needs a form or login.

Record the outcome of each check (accepted or rejected, and why) in `docs/notes/`.

**Pipeline for each accepted source:**
- a fetch script that downloads by pinned URL and verifies a SHA-256 for each file (like
  `bsd/fetch_sample.py`);
- a loader to the existing `Tracking` dataclass;
- a source-specific events-to-labels function. Document its rules the way R5–R7 are documented, and
  give it unit tests on small synthetic event tables;
- the same `degrade` step.

New games join the training pool.

**Bench for this stage:**
- **Primary (gate):** the 6 Metrica cases, now trained on the other 2 Metrica games plus all new
  games.
- **Secondary (reported, and must not get worse by more than 0.010):** LOGO over all games, with
  each new game held out once, at seed 0.

If a source's labels look unreliable (for example, label share or possession length far from the
Metrica games'), report it and leave the source out rather than tune around it.

**Memory and time.** Use stride and `float32` as needed. `bsd.run` end-to-end should stay under
~20 min, and a bench run under ~45 min.

## Stage 3: second-stage model

A stacked classifier over **out-of-fold** first-stage outputs (the H/A/D log-scores, plus the
change probability if stage 1 was adopted), sampled at offsets across a centred window (for
example ±4 s every 0.5 s), plus the current frame's features. Output: new H/A/D evidence for the
same decoder.

The first-stage outputs used to train the second stage must come from models that did not see that
game (nested folds). Otherwise the stacker learns from overconfident inputs. Window shape, offsets
and model settings are the builder's call; they count as the stage's variants.

## Stage 4: cheap tweaks

Variants:
- an ensemble of K emitters with different `random_state` values and training degrade seeds,
  averaging probabilities before `log(p/prior)`;
- `max_iter` / `learning_rate` changes;
- `STRIDE` 1 vs 3;
- more training degrade seeds.

Same gate as the other stages. Training time must stay within the stage-2 limits.

## End of all stages

1. Run the clean reproduction check, as last time: `git archive` the final commit into a temp
   folder with an empty cache, run `python -m bsd.run`, and confirm the same SHA.
2. Run the full test suite.
3. Run the public-window score **once**, as a correctness check only. Report it, and do not change
   anything because of it.
4. Update the memory file `bsd-entry-status` with the new SHA and the adopted stages.
5. Final whole-branch review. Then hand back to the user with the finishing menu. Merging, and
   which branch to publish, is the user's decision.

## Decisions settled after the builder check (design author's calls)

- **Time-box.** The stages run in the user's order. When the cutoff the user gives is near, stop
  the stage in progress at its last committed point. If it is not adopted by then, discard it. The
  last adopted state is the submission. Leave at least **1 hour** before the cutoff for the final
  checks and the upload. Benches may run in the background while the next variant is being coded.
- **Stage 2 labels.** Prefer a source's per-frame **ball-status and ball-possession flags** in its
  tracking data, if it has them (DFL does). The BAR-ATH GT is built from corrected tracking, which
  is closer to these flags than to event logs. Where possible, cross-check the flags against the
  source's events and report how well they agree. A source with possession or ball-status data but
  no events (for example SkillCorner) qualifies under the same test.
- **Stage 2 secondary bench.** Hold out each new game once, at seed 0. Compare:
  - **incumbent:** trained on the 3 Metrica games only;
  - **candidate:** trained on the 3 Metrica games plus the other new games.

  The Metrica games are not part of the secondary bench (they are the primary). The candidate's
  mean spellF1 there must not be more than 0.010 below the incumbent's.
- **Stage 3 with only 3 games** (if Stage 2 is rejected). Still run it, using inner leave-one-out:
  the first stage is trained on 1 game to produce out-of-fold inputs for the stacker. Note the
  train/test mismatch in the bench notes. The gate decides.
- **Reproduction commands.** Downloads stay separate fetch commands, as `bsd.fetch_sample` is
  today. `REPRODUCE.md` lists the fetch commands, then `python -m bsd.run`. The runtime budgets
  exclude download time. The clean check fetches into the archive copy, or symlinks the already
  verified `data/` folders (their SHA-256 checksums are checked by the fetch script).

## Left to the builder

Exact feature formulas, model hyperparameters inside each variant, variant order within a stage,
grid shapes (within the time targets), file and function names (following existing conventions),
and whether to vectorise `viterbi`.

## Approaches considered (for the whole effort)

- **Chosen: the four ideas in sequence, each gated on the bench** (the user's choice). Each stage
  builds on the last, and a rejected stage costs only its own time.
- **All at once, then one bench:** faster, but a loss from one idea can hide a gain from another,
  and nothing is attributable.
- **Stage 1 only:** the cheapest path to the measured weakness. Not chosen, because the user wants
  all four.

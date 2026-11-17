# BSD Improvements (Stages 1 and 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the bench spellF1 of the BSD entry with a hand-over detector (Stage 1) and cheap
training tweaks (Stage 4). Each stage is gated on the leave-one-game-out (LOGO) bench and adopted
into `bsd.run` only if it passes.

**Architecture:** New hand-over features are added to `bsd/features.py` behind a flag. A binary
"possession changes near here" classifier lives in `bsd/handover.py`, and its probability becomes a
per-frame H↔A switch cost in `bsd/decode.py`. An experiment driver, `bsd/experiments.py`, benches
named variants against the incumbent and applies the design's gate. The adopted configuration lives
in `bsd/decode_params.json` and is read by `bsd.run`.

**Tech Stack:** Python 3.14.6 (`venv/bin/python`), numpy, pandas, scikit-learn
`HistGradientBoostingClassifier`, pytest. All are pinned in `requirements-bsd.txt`, and no new
dependencies are added.

**Spec:** `docs/superpowers/specs/2026-11-17-bsd-improvements-design.md`. Read its "How a stage
works", "Stage 1", "Stage 4" and "Decisions settled" sections.

**Scope ruling (deadline):** the deadline is ~18:43 EST on 2026-11-17, and this plan started at
12:43 EST. This plan covers Stage 1, then Stage 4. Stage 4 runs before Stage 2: it is cheap and
low-risk, whereas Stage 2 (new datasets: fetch script, loader, labels) cannot finish and pass the
gate in the time left. Stages 2 and 3 get a follow-up plan only if at least 2.5 h remain after
Task 6. All work stops by 17:40 EST, which leaves the final checks (Task 7) and the upload margin.

## Global Constraints

- Run everything from `more-than-a-hack-2027-main/` with `venv/bin/python`.
- The branch is `bsd-improve`. Never `git push`. Never commit `data/`, `cache/`, or any `*.csv` under
  `evaluation/results/`.
- **Competition terms:** never print raw BAR-ATH data (anything under `data/tactical` or
  `data/broadcast`, or arrays derived from it frame by frame). Scripts print only aggregates:
  scores, counts, means, SHAs.
- Never read, score against, or tune on `evaluation/gt/BAR-ATH_possession_gt_public.csv` (Task 7
  runs it once, as a correctness check only).
- No frame-number literals for this match in `bsd/*.py`. `tests/test_no_literals.py` (regex
  `\b\d{5,6}\b`) must pass.
- Before a stage is adopted, `python -m bsd.run` must still give SHA-256
  `c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee`. Default code paths must be
  bit-identical until adoption.
- Determinism: seeded RNG, single-threaded fits (`threadpool_limits(1)`), log-scores and costs
  rounded to 6 decimals.
- **Gate (verbatim from spec):** adopt only if mean spellF1 improves by **≥ 0.010**, spellF1
  improves on **at least 4 of the 6** Metrica cases (paired), and mean macroF1 drops by no more
  than **0.010**. At most **5 variants** per stage. Stop early when a new variant adds less than
  **+0.005** over the stage's best so far.
- The full test suite (`venv/bin/python -m pytest -q`, 95 tests today) stays green after every task.

## Review Focus

- **Ball missing for long gaps.** Hand-over features must be NaN, never ±inf or 0-filled, so the
  classifier treats them as missing. Pinned by `test_handover_features_nan_when_ball_missing`.
- **Mirror symmetry.** Every new column must have a valid partner in `mirror`, and the detector's
  change probability must be identical for a frame and its mirror. Pinned by
  `test_handover_features_mirror_partners` and `test_detector_mirror_symmetric`.
- **`switch_cost=None` must not move the shipped output.** Pinned by
  `test_viterbi_switch_cost_none_identical` and by the SHA check in Task 3.
- **Period boundaries and X frames.** Change labels must not mark a "change" across a period
  boundary or into or out of X-labelled frames. Pinned by
  `test_change_labels_ignore_period_boundary`.
- **NaN change probability** (rows outside periods). The switch cost must fall back to the constant
  `switch_ha`, not NaN, which `viterbi` would reject. Pinned by `test_switch_cost_nan_falls_back`.

---

### Task 1: Hand-over features and change labels

**Files:**
- Modify: `bsd/features.py` (new constants plus the `handover` flag on `frame_features`)
- Create: `bsd/handover.py` (`change_labels` here; the detector comes in Task 2)
- Test: `tests/test_features.py`, `tests/test_handover.py` (new)

**Interfaces:**
- Produces: `frame_features(tr, handover=False) -> (X, names)`. With `handover=False`, the output is
  identical to today's (same columns, same values).

  With `handover=True`, these columns are appended after the existing ones, in this order:
  - `ball_acc`: magnitude of the change in ball velocity (m/s²), using the same centred differences
    as `ball_speed`;
  - `ball_turn`: angle in [0, π] between the ball's velocity over the previous
    `TURN_SPAN` frames and over the next `TURN_SPAN` frames;
  - `near_diff`: `near_home − near_away`;
  - `near_flip`: 1.0 if the sign of `near_diff` at t − `FLIP_SPAN` differs from the sign at
    t + `FLIP_SPAN`, otherwise 0.0, and NaN if either side is NaN;
  - `closest_home`, `closest_away`: 1.0 if that team holds the closest player and that player is
    within `CLOSE_R` m of the ball, otherwise 0.0, and NaN when there is no ball;
  - centred 1 s rolling means of the above, suffixed `_m1s` (use the existing `_window_mean`). Do
    this for `ball_acc`, `ball_turn`, `near_flip`, `closest_home` and `closest_away`.

  New module constants: `TURN_SPAN = 5` frames, `FLIP_SPAN = 6` frames, `CLOSE_R = 2.0` m.
- Produces: `bsd.handover.change_labels(labels, periods, fps, k_s=0.5, through_dead_s=None) -> np.ndarray`.
  The result is int8 of length `len(labels)`:
  - **1** if an H↔A change happens within ±round(k_s·fps) frames;
  - **0** otherwise;
  - **−1** (unscored) where the label is X or outside periods.

  A change happens at frame t when labels[t−1] and labels[t] are H and A in either order, inside the
  same period. With `through_dead_s` set, an alive→D→alive run whose D part is ≤ `through_dead_s`
  seconds, and whose team differs on the two sides, also counts. That change is placed at the first
  alive frame after the D run.

- [ ] **Step 1: Write failing tests**
  - In `tests/test_features.py`:
    - `test_handover_flag_off_identical`: `frame_features(tr)` and
      `frame_features(tr, handover=False)` return equal names and `np.array_equal(..., equal_nan=True)`.
      Use the existing synthetic-tracking fixture/helper in that file.
    - `test_handover_columns_appended`: with `handover=True`, the names start with the default names
      and end with exactly these 11 columns, in order: `ball_acc`, `ball_turn`, `near_diff`,
      `near_flip`, `closest_home`, `closest_away`, `ball_acc_m1s`, `ball_turn_m1s`, `near_flip_m1s`,
      `closest_home_m1s`, `closest_away_m1s`.
    - `test_handover_features_nan_when_ball_missing`: on frames with no ball, `ball_acc`,
      `ball_turn`, `near_flip`, `closest_home` and `closest_away` are NaN, and nothing in X is
      infinite.
    - `test_handover_features_mirror_partners`: `mirror(X, names)` doesn't raise. Mirroring swaps
      `closest_home`/`closest_away` and negates `near_diff`.
    - `test_ball_turn_reversal`: a synthetic ball moving +x and then −x gives `ball_turn` ≈ π at
      the reversal, and ≈ 0 on a straight path.
  - In `tests/test_handover.py`:
    - `test_change_labels_basic`: with labels `H×20, A×20` at fps 10, k_s 0.5, the frames 15–24 are
      1 and the rest are 0.
    - `test_change_labels_ignore_period_boundary`: an H run ending one period and an A run starting
      the next produce no 1s.
    - `test_change_labels_through_dead`: with labels `H×20, D×5, A×20` at fps 10, there are no 1s
      when `through_dead_s=None`, and 1s around frame 25 when `through_dead_s=1.0`.
    - `test_change_labels_x_unscored`: X frames get −1.

- [ ] **Step 2: Run the tests to verify they fail.**
  `venv/bin/python -m pytest tests/test_features.py tests/test_handover.py -q`. Expect failures
  (unknown kwarg, missing module).

- [ ] **Step 3: Implement both interfaces exactly as specified above.** Windows never cross a
  period boundary: use the existing `_segments`/`_window_mean` helpers. Rows outside periods are
  NaN, as today.

- [ ] **Step 4: Run the full suite.** `venv/bin/python -m pytest -q` passes, and `tests/test_no_literals.py` passes.

- [ ] **Step 5: Commit.** `git add bsd/features.py bsd/handover.py tests/test_features.py tests/test_handover.py && git commit -m "Stage 1: hand-over features (flagged) and change labels"`

### Task 2: Change detector and per-frame switch cost

**Files:**
- Modify: `bsd/handover.py` (detector), `bsd/decode.py` (per-frame switch cost)
- Test: `tests/test_handover.py`, `tests/test_decode.py`

**Interfaces:**
- Consumes: `change_labels` and `frame_features(..., handover=True)` from Task 1; `mirror`,
  `visibility_weights`, `STRIDE` and `HGB_PARAMS` from `bsd/model.py`.
- Produces: `bsd.handover.train_detector(Xs, cs, names, seed=0, visible_rate=None) -> Detector`.
  - Training rows: every `STRIDE`-th frame, with −1 rows dropped, plus mirrored copies that keep
    the same label (a change is team-symmetric).
  - Model: `HistGradientBoostingClassifier(random_state=seed, **HGB_PARAMS)`, fitted single-threaded.
  - Visibility weights are computed from `ball_visible` against the change label when
    `visible_rate` is set.
  - The training data must contain both classes; raise `ValueError` otherwise.
- Produces: `Detector.p_change(X, names) -> (F,) float64`.
  - It is the mean of the predictions on X and on `mirror(X)`, rounded to 6 decimals.
  - It raises `ValueError` if `names` differ from the training names.
  - Rows whose features are all NaN may get any finite value; callers handle out-of-period rows
    through `periods`.
- Produces: `bsd.decode.switch_cost_from(p_change, switch_ha, alpha, floor) -> (F,) float64`, where
  `max(floor, switch_ha − alpha·logit(clip(p, 1e-4, 1 − 1e-4)))`.
  - NaN p gives `switch_ha`.
  - The result is rounded to 6 decimals.
  - `alpha = 0` gives the constant `switch_ha` (even if `floor > switch_ha`, return `switch_ha`
    when alpha is 0).
- Produces: `viterbi(logp, p, switch_cost=None)` and
  `decode_periods(logp, p, periods, n_frames, switch_cost=None)`.
  - `switch_cost[t]` is the H↔A cost of a transition from frame t−1 into frame t.
  - `decode_periods` slices it per period.
  - `None` means `p.switch_ha` everywhere, with the output bit-identical to today.
  - Raise `ValueError` if `switch_cost` has the wrong length or non-finite values.

- [ ] **Step 1: Write failing tests**
  - `test_viterbi_switch_cost_none_identical`: on random logp (seeded) with
    `DecodeParams(20, 20, 40)`, `viterbi(logp, p)` equals `viterbi(logp, p, switch_cost=None)`, and
    equals `viterbi(logp, p, switch_cost=np.full(n, 20.0))`.
  - `test_viterbi_local_cheap_switch`:
    - Evidence: weak H evidence (+1 per frame for H) on frames 0–99, then weak A evidence on frames
      100–199.
    - With a constant switch cost of 1000, the path is all one state.
    - With `switch_cost[100] = 0.0` and 1000 elsewhere, the path switches at frame 100.
  - `test_switch_cost_nan_falls_back`: `switch_cost_from(np.array([np.nan, 0.5]), 20, 4, 1)` gives
    `[20, 20]`, since logit(0.5) = 0.
  - `test_switch_cost_monotone`: for increasing p, the cost is non-increasing and never below the
    floor.
  - `test_detector_mirror_symmetric`:
    - Train on a small synthetic feature table (≥ 500 rows, two classes) with `_home`/`_away`
      columns.
    - `p_change(X)` equals `p_change(mirror(X))` to 1e-9.
    - It raises on reordered names.
  - `test_detector_needs_both_classes`: all-zero labels raise `ValueError`.

- [ ] **Step 2: Run the tests to verify they fail.**
  `venv/bin/python -m pytest tests/test_decode.py tests/test_handover.py -q`

- [ ] **Step 3: Implement.** Keep the `viterbi` inner loop's structure and its tie rule (strict
  `>`, ties to the lower state index). Read the H↔A entry of the cost matrix from
  `switch_cost[t]`, and keep the other entries constant. Keep `test_speed_80k` passing.

- [ ] **Step 4: Run the full suite.** `venv/bin/python -m pytest -q` passes.

- [ ] **Step 5: Commit.** `git commit -m "Stage 1: change detector and per-frame H<->A switch cost"`

### Task 3: Stage 1 experiments and gate

**Files:**
- Create: `bsd/experiments.py` (CLI `python -m bsd.experiments stage1|stage4`)
- Modify: `bsd/bench.py` (only small refactors needed to reuse `prepare`/`heldout_logp` with a
  feature flag; default behaviour unchanged)
- Modify: `docs/notes/bench-results.md` (new "Stage 1" section)
- Test: `tests/test_experiments.py` (new)

**Interfaces:**
- Consumes: Tasks 1–2; `bsd.bench.prepare`, `heldout_logp`, `score`, `learned_emitter`.
- Produces: `bsd.experiments.gate(inc_rows, cand_rows) -> dict`.
  - Inputs: two per-case DataFrames with columns `game`, `seed`, `spell_f1`, `macro_f1`.
  - Returns keys `d_spell`, `n_better`, `d_macro`, `passed`, applying the Global Constraints gate
    exactly.
  - Cases are matched on (game, seed).
- Produces: `bsd/experiments_stage1.json` (committed).
  - It holds the incumbent's and every variant's tuned params, mean scores and per-case scores,
    plus the gate result.
  - It names the winning variant's config: `{"features": bool, "detector": bool, "k_s": float,
    "through_dead_s": float|null, "alpha": float, "floor": float, "switch_ha": float,
    "alive_to_dead": float, "dead_to_alive": float}`. If the gate fails, the winner is `null`.

**Procedure** (the variant order is fixed so the run is reproducible):
- **Incumbent:** re-bench today's learned emitter, with the decoder tuned on the stage grid at
  alpha = 0. Its numbers must be close to `docs/notes/bench-results.md` (0.5440). If they differ by
  more than 0.002, stop and report.
- **Stage grid:**
  - `switch_ha` ∈ {10, 20, 40};
  - (`alive_to_dead`, `dead_to_alive`) ∈ {(10, 20), (20, 40), (40, 80)};
  - `alpha` ∈ {0, 1, 2, 4, 8};
  - `floor` ∈ {1, 5} (only distinct when alpha > 0).

  Extend an axis once if its best value is on the edge, like `tune_decode`. Decoder points are
  decoded from cached held-out evidence; train each fold's emitter and detector once per variant.
- **Variants**, in this order. Stop early per the +0.005 rule, after at least (a)–(c):
  - (a) emitter with `handover=True` features, no detector;
  - (b) incumbent emitter plus detector (`k_s` 0.5, `through_dead_s` None);
  - (c) `handover=True` emitter plus detector;
  - (d) the best of (a)–(c) with `k_s` 1.0;
  - (e) the best so far with `through_dead_s` 2.0. Only for detector variants; otherwise skip.
- Detector training uses the same training pools and visibility rate as the emitter. Its features
  are `frame_features(handover=True)`.
- Write per-case CSVs to `cache/stage1_<variant>.csv`. Print one summary line per variant, plus the
  gate result.

- [ ] **Step 1: Write a failing test.** `test_gate_rule`:
  - Synthetic 6-case frames: +0.012 mean with 4 of 6 cases better and macro −0.005 passes.
  - +0.012 mean with 3 of 6 better fails.
  - +0.009 fails.
  - Macro −0.011 fails.
- [ ] **Step 2: Implement `gate` and the stage1 driver.** The test passes and the full suite is
  green.
- [ ] **Step 3: Check that the shipped output hasn't moved.** Run
  `venv/bin/python -m bsd.run --check-sha c22cc48bd24841d9dffb61f04aa1df26fac3647d17bb8a676c9f9d5d840f9eee`.
  It must exit 0.
- [ ] **Step 4: Run the experiments.** `venv/bin/python -m bsd.experiments stage1`, in the
  background if long. Target: ≤ 45 min in total. If a single variant takes more than 12 min,
  narrow the grid (drop `floor` 5 first, then the D pair (40, 80)) and note it.
- [ ] **Step 5: Record the results.** Write every variant's numbers, the gate result and the
  runtime into `docs/notes/bench-results.md` under "Stage 1 (2026-11-17)". State plainly whether
  the stage passed.
- [ ] **Step 6: Commit.** Commit the code, the test, `bsd/experiments_stage1.json` and the notes.

### Task 4: Adopt Stage 1 (only if the gate passed)

If `experiments_stage1.json` has `winner: null`, skip this task. Record "Task 4: skipped (gate
failed)" in the ledger.

**Files:**
- Modify: `bsd/decode_params.json` (add `"handover": <winner config>` to the learned block, and set
  the decode costs to the winner's)
- Modify: `bsd/run.py` and `bsd/bench.py` (`load_decode_params` returns the handover config)
- Modify: `docs/BSD.md`, `docs/notes/bench-results.md`, `REPRODUCE.md`, `README.md`, and the BSD SHA
  in the parent `../CLAUDE.md`
- Test: `tests/test_run.py`

**Interfaces:**
- When `decode_params.json` has a `handover` block, `bsd.run` in learned mode:
  - builds features with `handover=<features>`;
  - trains the detector on `training_data()` (with the same handover features) when
    `detector` is true;
  - builds `switch_cost_from(p_change, switch_ha, alpha, floor)`;
  - adds the Smart Tagging bonus as today;
  - calls `decode_periods(..., switch_cost=...)`.
- Without the block, it behaves as today.
- `load_decode_params` stays backward compatible.

- [ ] **Step 1: Write a test.** `test_load_decode_params_handover`: a temp JSON with and without
  the `handover` block loads correctly.
- [ ] **Step 2: Implement and wire it up.** The suite is green.
- [ ] **Step 3: Regenerate.** Run `venv/bin/python -m bsd.run` twice and confirm that the
  validator passes and both SHAs are identical. Record the SHA, possession count, mean length and
  D share. These are aggregates only.
- [ ] **Step 4: Update the docs.** Update `docs/BSD.md`: method, the bench table with the new row,
  limitations, and the SHA. Also update `REPRODUCE.md`, `README.md` and `../CLAUDE.md` (SHA only).
- [ ] **Step 5: Commit.** `git commit -m "Adopt Stage 1 hand-over detector"`

### Task 5: Stage 4 experiments and gate

**Files:**
- Modify: `bsd/experiments.py` (`stage4` subcommand), `bsd/model.py` (optional ensemble and
  parameter overrides; defaults unchanged)
- Modify: `docs/notes/bench-results.md` ("Stage 4" section)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `train_emitter(Xs, ys, names, seed=0, visible_rate=None, hgb=None, stride=None)`.
  - `hgb` is a dict merged over `HGB_PARAMS`, and `stride` overrides `STRIDE`.
  - Defaults are unchanged and bit-identical.
- Produces: `EnsembleEmitter(emitters)`, with the same `.emit(X, names)` and `.logp(X)` interface.
  - It averages the members' mirror-averaged probabilities, then applies
    `log(p/prior)` exactly as `Emitter.logp` does (using the members' common prior).
  - It rounds to 6 decimals.
- Produces: `bsd/experiments_stage4.json` (committed), with the same schema idea as stage 1.
  - Its winner config is `{"ensemble_k": int, "hgb": dict, "stride": int,
    "train_seeds": [int], "decode": {...}, ...plus Stage 1 winner config if adopted}`.

**Procedure:** the incumbent is whatever Tasks 3–4 left: Stage 1's winner if adopted, otherwise
the shipped model. Use the same 6 cases and the same decoder tuning (Stage 1 grid; alpha axis only
if the detector was adopted). Variants, in order, with the same caps and early-stop rule:
- (a) ensemble of K = 3, with `random_state` 0, 1, 2 and the training degrade seeds shifted per
  member (100–102, 103–105, 106–108);
- (b) `hgb={"max_iter": 300, "learning_rate": 0.05}`;
- (c) `stride=1`;
- (d) training seeds 100–105 (6 instead of 3);
- (e) the best two of (a)–(d) combined.

Any variant that would push `bsd.run` past ~20 min end-to-end is skipped and noted.

- [ ] **Step 1: Write failing tests.**
  - `test_train_emitter_defaults_unchanged`: same seed and same data, with explicit
    `hgb=None, stride=None`, give equal `logp`.
  - `test_ensemble_single_member_equals_emitter`: an ensemble of one gives the same `logp` as the
    member.
- [ ] **Step 2: Implement.** The suite is green.
- [ ] **Step 3: Run the experiments.** `venv/bin/python -m bsd.experiments stage4`. Target ≤ 60 min.
- [ ] **Step 4: Record the results.** Add every variant's numbers, the gate result and the runtime
  to `docs/notes/bench-results.md`.
- [ ] **Step 5: Commit.**

### Task 6: Adopt Stage 4 (only if the gate passed)

This is the same pattern as Task 4:
- `decode_params.json` gains a `"training"` block (`ensemble_k`, `hgb`, `stride`, `train_seeds`);
- `bsd.run` and `training_data` honour it;
- its absence means today's behaviour;
- add a test for loading the block;
- run `bsd.run` twice to confirm the same SHA;
- update the docs and the SHA everywhere;
- commit with `git commit -m "Adopt Stage 4 training tweaks"`.

If the gate failed, skip this task and note it in the ledger.

### Task 7: Final checks

- [ ] Run the full test suite.
- [ ] Clean reproduction:
  - `git archive HEAD` into a temp folder;
  - symlink this repo's `data/` into the temp folder (the sample data checksums were verified by
    `bsd.fetch_sample`);
  - make sure the temp folder has an empty `cache/`;
  - run `venv/bin/python -m bsd.run` there, with this repo's venv interpreter and the temp folder
    as the working directory;
  - check that the SHA equals the committed one.
- [ ] Validate `evaluation/results/BSD.csv`: run `venv/bin/python evaluation/validate_submission.py evaluation/results/BSD.csv`.
- [ ] Public-window correctness check, **once**: `venv/bin/python evaluation/run_evaluation.py`.
  Report the BSD and baseline rows. Change nothing because of it, and record it in
  `docs/notes/bench-results.md`.
- [ ] Commit the docs.

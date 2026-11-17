# BSD Possession Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `evaluation/results/BSD.csv`, a per-frame H/A/D/X possession file for BAR-ATH that maximises spellF1, from one reproducible command.

**Architecture:** Parse tracking (BAR-ATH + 3 Metrica sample games) into one `Tracking` container → team-level per-frame features → per-frame evidence (`log P(H/A/D)`), first hand-set (fallback), then learned on degraded sample games → Viterbi decoder with switch costs tuned leave-one-game-out on spellF1 → Smart Tagging hints → CSV.

**Tech Stack:** Python 3.14 (`venv/`), numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, pytest 9.2.0. Repo scorer `evaluation/possession_metrics.py` reused as the tuning objective.

**Spec:** `docs/superpowers/specs/2026-11-16-possession-hmm-design.md`

## Global Constraints

- All paths below are relative to `more-than-a-hack-2027-main/`. Run everything with `venv/bin/python`.
- Label codes everywhere are the scorer's: `H=0, A=1, D=2, X=3` (`possession_metrics.H/A/D/X`). Frame index `i` = video frame `i` = submission row `i`.
- BAR-ATH ATD `frame_id` counts from 1: index = `frame_id - 1`.
- Coordinates in metres inside `Tracking`; BAR-ATH pitch 105.3 × 68.0 (`games.GAMES["BAR-ATH"]["pitch"]`); sample games 105 × 68.
- **Never** tune any number on `evaluation/gt/BAR-ATH_possession_gt_public.csv`; it is read in Task 8 for conventions and scored once in Task 11.
- **No literal BAR-ATH frame numbers in `bsd/`** except via `games.half_video_blocks("BAR-ATH")`.
- Never send match data anywhere; never `git push`. Local `git init`/commits are fine; `data/`, `cache/`, `evaluation/results/*.csv` must stay untracked.
- Determinism: every RNG is `np.random.default_rng(seed)` with an explicit seed; models single-threaded (`n_jobs=1`); probabilities rounded to 6 decimals before decoding.
- Sample data pinned to commit `e706dd506b360d69d9d123d5b8026e7294b13996` of `metrica-sports/sample-data`, stored in `data/sample/Sample_Game_{1,2,3}/` (already downloaded once by hand; Task 0 makes it a script).
- Submission file: `evaluation/results/BSD.csv`; tech doc: `docs/BSD.md` (PDF export if a converter is available).

## Review Focus

1. **Sample-game ball missingness leaks the label** (measured: P(ball missing | dead) = 0.85 vs 0.15 alive in Game 1). Expect the degraded sample games to have ball missingness independent of state → test in Task 7 (`test_mask_independent_of_label`).
2. **Off-by-one alignment** between ATD `frame_id` and submission rows → tests in Task 1 (`test_epts_first_record_is_index_0`) and Task 6 (`test_submission_rows_and_frames`).
3. **Direction of play flips at half time**; team-relative "forward" features must flip with it → Task 4 (`test_attack_direction_per_period`).
4. **Second-half pre-kickoff stretch** must be produced by the model, not a constant → Task 6 (`test_no_hardcoded_frames` greps `bsd/` for 5–6-digit literals); the stretch's label mix is printed as a diagnostic only.
5. **Non-determinism** (thread counts, float noise) changing the file → Task 9 (`test_training_deterministic`) and Task 11 (two full runs, identical SHA-256).

---

### Task 0: Project setup

**Files:**
- Create: `requirements-bsd.txt`, `bsd/__init__.py`, `bsd/fetch_sample.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_fetch.py`
- Modify: `.gitignore` (add `cache/`)

**Interfaces:**
- Produces: `bsd.fetch_sample.SAMPLE_SHA: str`, `sample_urls() -> dict[str, str]` (relative path under `data/sample/` → raw URL), `fetch(dest: str = "data/sample") -> None` (skips files that exist).

- [ ] **Step 1:** `requirements-bsd.txt` = `-r requirements.txt` plus `numpy==2.5.5`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `pytest==9.2.0`. Run `venv/bin/pip install -r requirements-bsd.txt`. Expected: installs cleanly.
- [ ] **Step 2:** `git init` (local only), add `cache/` to `.gitignore`. Run `git status --porcelain | grep -E "^\?\? data/|\.csv$|\.mp4$"` → expected: only `evaluation/results/baseline_*.csv` and nothing under `data/`.
- [ ] **Step 3: Failing test** `tests/test_fetch.py::test_urls_pinned`: every URL from `sample_urls()` contains `SAMPLE_SHA`, there are 9 files, keys include `Sample_Game_3/Sample_Game_3_tracking.txt`. `tests/conftest.py` puts the repo root on `sys.path`.
- [ ] **Step 4:** Run `venv/bin/python -m pytest tests/test_fetch.py -v` → FAIL (module missing). Implement `bsd/fetch_sample.py` (urllib, no network in tests). Re-run → PASS.
- [ ] **Step 5: Commit** `git add -A && git commit -m "chore: bsd package skeleton, pinned deps, sample-data fetcher"` (check `git status` shows no data first).

### Task 1: Tracking container + FIFA/EPTS parser

**Files:**
- Create: `bsd/tracking.py`, `bsd/io_epts.py`, `tests/test_io_epts.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class Tracking:
      players: np.ndarray      # (F, S, 2) float32 metres, NaN missing
      team: np.ndarray         # (S,) int8: 0 home, 1 away
      ball: np.ndarray         # (F, 2) float32 metres, NaN missing
      periods: list[tuple[int, int, int]]  # (period, first_index, last_index) inclusive
      pitch: tuple[float, float]
      fps: int = 25
  ```
  `read_epts(raw_path, meta_path, pitch, periods=None) -> Tracking`; groups may be `x,y` or `x,y,speed` (speed ignored); first `<Team>` in metadata = home; slot order = `<Player>` order. `periods=None` → read from metadata `first_half_start/end`, `second_half_start/end` frame values (Sample Game 3), converted to index = frame − 1.
  `bar_ath_tracking() -> Tracking` (uses `games.require`, `games.GAMES` pitch, periods from `games.half_video_blocks("BAR-ATH")`), cached to `cache/bar_ath.npz`.

- [ ] **Step 1: Failing tests** with a 4-line synthetic raw file + minimal XML written to `tmp_path`:
  - `test_epts_first_record_is_index_0`: line `1:...` lands in `players[0]`.
  - `test_epts_metres`: `0.5,0.5` → `(52.65, 34.0)` for pitch (105.3, 68.0) within 1e-3.
  - `test_epts_nan_and_two_value_groups`: `NaN,NaN` → NaN; `x,y` groups parse like `x,y,s`.
  - `test_epts_team_from_meta`: slots of the first `<Team>` get 0.
- [ ] **Step 2:** Run → FAIL. Implement streaming line parser (no kloppy). Run → PASS.
- [ ] **Step 3: Real-data check** `venv/bin/python -c "from bsd.io_epts import bar_ath_tracking as f; t=f(); print(t.players.shape, t.periods)"` → `(199319, 46, 2) [(1, 11185, 90651), (2, 110808, 190197)]`; Sample Game 3 → 143761 frames, 2 periods.
- [ ] **Step 4: Commit.**

### Task 2: Metrica CSV parser (Sample Games 1–2)

**Files:**
- Create: `bsd/io_metrica_csv.py`, `tests/test_io_metrica_csv.py`

**Interfaces:**
- Produces: `read_metrica_csv(home_csv, away_csv, pitch=(105.0, 68.0)) -> Tracking`. Header is 3 rows (team row, jersey row, column names); columns `Period, Frame, Time [s]`, then `PlayerN` x/y pairs, last pair `Ball`. Ball taken from the home file. Periods from the `Period` column (index = Frame − 1).

- [ ] **Step 1: Failing test** `test_metrica_csv_shapes` on a 3-frame synthetic pair (2 home players, 1 away, ball): shapes `(3, 3, 2)`, `team == [0,0,1]`, metres scaling, periods `[(1,0,1),(2,2,2)]`.
- [ ] **Step 2:** Run → FAIL; implement with `pd.read_csv(skiprows=2)`; run → PASS.
- [ ] **Step 3:** Real check: Game 1 → 145006 frames; Game 2 → 141156 frames.
- [ ] **Step 4: Commit.**

### Task 3: Sample-game events → labels

**Files:**
- Create: `bsd/events.py`, `tests/test_events.py`

**Interfaces:**
- Produces: `load_events_csv(path) -> pd.DataFrame`, `load_events_json(path, home_team_id) -> pd.DataFrame`, both with columns `team ('H'|'A'), type (upper), subtype (upper, '' if none, list subtypes joined with '-'), period (int), start (int index), end (int index)`, index = frame − 1, sorted by `start`.
  `labels_from_events(ev: pd.DataFrame, n_frames: int, periods) -> np.ndarray` (int8 codes).
  `sample_game(n: int) -> tuple[Tracking, np.ndarray]` (tracking via Task 1/2 readers + labels), tracking cached to `cache/sample_{n}.npz`. Game 3 home team id = first `<Team>` in its metadata.
- Rules (from spec §2; Task 8 may revise them, and the tests here change with them):
  - Outside periods and before the first event of each period → `X`.
  - **Stoppage triggers:** `BALL OUT`; `SHOT` whose subtype contains `OUT` or `GOAL`; `FAULT RECEIVED`; `CHALLENGE` whose subtype contains `FAULT`. Dead from the trigger's `end` (or `start` if `end < start`) up to (excluding) the next **restart**: any `SET PIECE`, or `PASS` with subtype `GOAL KICK`. A trigger with no restart before the period ends → dead to period end.
  - **Possession owner while alive:** team of the latest `PASS`, `CARRY`, `RECOVERY`, `SET PIECE`, `SHOT`; `BALL LOST` and `CHALLENGE` never switch on their own.

- [ ] **Step 1: Failing tests** (synthetic event frames):
  - `test_ball_out_until_throw_in`: A passes at 10, BALL OUT A at 20–22, SET PIECE THROW IN H at 40 → frames 22..39 `D`, 40.. `H`, 10..21 `A`.
  - `test_recovery_switches_ball_lost_does_not`: H pass 0, H BALL LOST 10, A RECOVERY 15 → 10..14 `H`, 15.. `A`.
  - `test_goal_dead_until_kickoff`: SHOT GOAL H at 50, SET PIECE KICK OFF A at 90 → 50..89 `D`.
  - `test_outside_periods_x`: frames between periods are `X`.
  - `test_json_subtype_lists`: JSON subtypes list `[{"name":"FREE KICK"},{"name":"INDIRECT"}]` → `"FREE KICK-INDIRECT"`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Real check prints per game: dead share, possession count, mean possession seconds (expected roughly 25–40 % dead, possessions of ~10–20 s; wildly different numbers mean a rule bug).
- [ ] **Step 4: Commit.**

### Task 4: Per-frame features

**Files:**
- Create: `bsd/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `Tracking`.
- Produces: `frame_features(tr: Tracking) -> tuple[np.ndarray, list[str]]` → `(F, K)` float32, NaN allowed. `mirror(X, names) -> np.ndarray` swaps every `*_home`/`*_away` pair and negates every `*_diff` column. `attack_sign(tr) -> np.ndarray` `(F,)` +1 if home attacks +x in that frame's period else −1, inferred per period from mean team x over the period (home centroid left of away → home attacks +x).
- Required feature groups (names suffixed `_home`/`_away` or `_diff` = home − away):
  - ball: `ball_visible`, `ball_out` (outside pitch by > 0.5 m), `ball_speed` (from positions, 5-frame centred difference, m/s), `ball_dist_edge`, `near_{home,away}` (nearest player distance), `n2_{home,away}`/`n5_{home,away}` (players within 2 m / 5 m), `frames_since_ball`.
  - shape: `cx`/`cy`/`spread`/`speed`/`vx_fwd` (mean velocity toward own attacking goal) per team, `depth_diff` (forward-ness of team centroids), `n_tracked`.
  - stoppage: `speed_all`, `speed_all_trend` (2 s mean minus previous 2 s), `ball_static` (ball speed < 1 m/s over 1 s).
  - context: centred rolling means of a chosen subset over ±1 s and ±3 s (NaN-aware).
- [ ] **Step 1: Failing tests:**
  - `test_ball_out_flag`: ball at x = −1 m → `ball_out == 1`; ball at centre → 0.
  - `test_near_distances`: home player 1 m from ball, away 3 m → `near_home≈1, near_away≈3`.
  - `test_missing_ball_nan_safe`: all-NaN ball → no exception, ball columns NaN, `ball_visible == 0`.
  - `test_mirror_swaps`: `mirror(frame_features(tr))` equals `frame_features(tr_with_teams_swapped)` within 1e-5 (NaN-equal).
  - `test_attack_direction_per_period`: home centroid at x=30 in period 1 and x=75 in period 2 → `attack_sign` +1 then −1.
- [ ] **Step 2:** Run → FAIL; implement vectorised numpy; run → PASS.
- [ ] **Step 3:** Real check: BAR-ATH features computed in < 2 min, `X.shape == (199319, K)`.
- [ ] **Step 4: Commit.**

### Task 5: Viterbi decoder

**Files:**
- Create: `bsd/decode.py`, `tests/test_decode.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class DecodeParams:
      switch_ha: float      # cost of H<->A directly
      alive_to_dead: float  # H/A -> D
      dead_to_alive: float  # D -> H/A
  def viterbi(logp: np.ndarray, p: DecodeParams) -> np.ndarray      # (F,3) -> (F,) codes 0..2
  def decode_periods(logp, p, periods, n_frames) -> np.ndarray       # X outside periods
  ```
  Costs are subtracted log-score penalties; ties broken toward the lower state index; logp rounded to 6 decimals first.
- [ ] **Step 1: Failing tests:**
  - `test_constant_evidence_single_state`.
  - `test_short_flicker_ignored`: 1000 frames favouring H by 0.5 nats, 10 frames favouring A by 0.5, `switch_ha=20` → all H.
  - `test_sustained_switch`: 200 frames of A evidence after 500 of H → switch within 50 frames of the true change.
  - `test_periods_independent`: X outside periods; each period decoded separately.
  - `test_deterministic_ties`: equal evidence → all H.
- [ ] **Step 2:** Run → FAIL; implement O(F·9) numpy loop; run → PASS. Must decode 80k frames in < 5 s.
- [ ] **Step 3: Commit.**

### Task 6: Milestone 1 — hand-set emitter, end-to-end `BSD.csv`

**Files:**
- Create: `bsd/heuristic.py`, `bsd/output.py`, `bsd/run.py`, `tests/test_output.py`, `tests/test_no_literals.py`

**Interfaces:**
- Produces: `heuristic_logp(X, names) -> np.ndarray (F,3)`; `write_submission(codes, path) -> str` (returns SHA-256 hex); `python -m bsd.run --mode heuristic|learned [--no-tagging]` writing `evaluation/results/BSD.csv`, running `evaluation/validate_submission.py` on it, printing SHA-256, possession count, mean possession seconds, D share per half, and the label mix of the second half's first 5 minutes of graded frames (diagnostic only).
- Heuristic (first-principles, documented in the docstring with reasons): team evidence from `near_home − near_away` and `n2` counts when the ball is visible; when not visible, from `vx_fwd_diff`/`depth_diff` with low weight; D evidence from `ball_out`, low `speed_all`, `ball_static` near an edge; decode with switch cost equivalent to ~2 s of moderate evidence.
- [ ] **Step 1: Failing tests:**
  - `test_submission_rows_and_frames`: writing 199319 codes gives header `frame,label`, frames `0..199318`, labels in `HADX`.
  - `test_no_hardcoded_frames`: no 5–6-digit integer literals in `bsd/*.py` (regex `\b\d{5,6}\b`).
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** `venv/bin/python -m bsd.run --mode heuristic --no-tagging` → validator prints OK; record SHA. **This is the fallback submission.**
- [ ] **Step 4: Commit.**

### Task 7: Degrading the sample games

**Files:**
- Create: `bsd/degrade.py`, `tests/test_degrade.py`

**Interfaces:**
- Produces: `gap_profile(tr: Tracking) -> dict` with `gap_lengths` (int array of BAR-ATH missing-ball run lengths inside periods), `missing_rate` (float), `n_tracked` (array of per-frame tracked-object counts inside periods); computed from BAR-ATH tracking only (label-free), cached in `cache/gap_profile.npz`.
  `fill_ball(tr) -> Tracking`: linear interpolation of ball positions across gaps within each period (edges held constant).
  `degrade(tr, profile, seed) -> Tracking`: (1) `fill_ball`; (2) ball mask built by alternating visible/missing runs, missing lengths sampled from `gap_lengths`, visible lengths sampled so the overall rate ≈ `missing_rate`, **independent of labels**; (3) per frame drop random player observations to match a sampled `n_tracked`; (4) Gaussian jitter σ = 0.3 m; (5) re-assign slot columns in random blocks of 10–60 s within each team.
- [ ] **Step 1: Failing tests:**
  - `test_missing_rate`: degraded missing rate within ±0.03 of profile's.
  - `test_mask_independent_of_label`: with a synthetic label array, |P(miss|D) − P(miss|alive)| < 0.05.
  - `test_degrade_deterministic`: same seed → identical arrays; different seed → different.
  - `test_fill_ball_no_nan_inside_period`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3: Commit.**

### Task 8: Convention check + leave-one-game-out bench

**Files:**
- Create: `bsd/bench.py`, `docs/notes/label-conventions.md`, `tests/test_bench.py`
- Modify (only if the check demands it): `bsd/events.py` rules and their tests from Task 3.

**Interfaces:**
- Produces: `score(gt_codes, pred_codes) -> dict` with `spell_f1`, `macro_f1`, `n_poss`, `mean_poss_s` using `possession_metrics` (`valid_runs`, `spell_scores`, `evaluate`); `bench(emit_fn, params: DecodeParams, seeds=(0,1)) -> pd.DataFrame` one row per held-out game×seed; `python -m bsd.bench --mode heuristic|learned`.
- [ ] **Step 1: Convention check (qualitative, no numbers fitted).** For the public window, list the GT's D runs and hand-overs next to BAR-ATH cues (ball out flag, speed drop, Smart Tagging intervals) and look at a few in the video. Write findings to `docs/notes/label-conventions.md`: when D starts (line crossing vs later), when it ends (restart touch?), whether contested touches switch. If they contradict Task 3's rules, change the rules and their tests.
- [ ] **Step 2: Failing test** `test_score_perfect`: `score(x, x)["spell_f1"] == 1.0` on a synthetic label array; `test_score_constant_zero`.
- [ ] **Step 3:** Run → FAIL; implement; run → PASS.
- [ ] **Step 4:** `venv/bin/python -m bsd.bench --mode heuristic` → table printed and saved to `cache/bench_heuristic.csv`. This is the bar Task 9 must beat.
- [ ] **Step 5: Commit.**

### Task 9: Learned emitter + decoder tuning

**Files:**
- Create: `bsd/model.py`, `tests/test_model.py`
- Modify: `bsd/bench.py` (add tuning), `bsd/run.py` (`--mode learned`)

**Interfaces:**
- Produces: `train_emitter(Xs: list[np.ndarray], ys: list[np.ndarray], names, seed=0) -> Emitter` (trains on X ∪ mirror(X) with mirrored labels H↔A, drops `X`-label frames; `HistGradientBoostingClassifier(random_state=seed)`, threads pinned to 1 via `threadpoolctl` or `OMP_NUM_THREADS=1` set in `bsd/__init__.py`); `Emitter.logp(X) -> np.ndarray (F,3)` = log of class probabilities divided by training class priors, rounded to 6 decimals; `tune_decode(emit_fn, grid) -> DecodeParams` maximising mean held-out spellF1 over a grid of `switch_ha ∈ {5,10,20,40,80}`, `alive_to_dead ∈ {5,10,20,40}`, `dead_to_alive ∈ {5,10,20,40}`; final params saved to `cache/decode_params.json` and printed.
- [ ] **Step 1: Failing tests:** `test_logp_shape_and_normalisation` (probabilities sum to 1 before prior division); `test_training_deterministic` (two trainings, identical `logp`); `test_mirror_symmetry` (logp(mirror(X)) ≈ logp(X) with H/A columns swapped, within 0.05).
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** `venv/bin/python -m bsd.bench --mode learned` (leave-one-game-out: train on 2 degraded games × 3 seeds, tune decode on held-out). Adopt learned mode **only if** mean spellF1 beats `cache/bench_heuristic.csv`; record both in `docs/notes/bench-results.md`.
- [ ] **Step 4:** Final model: train on all 3 games, tuned params; `python -m bsd.run --mode learned --no-tagging` → validated `BSD.csv`.
- [ ] **Step 5: Commit.**

### Task 10: Smart Tagging hints

**Files:**
- Create: `bsd/tagging.py`, `tests/test_tagging.py`
- Modify: `bsd/run.py` (tagging on by default)

**Interfaces:**
- Consumes: `starter/load_smart_tagging.load()` output (or parse the XML directly, case-insensitive).
- Produces: `tag_bonus(n_frames, fps=25) -> np.ndarray (F,3)` added to `logp` before decoding. Fixed weights (first principles, not tuned): `SET PIECES` instance → +2.0 nats to D over the 3 s before its start, +1.0 to the tagged team over its first 2 s (none if `Team` is `N/A`); `GOALS` → +2.0 D from its end to the next `Kick Off` set piece start (cap 90 s); team-tagged phase codes → +0.3 to that team over the interval.
- [ ] **Step 1: Failing tests** on a synthetic XML: `test_set_piece_dead_before`, `test_na_team_no_team_bonus`, `test_case_insensitive_codes`, `test_goal_dead_until_kickoff`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** `python -m bsd.run --mode <adopted>` with and without tagging; keep tagging unless the possession count or D share becomes implausible (> 2× the no-tagging run). Log both in `docs/notes/bench-results.md`.
- [ ] **Step 4: Commit.**

### Task 11: Reproducibility, public-window check, write-up

**Files:**
- Create: `docs/BSD.md`, `REPRODUCE.md`
- Modify: `bsd/run.py` (`--check-sha <hex>`), `CLAUDE.md` in the parent folder (commands)

- [ ] **Step 1:** Run `venv/bin/python -m bsd.run` twice from a clean `cache/` → identical SHA-256. Record it in `REPRODUCE.md` together with the exact commands (`pip install -r requirements-bsd.txt`, `python -m bsd.fetch_sample`, `python -m bsd.run`).
- [ ] **Step 2:** Score once against the public window: `venv/bin/python evaluation/run_evaluation.py --no-detail`. Record the line in `docs/BSD.md` as a correctness check. No changes are made because of it.
- [ ] **Step 3:** `venv/bin/python evaluation/validate_submission.py evaluation/results/BSD.csv` → OK.
- [ ] **Step 4:** Write `docs/BSD.md` per SUBMISSION.md: approach, assets used/dropped (video, broadcast, homography dropped and why), training data (Metrica sample games, pinned commit, labels derived by code from their event data, credit Metrica), possession rules (Task 3/8 conventions, tagging weights), limitations, reproduction. Export `BSD.pdf` if `pandoc` is installed; otherwise leave `.md` (accepted format).
- [ ] **Step 5:** `git status` shows no data/CSV/cache tracked; commit. **Do not push.**

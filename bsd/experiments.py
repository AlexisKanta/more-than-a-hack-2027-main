"""Gated stage experiments on the leave-one-game-out bench: ``python -m bsd.experiments stage1|stage2|stage4``.

Stage 1 (hand-over detector; ``docs/superpowers/specs/2026-11-17-bsd-improvements-design.md``)
compares the incumbent (learned emitter, constant decoder costs) with hand-over variants on the
same 6 cases as ``python -m bsd.bench`` (sample games 1-3 held out, degradation seeds 0 and 1;
each fold trained on the two other games degraded with ``bench.TRAIN_SEEDS``):

- incumbent: today's learned emitter, decoder tuned on ``STAGE1_GRID`` at alpha = 0;
- (a) emitter with the hand-over feature columns, no detector;
- (b) incumbent emitter plus the change detector (k_s 0.5, no through-dead changes);
- (c) hand-over emitter plus detector;
- (d) the best of (a)-(c) with k_s 1.0 (detector variants only);
- (e) the best so far with through_dead_s 2.0 (detector variants only).

(d) and (e) are dropped once a variant adds less than ``EARLY_STOP_GAIN`` mean spellF1 over the
best variant so far. The detector always sees ``frame_features(handover=True)`` and is trained on
the emitter's training pool with the same visibility weights. Each fold's emitter and detector
are trained once (one process per fold, single-threaded fits); the decoder grid is decoded from
the cached held-out evidence, one process per case. With a detector the H<->A cost of frame t is
``switch_cost_from(p_change, switch_ha, alpha, floor)``; alpha = 0 means constant costs (decoded
once, whatever the floor). The best variant is compared with the incumbent by ``gate``.

Writes ``cache/stage1_<variant>.csv`` (tuned per-case scores), ``cache/stage1_<variant>_grid.csv``
(every decoded point) and ``bsd/experiments_stage1.json`` (committed record).

Stage 4 (cheap training tweaks) uses the same machinery. The incumbent is the committed model
(``bsd/decode_params.json``: learned emitter plus its ``handover`` block, i.e. Stage 1's winner);
each variant changes how it is trained and is re-tuned on ``STAGE1_GRID`` (with the alpha axis
when the detector is on):

- (a) an ensemble of 3 H/A/D emitters (``model.EnsembleEmitter``); member m has random_state m
  and the training degrade seeds shifted by m x 3 (100-102, 103-105, 106-108);
- (b) ``hgb`` override max_iter 300, learning_rate 0.05 for the H/A/D emitter;
- (c) ``stride`` 1 for the H/A/D emitter;
- (d) training degrade seeds 100-105 for the emitter and the detector;
- (e) the best two of (a)-(d) combined.

(a)-(c) leave the detector exactly as the incumbent's. (a)-(d) are separate tweaks and all run;
(e) is recorded as an early stop if it adds less than ``EARLY_STOP_GAIN`` over the best of
(a)-(d). Ranking uses unrounded means. Each variant's ``bsd.run`` time is estimated from the
per-fold training CPU time ratio to the incumbent (``est_run_seconds``); a variant estimated past
``RUN_LIMIT_S`` is skipped. Writes ``cache/stage4_<variant>[_grid].csv`` and
``bsd/experiments_stage4.json``.

Stage 2 (more training games) adds the 7 public DFL Bundesliga matches (``bsd.io_dfl``; labels are
the provider's per-frame BallStatus / BallPossession flags) to the training pool of the emitter
and the detector, each DFL game degraded like the Metrica games. Primary (gate): the 6 Metrica
cases, each fold trained on the 2 other Metrica games x ``TRAIN_SEEDS`` plus the DFL games.
Secondary: each DFL game held out once at seed 0, incumbent (3 Metrica games x ``TRAIN_SEEDS``)
vs candidate (the same plus the other DFL games), decoded at the variant's primary-tuned params;
the candidate's mean spellF1 there must not be more than ``SECONDARY_MAX_DROP`` below the
incumbent's (``secondary_check``), on top of the gate. Variants (``STAGE2_VARIANTS``): (a) 1 DFL
training seed, (b) DFL rows weighted 0.5 (``weights`` of ``train_emitter`` / ``train_detector``),
(c) 3 DFL training seeds (dropped by the early-stop rule if (b) adds < ``EARLY_STOP_GAIN`` over
(a)). Training rows are pre-strided by ``STRIDE`` in the parent (same rows as the default fits).
Writes ``cache/stage2_<variant>[_grid|_secondary].csv`` and ``bsd/experiments_stage2.json``.
"""
import argparse
import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bsd import bench
from bsd.decode import DecodeParams, decode_periods, switch_cost_from
from bsd.features import frame_features
from bsd.model import STRIDE, EnsembleEmitter
from bsd.tracking import Tracking

ROOT = bench.ROOT
CACHE = ROOT / "cache"
STAGE1_JSON = ROOT / "bsd" / "experiments_stage1.json"
STAGE4_JSON = ROOT / "bsd" / "experiments_stage4.json"
WORKERS = 6

# Gate (design spec, verbatim): adopt only if all three hold.
GATE_MIN_D_SPELL = 0.010  # mean spellF1 gain
GATE_MIN_BETTER = 4  # cases (of 6) with a higher spellF1, paired on (game, seed)
GATE_MAX_MACRO_DROP = 0.010  # mean macroF1 loss
EARLY_STOP_GAIN = 0.005
INCUMBENT_TOL = 0.002  # allowed distance of the re-benched incumbent from the committed record

STAGE1_GRID = {
    "switch_ha": (10, 20, 40),
    "dead": ((10, 20), (20, 40), (40, 80)),  # (alive_to_dead, dead_to_alive)
    "alpha": (0, 1, 2, 4, 8),
    "floor": (1, 5),  # only matters when alpha > 0
}
PARAMS = ("switch_ha", "alive_to_dead", "dead_to_alive", "alpha", "floor")
# Stage 4 tweaks (design spec / task brief, verbatim), each applied on top of the incumbent.
STAGE4_SEEDS = tuple(range(bench.TRAIN_SEEDS[0], bench.TRAIN_SEEDS[0] + 2 * len(bench.TRAIN_SEEDS)))
STAGE4_TWEAKS = {
    "a_ensemble": {"ensemble_k": 3},
    "b_hgb": {"hgb": {"max_iter": 300, "learning_rate": 0.05}},
    "c_stride1": {"stride": 1},
    "d_seeds": {"train_seeds": list(STAGE4_SEEDS)},
}
# `bsd.run` timing of the Stage 1 model on the development machine: whole run 4 min 14 s (Task 4);
# its final-model training alone (all 3 sample games x TRAIN_SEEDS, one process) 176 s for the
# emitter and 60 s for the detector.
RUN_SECONDS = 254
FINAL_EMIT_S, FINAL_DET_S = 176, 60
RUN_LIMIT_S = 20 * 60  # skip a variant whose `bsd.run` would exceed ~20 min
CASE_COLS = ["game", "seed", "spell_f1", "macro_f1", "n_poss", "mean_poss_s", "gt_n_poss", "gt_mean_poss_s"]


def gate(inc_rows, cand_rows):
    """Apply the stage gate to two per-case frames (columns game, seed, spell_f1, macro_f1).

    Differences are candidate minus incumbent, paired on (game, seed) and rounded to 6 decimals.
    """
    key = ["game", "seed"]
    cols = key + ["spell_f1", "macro_f1"]
    m = inc_rows[cols].merge(cand_rows[cols], on=key, suffixes=("_inc", "_cand"), validate="one_to_one")
    if not len(m) == len(inc_rows) == len(cand_rows):
        raise ValueError("incumbent and candidate must cover the same (game, seed) cases")
    d_spell = round(float((m["spell_f1_cand"] - m["spell_f1_inc"]).mean()), 6)
    n_better = int((m["spell_f1_cand"] > m["spell_f1_inc"]).sum())
    d_macro = round(float((m["macro_f1_cand"] - m["macro_f1_inc"]).mean()), 6)
    passed = d_spell >= GATE_MIN_D_SPELL and n_better >= GATE_MIN_BETTER and d_macro >= -GATE_MAX_MACRO_DROP
    return {"d_spell": d_spell, "n_better": n_better, "n_cases": len(m), "d_macro": d_macro,
            "passed": bool(passed)}


# --- decoder tuning --------------------------------------------------------------------------

def _key(pt):
    return tuple(pt[p] for p in PARAMS)


def _points(grid, detector, floor_na):
    """Grid points in evaluation order; alpha = 0 points carry ``floor_na`` and appear once."""
    alphas = grid["alpha"] if detector else (0,)
    out, seen = [], set()
    for sw, (ad, da), al, fl in itertools.product(grid["switch_ha"], grid["dead"], alphas, grid["floor"]):
        pt = dict(switch_ha=sw, alive_to_dead=ad, dead_to_alive=da, alpha=al,
                  floor=floor_na if al == 0 else fl)
        if _key(pt) not in seen:
            seen.add(_key(pt))
            out.append(pt)
    return out


def _best(rows, points):
    """First point with the highest mean spellF1, then mean macroF1."""
    acc = {}
    for r in rows:
        acc.setdefault(_key(r), []).append((r["spell_f1"], r["macro_f1"]))
    means = {k: np.mean(v, axis=0) for k, v in acc.items()}
    keyed = [(means[_key(pt)][0], means[_key(pt)][1], -i) for i, pt in enumerate(points)]
    return points[keyed.index(max(keyed))]


def _grow(vals, v):
    """Axis values with one more step past ``v`` if it is on the edge (double / halve)."""
    vals = sorted(vals)
    if len(vals) < 2:
        return vals

    def mul(x, f):
        return tuple(e * f for e in x) if isinstance(x, tuple) else x * f
    if v == vals[-1]:
        vals.append(mul(vals[-1], 2))
    elif v == vals[0] and mul(vals[0], 0.5) != vals[0]:  # alpha 0 cannot be halved
        vals.insert(0, mul(vals[0], 0.5))
    return vals


def tune_stage(score_points, grid, detector):
    """Grid-search decoder costs, like ``bench.tune_decode`` with alpha and floor added.

    ``score_points(points)`` returns one row per (point, case) with the point's ``PARAMS`` plus
    ``spell_f1`` and ``macro_f1``. Without a detector only alpha = 0 is searched. An axis whose
    best value is on its edge is extended once (floor only when the best alpha is > 0) and the
    new points are decoded. Returns ``(best point dict, rows)``.
    """
    floor_na = grid["floor"][0]
    points = _points(grid, detector, floor_na)
    rows = list(score_points(points))
    best = _best(rows, points)
    ext = dict(grid)
    ext["switch_ha"] = _grow(grid["switch_ha"], best["switch_ha"])
    ext["dead"] = _grow(grid["dead"], (best["alive_to_dead"], best["dead_to_alive"]))
    if detector:
        ext["alpha"] = _grow(grid["alpha"], best["alpha"])
        if best["alpha"] > 0:
            ext["floor"] = _grow(grid["floor"], best["floor"])
    done = {_key(pt) for pt in points}
    new = [pt for pt in _points(ext, detector, floor_na) if _key(pt) not in done]
    if new:
        rows += list(score_points(new))
        points += new
        best = _best(rows, points)
    return best, rows


# --- workers ---------------------------------------------------------------------------------

def _fit_member(job):
    """Train one fold's H/A/D emitter (one ensemble member); return (emitter, CPU seconds)."""
    train, names, seed, hgb, stride = job
    t0 = time.process_time()
    em = bench.learned_emitter([(t[0], t[1], names) for t in train], seed=seed, hgb=hgb, stride=stride)
    return em, time.process_time() - t0


def _fit_det(job):
    """Train one fold's detector; return (its outputs on the fold's held-out X's, CPU seconds)."""
    train, tests, names, cfg = job
    t0 = time.process_time()
    det = bench.learned_detector(train, names, cfg["k_s"], cfg["through_dead_s"])
    secs = time.process_time() - t0
    return [det.p_change(X, names) for X in tests], secs


def _score(job):
    """One fold's emitter (or ensemble) evidence on its held-out X's."""
    emitter, tests, names = job
    return [emitter.emit(X, names) for X in tests]


def _decode_case(job):
    """Decode one case at every point; one score row per point."""
    case, points = job
    rows = []
    for pt in points:
        p = DecodeParams(pt["switch_ha"], pt["alive_to_dead"], pt["dead_to_alive"])
        sc = None
        if pt["alpha"] != 0:
            sc = switch_cost_from(case["pchg"], pt["switch_ha"], pt["alpha"], pt["floor"])
        pred = decode_periods(case["logp"], p, case["periods"], case["n"], switch_cost=sc)
        rows.append({**pt, "game": case["game"], "seed": case["seed"], **bench.score(case["labels"], pred)})
    return rows


def default_names():
    """Names of the default (``handover=False``) feature columns, from a two-frame dummy game."""
    tr = Tracking(np.zeros((2, 2, 2), np.float32), np.array([0, 1], np.int8), np.zeros((2, 2), np.float32),
                  [(1, 0, 1)], (105.0, 68.0))
    return frame_features(tr)[1]


class Evidence:
    """Held-out emitter log-scores and detector probabilities, trained once per fold and cached.

    Emitter evidence is keyed by the training config (features, ensemble size, ``hgb``, stride,
    training seeds); each ensemble member is trained (and cached) once per fold, so a member
    shared by two configs is not trained twice. Pool entries for new training seeds are added
    on demand. Per-fold training CPU seconds are kept for ``train_cost``.
    """

    def __init__(self, held, pool, executor):
        self.held, self.pool, self.ex = held, dict(pool), executor
        self.names = list(held[(bench.GAMES[0], bench.SEEDS[0])][2])
        base = default_names()
        if self.names[:len(base)] != base or len(self.names) == len(base):
            raise ValueError("expected the hand-over columns after the default ones")
        self.nb = len(base)
        self.seeds = sorted({s for _, s in held})
        self.cases = [(g, s) for s in self.seeds for g in bench.GAMES]  # heldout_logp order
        self.cache, self.models, self.secs = {}, {}, {}

    @staticmethod
    def emit_key(cfg):
        return ("emit", bool(cfg["features"]), int(cfg.get("ensemble_k", 1)),
                tuple(sorted((cfg.get("hgb") or {}).items())), int(cfg.get("stride") or STRIDE),
                tuple(cfg.get("train_seeds") or bench.TRAIN_SEEDS))

    @staticmethod
    def det_key(cfg):
        return ("det", float(cfg["k_s"]), cfg["through_dead_s"],
                tuple(cfg.get("train_seeds") or bench.TRAIN_SEEDS))

    @staticmethod
    def keys(cfg):
        return [Evidence.emit_key(cfg)] + ([Evidence.det_key(cfg)] if cfg["detector"] else [])

    @staticmethod
    def members(key):
        """Member keys of an emitter key: member m has random_state m and the training seeds
        shifted by m x len(seeds)."""
        _, features, k, hgb, stride, seeds = key
        return [("member", features, hgb, stride, tuple(s + m * len(seeds) for s in seeds), m)
                for m in range(k)]

    def _cols(self, features):
        return slice(None) if features else slice(0, self.nb)

    def _ensure_pool(self, seeds):
        missing = sorted(s for s in seeds if (bench.GAMES[0], s) not in self.pool)
        if missing:
            self.pool.update(bench.prepare(seeds=(), train_seeds=tuple(missing), handover=True,
                                           pool_meta=True)[1])

    def _member_jobs(self, mkey):
        _, features, hgb, stride, seeds, m = mkey
        cols = self._cols(features)
        for g in bench.GAMES:
            train = [(e[0][:, cols], e[1]) for e in bench.fold_train(self.pool, g, seeds)]
            yield (train, self.names[cols], m, dict(hgb) or None, stride)

    def _det_jobs(self, key):
        _, k_s, through_dead_s, seeds = key
        cfg = {"k_s": k_s, "through_dead_s": through_dead_s}
        for g in bench.GAMES:
            train = [(e[0], e[1], e[3], e[5]) for e in bench.fold_train(self.pool, g, seeds)]
            yield (train, [self.held[(g, s)][0] for s in self.seeds], self.names, cfg)

    def _per_case(self, per_game):
        per_game = dict(zip(bench.GAMES, per_game))
        return {(g, s): per_game[g][self.seeds.index(s)] for g, s in self.cases}

    def fetch(self, keys):
        """Compute every missing key, all folds (and ensemble members) in parallel."""
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        emits = [k for k in todo if k[0] == "emit"]
        dets = [k for k in todo if k[0] == "det"]
        mkeys = [m for m in dict.fromkeys(m for k in emits for m in self.members(k)) if m not in self.models]
        self._ensure_pool({s for m in mkeys for s in m[4]} | {s for k in dets for s in k[3]})
        mfuts = {m: [self.ex.submit(_fit_member, j) for j in self._member_jobs(m)] for m in mkeys}
        dfuts = {k: [self.ex.submit(_fit_det, j) for j in self._det_jobs(k)] for k in dets}
        for m in mkeys:
            res = [f.result() for f in mfuts[m]]
            self.models[m] = dict(zip(bench.GAMES, (r[0] for r in res)))
            self.secs[m] = [r[1] for r in res]
        sfuts = {}
        for k in emits:
            ms, cols = self.members(k), self._cols(k[1])
            sfuts[k] = []
            for g in bench.GAMES:
                fold = [self.models[m][g] for m in ms]
                em = fold[0] if len(fold) == 1 else EnsembleEmitter(fold)
                tests = [self.held[(g, s)][0][:, cols] for s in self.seeds]
                sfuts[k].append(self.ex.submit(_score, (em, tests, self.names[cols])))
        for k in dets:
            res = [f.result() for f in dfuts[k]]
            self.cache[k] = self._per_case(r[0] for r in res)
            self.secs[k] = [r[1] for r in res]
        for k in emits:
            self.cache[k] = self._per_case(f.result() for f in sfuts[k])

    def get(self, key):
        self.fetch([key])
        return self.cache[key]

    def train_cost(self, cfg):
        """Mean per-fold training CPU seconds of ``cfg``: {"emit": all members, "det": detector}."""
        ek = self.emit_key(cfg)
        self.fetch(self.keys(cfg))
        emit = float(np.mean(np.sum([self.secs[m] for m in self.members(ek)], axis=0)))
        det = float(np.mean(self.secs[self.det_key(cfg)])) if cfg["detector"] else 0.0
        return {"emit": emit, "det": det}


def est_run_seconds(cost, inc_cost):
    """``bsd.run`` wall time implied by a variant: today's run plus today's final-model training
    times scaled by the variant's bench training-cost ratio (per-fold CPU seconds, variant /
    incumbent). Ratios, not bench seconds: the bench's parallel workers run slower per job."""
    out = RUN_SECONDS + FINAL_EMIT_S * (cost["emit"] / inc_cost["emit"] - 1)
    if inc_cost["det"]:
        out += FINAL_DET_S * (cost["det"] / inc_cost["det"] - 1)
    return out


def _config(features, detector, k_s=0.5, through_dead_s=None):
    return {"features": features, "detector": detector, "k_s": k_s, "through_dead_s": through_dead_s}


def _stage1_config(cfg, best):
    full = {**cfg, **{p: float(best[p]) for p in PARAMS}}
    return {k: full[k] for k in ("features", "detector", "k_s", "through_dead_s", "alpha", "floor",
                                 "switch_ha", "alive_to_dead", "dead_to_alive")}


def _stage4_config(cfg, best):
    return {"ensemble_k": int(cfg["ensemble_k"]), "hgb": dict(cfg["hgb"]), "stride": int(cfg["stride"]),
            "train_seeds": [int(s) for s in cfg["train_seeds"]],
            **{k: cfg[k] for k in ("features", "detector", "k_s", "through_dead_s")},
            "decode": {p: float(best[p]) for p in PARAMS}}


def _run_variant(name, cfg, ev, executor, stage="stage1", fmt=_stage1_config):
    t0 = time.time()
    keys = Evidence.keys(cfg)
    ev.fetch(keys)
    logp = ev.cache[keys[0]]
    pchg = ev.cache[keys[1]] if cfg["detector"] else None
    return _tune_cases(name, cfg, ev.held, ev.cases, logp, pchg, executor, stage, fmt, t0)


def _tune_cases(name, cfg, held, keys, logp, pchg, executor, stage, fmt, t0):
    """Tune the decoder on the cases ``keys`` (``held[(g, s)]`` = (X, labels, names, periods, n))
    with evidence ``logp[(g, s)]`` (and ``pchg[(g, s)]`` if a detector); save and return the record."""
    cases = []
    for g, s in keys:
        _, labels, _, periods, n = held[(g, s)]
        cases.append({"game": g, "seed": s, "logp": logp[(g, s)], "labels": labels, "periods": periods,
                      "n": n, "pchg": None if pchg is None else pchg[(g, s)]})

    def score_points(points):
        per_case = list(executor.map(_decode_case, [(c, points) for c in cases]))
        return [per_case[j][i] for i in range(len(points)) for j in range(len(cases))]

    best, rows = tune_stage(score_points, STAGE1_GRID, cfg["detector"])
    grid_df = pd.DataFrame(rows)
    sel = np.ones(len(grid_df), dtype=bool)
    for p in PARAMS:
        sel &= grid_df[p].to_numpy() == best[p]
    table = grid_df.loc[sel, CASE_COLS].reset_index(drop=True)
    CACHE.mkdir(exist_ok=True)
    table.to_csv(CACHE / ("%s_%s.csv" % (stage, name)), index=False)
    grid_df.to_csv(CACHE / ("%s_%s_grid.csv" % (stage, name)), index=False)
    means = table[CASE_COLS[2:]].mean()
    secs = time.time() - t0
    print("%-14s spellF1 %.4f  macroF1 %.4f  n_poss %.1f  sw %g  D (%g, %g)  alpha %g  floor %g"
          "  [%d points, %.0f s]" % (name, means["spell_f1"], means["macro_f1"], means["n_poss"],
                                     best["switch_ha"], best["alive_to_dead"], best["dead_to_alive"],
                                     best["alpha"], best["floor"], len(grid_df) // len(cases), secs),
          flush=True)
    return {
        "config": fmt(cfg, best),
        "n_points": len(grid_df) // len(cases),
        "mean": {k: round(float(v), 4) for k, v in means.items()},
        "cases": [{k: (int(r[k]) if k in ("game", "seed", "n_poss", "gt_n_poss") else round(float(r[k]), 6))
                   for k in CASE_COLS} for _, r in table.iterrows()],
        "_table": table,
        "_means": (float(means["spell_f1"]), float(means["macro_f1"])),
    }


def _rank(rec):
    """Unrounded (mean spellF1, mean macroF1)."""
    return rec["_means"]


def _ordered(variants):
    """Variant names, best first (unrounded means; ties: macroF1, then the earlier one)."""
    names = list(variants)
    return sorted(names, key=lambda v: (*_rank(variants[v]), -names.index(v)), reverse=True)


def _best_of(variants):
    return _ordered(variants)[0]


def _public(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def stage1(workers=WORKERS):
    t_start = time.time()
    ref = json.loads(bench.PARAMS_PATH.read_text())["learned"]["bench_spell_f1"]
    held, pool = bench.prepare(handover=True, pool_meta=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        ev = Evidence(held, pool, executor)
        base, hand = _config(False, True), _config(True, True)
        ev.fetch(Evidence.keys(_config(False, False)) + Evidence.keys(hand) + Evidence.keys(base))
        print("evidence for incumbent, (a)-(c): %.0f s" % (time.time() - t_start), flush=True)

        inc = _run_variant("incumbent", _config(False, False), ev, executor)
        if abs(inc["mean"]["spell_f1"] - ref) > INCUMBENT_TOL:
            print("incumbent spellF1 %.4f differs from the committed %.4f by more than %.3f; stopping"
                  % (inc["mean"]["spell_f1"], ref, INCUMBENT_TOL))
            return 1

        variants, skipped = {}, {}
        variants["a_features"] = _run_variant("a_features", _config(True, False), ev, executor)
        variants["b_detector"] = _run_variant("b_detector", base, ev, executor)
        variants["c_both"] = _run_variant("c_both", hand, ev, executor)

        def best_name():
            return _best_of(variants)

        stopped = None
        top = best_name()
        if not variants[top]["config"]["detector"]:
            skipped["d_k1"] = skipped["e_through_dead"] = "best of (a)-(c) has no detector"
        else:
            c = variants[top]["config"]
            variants["d_k1"] = _run_variant("d_k1", _config(c["features"], True, 1.0, None), ev, executor)
            if variants["d_k1"]["mean"]["spell_f1"] < variants[top]["mean"]["spell_f1"] + EARLY_STOP_GAIN:
                stopped = "d_k1"
                skipped["e_through_dead"] = "early stop: d_k1 added < %.3f" % EARLY_STOP_GAIN
            else:
                c = variants[best_name()]["config"]
                variants["e_through_dead"] = _run_variant(
                    "e_through_dead", _config(c["features"], True, c["k_s"], 2.0), ev, executor)

    winner = best_name()
    g = gate(inc["_table"], variants[winner]["_table"])
    runtime = time.time() - t_start
    print("best variant: %s   gate vs incumbent: d_spell %+.4f, %d/%d better, d_macro %+.4f -> %s"
          % (winner, g["d_spell"], g["n_better"], g["n_cases"], g["d_macro"],
             "PASSED" if g["passed"] else "FAILED"))
    print("runtime: %.0f s" % runtime)
    inc = _public(inc)
    variants = {k: _public(v) for k, v in variants.items()}
    out = {
        "stage": 1,
        "protocol": {
            "games": list(bench.GAMES), "heldout_seeds": list(bench.SEEDS), "train_seeds": list(bench.TRAIN_SEEDS),
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in STAGE1_GRID.items()},
            "gate": {"min_d_spell": GATE_MIN_D_SPELL, "min_better": GATE_MIN_BETTER,
                     "max_macro_drop": GATE_MAX_MACRO_DROP},
            "early_stop_gain": EARLY_STOP_GAIN,
        },
        "reference_spell_f1": ref,
        "incumbent": inc,
        "variants": variants,
        "skipped": skipped,
        "stopped_early_after": stopped,
        "best_variant": winner,
        "gate": g,
        "winner_variant": winner if g["passed"] else None,
        "winner": variants[winner]["config"] if g["passed"] else None,
    }
    STAGE1_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print("saved %s" % STAGE1_JSON.relative_to(ROOT))
    return 0


def combine_tweaks(names):
    """Merge Stage 4 tweaks (they set disjoint fields)."""
    out = {}
    for n in names:
        if set(out) & set(STAGE4_TWEAKS[n]):
            raise ValueError("tweaks %s overlap" % (names,))
        out.update(STAGE4_TWEAKS[n])
    return out


def _prior_cost(names, costs, inc_cost):
    """A-priori training cost of combined tweaks: emitter cost factors multiply; the detector cost
    is that of the tweak that changes the training seeds (the detector's only input here)."""
    emit = inc_cost["emit"]
    det = inc_cost["det"]
    for n in names:
        emit *= costs[n]["emit"] / inc_cost["emit"]
        if "train_seeds" in STAGE4_TWEAKS[n]:
            det = costs[n]["det"]
    return {"emit": emit, "det": det}


def stage4(workers=WORKERS, skip=()):
    t_start = time.time()
    learned = json.loads(bench.PARAMS_PATH.read_text())["learned"]
    ref = {"spell_f1": learned["bench_spell_f1"], "macro_f1": learned["bench_macro_f1"]}
    ho = learned["handover"]
    base = {**_config(ho["features"], ho["detector"], ho["k_s"], ho["through_dead_s"]),
            "ensemble_k": 1, "hgb": {}, "stride": STRIDE, "train_seeds": list(bench.TRAIN_SEEDS)}
    cfgs = {n: {**base, **t} for n, t in STAGE4_TWEAKS.items() if n not in skip}
    skipped = {n: "dropped for time (--skip)" for n in STAGE4_TWEAKS if n in skip}
    held, pool = bench.prepare(handover=True, pool_meta=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        ev = Evidence(held, pool, executor)
        ev.fetch([k for c in [base, *cfgs.values()] for k in Evidence.keys(c)])
        print("evidence for incumbent, %s: %.0f s" % (", ".join(cfgs), time.time() - t_start), flush=True)

        def run(name, cfg):
            rec = _run_variant(name, cfg, ev, executor, "stage4", _stage4_config)
            cost = ev.train_cost(cfg)
            rec["train_cpu_s"] = {k: round(v, 1) for k, v in cost.items()}
            rec["est_run_s"] = round(est_run_seconds(cost, inc_cost))
            print("%-14s train CPU per fold: emitter %.0f s, detector %.0f s; est. bsd.run %.0f s"
                  % (name, cost["emit"], cost["det"], rec["est_run_s"]), flush=True)
            return rec, cost

        inc_cost = ev.train_cost(base)
        inc, _ = run("incumbent", base)
        off = {k: abs(inc["_means"][i] - ref[k]) for i, k in enumerate(("spell_f1", "macro_f1"))}
        if max(off.values()) > INCUMBENT_TOL:
            print("incumbent %.4f / %.4f differs from the committed %.4f / %.4f by more than %.3f; stopping"
                  % (*inc["_means"], ref["spell_f1"], ref["macro_f1"], INCUMBENT_TOL))
            return 1

        variants, costs = {}, {}
        for name, cfg in cfgs.items():
            cost = ev.train_cost(cfg)
            est = est_run_seconds(cost, inc_cost)
            if est > RUN_LIMIT_S:
                skipped[name] = "est. bsd.run %.0f s > %d s" % (est, RUN_LIMIT_S)
                print("%-14s skipped: %s" % (name, skipped[name]), flush=True)
                continue
            variants[name], costs[name] = run(name, cfg)

        stopped, combined = None, None
        if "e_combined" in skip:
            skipped["e_combined"] = "dropped for time (--skip)"
        elif len(variants) < 2:
            skipped["e_combined"] = "fewer than two of (a)-(d) ran"
        else:
            combined = sorted(_ordered(variants)[:2], key=list(STAGE4_TWEAKS).index)
            est = est_run_seconds(_prior_cost(combined, costs, inc_cost), inc_cost)
            if est > RUN_LIMIT_S:
                skipped["e_combined"] = "%s: a-priori est. bsd.run %.0f s > %d s" % ("+".join(combined), est,
                                                                                    RUN_LIMIT_S)
                print("e_combined     skipped: %s" % skipped["e_combined"], flush=True)
            else:
                top = _best_of(variants)
                variants["e_combined"], _ = run("e_combined", {**base, **combine_tweaks(combined)})
                if _rank(variants["e_combined"])[0] < _rank(variants[top])[0] + EARLY_STOP_GAIN:
                    stopped = "e_combined"

    winner = _best_of(variants)
    g = gate(inc["_table"], variants[winner]["_table"])
    runtime = time.time() - t_start
    print("best variant: %s   gate vs incumbent: d_spell %+.4f, %d/%d better, d_macro %+.4f -> %s"
          % (winner, g["d_spell"], g["n_better"], g["n_cases"], g["d_macro"],
             "PASSED" if g["passed"] else "FAILED"))
    print("runtime: %.0f s" % runtime)
    inc = _public(inc)
    variants = {k: _public(v) for k, v in variants.items()}
    out = {
        "stage": 4,
        "protocol": {
            "games": list(bench.GAMES), "heldout_seeds": list(bench.SEEDS),
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in STAGE1_GRID.items()},
            "tweaks": STAGE4_TWEAKS,
            "ensemble_members": "member m: random_state m, train seeds shifted by m x len(train_seeds)",
            "gate": {"min_d_spell": GATE_MIN_D_SPELL, "min_better": GATE_MIN_BETTER,
                     "max_macro_drop": GATE_MAX_MACRO_DROP},
            "early_stop_gain": EARLY_STOP_GAIN,
            "run_limit_s": RUN_LIMIT_S, "run_seconds_today": RUN_SECONDS,
        },
        "reference": ref,
        "incumbent": inc,
        "variants": variants,
        "e_combines": combined,
        "skipped": skipped,
        "stopped_early_after": stopped,
        "best_variant": winner,
        "gate": g,
        "winner_variant": winner if g["passed"] else None,
        "winner": variants[winner]["config"] if g["passed"] else None,
        "runtime_s": round(runtime),
    }
    STAGE4_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print("saved %s" % STAGE4_JSON.relative_to(ROOT))
    return 0


# --- Stage 2: public DFL Bundesliga games in the training pool ------------------------------

STAGE2_JSON = ROOT / "bsd" / "experiments_stage2.json"
STAGE2_WORKERS = 4
SECONDARY_MAX_DROP = 0.010  # candidate's secondary mean spellF1 may be at most this below the incumbent's
SECONDARY_SEED = 0  # degradation seed of a held-out DFL game
STAGE2_VARIANTS = {
    "a_dfl": {"dfl_seeds": [bench.TRAIN_SEEDS[0]], "dfl_weight": 1.0},
    "b_dfl_w05": {"dfl_seeds": [bench.TRAIN_SEEDS[0]], "dfl_weight": 0.5},
    "c_dfl_seeds3": {"dfl_seeds": list(bench.TRAIN_SEEDS), "dfl_weight": 1.0},
}


def secondary_check(inc_rows, cand_rows):
    """Stage 2's extra condition on the held-out DFL games (columns game, spell_f1): the
    candidate's mean spellF1 must not be more than ``SECONDARY_MAX_DROP`` below the incumbent's.
    Paired on game; the difference is rounded to 6 decimals."""
    m = inc_rows[["game", "spell_f1"]].merge(cand_rows[["game", "spell_f1"]], on="game",
                                              suffixes=("_inc", "_cand"), validate="one_to_one")
    if not len(m) == len(inc_rows) == len(cand_rows):
        raise ValueError("incumbent and candidate must cover the same held-out games")
    d = round(float((m["spell_f1_cand"] - m["spell_f1_inc"]).mean()), 6)
    return {"d_spell": d, "n_better": int((m["spell_f1_cand"] > m["spell_f1_inc"]).sum()),
            "n_games": len(m), "passed": bool(d >= -SECONDARY_MAX_DROP)}


def stage2_adopt(gate_result, secondary_result):
    """Stage 2 adopts only if the gate passes on the primary and the secondary condition holds."""
    return bool(gate_result["passed"] and secondary_result is not None and secondary_result["passed"])


def stage2_est_run_seconds(cost, inc_cost, extra_s=0.0):
    """``bsd.run`` time of a Stage 2 variant from per-fold training CPU seconds of the primary.

    A primary fold trains on 2 Metrica games plus the DFL rows; the final model on all 3 Metrica
    games plus the same DFL rows, i.e. 1.5x the fold's Metrica cost plus the DFL increment.
    ``extra_s`` adds the final run's DFL featurisation time.
    """
    f = len(bench.GAMES) / (len(bench.GAMES) - 1)
    final = {k: f * inc_cost[k] + cost[k] - inc_cost[k] for k in ("emit", "det")}
    return est_run_seconds(final, {k: f * inc_cost[k] for k in ("emit", "det")}) + extra_s


def _prep_dfl(job):
    """One DFL game: pre-strided (X, labels, change labels) per training degrade seed, the full
    held-out (X, labels, names, periods, n) at ``held_seed`` (or None), names, and the CPU
    seconds of preparing one training copy."""
    from bsd.degrade import degrade, gap_profile
    from bsd.handover import change_labels
    from bsd.io_dfl import dfl_game

    match_id, train_seeds, held_seed, k_s, through_dead_s = job
    tr, labels = dfl_game(match_id)
    profile = gap_profile()
    train, secs, names = {}, [], None
    for s in train_seeds:
        t0 = time.process_time()
        d = degrade(tr, profile, s)
        X, names = frame_features(d, handover=True)
        c = change_labels(labels, d.periods, int(d.fps), k_s=k_s, through_dead_s=through_dead_s)
        secs.append(time.process_time() - t0)
        train[s] = (X[::STRIDE], labels[::STRIDE], c[::STRIDE])
    held = None
    if held_seed is not None:
        d = degrade(tr, profile, held_seed)
        X, names = frame_features(d, handover=True)
        held = (X, labels, names, d.periods, len(d.ball))
    return names, train, held, float(np.mean(secs)) if secs else 0.0


def _fit_s2(job):
    """Train an emitter ("emit") or detector ("det") on pre-strided ``(X, target, weight)``
    entries (stride 1, BAR-ATH visibility weights, dataset weights on top); return its outputs
    on ``tests`` and the training CPU seconds."""
    from bsd.degrade import gap_profile
    from bsd.handover import train_detector
    from bsd.model import train_emitter

    kind, entries, tests, names = job
    vr = 1.0 - float(gap_profile()["missing_rate"])
    Xs, ts, ws = [e[0] for e in entries], [e[1] for e in entries], [e[2] for e in entries]
    t0 = time.process_time()
    if kind == "emit":
        m = train_emitter(Xs, ts, names, seed=0, visible_rate=vr, stride=1, weights=ws)
        secs = time.process_time() - t0
        return [m.emit(X, names) for X in tests], secs
    m = train_detector(Xs, ts, names, seed=0, visible_rate=vr, stride=1, weights=ws)
    secs = time.process_time() - t0
    return [m.p_change(X, names) for X in tests], secs


def _stage2_config(cfg, best):
    return {**{k: cfg[k] for k in ("features", "detector", "k_s", "through_dead_s")},
            "train_seeds": [int(s) for s in cfg["train_seeds"]],
            "dfl_games": list(cfg.get("dfl_games") or []), "dfl_seeds": [int(s) for s in cfg.get("dfl_seeds") or []],
            "dfl_weight": float(cfg.get("dfl_weight", 1.0)),
            "decode": {p: float(best[p]) for p in PARAMS}}


def _point(rec):
    return {p: rec["config"]["decode"][p] for p in PARAMS}


def _score_secondary(name, held, ids, logp, pchg, point, executor):
    """Decode each held-out DFL game at ``point`` (no tuning on DFL) and score it."""
    cases = [{"game": m, "seed": SECONDARY_SEED, "logp": logp[m], "labels": held[m][1], "periods": held[m][3],
              "n": held[m][4], "pchg": pchg[m]} for m in ids]
    rows = [r[0] for r in executor.map(_decode_case, [(c, [point]) for c in cases])]
    table = pd.DataFrame(rows)[CASE_COLS]
    table.to_csv(CACHE / ("stage2_%s_secondary.csv" % name), index=False)
    means = table[CASE_COLS[2:]].mean()
    print("%-14s secondary (%d DFL games) spellF1 %.4f  macroF1 %.4f" % (name, len(ids), means["spell_f1"],
                                                                        means["macro_f1"]), flush=True)
    return {"mean": {k: round(float(v), 4) for k, v in means.items()},
            "cases": [{k: (r[k] if k == "game" else int(r[k]) if k in ("seed", "n_poss", "gt_n_poss")
                           else round(float(r[k]), 6)) for k in CASE_COLS} for _, r in table.iterrows()],
            "_table": table, "_mean": float(means["spell_f1"])}


def stage2(workers=STAGE2_WORKERS, skip=(), secondary_n=None):
    from bsd.fetch_dfl import MATCH_IDS
    from bsd.handover import change_labels

    t_start = time.time()
    learned = json.loads(bench.PARAMS_PATH.read_text())["learned"]
    ref = {"spell_f1": learned["bench_spell_f1"], "macro_f1": learned["bench_macro_f1"]}
    ho = learned["handover"]
    if not (ho["features"] and ho["detector"]):
        raise ValueError("stage2 expects the incumbent to use hand-over features and the detector")
    base = {**_config(ho["features"], ho["detector"], ho["k_s"], ho["through_dead_s"]),
            "ensemble_k": 1, "hgb": {}, "stride": STRIDE, "train_seeds": list(bench.TRAIN_SEEDS)}
    dfl_ids = list(MATCH_IDS)
    sec_ids = dfl_ids[:secondary_n] if secondary_n else dfl_ids
    planned = [n for n in STAGE2_VARIANTS if n not in skip]
    skipped = {n: "dropped for time (--skip)" for n in STAGE2_VARIANTS if n in skip}
    cfgs = {n: {**base, **STAGE2_VARIANTS[n], "dfl_games": dfl_ids} for n in STAGE2_VARIANTS}
    dfl_seeds = sorted({s for n in planned for s in cfgs[n]["dfl_seeds"]})
    held, pool = bench.prepare(handover=True, pool_meta=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        ev = Evidence(held, pool, executor)
        names = ev.names
        # DFL games: training copies (pre-strided) and the secondary held-out copy at seed 0.
        prep = list(executor.map(_prep_dfl, [(m, dfl_seeds, SECONDARY_SEED if m in sec_ids else None,
                                              base["k_s"], base["through_dead_s"]) for m in dfl_ids]))
        if any(list(p[0]) != names for p in prep):
            raise ValueError("DFL feature names differ from the Metrica ones")
        dtrain = {(m, s): p[1][s] for m, p in zip(dfl_ids, prep) for s in dfl_seeds}
        dheld = {m: p[2] for m, p in zip(dfl_ids, prep) if p[2] is not None}
        prep_s = float(np.mean([p[3] for p in prep]))
        del prep
        label_shares = {m: {lab: round(float(np.mean(dheld[m][1] == code)), 4)
                            for lab, code in (("H", 0), ("A", 1), ("D", 2), ("X", 3))} for m in dheld}
        print("DFL prepared (%d games, seeds %s): %.0f s" % (len(dfl_ids), dfl_seeds, time.time() - t_start),
              flush=True)
        # Metrica pool entries, pre-strided, with change labels (as bench.learned_detector makes them).
        mtrain = {}
        for (g, s), e in ev.pool.items():
            c = change_labels(e[1], e[3], e[5], k_s=base["k_s"], through_dead_s=base["through_dead_s"])
            mtrain[(g, s)] = (e[0][::STRIDE], e[1][::STRIDE], c[::STRIDE])

        def entries(kind, metrica, dfl, w):
            i = 1 if kind == "emit" else 2
            return ([(mtrain[k][0], mtrain[k][i], 1.0) for k in metrica]
                    + [(dtrain[k][0], dtrain[k][i], w) for k in dfl])

        all_metrica = [(g, s) for g in bench.GAMES for s in base["train_seeds"]]

        def submit(name):
            """Primary folds and secondary holdouts of a variant (None: the incumbent's secondary)."""
            futs = {"primary": {}, "secondary": {}}
            if name is None:
                tests = [dheld[m][0] for m in sec_ids]
                futs["secondary"]["all"] = {k: executor.submit(_fit_s2, (k, entries(k, all_metrica, [], 1.0),
                                                                         tests, names)) for k in ("emit", "det")}
                return futs
            c = cfgs[name]
            for m in sec_ids:  # longest jobs first
                dfl = [(o, s) for o in dfl_ids if o != m for s in c["dfl_seeds"]]
                futs["secondary"][m] = {k: executor.submit(
                    _fit_s2, (k, entries(k, all_metrica, dfl, c["dfl_weight"]), [dheld[m][0]], names))
                    for k in ("emit", "det")}
            dfl = [(o, s) for o in dfl_ids for s in c["dfl_seeds"]]
            for g in bench.GAMES:
                metrica = [(o, s) for o in bench.GAMES if o != g for s in base["train_seeds"]]
                tests = [held[(g, s)][0] for s in ev.seeds]
                futs["primary"][g] = {k: executor.submit(
                    _fit_s2, (k, entries(k, metrica, dfl, c["dfl_weight"]), tests, names)) for k in ("emit", "det")}
            return futs

        def collect(name, futs):
            t0 = time.time()
            res = {g: {k: f.result() for k, f in d.items()} for g, d in futs["primary"].items()}
            logp = {(g, s): res[g]["emit"][0][j] for g in bench.GAMES for j, s in enumerate(ev.seeds)}
            pchg = {(g, s): res[g]["det"][0][j] for g in bench.GAMES for j, s in enumerate(ev.seeds)}
            rec = _tune_cases(name, cfgs[name], held, ev.cases, logp, pchg, executor, "stage2",
                              _stage2_config, t0)
            cost = {k: float(np.mean([res[g][k][1] for g in bench.GAMES])) for k in ("emit", "det")}
            rec["train_cpu_s"] = {k: round(v, 1) for k, v in cost.items()}
            extra = prep_s * len(dfl_ids) * len(cfgs[name]["dfl_seeds"])
            rec["est_run_s"] = round(stage2_est_run_seconds(cost, inc_cost, extra))
            print("%-14s train CPU per fold: emitter %.0f s, detector %.0f s; est. bsd.run %.0f s"
                  % (name, cost["emit"], cost["det"], rec["est_run_s"]), flush=True)
            sres = {m: {k: f.result() for k, f in d.items()} for m, d in futs["secondary"].items()}
            rec["secondary"] = _score_secondary(name, dheld, sec_ids, {m: sres[m]["emit"][0][0] for m in sec_ids},
                                                {m: sres[m]["det"][0][0] for m in sec_ids}, _point(rec), executor)
            return rec

        futs = {n: submit(n) for n in planned if n != "c_dfl_seeds3"}
        inc_sec = submit(None)
        inc_cost = ev.train_cost(base)
        inc = _run_variant("incumbent", base, ev, executor, "stage2", _stage2_config)
        inc["train_cpu_s"] = {k: round(v, 1) for k, v in inc_cost.items()}
        sres = {k: f.result() for k, f in inc_sec["secondary"]["all"].items()}
        inc["secondary"] = _score_secondary(
            "incumbent", dheld, sec_ids, {m: sres["emit"][0][i] for i, m in enumerate(sec_ids)},
            {m: sres["det"][0][i] for i, m in enumerate(sec_ids)}, _point(inc), executor)
        off = {k: abs(inc["_means"][i] - ref[k]) for i, k in enumerate(("spell_f1", "macro_f1"))}
        if max(off.values()) > INCUMBENT_TOL:
            print("incumbent %.4f / %.4f differs from the committed %.4f / %.4f by more than %.3f; stopping"
                  % (*inc["_means"], ref["spell_f1"], ref["macro_f1"], INCUMBENT_TOL))
            return 1

        variants, stopped = {}, None
        for n in futs:
            variants[n] = collect(n, futs[n])
        if "c_dfl_seeds3" in planned:
            top = _best_of(variants) if variants else None
            if "b_dfl_w05" in variants and _rank(variants["b_dfl_w05"])[0] < _rank(variants["a_dfl"])[0] + EARLY_STOP_GAIN:
                stopped = "b_dfl_w05"
                skipped["c_dfl_seeds3"] = "early stop: b_dfl_w05 added < %.3f over a_dfl" % EARLY_STOP_GAIN
            else:
                variants["c_dfl_seeds3"] = collect("c_dfl_seeds3", submit("c_dfl_seeds3"))
                if top and _rank(variants["c_dfl_seeds3"])[0] < _rank(variants[top])[0] + EARLY_STOP_GAIN:
                    stopped = "c_dfl_seeds3"
        elif "b_dfl_w05" in variants and _rank(variants["b_dfl_w05"])[0] < _rank(variants["a_dfl"])[0] + EARLY_STOP_GAIN:
            stopped = "b_dfl_w05"
        for n in list(variants):
            if variants[n]["est_run_s"] > RUN_LIMIT_S:
                skipped[n] = "est. bsd.run %d s > %d s" % (variants[n]["est_run_s"], RUN_LIMIT_S)
                del variants[n]

    winner = _best_of(variants) if variants else None
    g = gate(inc["_table"], variants[winner]["_table"]) if winner else None
    sec = secondary_check(inc["secondary"]["_table"], variants[winner]["secondary"]["_table"]) if winner else None
    adopt = stage2_adopt(g, sec) if winner else False
    runtime = time.time() - t_start
    if winner:
        print("best variant: %s   gate vs incumbent: d_spell %+.4f, %d/%d better, d_macro %+.4f -> %s;"
              "  secondary d_spell %+.4f -> %s;  adopt: %s"
              % (winner, g["d_spell"], g["n_better"], g["n_cases"], g["d_macro"], "PASSED" if g["passed"] else "FAILED",
                 sec["d_spell"], "OK" if sec["passed"] else "FAILED", adopt))
    print("runtime: %.0f s" % runtime)
    for n, v in variants.items():
        v["secondary_check"] = secondary_check(inc["secondary"]["_table"], v["secondary"]["_table"])
        v["gate"] = gate(inc["_table"], v["_table"])

    def pub(rec):
        out = _public(rec)
        out["secondary"] = _public(rec["secondary"])
        return out
    out = {
        "stage": 2,
        "protocol": {
            "games": list(bench.GAMES), "heldout_seeds": list(bench.SEEDS), "train_seeds": list(bench.TRAIN_SEEDS),
            "dfl_games": dfl_ids, "secondary_games": sec_ids, "secondary_seed": SECONDARY_SEED,
            "secondary": "each DFL game held out once; incumbent trained on 3 Metrica games x train_seeds, "
                         "candidate on those plus the other DFL games; decoded at the variant's primary-tuned "
                         "params (no tuning on DFL)",
            "variants": STAGE2_VARIANTS,
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in STAGE1_GRID.items()},
            "gate": {"min_d_spell": GATE_MIN_D_SPELL, "min_better": GATE_MIN_BETTER,
                     "max_macro_drop": GATE_MAX_MACRO_DROP},
            "secondary_max_drop": SECONDARY_MAX_DROP,
            "early_stop_gain": EARLY_STOP_GAIN,
            "run_limit_s": RUN_LIMIT_S, "run_seconds_today": RUN_SECONDS,
            "labels": "DFL labels are the provider's per-frame BallStatus/BallPossession flags (more dead time, "
                      "shorter spells than the Metrica event-derived labels)",
        },
        "dfl_label_shares": label_shares,
        "reference": ref,
        "incumbent": pub(inc),
        "variants": {k: pub(v) for k, v in variants.items()},
        "skipped": skipped,
        "stopped_early_after": stopped,
        "best_variant": winner,
        "gate": g,
        "secondary_check": sec,
        "winner_variant": winner if adopt else None,
        "winner": variants[winner]["config"] if adopt else None,
        "runtime_s": round(runtime),
    }
    STAGE2_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print("saved %s" % STAGE2_JSON.relative_to(ROOT))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("stage1", "stage2", "stage4"))
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default %d; stage2 %d)" % (WORKERS, STAGE2_WORKERS))
    ap.add_argument("--skip", nargs="*", default=(), choices=(*STAGE4_TWEAKS, "e_combined", *STAGE2_VARIANTS),
                    help="stage2/stage4: variants to leave out (time budget)")
    ap.add_argument("--secondary-n", type=int, default=None,
                    help="stage2 only: hold out just the first N DFL games in the secondary bench (time budget)")
    args = ap.parse_args(argv)
    if args.stage == "stage2":
        return stage2(args.workers or STAGE2_WORKERS, tuple(args.skip), args.secondary_n)
    args.workers = args.workers or WORKERS
    if args.stage == "stage4":
        return stage4(args.workers, tuple(args.skip))
    return stage1(args.workers)


if __name__ == "__main__":
    sys.exit(main())

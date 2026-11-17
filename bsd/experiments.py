"""Gated stage experiments on the leave-one-game-out bench: ``python -m bsd.experiments stage1|stage4``.

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
    cases = []
    for g, s in ev.cases:
        _, labels, _, periods, n = ev.held[(g, s)]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("stage1", "stage4"))
    ap.add_argument("--workers", type=int, default=WORKERS, help="worker processes (default %d)" % WORKERS)
    ap.add_argument("--skip", nargs="*", default=(), choices=(*STAGE4_TWEAKS, "e_combined"),
                    help="stage4 only: variants to leave out (time budget)")
    args = ap.parse_args(argv)
    if args.stage == "stage4":
        return stage4(args.workers, tuple(args.skip))
    return stage1(args.workers)


if __name__ == "__main__":
    sys.exit(main())

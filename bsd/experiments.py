"""Gated stage experiments on the leave-one-game-out bench: ``python -m bsd.experiments stage1``.

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
from bsd.degrade import gap_profile
from bsd.features import frame_features
from bsd.handover import change_labels, train_detector
from bsd.tracking import Tracking

ROOT = bench.ROOT
CACHE = ROOT / "cache"
STAGE1_JSON = ROOT / "bsd" / "experiments_stage1.json"
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

def _fit_fold(job):
    """Train one fold's emitter or detector; return its outputs on the fold's held-out X's."""
    kind, train, tests, names, cfg = job
    if kind == "emit":
        em = bench.learned_emitter([(t[0], t[1], names) for t in train])
        return [em.emit(X, names) for X in tests]
    visible_rate = 1.0 - float(gap_profile()["missing_rate"])
    cs = [change_labels(t[1], t[2], t[3], k_s=cfg["k_s"], through_dead_s=cfg["through_dead_s"])
          for t in train]
    det = train_detector([t[0] for t in train], cs, names, seed=0, visible_rate=visible_rate)
    return [det.p_change(X, names) for X in tests]


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
    """Held-out emitter log-scores and detector probabilities, trained once per fold and cached."""

    def __init__(self, held, pool, executor):
        self.held, self.pool, self.ex = held, pool, executor
        self.names = list(held[(bench.GAMES[0], bench.SEEDS[0])][2])
        base = default_names()
        if self.names[:len(base)] != base or len(self.names) == len(base):
            raise ValueError("expected the hand-over columns after the default ones")
        self.nb = len(base)
        self.seeds = sorted({s for _, s in held})
        self.cases = [(g, s) for s in self.seeds for g in bench.GAMES]  # heldout_logp order
        self.cache = {}

    @staticmethod
    def emit_key(features):
        return ("emit", bool(features))

    @staticmethod
    def det_key(cfg):
        return ("det", float(cfg["k_s"]), cfg["through_dead_s"])

    def _jobs(self, key):
        if key[0] == "emit":
            cols = slice(None) if key[1] else slice(0, self.nb)
            names = self.names[cols]
            for g in bench.GAMES:
                train = [(e[0][:, cols], e[1]) for e in bench.fold_train(self.pool, g)]
                yield ("emit", train, [self.held[(g, s)][0][:, cols] for s in self.seeds], names, None)
        else:
            cfg = {"k_s": key[1], "through_dead_s": key[2]}
            for g in bench.GAMES:
                train = [(e[0], e[1], e[3], e[5]) for e in bench.fold_train(self.pool, g)]
                yield ("det", train, [self.held[(g, s)][0] for s in self.seeds], self.names, cfg)

    def fetch(self, keys):
        """Compute every missing key, all folds in parallel."""
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        futs = {k: [self.ex.submit(_fit_fold, j) for j in self._jobs(k)] for k in todo}
        for k in todo:
            per_game = dict(zip(bench.GAMES, (f.result() for f in futs[k])))
            self.cache[k] = {(g, s): per_game[g][self.seeds.index(s)] for g, s in self.cases}

    def get(self, key):
        self.fetch([key])
        return self.cache[key]


def _config(features, detector, k_s=0.5, through_dead_s=None):
    return {"features": features, "detector": detector, "k_s": k_s, "through_dead_s": through_dead_s}


def _run_variant(name, cfg, ev, executor):
    t0 = time.time()
    keys = [Evidence.emit_key(cfg["features"])] + ([Evidence.det_key(cfg)] if cfg["detector"] else [])
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
    table.to_csv(CACHE / ("stage1_%s.csv" % name), index=False)
    grid_df.to_csv(CACHE / ("stage1_%s_grid.csv" % name), index=False)
    means = table[CASE_COLS[2:]].mean()
    secs = time.time() - t0
    full = {**cfg, **{p: float(best[p]) for p in PARAMS}}
    print("%-14s spellF1 %.4f  macroF1 %.4f  n_poss %.1f  sw %g  D (%g, %g)  alpha %g  floor %g"
          "  [%d points, %.0f s]" % (name, means["spell_f1"], means["macro_f1"], means["n_poss"],
                                     best["switch_ha"], best["alive_to_dead"], best["dead_to_alive"],
                                     best["alpha"], best["floor"], len(grid_df) // len(cases), secs),
          flush=True)
    return {
        "config": {k: full[k] for k in ("features", "detector", "k_s", "through_dead_s", "alpha", "floor",
                                        "switch_ha", "alive_to_dead", "dead_to_alive")},
        "n_points": len(grid_df) // len(cases),
        "mean": {k: round(float(v), 4) for k, v in means.items()},
        "cases": [{k: (int(r[k]) if k in ("game", "seed", "n_poss", "gt_n_poss") else round(float(r[k]), 6))
                   for k in CASE_COLS} for _, r in table.iterrows()],
        "_table": table,
    }


def _rank(rec):
    return rec["mean"]["spell_f1"], rec["mean"]["macro_f1"]


def stage1(workers=WORKERS):
    t_start = time.time()
    ref = json.loads(bench.PARAMS_PATH.read_text())["learned"]["bench_spell_f1"]
    held, pool = bench.prepare(handover=True, pool_meta=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        ev = Evidence(held, pool, executor)
        base, hand = _config(False, True), _config(True, True)
        ev.fetch([Evidence.emit_key(False), Evidence.emit_key(True), Evidence.det_key(base)])
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
            return max(variants, key=lambda v: (*_rank(variants[v]), -list(variants).index(v)))

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
    for rec in [inc, *variants.values()]:
        del rec["_table"]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("stage1",))
    ap.add_argument("--workers", type=int, default=WORKERS, help="worker processes (default %d)" % WORKERS)
    args = ap.parse_args(argv)
    return stage1(args.workers)


if __name__ == "__main__":
    sys.exit(main())

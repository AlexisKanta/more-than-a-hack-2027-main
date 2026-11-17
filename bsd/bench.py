"""Leave-one-game-out bench on the Metrica sample games: ``python -m bsd.bench --mode heuristic|learned``.

The only place versions of the method are compared. For each held-out sample game ``g`` and seed
``s``: the other two games are degraded with seed ``s`` to look like BAR-ATH (``bsd.degrade``),
turned into features and paired with their event-derived labels (``bsd.events``); that training
list goes to ``make_emit``; the returned emitter scores the degraded held-out game, whose evidence
is Viterbi-decoded per period and scored against its labels with the official metric.

Degraded features are not cached: degrading and featurising a sample game takes under a second.

``--mode heuristic`` scores the hand-set emitter with its hand-set decoder costs.
``--mode learned`` tunes the decoder (``tune_decode``) for both the heuristic and the learned
emitter (``bsd.model``) on the same held-out evidence and adopts the one with the higher mean
held-out spellF1. Each fold's learned emitter is trained once, on the two other games degraded
with ``TRAIN_SEEDS`` (distinct from the held-out degradation seeds), and its held-out log-scores
are cached; the whole grid is then decoded from them. Training rows are weighted so that the
ball is visible equally often under every label (``learned_emitter``). The adopted mode and
both tuned parameter sets are written to ``bsd/decode_params.json`` (committed; read by
``bsd.run``) and copied to ``cache/decode_params.json``.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bsd.decode import DecodeParams, decode_periods
from bsd.degrade import degrade, gap_profile
from bsd.events import sample_game
from bsd.features import frame_features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

GAMES = (1, 2, 3)
SEEDS = (0, 1)  # held-out degradation seeds
TRAIN_SEEDS = (100, 101, 102)  # degradation seeds of the training copies of the other games
GRID = {
    "switch_ha": (5, 10, 20, 40, 80),
    "alive_to_dead": (5, 10, 20, 40),
    "dead_to_alive": (5, 10, 20, 40),
}
AXES = tuple(GRID)
MODES = ("heuristic", "learned")
OUT_PATH = ROOT / "cache" / "bench_heuristic.csv"
PARAMS_PATH = ROOT / "bsd" / "decode_params.json"
CACHE_PARAMS_PATH = ROOT / "cache" / "decode_params.json"


def score(gt_codes, pred_codes):
    """Official metrics of ``pred_codes`` against ``gt_codes`` (frames where the GT is X are unscored)."""
    r = pm.evaluate(np.asarray(gt_codes, dtype=np.int8), np.asarray(pred_codes, dtype=np.int8))
    return {
        "spell_f1": r["spell_f1_mean"],
        "macro_f1": r["macro_f1"],
        "n_poss": r["n_pred_spells"],
        "mean_poss_s": r["mean_spell_duration_s"],
        "gt_n_poss": r["n_gt_spells"],
        "gt_mean_poss_s": r["gt_mean_spell_duration_s"],
    }


def _degraded(game, profile, seed):
    """(X, labels, names, periods, n_frames) of a sample game after BAR-ATH-like degradation."""
    tr, labels = sample_game(game)
    d = degrade(tr, profile, seed)
    X, names = frame_features(d)
    return X, labels, names, d.periods, len(d.ball)


def bench(make_emit, params, seeds=(0, 1)):
    """One row per (held-out game, seed).

    ``make_emit(train)`` receives ``[(X, labels, names), ...]`` for the two training games and
    returns ``emit(X, names) -> (F, 3)`` log-scores for H, A, D.
    """
    profile = gap_profile()
    rows = []
    for seed in seeds:
        data = {g: _degraded(g, profile, seed) for g in GAMES}
        for g in GAMES:
            train = [data[o][:3] for o in GAMES if o != g]
            emit = make_emit(train)
            X, labels, names, periods, n = data[g]
            pred = decode_periods(emit(X, names), params, periods, n)
            rows.append({"game": g, "seed": seed, **score(labels, pred)})
    return pd.DataFrame(rows)


def prepare(seeds=SEEDS, train_seeds=TRAIN_SEEDS):
    """Degraded sample games: ``held[(game, seed)]`` = (X, labels, names, periods, n) for scoring,
    ``pool[(game, train_seed)]`` = (X, labels, names) for training."""
    profile = gap_profile()
    held = {(g, s): _degraded(g, profile, s) for s in seeds for g in GAMES}
    pool = {(g, s): _degraded(g, profile, s)[:3] for g in GAMES for s in train_seeds}
    return held, pool


def training_data(games=GAMES, seeds=TRAIN_SEEDS):
    """[(X, labels, names)] of every sample game degraded with every seed (the final model's data)."""
    profile = gap_profile()
    return [_degraded(g, profile, s)[:3] for g in games for s in seeds]


def heldout_logp(make_emit, held, pool):
    """Cached held-out evidence, one dict per (seed, game) in that order.

    ``make_emit`` is called once per held-out game with every ``pool`` entry of the other games;
    the emitter it returns scores that game under every held-out seed.
    """
    seeds = sorted({s for _, s in held})
    train_seeds = sorted({s for _, s in pool})
    emits = {g: make_emit([pool[(o, s)] for o in GAMES if o != g for s in train_seeds]) for g in GAMES}
    cases = []
    for s in seeds:
        for g in GAMES:
            X, labels, names, periods, n = held[(g, s)]
            cases.append({"game": g, "seed": s, "logp": emits[g](X, names), "labels": labels,
                          "periods": periods, "n": n})
    return cases


def _score_points(cases, points):
    rows = []
    for vals in points:
        p = DecodeParams(**dict(zip(AXES, vals)))
        for c in cases:
            pred = decode_periods(c["logp"], p, c["periods"], c["n"])
            rows.append({**dict(zip(AXES, vals)), "game": c["game"], "seed": c["seed"],
                         **score(c["labels"], pred)})
    return rows


def _best(rows, points):
    """First point (in evaluation order) with the highest mean spellF1, then mean macroF1."""
    df = pd.DataFrame(rows)
    means = df.groupby(list(AXES), sort=False)[["spell_f1", "macro_f1"]].mean()
    keyed = [(means.loc[v, "spell_f1"], means.loc[v, "macro_f1"], -i) for i, v in enumerate(points)]
    return points[keyed.index(max(keyed))]


def tune_decode(cases, grid=GRID):
    """Grid-search decoder costs on cached held-out evidence (``heldout_logp``).

    Maximises mean held-out spellF1 (ties: mean macroF1, then the earlier grid point). If the
    best value of an axis is on the edge of its grid, that axis is extended once by doubling the
    largest (or halving the smallest) value and the search is repeated over the extended grid;
    points already decoded are not decoded again. Returns ``(DecodeParams, rows)`` with one row
    per (grid point, case).
    """
    points = list(itertools.product(*(tuple(grid[a]) for a in AXES)))
    rows = _score_points(cases, points)
    best = _best(rows, points)
    ext = {}
    for a, v in zip(AXES, best):
        vals = sorted(grid[a])
        if len(vals) > 1 and v == vals[-1]:
            vals.append(vals[-1] * 2)
        elif len(vals) > 1 and v == vals[0]:
            vals.insert(0, vals[0] / 2)
        ext[a] = tuple(vals)
    if any(len(ext[a]) != len(grid[a]) for a in AXES):
        done = set(points)
        new = [v for v in itertools.product(*(ext[a] for a in AXES)) if v not in done]
        rows += _score_points(cases, new)
        points += new
        best = _best(rows, points)
    return DecodeParams(**dict(zip(AXES, best))), pd.DataFrame(rows)


def _at(grid_df, p):
    sel = np.ones(len(grid_df), dtype=bool)
    for a in AXES:
        sel &= grid_df[a].to_numpy() == getattr(p, a)
    return grid_df[sel].drop(columns=list(AXES)).reset_index(drop=True)


def _print_table(df):
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


def learned_emitter(train):
    """``bsd.model`` emitter fitted on ``[(X, labels, names)]``, with ball visibility weighted to
    BAR-ATH's in-period visible rate so that a missing ball is no dead-ball cue."""
    from bsd.model import train_emitter

    visible_rate = 1.0 - float(gap_profile()["missing_rate"])
    return train_emitter([t[0] for t in train], [t[1] for t in train], train[0][2],
                         visible_rate=visible_rate)


def tune_all():
    """Tune both emitters on the same held-out data; save and return the adoption record."""
    from bsd.heuristic import heuristic_logp

    held, pool = prepare()
    makers = {"heuristic": lambda train: heuristic_logp, "learned": lambda train: learned_emitter(train).emit}
    record = {}
    for mode in MODES:
        print("\n=== %s ===" % mode, flush=True)
        cases = heldout_logp(makers[mode], held, pool)
        p, grid_df = tune_decode(cases)
        table = _at(grid_df, p)
        grid_df.to_csv(ROOT / "cache" / ("grid_%s.csv" % mode), index=False)
        table.to_csv(ROOT / "cache" / ("bench_tuned_%s.csv" % mode), index=False)
        print("tuned %s" % (p,))
        _print_table(table)
        means = table[["spell_f1", "macro_f1", "n_poss", "mean_poss_s", "gt_n_poss", "gt_mean_poss_s"]].mean()
        print("mean spell_f1: %.4f   mean macro_f1: %.4f" % (means["spell_f1"], means["macro_f1"]))
        record[mode] = {**{a: float(getattr(p, a)) for a in AXES},
                        **{"bench_" + k: round(float(v), 4) for k, v in means.items()}}
        del cases
    adopted = max(MODES, key=lambda m: record[m]["bench_spell_f1"])  # tie -> heuristic (listed first)
    out = {"adopted": adopted, **record}
    text = json.dumps(out, indent=2) + "\n"
    PARAMS_PATH.write_text(text)
    CACHE_PARAMS_PATH.parent.mkdir(exist_ok=True)
    CACHE_PARAMS_PATH.write_text(text)
    print("\nadopted: %s" % adopted)
    print(text, end="")
    print("saved %s and %s" % (PARAMS_PATH.relative_to(ROOT), CACHE_PARAMS_PATH.relative_to(ROOT)))
    return out


def load_decode_params(path=PARAMS_PATH):
    """(adopted mode, {mode: DecodeParams}) from the committed tuning record."""
    rec = json.loads(Path(path).read_text())
    return rec["adopted"], {m: DecodeParams(**{a: rec[m][a] for a in AXES}) for m in MODES}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("heuristic", "learned"), default="heuristic")
    args = ap.parse_args(argv)
    if args.mode == "learned":
        tune_all()
        return 0
    from bsd.heuristic import HEURISTIC_PARAMS, heuristic_logp

    df = bench(lambda train: heuristic_logp, HEURISTIC_PARAMS)
    _print_table(df)
    print("mean spell_f1: %.4f" % df["spell_f1"].mean())
    print("mean macro_f1: %.4f" % df["macro_f1"].mean())
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("saved %s" % OUT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

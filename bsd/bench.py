"""Leave-one-game-out bench on the Metrica sample games: ``python -m bsd.bench --mode heuristic``.

The only place versions of the method are compared. For each held-out sample game ``g`` and seed
``s``: the other two games are degraded with seed ``s`` to look like BAR-ATH (``bsd.degrade``),
turned into features and paired with their event-derived labels (``bsd.events``); that training
list goes to ``make_emit``; the returned emitter scores the degraded held-out game, whose evidence
is Viterbi-decoded per period and scored against its labels with the official metric.

Degraded features are not cached: degrading and featurising a sample game takes under a second.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bsd.decode import decode_periods
from bsd.degrade import degrade, gap_profile
from bsd.events import sample_game
from bsd.features import frame_features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

GAMES = (1, 2, 3)
OUT_PATH = ROOT / "cache" / "bench_heuristic.csv"


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("heuristic", "learned"), default="heuristic")
    args = ap.parse_args(argv)
    if args.mode == "learned":
        sys.exit("--mode learned is not implemented yet (added in Task 9); use --mode heuristic")
    from bsd.heuristic import HEURISTIC_PARAMS, heuristic_logp

    df = bench(lambda train: heuristic_logp, HEURISTIC_PARAMS)
    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print("mean spell_f1: %.4f" % df["spell_f1"].mean())
    print("mean macro_f1: %.4f" % df["macro_f1"].mean())
    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("saved %s" % OUT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

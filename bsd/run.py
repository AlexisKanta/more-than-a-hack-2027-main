"""End-to-end BAR-ATH labelling: ``python -m bsd.run [--mode heuristic|learned] [--no-tagging]``.

Builds tracking -> features -> per-frame evidence -> Viterbi labels, writes
``evaluation/results/BSD.csv``, runs the official validator on it (non-zero exit if it fails)
and prints label-free diagnostics. With ``--check-sha <hex>`` it also exits non-zero unless the
written file's SHA-256 equals ``<hex>`` (the reproducibility check in REPRODUCE.md). Run from the
repository root.

The decoder costs of each mode, and the default mode (the one that won the sample-game bench),
come from the committed ``bsd/decode_params.json``, written by ``python -m bsd.experiments
stage1``. ``--mode learned`` trains the emitter on all three degraded sample games
(``bsd.bench.training_data``) before scoring BAR-ATH. If the learned entry has a ``handover`` block
the emitter also gets the hand-over features and, when ``detector`` is true, a change detector
trained on the same data sets a per-frame H<->A switch cost (``bsd.decode.switch_cost_from``).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "evaluation" / "results" / "BSD.csv"
VALIDATOR = ROOT / "evaluation" / "validate_submission.py"
GAME = "BAR-ATH"
DIAG_MINUTES = 5


def _spells(codes, periods):
    """[(start, stop, code)] maximal H or A runs inside each inclusive period."""
    out = []
    for _, first, last in periods:
        seg = codes[first:last + 1]
        cut = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate(([0], cut))
        stops = np.concatenate((cut, [len(seg)]))
        out += [(first + a, first + b, int(seg[a])) for a, b in zip(starts, stops) if seg[a] in (0, 1)]
    return out


def _mix(seg):
    n = max(len(seg), 1)
    return "  ".join("%s %5.1f%%" % (s, 100.0 * np.sum(seg == k) / n) for k, s in enumerate("HADX"))


def check_sha(actual, expected):
    """Exit non-zero unless hex digest ``actual`` equals ``expected`` (case/whitespace ignored)."""
    if actual.strip().lower() != expected.strip().lower():
        sys.exit("SHA-256 mismatch: wrote %s, expected %s" % (actual, expected.strip()))


def summarize(codes, periods, blocks, fps):
    spells = _spells(codes, periods)
    lengths = np.array([b - a for a, b, _ in spells], dtype=float)
    print("possessions: %d" % len(spells))
    print("mean possession: %.2f s" % (lengths.mean() / fps if len(lengths) else 0.0))
    for half, first, last in periods:
        seg = codes[first:last + 1]
        print("half %d: D share %.1f%%  (%s)" % (half, 100.0 * np.mean(seg == 2), _mix(seg)))
    half, first, last = blocks[1]
    n = DIAG_MINUTES * 60 * fps
    seg = codes[first:min(first + n, last + 1)]
    print("diagnostic, first %d min of half %d graded block: %s" % (DIAG_MINUTES, half, _mix(seg)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sys.path.insert(0, str(ROOT))
    from bsd.bench import load_decode_params, load_handover_config

    adopted, tuned = load_decode_params()
    ap.add_argument("--mode", choices=("heuristic", "learned"), default=adopted,
                    help="evidence model (default: %(default)s, the bench winner)")
    ap.add_argument("--no-tagging", action="store_true", help="do not use Smart Tagging")
    ap.add_argument("--check-sha", metavar="HEX",
                    help="exit non-zero unless the written file has this SHA-256")
    args = ap.parse_args(argv)
    params = tuned[args.mode]
    ho = load_handover_config(mode=args.mode) if args.mode == "learned" else None
    print("mode: %s  decode: %s" % (args.mode, params), flush=True)
    print("hand-over detector: %s" % (ho if ho else "off"), flush=True)
    print("tagging: %s" % ("off (--no-tagging)" if args.no_tagging else "on"), flush=True)

    import games
    from bsd.decode import decode_periods, switch_cost_from
    from bsd.features import frame_features
    from bsd.io_epts import bar_ath_tracking
    from bsd.output import write_submission

    tr = bar_ath_tracking()
    X, names = frame_features(tr, handover=bool(ho and ho["features"]))
    switch_cost = None
    if args.mode == "learned":
        from bsd.bench import learned_detector, learned_emitter, training_data

        print("training emitter on the degraded sample games ...", flush=True)
        train = training_data(handover=bool(ho and ho["features"]), meta=bool(ho and ho["detector"]))
        logp = learned_emitter(train).emit(X, names)
        if ho and ho["detector"]:
            print("training hand-over detector ...", flush=True)
            det = learned_detector([(t[0], t[1], t[3], t[5]) for t in train], names,
                                   ho["k_s"], ho["through_dead_s"])
            switch_cost = switch_cost_from(det.p_change(X, names), params.switch_ha,
                                           ho["alpha"], ho["floor"])
        del train
    else:
        from bsd.heuristic import heuristic_logp

        logp = heuristic_logp(X, names)
    if not args.no_tagging:
        from bsd.tagging import tag_bonus

        logp = logp + tag_bonus(len(tr.ball), int(tr.fps), tr.periods)
        assert np.isfinite(logp).all()
    codes = decode_periods(logp, params, tr.periods, len(tr.ball), switch_cost=switch_cost)
    sha = write_submission(codes, OUT_PATH)

    res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_PATH)], cwd=str(ROOT))
    if res.returncode != 0:
        sys.exit("validator failed on %s (exit %d)" % (OUT_PATH, res.returncode))

    print("\nwrote %s" % OUT_PATH.relative_to(ROOT))
    print("SHA-256: %s" % sha)
    summarize(codes, tr.periods, games.half_video_blocks(GAME), int(tr.fps))
    if args.check_sha:
        check_sha(sha, args.check_sha)
        print("SHA-256 matches --check-sha")
    return 0


if __name__ == "__main__":
    sys.exit(main())

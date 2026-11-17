"""End-to-end BAR-ATH labelling: ``python -m bsd.run --mode heuristic [--no-tagging]``.

Builds tracking -> features -> per-frame evidence -> Viterbi labels, writes
``evaluation/results/BSD.csv``, runs the official validator on it (non-zero exit if it fails)
and prints label-free diagnostics. Run from the repository root.
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
    ap.add_argument("--mode", choices=("heuristic", "learned"), default="heuristic")
    ap.add_argument("--no-tagging", action="store_true", help="do not use Smart Tagging")
    args = ap.parse_args(argv)
    if args.mode == "learned":
        sys.exit("--mode learned is not implemented yet (added in Task 9); use --mode heuristic")
    print("tagging: off (%s)" % ("--no-tagging" if args.no_tagging else "not implemented yet"), flush=True)

    sys.path.insert(0, str(ROOT))
    import games
    from bsd.decode import decode_periods
    from bsd.features import frame_features
    from bsd.heuristic import HEURISTIC_PARAMS, heuristic_logp
    from bsd.io_epts import bar_ath_tracking
    from bsd.output import write_submission

    tr = bar_ath_tracking()
    X, names = frame_features(tr)
    logp = heuristic_logp(X, names)
    codes = decode_periods(logp, HEURISTIC_PARAMS, tr.periods, len(tr.ball))
    sha = write_submission(codes, OUT_PATH)

    res = subprocess.run([sys.executable, str(VALIDATOR), str(OUT_PATH)], cwd=str(ROOT))
    if res.returncode != 0:
        sys.exit("validator failed on %s (exit %d)" % (OUT_PATH, res.returncode))

    print("\nwrote %s" % OUT_PATH.relative_to(ROOT))
    print("SHA-256: %s" % sha)
    summarize(codes, tr.periods, games.half_video_blocks(GAME), int(tr.fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())

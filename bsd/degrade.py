"""Degrade clean sample-game tracking so it looks like BAR-ATH's automatic tracking.

A model trained on the sample games is applied to BAR-ATH, so the sample tracking is made to
share BAR-ATH's defects -- and, crucially, *not* the sample games' own artefact that the ball
is NaN mostly when it is dead (which would teach "no ball => dead ball", false on BAR-ATH,
where a missing ball is mostly occlusion). ``degrade`` therefore

1. fills the sample ball across all its gaps (``fill_ball``), erasing its dead-ball NaN pattern;
2. hides the ball with a new mask of alternating visible / missing runs resampled from BAR-ATH's
   own run lengths -- built without looking at labels or at the source ball;
3. drops player observations down to a per-frame count resampled from BAR-ATH;
4. adds smooth Gaussian position jitter;
5. reassigns identities by permuting each team's slot columns in blocks of 10-60 s
   (tracklet identity swaps; the feature code treats the implied jumps as identity jumps).

Every random draw comes from ``np.random.default_rng(seed)``; inputs are never mutated.
"""
import dataclasses
from pathlib import Path

import numpy as np

from bsd.tracking import Tracking

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "gap_profile.npz"

RATE_TOL = 0.03  # degraded in-period missing-ball rate must be within this of BAR-ATH's
MASK_TRIES = 200  # whole-mask redraws before falling back to rescaling the runs
DROP_BLOCK_S = (1, 5)  # player-count blocks: the sampled count is constant for 1-5 s
SWAP_BLOCK_S = (10, 60)  # identity-reassignment blocks per team
JITTER_SIGMA = 0.3  # metres, players and ball
JITTER_KNOT_S = 1.0  # jitter is linear between independent Gaussian knots this far apart


def _period_mask(tr):
    inside = np.zeros(len(tr.ball), dtype=bool)
    for _, a, b in tr.periods:
        inside[a:b + 1] = True
    return inside


def _runs(flag):
    """Run lengths of True and of False in a 1-D bool array."""
    if len(flag) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    edges = np.flatnonzero(np.diff(flag.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    lengths = np.diff(np.r_[starts, len(flag)])
    values = flag[starts]
    return lengths[values], lengths[~values]


def _profile_from(tr):
    gaps, vis, n_tracked = [], [], []
    for _, a, b in tr.periods:
        miss = np.isnan(tr.ball[a:b + 1]).any(axis=1)
        g, v = _runs(miss)
        gaps.append(g)
        vis.append(v)
        n_tracked.append(np.isfinite(tr.players[a:b + 1]).all(axis=2).sum(axis=1))
    gaps, vis = np.concatenate(gaps), np.concatenate(vis)
    return {
        "gap_lengths": gaps.astype(np.int64),
        "visible_lengths": vis.astype(np.int64),
        "missing_rate": float(gaps.sum() / max(1, gaps.sum() + vis.sum())),
        "n_tracked": np.concatenate(n_tracked).astype(np.int16),
    }


def gap_profile(tr=None):
    """Label-free BAR-ATH tracking defects, as a dict.

    ``gap_lengths`` / ``visible_lengths``: missing / visible ball run lengths inside periods;
    ``missing_rate``: in-period fraction of missing-ball frames; ``n_tracked``: per in-period
    frame, the number of tracked players. With ``tr`` given the profile is computed from it;
    with ``tr=None`` it is BAR-ATH's, cached in ``cache/gap_profile.npz``.
    """
    if tr is not None:
        return _profile_from(tr)
    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        return {"gap_lengths": z["gap_lengths"], "visible_lengths": z["visible_lengths"],
                "missing_rate": float(z["missing_rate"]), "n_tracked": z["n_tracked"]}
    from bsd.io_epts import bar_ath_tracking

    p = _profile_from(bar_ath_tracking())
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.savez(CACHE_PATH, **p)
    return p


def fill_ball(tr):
    """New Tracking whose ball is linearly interpolated across gaps within each period.

    Leading / trailing gaps of a period hold the nearest valid position; a period without any
    valid ball gets the pitch centre. Frames outside periods are left as they are.
    """
    ball = tr.ball.copy()
    centre = np.array([tr.pitch[0] / 2, tr.pitch[1] / 2], dtype=ball.dtype)
    for _, a, b in tr.periods:
        seg = ball[a:b + 1]
        ok = np.isfinite(seg).all(axis=1)
        if not ok.any():
            seg[:] = centre
            continue
        idx = np.arange(len(seg))
        for c in range(2):
            seg[:, c] = np.interp(idx, idx[ok], seg[ok, c])
    return dataclasses.replace(tr, ball=ball)


def _draw_runs(n, vis_lengths, gap_lengths, rate, rng):
    """Alternating run lengths covering ``n`` frames: (lengths, first_run_is_missing)."""
    first_missing = bool(rng.random() < rate)
    mean_cycle = vis_lengths.mean() + gap_lengths.mean()
    lengths, total, missing = [], 0, first_missing
    first = True
    while total < n:
        k = int(2 * n / mean_cycle) + 4
        g = rng.choice(gap_lengths, k)
        v = rng.choice(vis_lengths, k)
        for i in range(k):
            run = int(g[i] if missing else v[i])
            if first:  # start part-way into a run (random phase)
                run = int(rng.integers(1, run + 1))
                first = False
            lengths.append(run)
            total += run
            missing = not missing
            if total >= n:
                break
    lengths = np.array(lengths, dtype=np.int64)
    lengths[-1] -= total - n
    return lengths, first_missing


def _missing_in(lengths, first_missing):
    start = 0 if first_missing else 1
    return int(lengths[start::2].sum())


def _rescale_runs(lengths, first_missing, rate, rng):
    """Stretch missing runs (visible runs fixed) so the period's missing fraction is ``rate``."""
    n = int(lengths.sum())
    is_miss = np.zeros(len(lengths), dtype=bool)
    is_miss[0 if first_missing else 1::2] = True
    m, v = lengths[is_miss].sum(), lengths[~is_miss].sum()
    if rate <= 0 or rate >= 1 or m == 0 or v == 0:
        # degenerate (single run): one missing run of the right length at a random place
        n_miss = int(round(min(max(rate, 0.0), 1.0) * n))
        start = int(rng.integers(0, n - n_miss + 1))
        mask = np.zeros(n, dtype=bool)
        mask[start:start + n_miss] = True
        return mask
    scaled = np.where(is_miss, lengths * (rate * v / ((1 - rate) * m)), lengths).astype(float)
    bounds = np.round(np.cumsum(scaled) * n / scaled.sum()).astype(np.int64)
    return np.repeat(is_miss, np.diff(np.r_[0, bounds]))


def _ball_mask(tr, profile, rng):
    """In-period missing-ball mask from BAR-ATH run lengths; never sees labels or the source ball."""
    gaps = np.asarray(profile["gap_lengths"], dtype=np.int64)
    vis = np.asarray(profile["visible_lengths"], dtype=np.int64)
    rate = float(profile["missing_rate"])
    sizes = [b - a + 1 for _, a, b in tr.periods]
    total = sum(sizes)
    best, best_err = None, np.inf
    for _ in range(MASK_TRIES):
        draws = [_draw_runs(n, vis, gaps, rate, rng) for n in sizes]
        err = abs(sum(_missing_in(*d) for d in draws) / max(1, total) - rate)
        if err < best_err:
            best, best_err = draws, err
        if err <= RATE_TOL:
            break
    mask = np.zeros(len(tr.ball), dtype=bool)
    for (_, a, b), (lengths, first_missing) in zip(tr.periods, best):
        if best_err <= RATE_TOL:
            states = np.zeros(len(lengths), dtype=bool)
            states[0 if first_missing else 1::2] = True
            mask[a:b + 1] = np.repeat(states, lengths)
        else:
            mask[a:b + 1] = _rescale_runs(lengths, first_missing, rate, rng)
    return mask


def _blocks(n, lo, hi, rng):
    """Block id per frame for consecutive blocks of uniform random length in [lo, hi] frames."""
    lens = rng.integers(lo, hi + 1, n // lo + 1)
    return np.repeat(np.arange(len(lens)), lens)[:n]


def _drop_players(players, n_tracked, fps, rng):
    """Keep, per frame, a random subset of observed players of a size resampled from BAR-ATH.

    Count and subset preference are held over 1-5 s blocks so tracks do not flicker.
    """
    n, s = players.shape[:2]
    block = _blocks(n, DROP_BLOCK_S[0] * fps, DROP_BLOCK_S[1] * fps, rng)
    nb = int(block[-1]) + 1 if n else 0
    target = rng.choice(np.asarray(n_tracked), nb)[block]
    keys = rng.random((nb, s), dtype=np.float32)[block]
    observed = np.isfinite(players).all(axis=2)
    keys[~observed] = np.inf
    rank = np.argsort(np.argsort(keys, axis=1, kind="stable"), axis=1, kind="stable")
    drop = ~(observed & (rank < target[:, None]))
    players[drop] = np.nan


def _smooth_noise(n, shape, fps, rng):
    """(n, *shape) Gaussian noise, linear between independent knots, mean variance JITTER_SIGMA**2.

    Per-frame i.i.d. jitter of 0.3 m would add ~10 m/s of fake speed (BAR-ATH tracks are smooth
    frame to frame), so the jitter is a slowly drifting offset instead.
    """
    k = max(1, int(round(JITTER_KNOT_S * fps)))
    t = (np.arange(n) % k) / k
    scale = JITTER_SIGMA / np.sqrt(np.mean((1 - t[:k]) ** 2 + t[:k] ** 2))
    knots = rng.normal(0.0, scale, (n // k + 2,) + tuple(shape)).astype(np.float32)
    i = np.arange(n) // k
    w = t.astype(np.float32).reshape((n,) + (1,) * len(shape))
    return knots[i] * (1 - w) + knots[i + 1] * w


def _swap_slots(players, team, fps, rng):
    """Permute each team's columns in random blocks of 10-60 s (tracklet identity swaps)."""
    out = players.copy()
    n = len(players)
    for t in np.unique(team):
        cols = np.flatnonzero(team == t)
        block = _blocks(n, SWAP_BLOCK_S[0] * fps, SWAP_BLOCK_S[1] * fps, rng)
        starts = np.flatnonzero(np.r_[True, np.diff(block) != 0])
        for a, b in zip(starts, np.r_[starts[1:], n]):
            out[a:b, cols] = players[a:b, cols[rng.permutation(len(cols))]]
    return out


def degrade(tr, profile, seed):
    """New Tracking: ``tr`` with BAR-ATH-like ball gaps, player drop-outs, jitter and id swaps."""
    rng = np.random.default_rng(seed)
    fps = int(tr.fps)
    n = len(tr.ball)
    ball = fill_ball(tr).ball
    missing = _ball_mask(tr, profile, rng) | ~_period_mask(tr)
    players = tr.players.copy()
    _drop_players(players, profile["n_tracked"], fps, rng)
    players += _smooth_noise(n, players.shape[1:], fps, rng)  # NaN stays NaN
    ball = ball + _smooth_noise(n, (2,), fps, rng)
    ball[missing] = np.nan
    players = _swap_slots(players, tr.team, fps, rng)
    return Tracking(players, tr.team.copy(), ball.astype(np.float32), list(tr.periods),
                    tr.pitch, tr.fps)

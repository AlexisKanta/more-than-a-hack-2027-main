"""Viterbi decoding of per-frame H/A/D evidence into stable label runs."""
from dataclasses import dataclass

import numpy as np

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class DecodeParams:
    switch_ha: float      # cost of H<->A directly
    alive_to_dead: float  # H/A -> D
    dead_to_alive: float  # D -> H/A


def _cost_matrix(p):
    c = np.zeros((3, 3))  # c[prev, cur]
    c[H, A] = c[A, H] = p.switch_ha
    c[H, D] = c[A, D] = p.alive_to_dead
    c[D, H] = c[D, A] = p.dead_to_alive
    return c.tolist()


def viterbi(logp, p):
    """Max-score path of Σ logp[t, s_t] − Σ transition costs; (F,3) -> (F,) int8."""
    logp = np.asarray(logp, dtype=float)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (F, 3)")
    if not np.isfinite(logp).all():
        raise ValueError("logp must be finite")
    n = len(logp)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    rows = np.round(logp, 6).tolist()
    c = _cost_matrix(p)
    back = np.zeros((n, 3), dtype=np.int8)
    score = rows[0]
    for t in range(1, n):
        new = [0.0, 0.0, 0.0]
        bt = back[t]
        for s in range(3):
            best, arg = score[0] - c[0][s], 0
            for q in (1, 2):
                v = score[q] - c[q][s]
                if v > best:  # strict: ties go to the lower state index
                    best, arg = v, q
            new[s] = best + rows[t][s]
            bt[s] = arg
        score = new
    out = np.zeros(n, dtype=np.int8)
    state = int(np.argmax(score))
    for t in range(n - 1, -1, -1):
        out[t] = state
        state = back[t, state]
    return out


def decode_periods(logp, p, periods, n_frames):
    """Decode each (period, first, last) inclusive range independently; X elsewhere."""
    out = np.full(n_frames, X, dtype=np.int8)
    for _, first, last in periods:
        out[first:last + 1] = viterbi(logp[first:last + 1], p)
    return out

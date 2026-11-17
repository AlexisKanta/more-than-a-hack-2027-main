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


def switch_cost_from(p_change, switch_ha, alpha, floor):
    """Per-frame H<->A cost ``max(floor, switch_ha - alpha*logit(p))``; NaN p gives ``switch_ha``."""
    p = np.asarray(p_change, dtype=float)
    if alpha == 0:
        return np.full(len(p), float(switch_ha))
    q = np.clip(np.where(np.isnan(p), 0.5, p), 1e-4, 1 - 1e-4)
    cost = np.maximum(floor, switch_ha - alpha * np.log(q / (1 - q)))
    return np.round(np.where(np.isnan(p), float(switch_ha), cost), 6)


def viterbi(logp, p, switch_cost=None):
    """Max-score path of Σ logp[t, s_t] − Σ transition costs; (F,3) -> (F,) int8.

    ``switch_cost[t]`` (optional) replaces ``p.switch_ha`` for the H<->A move from t-1 into t.
    """
    logp = np.asarray(logp, dtype=float)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (F, 3)")
    if not np.isfinite(logp).all():
        raise ValueError("logp must be finite")
    n = len(logp)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    sc = None
    if switch_cost is not None:
        switch_cost = np.asarray(switch_cost, dtype=float)
        if switch_cost.shape != (n,):
            raise ValueError("switch_cost must have shape (F,)")
        if not np.isfinite(switch_cost).all():
            raise ValueError("switch_cost must be finite")
        sc = np.round(switch_cost, 6).tolist()
    rows = np.round(logp, 6).tolist()
    c = _cost_matrix(p)
    back = np.zeros((n, 3), dtype=np.int8)
    score = rows[0]
    for t in range(1, n):
        if sc is not None:
            c[0][1] = c[1][0] = sc[t]
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


def decode_periods(logp, p, periods, n_frames, switch_cost=None):
    """Decode each (period, first, last) inclusive range independently; X elsewhere."""
    if switch_cost is not None and len(switch_cost) != n_frames:
        raise ValueError("switch_cost must have length n_frames")
    out = np.full(n_frames, X, dtype=np.int8)
    for _, first, last in periods:
        sc = None if switch_cost is None else np.asarray(switch_cost)[first:last + 1]
        out[first:last + 1] = viterbi(logp[first:last + 1], p, sc)
    return out

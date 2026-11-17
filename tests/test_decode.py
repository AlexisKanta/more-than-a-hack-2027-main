import time

import numpy as np
import pytest

from bsd.decode import DecodeParams, decode_periods, viterbi

P = DecodeParams(switch_ha=20.0, alive_to_dead=5.0, dead_to_alive=5.0)


def _ev(n, state, margin=0.5):
    out = np.zeros((n, 3))
    out[:, state] = margin
    return out


def test_constant_evidence_single_state():
    for s in range(3):
        assert (viterbi(_ev(300, s), P) == s).all()


def test_short_flicker_ignored():
    logp = np.vstack([_ev(500, 0), _ev(10, 1), _ev(500, 0)])
    assert (viterbi(logp, P) == 0).all()


def test_sustained_switch():
    logp = np.vstack([_ev(500, 0), _ev(200, 1)])
    out = viterbi(logp, P)
    change = int(np.argmax(out == 1))
    assert out[change:].tolist() == [1] * (700 - change)
    assert abs(change - 500) <= 50


def test_periods_independent():
    logp = np.vstack([_ev(100, 0), _ev(100, 1)])
    out = decode_periods(logp, P, [(1, 0, 99), (2, 100, 199)], 250)
    assert out.dtype == np.int8 and len(out) == 250
    assert (out[:100] == 0).all() and (out[100:200] == 1).all()
    assert (out[200:] == 3).all()


def test_deterministic_ties():
    assert (viterbi(np.zeros((50, 3)), P) == 0).all()


def test_nonfinite_raises():
    bad = np.zeros((5, 3))
    bad[2, 1] = np.nan
    with pytest.raises(ValueError):
        viterbi(bad, P)


def test_speed_80k():
    rng = np.random.default_rng(0)
    logp = rng.normal(size=(80000, 3))
    t = time.time()
    viterbi(logp, P)
    assert time.time() - t < 5

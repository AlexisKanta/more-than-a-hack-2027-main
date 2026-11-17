import numpy as np

from bsd.handover import change_labels

H, A, D, X = 0, 1, 2, 3


def _lab(*runs):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def test_change_labels_basic():
    lab = _lab((H, 20), (A, 20))
    out = change_labels(lab, [(1, 0, 39)], fps=10, k_s=0.5)
    assert out.dtype == np.int8
    assert (out[15:25] == 1).all()
    assert (out[:15] == 0).all() and (out[25:] == 0).all()


def test_change_labels_ignore_period_boundary():
    lab = _lab((H, 20), (A, 20))
    out = change_labels(lab, [(1, 0, 19), (2, 20, 39)], fps=10)
    assert not (out == 1).any()


def test_change_labels_through_dead():
    lab = _lab((H, 20), (D, 5), (A, 20))
    per = [(1, 0, 44)]
    assert not (change_labels(lab, per, fps=10) == 1).any()
    out = change_labels(lab, per, fps=10, through_dead_s=1.0)
    assert out[25] == 1 and out[20] == 1 and out[0] == 0 and out[44] == 0
    assert not (change_labels(lab, per, fps=10, through_dead_s=0.2) == 1).any()


def test_change_labels_x_unscored():
    lab = _lab((H, 20), (X, 5), (A, 20))
    out = change_labels(lab, [(1, 0, 40)], fps=10)
    assert (out[20:25] == -1).all()
    assert out[41:].tolist() == [-1] * (len(lab) - 41)  # outside periods

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


def _synthetic(n=900, seed=0):
    from bsd.features import mirror  # noqa: F401
    rng = np.random.default_rng(seed)
    names = ["a_home", "a_away", "d_diff", "ball_visible"]
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[:, 3] = (rng.random(n) > 0.3)
    y = (X[:, 0] + 0.5 * X[:, 2] > 0.8).astype(np.int8)
    return X, y, names


def test_detector_mirror_symmetric():
    import pytest
    from bsd.features import mirror
    from bsd.handover import train_detector
    X, y, names = _synthetic()
    det = train_detector([X], [y], names, visible_rate=0.6)
    p = det.p_change(X, names)
    assert p.shape == (len(X),) and p.dtype == np.float64
    pm = det.p_change(mirror(X, names), names)
    assert np.abs(p - pm).max() < 1e-9
    with pytest.raises(ValueError):
        det.p_change(X, names[::-1])


def test_detector_drops_unscored_rows():
    from bsd.handover import train_detector
    X, y, names = _synthetic()
    y = y.copy()
    y[::2] = -1
    assert train_detector([X], [y], names).names == tuple(names)


def test_detector_needs_both_classes():
    import pytest
    from bsd.handover import train_detector
    X, y, names = _synthetic()
    with pytest.raises(ValueError):
        train_detector([X], [np.zeros_like(y)], names)
